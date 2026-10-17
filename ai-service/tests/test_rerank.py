"""Tests for cross-encoder re-ranking and the two-stage retrieve (Issue #21).

Covers:
- _sigmoid: numeric stability and range
- CrossEncoderReranker: reordering, sigmoid scores, top_n, empty input,
  documents passed to the model, score-count mismatch, error wrapping,
  payload preservation, tie stability
- CrossEncoderReranker.available: import probing
- get_reranker: disabled setting, missing fastembed, caching
- retrieve: candidate pool sizing, re-rank application, fail-open paths,
  store error propagation, filter forwarding, settings defaults

The fastembed model is never loaded: a fake model object is injected, and
hybrid_search is mocked where the retrieval pipeline is under test.
"""

from __future__ import annotations

import math
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import settings
from app.retrieval.rerank import (
    CrossEncoderReranker,
    RerankError,
    _sigmoid,
    get_reranker,
    retrieve,
)
from app.retrieval.vector_store import SearchResult, VectorStoreError


def _make_result(chunk_id: str, content: str = "") -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        score=0.5,
        payload={"chunk_id": chunk_id, "content": content},
        content=content or f"content-{chunk_id}",
    )


class _FakeModel:
    """Stand-in for fastembed's TextCrossEncoder."""

    def __init__(self, scores: list[float] | Exception) -> None:
        self._scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query: str, documents):
        self.calls.append((query, list(documents)))
        if isinstance(self._scores, Exception):
            raise self._scores
        return iter(self._scores)


@pytest.fixture(autouse=True)
def _clear_reranker_cache():
    get_reranker.cache_clear()
    yield
    get_reranker.cache_clear()


def _make_reranker(scores: list[float] | Exception) -> tuple[CrossEncoderReranker, _FakeModel]:
    reranker = CrossEncoderReranker(model_name="fake-model")
    model = _FakeModel(scores)
    reranker._model = model
    return reranker, model


# ===========================================================================
# _sigmoid
# ===========================================================================


class TestSigmoid:
    def test_zero_maps_to_half(self):
        assert _sigmoid(0.0) == pytest.approx(0.5)

    def test_large_positive_saturates_at_one(self):
        assert _sigmoid(1000.0) == pytest.approx(1.0)

    def test_large_negative_saturates_at_zero(self):
        assert _sigmoid(-1000.0) == pytest.approx(0.0)

    def test_monotonic(self):
        assert _sigmoid(-11.4) < _sigmoid(0.0) < _sigmoid(9.8)

    def test_symmetry(self):
        assert _sigmoid(2.0) + _sigmoid(-2.0) == pytest.approx(1.0)

    def test_matches_reference_formula(self):
        for x in (-5.0, -0.5, 0.0, 1.5, 6.0):
            assert _sigmoid(x) == pytest.approx(1.0 / (1.0 + math.exp(-x)))


# ===========================================================================
# CrossEncoderReranker
# ===========================================================================


class TestCrossEncoderReranker:
    @pytest.mark.asyncio
    async def test_reorders_by_descending_score(self):
        results = [_make_result("a"), _make_result("b"), _make_result("c")]
        reranker, _ = _make_reranker([0.1, 5.0, 2.0])
        out = await reranker.rerank("query", results)
        assert [r.chunk_id for r in out] == ["b", "c", "a"]

    @pytest.mark.asyncio
    async def test_scores_are_sigmoid_of_logits(self):
        results = [_make_result("a")]
        reranker, _ = _make_reranker([6.0])
        out = await reranker.rerank("query", results)
        assert out[0].score == pytest.approx(1.0 / (1.0 + math.exp(-6.0)))

    @pytest.mark.asyncio
    async def test_scores_stay_within_zero_and_one(self):
        results = [_make_result("a"), _make_result("b")]
        reranker, _ = _make_reranker([1000.0, -1000.0])
        out = await reranker.rerank("query", results)
        assert all(0.0 <= r.score <= 1.0 for r in out)

    @pytest.mark.asyncio
    async def test_top_n_trims(self):
        results = [_make_result(c) for c in "abcde"]
        reranker, _ = _make_reranker([1.0, 2.0, 3.0, 4.0, 5.0])
        out = await reranker.rerank("query", results, top_n=2)
        assert [r.chunk_id for r in out] == ["e", "d"]

    @pytest.mark.asyncio
    async def test_empty_results_return_empty_without_model_use(self):
        reranker = CrossEncoderReranker(model_name="fake-model")
        assert await reranker.rerank("query", []) == []
        assert reranker._model is None  # model never loaded

    @pytest.mark.asyncio
    async def test_documents_passed_to_model_in_order(self):
        results = [
            _make_result("a", content="passage A"),
            _make_result("b", content="passage B"),
        ]
        reranker, model = _make_reranker([0.1, 0.2])
        await reranker.rerank("my query", results)
        assert model.calls == [("my query", ["passage A", "passage B"])]

    @pytest.mark.asyncio
    async def test_score_count_mismatch_raises(self):
        results = [_make_result("a"), _make_result("b")]
        reranker, _ = _make_reranker([0.5])
        with pytest.raises(RerankError, match="2 documents"):
            await reranker.rerank("query", results)

    @pytest.mark.asyncio
    async def test_model_error_wrapped_in_rerank_error(self):
        results = [_make_result("a")]
        reranker, _ = _make_reranker(RuntimeError("onnx exploded"))
        with pytest.raises(RerankError, match="onnx exploded"):
            await reranker.rerank("query", results)

    @pytest.mark.asyncio
    async def test_payload_and_content_preserved(self):
        original = _make_result("a", content="text")
        reranker, _ = _make_reranker([1.0])
        out = await reranker.rerank("query", [original])
        assert out[0].payload is original.payload
        assert out[0].content == "text"

    @pytest.mark.asyncio
    async def test_equal_scores_keep_candidate_order(self):
        results = [_make_result("a"), _make_result("b")]
        reranker, _ = _make_reranker([1.0, 1.0])
        out = await reranker.rerank("query", results)
        assert [r.chunk_id for r in out] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_does_not_mutate_input_scores(self):
        original = _make_result("a")
        reranker, _ = _make_reranker([1.0])
        await reranker.rerank("query", [original])
        assert original.score == pytest.approx(0.5)


# ===========================================================================
# CrossEncoderReranker.available / get_reranker
# ===========================================================================


class TestRerankerAvailability:
    def test_available_true_when_importable(self):
        assert CrossEncoderReranker.available() is True

    def test_available_false_when_import_fails(self):
        with patch.dict(sys.modules, {"fastembed.rerank.cross_encoder": None}):
            assert CrossEncoderReranker.available() is False


class TestGetReranker:
    def test_returns_none_when_disabled(self, monkeypatch):
        monkeypatch.setattr(settings, "rerank_enabled", False)
        assert get_reranker() is None

    def test_returns_none_when_fastembed_unavailable(self, monkeypatch):
        monkeypatch.setattr(settings, "rerank_enabled", True)
        with patch.object(CrossEncoderReranker, "available", return_value=False):
            assert get_reranker() is None

    def test_returns_configured_instance(self, monkeypatch):
        monkeypatch.setattr(settings, "rerank_enabled", True)
        monkeypatch.setattr(settings, "rerank_model", "test/reranker")
        monkeypatch.setattr(settings, "rerank_cache_dir", "/tmp/rerank-cache")
        reranker = get_reranker()
        assert isinstance(reranker, CrossEncoderReranker)
        assert reranker.model_name == "test/reranker"
        assert reranker.cache_dir == "/tmp/rerank-cache"

    def test_instance_is_cached(self, monkeypatch):
        monkeypatch.setattr(settings, "rerank_enabled", True)
        assert get_reranker() is get_reranker()


# ===========================================================================
# retrieve — two-stage pipeline
# ===========================================================================


class TestRetrieve:
    @pytest.mark.asyncio
    async def test_fetches_wide_candidate_pool(self, monkeypatch):
        monkeypatch.setattr(settings, "retrieval_top_k", 20)
        store = MagicMock()
        fake_hybrid = AsyncMock(return_value=[])
        with (
            patch("app.retrieval.rerank.hybrid_search", fake_hybrid),
            patch("app.retrieval.rerank.get_reranker", return_value=None),
        ):
            await retrieve(store, "q", [0.1], top_k=5)

        fake_hybrid.assert_awaited_once()
        assert fake_hybrid.call_args[1]["top_k"] == 20

    @pytest.mark.asyncio
    async def test_candidate_pool_at_least_top_k(self, monkeypatch):
        monkeypatch.setattr(settings, "retrieval_top_k", 5)
        store = MagicMock()
        fake_hybrid = AsyncMock(return_value=[])
        with (
            patch("app.retrieval.rerank.hybrid_search", fake_hybrid),
            patch("app.retrieval.rerank.get_reranker", return_value=None),
        ):
            await retrieve(store, "q", [0.1], top_k=12)

        assert fake_hybrid.call_args[1]["top_k"] == 12

    @pytest.mark.asyncio
    async def test_rerank_applied_and_limited(self):
        candidates = [_make_result(c) for c in "abcde"]
        reranked = [candidates[3], candidates[0], candidates[2]]
        fake_reranker = MagicMock()
        fake_reranker.rerank = AsyncMock(return_value=reranked)
        store = MagicMock()
        with (
            patch(
                "app.retrieval.rerank.hybrid_search",
                AsyncMock(return_value=candidates),
            ),
            patch("app.retrieval.rerank.get_reranker", return_value=fake_reranker),
        ):
            out = await retrieve(store, "q", [0.1], top_k=3)

        assert out == reranked
        assert fake_reranker.rerank.call_args[1]["top_n"] == 3

    @pytest.mark.asyncio
    async def test_without_reranker_returns_hybrid_top_k(self):
        candidates = [_make_result(c) for c in "abcde"]
        store = MagicMock()
        with (
            patch(
                "app.retrieval.rerank.hybrid_search",
                AsyncMock(return_value=candidates),
            ),
            patch("app.retrieval.rerank.get_reranker", return_value=None),
        ):
            out = await retrieve(store, "q", [0.1], top_k=3)

        assert out == candidates[:3]

    @pytest.mark.asyncio
    async def test_rerank_failure_falls_back_to_hybrid_order(self):
        candidates = [_make_result(c) for c in "abcde"]
        fake_reranker = MagicMock()
        fake_reranker.rerank = AsyncMock(side_effect=RerankError("model failed"))
        store = MagicMock()
        with (
            patch(
                "app.retrieval.rerank.hybrid_search",
                AsyncMock(return_value=candidates),
            ),
            patch("app.retrieval.rerank.get_reranker", return_value=fake_reranker),
        ):
            out = await retrieve(store, "q", [0.1], top_k=2)

        assert out == candidates[:2]

    @pytest.mark.asyncio
    async def test_store_error_propagates(self):
        store = MagicMock()
        with patch(
            "app.retrieval.rerank.hybrid_search",
            AsyncMock(side_effect=VectorStoreError("qdrant down")),
        ):
            with pytest.raises(VectorStoreError, match="qdrant down"):
                await retrieve(store, "q", [0.1], top_k=5)

    @pytest.mark.asyncio
    async def test_filter_forwarded_to_hybrid_search(self):
        qdrant_filter = object()
        store = MagicMock()
        fake_hybrid = AsyncMock(return_value=[])
        with (
            patch("app.retrieval.rerank.hybrid_search", fake_hybrid),
            patch("app.retrieval.rerank.get_reranker", return_value=None),
        ):
            await retrieve(store, "q", [0.1], top_k=5, filter=qdrant_filter)

        assert fake_hybrid.call_args[1]["filter"] is qdrant_filter

    @pytest.mark.asyncio
    async def test_default_top_k_from_rerank_top_n(self, monkeypatch):
        monkeypatch.setattr(settings, "rerank_top_n", 4)
        monkeypatch.setattr(settings, "retrieval_top_k", 20)
        candidates = [_make_result(c) for c in "abcdefgh"]
        store = MagicMock()
        with (
            patch(
                "app.retrieval.rerank.hybrid_search",
                AsyncMock(return_value=candidates),
            ),
            patch("app.retrieval.rerank.get_reranker", return_value=None),
        ):
            out = await retrieve(store, "q", [0.1])

        assert len(out) == 4
