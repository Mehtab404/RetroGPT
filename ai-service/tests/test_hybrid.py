"""Tests for hybrid search: BM25 encoder, RRF fusion, stats loading (Issue #20).

Covers:
- bm25.tokenize: lowercasing, punctuation, digits, empty input
- Bm25Encoder: fit validation, deterministic vocabulary, encode semantics
  (term weights, unknown terms, empty queries, unfitted raises)
- Bm25Encoder persistence: dict/JSON round-trip, version checks, bad files
- rrf_fuse: rank fusion, normalisation, determinism, limits, empty inputs
- get_bm25_encoder: missing file, valid file, mtime-based cache invalidation
- hybrid_search: dense-only fallback, full hybrid path, sparse failure
  fail-open, dense error propagation, settings defaults

All Qdrant interactions are mocked; no Qdrant instance is required.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client.http import models as qdrant_models

from app.config import settings
from app.retrieval.bm25 import STATS_VERSION, Bm25Encoder, tokenize
from app.retrieval.hybrid import (
    RRF_K,
    _load_bm25_stats_cached,
    get_bm25_encoder,
    hybrid_search,
    rrf_fuse,
)
from app.retrieval.vector_store import SearchResult, VectorStoreError

# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def _clear_bm25_cache():
    """Keep the mtime-keyed encoder cache from leaking between tests."""
    _load_bm25_stats_cached.cache_clear()
    yield
    _load_bm25_stats_cached.cache_clear()


def _make_result(chunk_id: str, score: float = 0.5, content: str = "") -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        score=score,
        payload={"chunk_id": chunk_id, "content": content},
        content=content or f"content-{chunk_id}",
    )


def _make_store(
    dense: list[SearchResult] | None = None,
    sparse: list[SearchResult] | None = None,
    search_error: Exception | None = None,
    sparse_error: Exception | None = None,
) -> MagicMock:
    """Build a mock QdrantVectorStore for hybrid_search tests."""
    store = MagicMock()
    store.set_bm25_encoder = MagicMock()
    if search_error is not None:
        store.search = AsyncMock(side_effect=search_error)
    else:
        store.search = AsyncMock(return_value=dense or [])
    if sparse_error is not None:
        store.search_sparse = AsyncMock(side_effect=sparse_error)
    else:
        store.search_sparse = AsyncMock(return_value=sparse or [])
    return store


# ===========================================================================
# tokenize
# ===========================================================================


class TestTokenize:
    def test_lowercases(self):
        assert tokenize("Asteroid ASTEROID") == ["asteroid", "asteroid"]

    def test_splits_on_punctuation(self):
        assert tokenize("pac-man's score: 100!") == ["pac", "man", "s", "score", "100"]

    def test_keeps_digits(self):
        assert tokenize("20 pts") == ["20", "pts"]

    def test_empty_string(self):
        assert tokenize("") == []

    def test_only_punctuation(self):
        assert tokenize("--- !!!") == []


# ===========================================================================
# Bm25Encoder.fit
# ===========================================================================


class TestBm25Fit:
    def test_not_fitted_by_default(self):
        assert Bm25Encoder().is_fitted is False

    def test_fit_marks_fitted(self):
        encoder = Bm25Encoder().fit(["hello world"])
        assert encoder.is_fitted is True

    def test_n_docs_and_vocabulary(self):
        encoder = Bm25Encoder().fit(["alpha beta", "beta gamma"])
        assert encoder.n_docs == 2
        # alpha, beta, gamma
        assert encoder.vocabulary_size == 3

    def test_empty_corpus_raises(self):
        with pytest.raises(ValueError, match="empty corpus"):
            Bm25Encoder().fit([])

    def test_vocabulary_indices_are_sorted_and_deterministic(self):
        encoder = Bm25Encoder().fit(["zeta alpha", "beta"])
        index = encoder.to_dict()["index"]
        assert index == {"alpha": 0, "beta": 1, "zeta": 2}

    def test_average_doc_length(self):
        encoder = Bm25Encoder().fit(["one two three four", "one two"])
        assert encoder.to_dict()["avgdl"] == pytest.approx(3.0)

    def test_fit_returns_self_for_chaining(self):
        encoder = Bm25Encoder()
        assert encoder.fit(["text"]) is encoder

    def test_refit_replaces_previous_state(self):
        encoder = Bm25Encoder().fit(["alpha beta", "beta"])
        encoder.fit(["gamma"])
        assert encoder.n_docs == 1
        assert encoder.vocabulary_size == 1


# ===========================================================================
# Bm25Encoder.encode
# ===========================================================================


class TestBm25Encode:
    def test_encode_before_fit_raises(self):
        with pytest.raises(ValueError, match="fitted"):
            Bm25Encoder().encode("hello")

    def test_encode_returns_sparse_vector(self):
        encoder = Bm25Encoder().fit(["asteroid scoring rules"])
        vector = encoder.encode("asteroid")
        assert isinstance(vector, qdrant_models.SparseVector)
        assert len(vector.indices) == len(vector.values) == 1
        assert vector.values[0] > 0

    def test_encode_unknown_terms_yields_empty_vector(self):
        encoder = Bm25Encoder().fit(["asteroid scoring"])
        vector = encoder.encode("completely unrelated word")
        assert vector.indices == []
        assert vector.values == []

    def test_encode_empty_text_yields_empty_vector(self):
        encoder = Bm25Encoder().fit(["asteroid scoring"])
        assert encoder.encode("").indices == []
        assert encoder.encode(" --- ").indices == []

    def test_encode_uses_fitted_vocabulary_index(self):
        encoder = Bm25Encoder().fit(["beta alpha"])
        index = encoder.to_dict()["index"]
        vector = encoder.encode("alpha")
        assert vector.indices == [index["alpha"]]

    def test_rarer_terms_receive_higher_weights(self):
        # beta appears in 1 of 2 docs; alpha in both → beta idf higher.
        encoder = Bm25Encoder().fit(["alpha beta", "alpha"])
        vector = encoder.encode("alpha beta")
        index_to_term = {v: k for k, v in encoder.to_dict()["index"].items()}
        weights = {index_to_term[i]: w for i, w in zip(vector.indices, vector.values, strict=True)}
        assert weights["beta"] > weights["alpha"]

    def test_repeated_term_saturates_but_increases(self):
        encoder = Bm25Encoder().fit(["alpha", "alpha alpha"])
        single = encoder.encode("alpha").values[0]
        doubled = encoder.encode("alpha alpha").values[0]
        assert doubled > single

    def test_longer_documents_get_lower_relative_weight(self):
        # BM25 length normalisation: same tf in a longer doc scores lower.
        encoder = Bm25Encoder().fit(["alpha", "alpha", "alpha"])
        one_term = encoder.encode("alpha").values[0]
        long_text = "alpha " + " ".join(["filler"] * 20)
        padded = encoder.encode(long_text)
        index = encoder.to_dict()["index"]
        weights = dict(zip(padded.indices, padded.values, strict=True))
        weight = weights.get(index["alpha"], 0.0)
        assert weight < one_term

    def test_multiple_terms_sorted_by_index(self):
        encoder = Bm25Encoder().fit(["zeta alpha beta"])
        vector = encoder.encode("zeta beta alpha")
        assert vector.indices == sorted(vector.indices)


# ===========================================================================
# Bm25Encoder persistence
# ===========================================================================


class TestBm25Persistence:
    def test_to_dict_contains_version(self):
        encoder = Bm25Encoder().fit(["some text"])
        assert encoder.to_dict()["version"] == STATS_VERSION

    def test_round_trip_dict(self):
        original = Bm25Encoder().fit(["asteroid scoring rules", "pacman pellets"])
        restored = Bm25Encoder.from_dict(original.to_dict())
        assert restored.is_fitted
        assert restored.n_docs == original.n_docs
        assert restored.encode("asteroid").indices == original.encode("asteroid").indices

    def test_save_and_load_round_trip(self, tmp_path: Path):
        path = tmp_path / "bm25.json"
        original = Bm25Encoder().fit(["asteroid scoring"])
        original.save(path)
        assert path.exists()

        restored = Bm25Encoder.load(path)
        assert restored.n_docs == 1
        assert restored.encode("asteroid").values == pytest.approx(
            original.encode("asteroid").values
        )

    def test_load_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            Bm25Encoder.load(tmp_path / "missing.json")

    def test_load_rejects_wrong_version(self, tmp_path: Path):
        path = tmp_path / "bm25.json"
        payload = Bm25Encoder().fit(["text"]).to_dict()
        payload["version"] = 999
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="version"):
            Bm25Encoder.load(path)

    def test_load_rejects_non_object_json(self, tmp_path: Path):
        path = tmp_path / "bm25.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ValueError, match="JSON object"):
            Bm25Encoder.load(path)

    def test_load_rejects_empty_stats(self, tmp_path: Path):
        path = tmp_path / "bm25.json"
        payload = Bm25Encoder().fit(["text"]).to_dict()
        payload["index"] = {}
        payload["n_docs"] = 0
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            Bm25Encoder.load(path)

    def test_save_creates_parent_directories(self, tmp_path: Path):
        path = tmp_path / "nested" / "dir" / "bm25.json"
        Bm25Encoder().fit(["text"]).save(path)
        assert path.exists()

    def test_save_output_is_valid_json_object(self, tmp_path: Path):
        path = tmp_path / "bm25.json"
        Bm25Encoder().fit(["text"]).save(path)
        assert isinstance(json.loads(path.read_text(encoding="utf-8")), dict)


# ===========================================================================
# rrf_fuse
# ===========================================================================


class TestRrfFuse:
    def test_empty_inputs_return_empty(self):
        assert rrf_fuse([], []) == []

    def test_single_list_preserves_order(self):
        dense = [_make_result("a"), _make_result("b")]
        fused = rrf_fuse(dense, [])
        assert [r.chunk_id for r in fused] == ["a", "b"]

    def test_sparse_only_list_preserves_order(self):
        sparse = [_make_result("x"), _make_result("y")]
        fused = rrf_fuse([], sparse)
        assert [r.chunk_id for r in fused] == ["x", "y"]

    def test_overlap_wins_over_single_list_hits(self):
        dense = [_make_result("a"), _make_result("b")]
        sparse = [_make_result("b"), _make_result("c")]
        fused = rrf_fuse(dense, sparse)
        # b: 1/62 + 1/61 beats a: 1/61 and c: 1/62
        assert fused[0].chunk_id == "b"

    def test_best_hit_normalised_to_one(self):
        dense = [_make_result("a"), _make_result("b")]
        fused = rrf_fuse(dense, [])
        assert fused[0].score == pytest.approx(1.0)

    def test_scores_strictly_decreasing(self):
        dense = [_make_result("a"), _make_result("b"), _make_result("c")]
        fused = rrf_fuse(dense, [])
        assert [r.score for r in fused] == sorted([r.score for r in fused], reverse=True)

    def test_expected_rrf_scores(self):
        dense = [_make_result("a"), _make_result("b")]
        sparse = [_make_result("b"), _make_result("c")]
        fused = {r.chunk_id: r.score for r in rrf_fuse(dense, sparse)}
        raw_b = 1.0 / (RRF_K + 2) + 1.0 / (RRF_K + 1)
        assert fused["b"] == pytest.approx(1.0)
        assert fused["a"] == pytest.approx((1.0 / (RRF_K + 1)) / raw_b)
        assert fused["c"] == pytest.approx((1.0 / (RRF_K + 2)) / raw_b)

    def test_limit_truncates(self):
        dense = [_make_result(c) for c in ("a", "b", "c", "d")]
        fused = rrf_fuse(dense, [], limit=2)
        assert [r.chunk_id for r in fused] == ["a", "b"]

    def test_custom_k_parameter(self):
        fused_k10 = rrf_fuse([_make_result("a")], [], k=10)
        fused_k60 = rrf_fuse([_make_result("a")], [], k=60)
        # Both are normalised to 1.0 for a single hit, order is stable;
        # verify via a two-item computation with different k.
        dense = [_make_result("a"), _make_result("b")]
        s10 = {r.chunk_id: r.score for r in rrf_fuse(dense, [], k=10)}
        s60 = {r.chunk_id: r.score for r in rrf_fuse(dense, [], k=60)}
        assert s10["b"] != s60["b"]
        assert fused_k10[0].score == fused_k60[0].score == 1.0

    def test_overlap_keeps_dense_payload(self):
        dense = [_make_result("b", content="dense content")]
        sparse = [_make_result("b", content="sparse content")]
        fused = rrf_fuse(dense, sparse)
        assert fused[0].content == "dense content"

    def test_ties_are_broken_by_chunk_id(self):
        # Two lists with disjoint single hits at the same rank → tie.
        dense = [_make_result("zz")]
        sparse = [_make_result("aa")]
        fused = rrf_fuse(dense, sparse)
        assert [r.chunk_id for r in fused] == ["aa", "zz"]

    def test_does_not_mutate_input_results(self):
        dense = [_make_result("a", score=0.42)]
        rrf_fuse(dense, [])
        assert dense[0].score == pytest.approx(0.42)


# ===========================================================================
# get_bm25_encoder
# ===========================================================================


class TestGetBm25Encoder:
    def test_missing_file_returns_none(self, tmp_path: Path):
        assert get_bm25_encoder(tmp_path / "missing.json") is None

    def test_valid_file_returns_fitted_encoder(self, tmp_path: Path):
        path = tmp_path / "bm25.json"
        Bm25Encoder().fit(["asteroid scoring"]).save(path)
        encoder = get_bm25_encoder(path)
        assert encoder is not None
        assert encoder.is_fitted

    def test_invalid_json_returns_none(self, tmp_path: Path):
        path = tmp_path / "bm25.json"
        path.write_text("not json", encoding="utf-8")
        assert get_bm25_encoder(path) is None

    def test_wrong_version_returns_none(self, tmp_path: Path):
        path = tmp_path / "bm25.json"
        payload = Bm25Encoder().fit(["text"]).to_dict()
        payload["version"] = 42
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert get_bm25_encoder(path) is None

    def test_result_is_cached_by_path_and_mtime(self, tmp_path: Path):
        path = tmp_path / "bm25.json"
        Bm25Encoder().fit(["alpha"]).save(path)
        first = get_bm25_encoder(path)
        second = get_bm25_encoder(path)
        assert first is second

    def test_cache_invalidated_when_file_changes(self, tmp_path: Path):
        path = tmp_path / "bm25.json"
        Bm25Encoder().fit(["alpha"]).save(path)
        first = get_bm25_encoder(path)

        # Rewrite with a different corpus and force a distinct mtime.
        Bm25Encoder().fit(["beta gamma delta"]).save(path)
        import os

        os.utime(path, (path.stat().st_atime, path.stat().st_mtime + 5))

        second = get_bm25_encoder(path)
        assert first is not second
        assert second.vocabulary_size == 3

    def test_default_path_comes_from_settings(self, tmp_path: Path, monkeypatch):
        path = tmp_path / "from_settings.json"
        Bm25Encoder().fit(["alpha"]).save(path)
        monkeypatch.setattr(settings, "bm25_stats_path", str(path))
        encoder = get_bm25_encoder()
        assert encoder is not None


# ===========================================================================
# hybrid_search
# ===========================================================================


class TestHybridSearch:
    @pytest.mark.asyncio
    async def test_dense_only_when_no_encoder(self):
        dense = [_make_result("a"), _make_result("b")]
        store = _make_store(dense=dense)
        with patch("app.retrieval.hybrid.get_bm25_encoder", return_value=None):
            results = await hybrid_search(store, "query", [0.1], top_k=5)

        assert results == dense
        store.search.assert_awaited_once()
        assert store.search.call_args[1]["top_k"] == 5
        store.search_sparse.assert_not_called()

    @pytest.mark.asyncio
    async def test_dense_only_when_hybrid_disabled(self, monkeypatch):
        dense = [_make_result("a")]
        store = _make_store(dense=dense)
        monkeypatch.setattr(settings, "hybrid_search_enabled", False)
        with patch("app.retrieval.hybrid.get_bm25_encoder") as mock_get:
            results = await hybrid_search(store, "query", [0.1], top_k=4)

        mock_get.assert_not_called()
        assert results == dense
        store.search_sparse.assert_not_called()

    @pytest.mark.asyncio
    async def test_fuses_when_encoder_available(self):
        dense = [_make_result("a"), _make_result("b")]
        sparse = [_make_result("b"), _make_result("c")]
        store = _make_store(dense=dense, sparse=sparse)
        encoder = MagicMock()

        with patch("app.retrieval.hybrid.get_bm25_encoder", return_value=encoder):
            results = await hybrid_search(store, "query", [0.1], top_k=3, prefetch_k=10)

        store.set_bm25_encoder.assert_called_once_with(encoder)
        assert store.search.call_args[1]["top_k"] == 10
        assert store.search_sparse.call_args[1]["top_k"] == 10
        assert [r.chunk_id for r in results][:1] == ["b"]
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_prefetch_defaults_from_settings(self, monkeypatch):
        store = _make_store(
            dense=[_make_result("a")],
            sparse=[_make_result("a")],
        )
        monkeypatch.setattr(settings, "hybrid_prefetch_k", 7)
        with patch("app.retrieval.hybrid.get_bm25_encoder", return_value=MagicMock()):
            await hybrid_search(store, "query", [0.1], top_k=5)

        assert store.search.call_args[1]["top_k"] == 7
        assert store.search_sparse.call_args[1]["top_k"] == 7

    @pytest.mark.asyncio
    async def test_top_k_defaults_from_settings(self, monkeypatch):
        dense = [_make_result("a")]
        store = _make_store(dense=dense)
        monkeypatch.setattr(settings, "retrieval_top_k", 3)
        with patch("app.retrieval.hybrid.get_bm25_encoder", return_value=None):
            await hybrid_search(store, "query", [0.1])

        assert store.search.call_args[1]["top_k"] == 3

    @pytest.mark.asyncio
    async def test_filter_forwarded_to_both_retrievers(self):
        qdrant_filter = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="game", match=qdrant_models.MatchValue(value="pong")
                )
            ]
        )
        store = _make_store(dense=[_make_result("a")], sparse=[_make_result("a")])
        with patch("app.retrieval.hybrid.get_bm25_encoder", return_value=MagicMock()):
            await hybrid_search(store, "query", [0.1], top_k=2, filter=qdrant_filter)

        assert store.search.call_args[1]["filter"] is qdrant_filter
        assert store.search_sparse.call_args[1]["filter"] is qdrant_filter

    @pytest.mark.asyncio
    async def test_sparse_failure_falls_back_to_dense(self):
        dense = [_make_result("a"), _make_result("b"), _make_result("c")]
        store = _make_store(dense=dense, sparse_error=VectorStoreError("no sparse vectors"))

        with patch("app.retrieval.hybrid.get_bm25_encoder", return_value=MagicMock()):
            results = await hybrid_search(store, "query", [0.1], top_k=2, prefetch_k=3)

        assert results == dense[:2]

    @pytest.mark.asyncio
    async def test_dense_failure_propagates(self):
        store = _make_store(search_error=VectorStoreError("qdrant down"))
        with patch("app.retrieval.hybrid.get_bm25_encoder", return_value=None):
            with pytest.raises(VectorStoreError, match="qdrant down"):
                await hybrid_search(store, "query", [0.1], top_k=5)

    @pytest.mark.asyncio
    async def test_encoder_not_attached_when_dense_only(self):
        store = _make_store(dense=[_make_result("a")])
        with patch("app.retrieval.hybrid.get_bm25_encoder", return_value=None):
            await hybrid_search(store, "query", [0.1], top_k=5)

        store.set_bm25_encoder.assert_not_called()
