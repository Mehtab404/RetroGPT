"""Tests for the generation package and POST /query endpoint (Issue #10).

Covers:
- LLMProvider ABC contract
- GeminiLLM: generate, retry logic, malformed response
- OpenAILLM: generate, retry logic, malformed response
- get_llm_provider factory
- LLMError raised on bad response shapes and exhausted retries
- prompts.build_rag_prompt: happy path, empty inputs, citation format
- schemas.QueryRequest / QueryResponse / SourceDoc: validation
- POST /query: end-to-end with mocked embedder, vector store, and LLM
  - happy path with sources
  - no results returns "I don't know" answer
  - embedding error → 502
  - vector store error → 503
  - LLM error → 502
  - game_slug filter forwarded
  - request validation errors
  - latency_ms populated

All external HTTP calls and service dependencies are mocked via
pytest-mock / unittest.mock.  asyncio tests use pytest-asyncio.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.generation.llm import (
    GeminiLLM,
    LLMError,
    LLMProvider,
    OpenAILLM,
    get_llm_provider,
)
from app.generation.prompts import RAG_SYSTEM_PROMPT, build_rag_prompt
from app.main import app
from app.retrieval.vector_store import SearchResult
from app.schemas import QueryRequest, QueryResponse, SourceDoc

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_httpx_response(json_data: dict, status_code: int = 200) -> MagicMock:
    """Return a MagicMock shaped like an httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.text = str(json_data)
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status_code}",
            request=MagicMock(),
            response=resp,
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


def _gemini_ok_response(text: str = "Test answer") -> dict[str, Any]:
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def _openai_ok_response(text: str = "Test answer") -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": text,
                }
            }
        ]
    }


def _make_search_result(
    chunk_id: str = "chunk_1",
    score: float = 0.9,
    content: str = "Sample content",
    title: str = "Test Manual",
    source: str = "/data/raw/asteroids/asteroids_manual.md",
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        score=score,
        payload={
            "chunk_id": chunk_id,
            "title": title,
            "source": source,
            "content": content,
        },
        content=content,
    )


@pytest.fixture(autouse=True)
def _deterministic_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force /query onto the deterministic dense-only, no-rerank path.

    Without this, the endpoint would try to load BM25 statistics from
    ``settings.bm25_stats_path`` and the fastembed cross-encoder model,
    neither of which is guaranteed (or appropriate) on the test host.
    """
    monkeypatch.setattr("app.retrieval.hybrid.get_bm25_encoder", lambda path=None: None)
    monkeypatch.setattr("app.retrieval.rerank.get_reranker", lambda: None)


# ---------------------------------------------------------------------------
# LLMProvider ABC
# ---------------------------------------------------------------------------


class TestLLMProviderABC:
    """Verify the ABC cannot be instantiated directly."""

    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            LLMProvider(model="test")  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_generate(self) -> None:
        class Partial(LLMProvider):
            pass  # Missing generate

        with pytest.raises(TypeError):
            Partial(model="test")  # type: ignore[abstract]

    def test_concrete_subclass_ok(self) -> None:
        class Complete(LLMProvider):
            async def generate(self, prompt: str) -> str:
                return "ok"

        provider = Complete(model="test")
        assert provider.model == "test"
        assert provider.temperature == 0.2
        assert provider.max_tokens == 1024
        assert provider.max_retries == 3


# ---------------------------------------------------------------------------
# LLMError
# ---------------------------------------------------------------------------


class TestLLMError:
    def test_is_runtime_error(self) -> None:
        err = LLMError("boom")
        assert isinstance(err, RuntimeError)

    def test_cause_preserved(self) -> None:
        cause = ValueError("root cause")
        err = LLMError("wrapper")
        err.__cause__ = cause
        assert err.__cause__ is cause


# ---------------------------------------------------------------------------
# GeminiLLM
# ---------------------------------------------------------------------------


class TestGeminiLLMGenerate:
    """Tests for GeminiLLM.generate()."""

    @pytest.mark.asyncio
    async def test_generate_success(self) -> None:
        """Successful Gemini generation returns text."""
        provider = GeminiLLM(model="gemini-1.5-flash", api_key="test-key", jitter=False)
        resp = _mock_httpx_response(_gemini_ok_response("Asteroids scores 20 pts."))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=resp)
            mock_client_cls.return_value = mock_client

            result = await provider.generate("How do you score in Asteroids?")

        assert result == "Asteroids scores 20 pts."

    @pytest.mark.asyncio
    async def test_generate_calls_correct_url(self) -> None:
        """Gemini API is called with model name in the URL."""
        provider = GeminiLLM(model="gemini-1.5-flash", api_key="my-key", jitter=False)
        resp = _mock_httpx_response(_gemini_ok_response("answer"))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=resp)
            mock_client_cls.return_value = mock_client

            await provider.generate("question")

        call_args = mock_client.post.call_args
        url = call_args[0][0] if call_args[0] else call_args[1].get("url", call_args[0][0])
        assert "gemini-1.5-flash" in url
        assert "my-key" in url

    @pytest.mark.asyncio
    async def test_generate_passes_temperature_and_max_tokens(self) -> None:
        """Generation config is included in the request payload."""
        provider = GeminiLLM(
            model="gemini-1.5-flash",
            api_key="key",
            temperature=0.5,
            max_tokens=512,
            jitter=False,
        )
        resp = _mock_httpx_response(_gemini_ok_response("answer"))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=resp)
            mock_client_cls.return_value = mock_client

            await provider.generate("q")

        payload = mock_client.post.call_args[1]["json"]
        assert payload["generationConfig"]["temperature"] == 0.5
        assert payload["generationConfig"]["maxOutputTokens"] == 512

    @pytest.mark.asyncio
    async def test_generate_malformed_response_raises_llm_error(self) -> None:
        """LLMError raised if API returns unexpected JSON structure."""
        provider = GeminiLLM(model="gemini-1.5-flash", api_key="key", jitter=False)
        resp = _mock_httpx_response({"unexpected": "response"})

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=resp)
            mock_client_cls.return_value = mock_client

            with pytest.raises(LLMError, match="Unexpected Gemini response"):
                await provider.generate("q")

    @pytest.mark.asyncio
    async def test_generate_retries_on_429(self) -> None:
        """Transient 429 is retried; eventual success returns text."""
        provider = GeminiLLM(
            model="gemini-1.5-flash",
            api_key="key",
            max_retries=2,
            base_delay=0.0,
            jitter=False,
        )
        fail_resp = _mock_httpx_response({}, status_code=429)
        ok_resp = _mock_httpx_response(_gemini_ok_response("ok"))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=[fail_resp, ok_resp])
            mock_client_cls.return_value = mock_client

            result = await provider.generate("q")

        assert result == "ok"
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_generate_raises_llm_error_after_exhausted_retries(self) -> None:
        """LLMError raised when all retries are exhausted."""
        provider = GeminiLLM(
            model="gemini-1.5-flash",
            api_key="key",
            max_retries=1,
            base_delay=0.0,
            jitter=False,
        )
        fail_resp = _mock_httpx_response({}, status_code=503)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=fail_resp)
            mock_client_cls.return_value = mock_client

            with pytest.raises(LLMError):
                await provider.generate("q")

        assert mock_client.post.call_count == 2  # 1 initial + 1 retry

    @pytest.mark.asyncio
    async def test_generate_non_retryable_4xx_raises_immediately(self) -> None:
        """Non-retryable 4xx (e.g. 400) raises LLMError immediately."""
        provider = GeminiLLM(
            model="gemini-1.5-flash",
            api_key="key",
            max_retries=3,
            base_delay=0.0,
            jitter=False,
        )
        fail_resp = _mock_httpx_response({}, status_code=400)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=fail_resp)
            mock_client_cls.return_value = mock_client

            with pytest.raises(LLMError):
                await provider.generate("q")

        assert mock_client.post.call_count == 1  # No retries

    @pytest.mark.asyncio
    async def test_generate_retries_on_request_error(self) -> None:
        """Network errors (RequestError) are retried."""
        provider = GeminiLLM(
            model="gemini-1.5-flash",
            api_key="key",
            max_retries=2,
            base_delay=0.0,
            jitter=False,
        )
        ok_resp = _mock_httpx_response(_gemini_ok_response("recovered"))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(
                side_effect=[
                    httpx.RequestError("timeout"),
                    ok_resp,
                ]
            )
            mock_client_cls.return_value = mock_client

            result = await provider.generate("q")

        assert result == "recovered"


# ---------------------------------------------------------------------------
# OpenAILLM
# ---------------------------------------------------------------------------


class TestOpenAILLMGenerate:
    """Tests for OpenAILLM.generate()."""

    @pytest.mark.asyncio
    async def test_generate_success(self) -> None:
        """Successful OpenAI generation returns text."""
        provider = OpenAILLM(model="gpt-4o-mini", api_key="test-key", jitter=False)
        resp = _mock_httpx_response(_openai_ok_response("OpenAI answer"))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=resp)
            mock_client_cls.return_value = mock_client

            result = await provider.generate("How do you score?")

        assert result == "OpenAI answer"

    @pytest.mark.asyncio
    async def test_generate_sends_bearer_token(self) -> None:
        """Authorization header uses Bearer token."""
        provider = OpenAILLM(model="gpt-4o-mini", api_key="sk-secret", jitter=False)
        resp = _mock_httpx_response(_openai_ok_response("answer"))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=resp)
            mock_client_cls.return_value = mock_client

            await provider.generate("q")

        headers = mock_client.post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer sk-secret"

    @pytest.mark.asyncio
    async def test_generate_malformed_response_raises_llm_error(self) -> None:
        """LLMError raised if OpenAI returns unexpected JSON structure."""
        provider = OpenAILLM(model="gpt-4o-mini", api_key="key", jitter=False)
        resp = _mock_httpx_response({"wrong": "structure"})

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=resp)
            mock_client_cls.return_value = mock_client

            with pytest.raises(LLMError, match="Unexpected OpenAI response"):
                await provider.generate("q")

    @pytest.mark.asyncio
    async def test_generate_retries_on_500(self) -> None:
        """500 server error triggers retry."""
        provider = OpenAILLM(
            model="gpt-4o-mini",
            api_key="key",
            max_retries=2,
            base_delay=0.0,
            jitter=False,
        )
        fail_resp = _mock_httpx_response({}, status_code=500)
        ok_resp = _mock_httpx_response(_openai_ok_response("recovered"))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=[fail_resp, ok_resp])
            mock_client_cls.return_value = mock_client

            result = await provider.generate("q")

        assert result == "recovered"


# ---------------------------------------------------------------------------
# get_llm_provider factory
# ---------------------------------------------------------------------------


class TestGetLLMProvider:
    def test_returns_gemini_by_default(self) -> None:
        with patch("app.generation.llm.settings") as mock_settings:
            mock_settings.llm_provider = "gemini"
            mock_settings.llm_model = "gemini-1.5-flash"
            mock_settings.google_api_key = "key"
            provider = get_llm_provider(provider="gemini")
        assert isinstance(provider, GeminiLLM)

    def test_returns_openai_when_specified(self) -> None:
        with patch("app.generation.llm.settings") as mock_settings:
            mock_settings.openai_api_key = "key"
            provider = get_llm_provider(provider="openai")
        assert isinstance(provider, OpenAILLM)

    def test_raises_on_unknown_provider(self) -> None:
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm_provider(provider="unknown")

    def test_kwargs_forwarded_to_provider(self) -> None:
        with patch("app.generation.llm.settings") as mock_settings:
            mock_settings.llm_model = "gemini-1.5-flash"
            mock_settings.google_api_key = "key"
            provider = get_llm_provider(provider="gemini", temperature=0.9)
        assert provider.temperature == 0.9


# ---------------------------------------------------------------------------
# build_rag_prompt
# ---------------------------------------------------------------------------


class TestBuildRagPrompt:
    def test_prompt_contains_system_instructions(self) -> None:
        """System prompt is included in the output."""
        prompt = build_rag_prompt(
            question="How do I score?",
            context_passages=[{"id": "p1", "content": "20 points per asteroid"}],
        )
        assert RAG_SYSTEM_PROMPT in prompt

    def test_prompt_contains_question(self) -> None:
        """Question appears verbatim in the prompt."""
        question = "What is the highest score in Pong?"
        prompt = build_rag_prompt(
            question=question,
            context_passages=[{"id": "p1", "content": "Max score is 99"}],
        )
        assert question in prompt

    def test_prompt_numbers_passages_from_one(self) -> None:
        """Passages are labelled [1], [2], etc."""
        prompt = build_rag_prompt(
            question="q",
            context_passages=[
                {"id": "a", "content": "First passage"},
                {"id": "b", "content": "Second passage"},
            ],
        )
        assert "[1]" in prompt
        assert "[2]" in prompt

    def test_prompt_includes_passage_source_ids(self) -> None:
        """Source IDs appear in the context block."""
        prompt = build_rag_prompt(
            question="q",
            context_passages=[{"id": "asteroids_manual_p3", "content": "content"}],
        )
        assert "asteroids_manual_p3" in prompt

    def test_prompt_uses_fallback_id_when_no_id(self) -> None:
        """Passages without 'id' get a fallback label."""
        prompt = build_rag_prompt(
            question="q",
            context_passages=[{"content": "no id here"}],
        )
        assert "passage_1" in prompt

    def test_prompt_contains_answer_marker(self) -> None:
        """Prompt ends with 'ANSWER:' to prime the LLM."""
        prompt = build_rag_prompt(
            question="q",
            context_passages=[{"id": "x", "content": "some context"}],
        )
        assert prompt.strip().endswith("ANSWER:")

    def test_empty_question_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            build_rag_prompt(
                question="",
                context_passages=[{"id": "p", "content": "ctx"}],
            )

    def test_whitespace_question_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            build_rag_prompt(
                question="   ",
                context_passages=[{"id": "p", "content": "ctx"}],
            )

    def test_empty_passages_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="at least one passage"):
            build_rag_prompt(question="How to score?", context_passages=[])

    def test_multiple_passages_all_appear_in_prompt(self) -> None:
        passages = [{"id": f"doc_{i}", "content": f"Content number {i}"} for i in range(5)]
        prompt = build_rag_prompt(question="q", context_passages=passages)
        for p in passages:
            assert p["content"] in prompt
            assert p["id"] in prompt


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class TestQueryRequest:
    def test_valid_request(self) -> None:
        req = QueryRequest(question="How do I play Pong?")
        assert req.question == "How do I play Pong?"
        assert req.top_k == 5
        assert req.game_slug is None

    def test_top_k_custom(self) -> None:
        req = QueryRequest(question="q", top_k=10)
        assert req.top_k == 10

    def test_game_slug_optional(self) -> None:
        req = QueryRequest(question="q", game_slug="asteroids")
        assert req.game_slug == "asteroids"

    def test_empty_question_fails_validation(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(question="")

    def test_too_long_question_fails_validation(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(question="x" * 1001)

    def test_top_k_below_minimum_fails(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(question="q", top_k=0)

    def test_top_k_above_maximum_fails(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(question="q", top_k=21)

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(question="q", unknown_field="x")


class TestSourceDoc:
    def test_valid_source_doc(self) -> None:
        doc = SourceDoc(id="chunk_1", score=0.95)
        assert doc.id == "chunk_1"
        assert doc.score == 0.95
        assert doc.title == ""
        assert doc.source == ""
        assert doc.snippet == ""

    def test_with_all_fields(self) -> None:
        doc = SourceDoc(
            id="chunk_1",
            title="Asteroids Manual",
            source="/data/raw/asteroids/asteroids_manual.md",
            score=0.91,
            snippet="Large asteroids: 20 points",
        )
        assert doc.title == "Asteroids Manual"
        assert doc.source == "/data/raw/asteroids/asteroids_manual.md"
        assert doc.snippet == "Large asteroids: 20 points"

    def test_score_out_of_range_fails(self) -> None:
        with pytest.raises(ValidationError):
            SourceDoc(id="x", score=1.5)

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            SourceDoc(id="x", score=0.5, surprise="field")


class TestQueryResponse:
    def test_valid_response(self) -> None:
        resp = QueryResponse(answer="42 points.", latency_ms=1500.0)
        assert resp.answer == "42 points."
        assert resp.latency_ms == 1500.0
        assert resp.sources == []
        assert resp.cached is False

    def test_with_sources(self) -> None:
        source = SourceDoc(id="doc_1", score=0.85)
        resp = QueryResponse(answer="answer", latency_ms=100.0, sources=[source])
        assert len(resp.sources) == 1
        assert resp.sources[0].id == "doc_1"

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            QueryResponse(answer="a", latency_ms=1.0, extra="bad")


# ---------------------------------------------------------------------------
# POST /query endpoint
# ---------------------------------------------------------------------------


client = TestClient(app)


class TestQueryEndpointHappyPath:
    """POST /query returns an answer with sources."""

    def test_query_returns_answer_and_sources(self) -> None:
        search_result = _make_search_result(
            chunk_id="asteroids_p3_c2",
            score=0.91,
            content="Large asteroids: 20 points each.",
            title="Asteroids Manual",
        )

        with (
            patch("app.main.get_embedding_provider") as mock_embed_factory,
            patch("app.main.QdrantVectorStore") as mock_vs_cls,
            patch("app.main.get_llm_provider") as mock_llm_factory,
        ):
            # Mock embedder
            mock_embedder = AsyncMock()
            mock_embedder.embed_query = AsyncMock(return_value=[0.1] * 768)
            mock_embed_factory.return_value = mock_embedder

            # Mock vector store
            mock_vs = AsyncMock()
            mock_vs.search = AsyncMock(return_value=[search_result])
            mock_vs_cls.return_value = mock_vs

            # Mock LLM
            mock_llm = AsyncMock()
            mock_llm.generate = AsyncMock(return_value="Large asteroids are worth 20 points [1].")
            mock_llm_factory.return_value = mock_llm

            response = client.post(
                "/query",
                json={"question": "How do you score in Asteroids?"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "20 points" in data["answer"]
        assert len(data["sources"]) == 1
        assert data["sources"][0]["id"] == "asteroids_p3_c2"
        assert data["sources"][0]["score"] == 0.91
        assert "latency_ms" in data
        assert data["latency_ms"] >= 0
        assert data["cached"] is False

    def test_query_source_snippet_truncated_to_300_chars(self) -> None:
        long_content = "A" * 500
        search_result = _make_search_result(content=long_content)

        with (
            patch("app.main.get_embedding_provider") as mock_embed_factory,
            patch("app.main.QdrantVectorStore") as mock_vs_cls,
            patch("app.main.get_llm_provider") as mock_llm_factory,
        ):
            mock_embedder = AsyncMock()
            mock_embedder.embed_query = AsyncMock(return_value=[0.1] * 768)
            mock_embed_factory.return_value = mock_embedder

            mock_vs = AsyncMock()
            mock_vs.search = AsyncMock(return_value=[search_result])
            mock_vs_cls.return_value = mock_vs

            mock_llm = AsyncMock()
            mock_llm.generate = AsyncMock(return_value="Answer.")
            mock_llm_factory.return_value = mock_llm

            response = client.post("/query", json={"question": "q?"})

        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"][0]["snippet"]) <= 300

    def test_query_title_populated_from_payload(self) -> None:
        search_result = _make_search_result(title="Missile Command FAQ")

        with (
            patch("app.main.get_embedding_provider") as mock_embed_factory,
            patch("app.main.QdrantVectorStore") as mock_vs_cls,
            patch("app.main.get_llm_provider") as mock_llm_factory,
        ):
            mock_embedder = AsyncMock()
            mock_embedder.embed_query = AsyncMock(return_value=[0.1] * 768)
            mock_embed_factory.return_value = mock_embedder

            mock_vs = AsyncMock()
            mock_vs.search = AsyncMock(return_value=[search_result])
            mock_vs_cls.return_value = mock_vs

            mock_llm = AsyncMock()
            mock_llm.generate = AsyncMock(return_value="Answer.")
            mock_llm_factory.return_value = mock_llm

            response = client.post("/query", json={"question": "q?"})

        assert response.status_code == 200
        data = response.json()
        assert data["sources"][0]["title"] == "Missile Command FAQ"

    def test_query_source_path_populated_from_payload(self) -> None:
        """SourceDoc.source exposes the payload source path (used by eval)."""
        search_result = _make_search_result(
            source="/data/raw/pacman/pacman_manual.md",
        )

        with (
            patch("app.main.get_embedding_provider") as mock_embed_factory,
            patch("app.main.QdrantVectorStore") as mock_vs_cls,
            patch("app.main.get_llm_provider") as mock_llm_factory,
        ):
            mock_embedder = AsyncMock()
            mock_embedder.embed_query = AsyncMock(return_value=[0.1] * 768)
            mock_embed_factory.return_value = mock_embedder

            mock_vs = AsyncMock()
            mock_vs.search = AsyncMock(return_value=[search_result])
            mock_vs_cls.return_value = mock_vs

            mock_llm = AsyncMock()
            mock_llm.generate = AsyncMock(return_value="Answer.")
            mock_llm_factory.return_value = mock_llm

            response = client.post("/query", json={"question": "q?"})

        assert response.status_code == 200
        data = response.json()
        assert data["sources"][0]["source"] == "/data/raw/pacman/pacman_manual.md"

    def test_query_top_k_forwarded_to_search(self) -> None:
        search_results = [_make_search_result(chunk_id=f"c{i}") for i in range(10)]

        with (
            patch("app.main.get_embedding_provider") as mock_embed_factory,
            patch("app.main.QdrantVectorStore") as mock_vs_cls,
            patch("app.main.get_llm_provider") as mock_llm_factory,
        ):
            mock_embedder = AsyncMock()
            mock_embedder.embed_query = AsyncMock(return_value=[0.1] * 768)
            mock_embed_factory.return_value = mock_embedder

            mock_vs = AsyncMock()
            mock_vs.search = AsyncMock(return_value=search_results)
            mock_vs_cls.return_value = mock_vs

            mock_llm = AsyncMock()
            mock_llm.generate = AsyncMock(return_value="Answer.")
            mock_llm_factory.return_value = mock_llm

            response = client.post("/query", json={"question": "q?", "top_k": 8})

        assert response.status_code == 200
        # The candidate pool (retrieval_top_k=20) is fetched wide, then the
        # requested top_k is kept after retrieval/re-ranking.
        mock_vs.search.assert_called_once()
        call_kwargs = mock_vs.search.call_args[1]
        assert call_kwargs["top_k"] == 20
        assert len(response.json()["sources"]) == 8

    def test_query_reranks_candidates_when_reranker_available(self) -> None:
        results = [
            _make_search_result(chunk_id="first", content="first passage"),
            _make_search_result(chunk_id="second", content="second passage"),
        ]
        fake_reranker = MagicMock()
        fake_reranker.rerank = AsyncMock(return_value=[results[1], results[0]])

        with (
            patch("app.main.get_embedding_provider") as mock_embed_factory,
            patch("app.main.QdrantVectorStore") as mock_vs_cls,
            patch("app.main.get_llm_provider") as mock_llm_factory,
            patch("app.retrieval.rerank.get_reranker", return_value=fake_reranker),
        ):
            mock_embedder = AsyncMock()
            mock_embedder.embed_query = AsyncMock(return_value=[0.1] * 768)
            mock_embed_factory.return_value = mock_embedder

            mock_vs = AsyncMock()
            mock_vs.search = AsyncMock(return_value=results)
            mock_vs_cls.return_value = mock_vs

            mock_llm = AsyncMock()
            mock_llm.generate = AsyncMock(return_value="Answer.")
            mock_llm_factory.return_value = mock_llm

            response = client.post("/query", json={"question": "q?"})

        assert response.status_code == 200
        data = response.json()
        assert [s["id"] for s in data["sources"]] == ["second", "first"]

    def test_query_game_slug_filter_applied(self) -> None:
        search_result = _make_search_result()

        with (
            patch("app.main.get_embedding_provider") as mock_embed_factory,
            patch("app.main.QdrantVectorStore") as mock_vs_cls,
            patch("app.main.get_llm_provider") as mock_llm_factory,
        ):
            mock_embedder = AsyncMock()
            mock_embedder.embed_query = AsyncMock(return_value=[0.1] * 768)
            mock_embed_factory.return_value = mock_embedder

            mock_vs = AsyncMock()
            mock_vs.search = AsyncMock(return_value=[search_result])
            mock_vs_cls.return_value = mock_vs

            mock_llm = AsyncMock()
            mock_llm.generate = AsyncMock(return_value="Answer.")
            mock_llm_factory.return_value = mock_llm

            client.post(
                "/query",
                json={"question": "q?", "game_slug": "asteroids"},
            )

        call_kwargs = mock_vs.search.call_args[1]
        # filter must be set (not None) when game_slug is provided
        assert call_kwargs["filter"] is not None
        # …and it must match the payload key used at ingestion time ("game",
        # not "game_slug").
        assert call_kwargs["filter"].must[0].key == "game"
        assert call_kwargs["filter"].must[0].match.value == "asteroids"


class TestQueryEndpointNoResults:
    """POST /query returns graceful answer when no passages are found."""

    def test_no_results_returns_i_dont_know(self) -> None:
        with (
            patch("app.main.get_embedding_provider") as mock_embed_factory,
            patch("app.main.QdrantVectorStore") as mock_vs_cls,
            patch("app.main.get_llm_provider") as mock_llm_factory,
        ):
            mock_embedder = AsyncMock()
            mock_embedder.embed_query = AsyncMock(return_value=[0.1] * 768)
            mock_embed_factory.return_value = mock_embedder

            mock_vs = AsyncMock()
            mock_vs.search = AsyncMock(return_value=[])
            mock_vs_cls.return_value = mock_vs

            mock_llm = AsyncMock()
            mock_llm_factory.return_value = mock_llm

            response = client.post("/query", json={"question": "q?"})

        assert response.status_code == 200
        data = response.json()
        assert "don't know" in data["answer"].lower()
        assert data["sources"] == []
        # LLM should NOT be called when there are no results
        mock_llm.generate.assert_not_called()


class TestQueryEndpointErrors:
    """POST /query error handling."""

    def test_embedding_error_returns_502(self) -> None:
        from app.ingestion.embeddings import EmbeddingError

        with patch("app.main.get_embedding_provider") as mock_embed_factory:
            mock_embedder = AsyncMock()
            mock_embedder.embed_query = AsyncMock(side_effect=EmbeddingError("API down"))
            mock_embed_factory.return_value = mock_embedder

            response = client.post("/query", json={"question": "q?"})

        assert response.status_code == 502
        assert "Embedding" in response.json()["detail"]

    def test_vector_store_error_returns_503(self) -> None:
        from app.retrieval.vector_store import VectorStoreError

        with (
            patch("app.main.get_embedding_provider") as mock_embed_factory,
            patch("app.main.QdrantVectorStore") as mock_vs_cls,
        ):
            mock_embedder = AsyncMock()
            mock_embedder.embed_query = AsyncMock(return_value=[0.1] * 768)
            mock_embed_factory.return_value = mock_embedder

            mock_vs = AsyncMock()
            mock_vs.search = AsyncMock(side_effect=VectorStoreError("Qdrant down"))
            mock_vs_cls.return_value = mock_vs

            response = client.post("/query", json={"question": "q?"})

        assert response.status_code == 503
        assert "Vector store" in response.json()["detail"]

    def test_llm_error_returns_502(self) -> None:
        search_result = _make_search_result()

        with (
            patch("app.main.get_embedding_provider") as mock_embed_factory,
            patch("app.main.QdrantVectorStore") as mock_vs_cls,
            patch("app.main.get_llm_provider") as mock_llm_factory,
        ):
            mock_embedder = AsyncMock()
            mock_embedder.embed_query = AsyncMock(return_value=[0.1] * 768)
            mock_embed_factory.return_value = mock_embedder

            mock_vs = AsyncMock()
            mock_vs.search = AsyncMock(return_value=[search_result])
            mock_vs_cls.return_value = mock_vs

            mock_llm = AsyncMock()
            mock_llm.generate = AsyncMock(side_effect=LLMError("LLM timed out"))
            mock_llm_factory.return_value = mock_llm

            response = client.post("/query", json={"question": "q?"})

        assert response.status_code == 502
        assert "LLM" in response.json()["detail"]

    def test_empty_question_returns_422(self) -> None:
        response = client.post("/query", json={"question": ""})
        assert response.status_code == 422

    def test_missing_question_returns_422(self) -> None:
        response = client.post("/query", json={})
        assert response.status_code == 422

    def test_top_k_too_large_returns_422(self) -> None:
        response = client.post("/query", json={"question": "q?", "top_k": 99})
        assert response.status_code == 422

    def test_no_game_slug_no_filter(self) -> None:
        """Without game_slug, filter should be None."""
        search_result = _make_search_result()

        with (
            patch("app.main.get_embedding_provider") as mock_embed_factory,
            patch("app.main.QdrantVectorStore") as mock_vs_cls,
            patch("app.main.get_llm_provider") as mock_llm_factory,
        ):
            mock_embedder = AsyncMock()
            mock_embedder.embed_query = AsyncMock(return_value=[0.1] * 768)
            mock_embed_factory.return_value = mock_embedder

            mock_vs = AsyncMock()
            mock_vs.search = AsyncMock(return_value=[search_result])
            mock_vs_cls.return_value = mock_vs

            mock_llm = AsyncMock()
            mock_llm.generate = AsyncMock(return_value="Answer.")
            mock_llm_factory.return_value = mock_llm

            client.post("/query", json={"question": "q?"})

        call_kwargs = mock_vs.search.call_args[1]
        assert call_kwargs["filter"] is None
