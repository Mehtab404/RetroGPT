"""Tests for token streaming (Issue #25).

Covers:
- ``LLMProvider.generate_stream`` default fallback (non-streaming generate).
- ``GeminiLLM.generate_stream``: SSE frame parsing, keep-alive / finish-only
  frames ignored, retry before the first chunk, no retry after the first
  chunk, non-retryable errors.
- ``OpenAILLM.generate_stream``: delta parsing, ``[DONE]`` handling,
  non-retryable errors.
- ``POST /query/stream``: ``meta`` / ``token`` / ``done`` ordering, source
  metadata, no-context path, pre-stream HTTP errors (502/503), mid-stream
  ``error`` events, and structured ``query`` logging (``streamed=True``,
  ``chunk_count``).

All external HTTP calls are mocked via unittest.mock / pytest-mock.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from app.generation.llm import GeminiLLM, LLMError, LLMProvider, OpenAILLM
from app.ingestion.embeddings import EmbeddingError
from app.logging_config import JsonFormatter
from app.main import app
from app.retrieval.vector_store import SearchResult, VectorStoreError

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_sse(text: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse an SSE response body into ``(event, payload)`` pairs."""
    events: list[tuple[str, dict[str, Any]]] = []
    for frame in text.split("\n\n"):
        frame = frame.strip()
        if not frame:
            continue
        event_name = "message"
        data_lines: list[str] = []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
        events.append((event_name, json.loads("\n".join(data_lines))))
    return events


def _stream_response(lines: list[str], status: int = 200) -> MagicMock:
    """Build a MagicMock shaped like a streaming ``httpx.Response``."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = status
    response.text = "error body"
    if status >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status}", request=MagicMock(), response=response
        )
    else:
        response.raise_for_status.return_value = None

    async def aiter_lines(_lines: list[str] = lines):
        for line in _lines:
            yield line

    response.aiter_lines = aiter_lines
    return response


def _stream_context(response: MagicMock) -> MagicMock:
    """Wrap a mocked response in an async context manager like client.stream()."""
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=response)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


def _mock_client(contexts: list[MagicMock]) -> AsyncMock:
    """Build an AsyncMock shaped like ``httpx.AsyncClient``."""
    client_mock = AsyncMock()
    client_mock.__aenter__ = AsyncMock(return_value=client_mock)
    client_mock.__aexit__ = AsyncMock(return_value=False)
    client_mock.stream = MagicMock(side_effect=contexts)
    return client_mock


def _gemini_line(text: str) -> list[str]:
    payload = {"candidates": [{"content": {"parts": [{"text": text}]}}]}
    return [f"data: {json.dumps(payload)}", ""]


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


def _streaming_llm(chunks: list[str]) -> MagicMock:
    """Build an LLM mock whose ``generate_stream`` yields *chunks*."""
    llm = MagicMock()

    async def stream(prompt: str):
        for chunk in chunks:
            yield chunk

    llm.generate_stream = stream
    llm.generate = AsyncMock(return_value="".join(chunks))
    return llm


def _failing_stream_llm(chunks: list[str], error: str) -> MagicMock:
    """Build an LLM mock that yields *chunks* then raises :class:`LLMError`."""
    llm = MagicMock()

    async def stream(prompt: str):
        for chunk in chunks:
            yield chunk
        raise LLMError(error)

    llm.generate_stream = stream
    llm.generate = AsyncMock(return_value="".join(chunks))
    return llm


class _JsonCaptureHandler(logging.Handler):
    """Collects :class:`JsonFormatter` payloads at emission time."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.setFormatter(JsonFormatter())
        self.payloads: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.payloads.append(json.loads(self.format(record)))


# ---------------------------------------------------------------------------
# LLMProvider.generate_stream (default fallback)
# ---------------------------------------------------------------------------


class TestGenerateStreamFallback:
    """The ABC default yields the whole (non-streamed) answer as one chunk."""

    async def test_falls_back_to_generate(self) -> None:
        class _NonStreamingProvider(LLMProvider):
            async def generate(self, prompt: str) -> str:
                return "whole answer"

        provider = _NonStreamingProvider(model="test-model")

        chunks = [chunk async for chunk in provider.generate_stream("prompt")]

        assert chunks == ["whole answer"]


# ---------------------------------------------------------------------------
# Gemini streaming
# ---------------------------------------------------------------------------


class TestGeminiStream:
    async def test_streams_text_chunks_and_ignores_metadata_frames(self) -> None:
        lines = [
            *_gemini_line("Large "),
            *_gemini_line("asteroids"),
            'data: {"candidates":[{"content":{"role":"model"},"finishReason":"STOP"}]}',
            "data: not-json",
            "",
        ]
        client_mock = _mock_client([_stream_context(_stream_response(lines))])
        provider = GeminiLLM(api_key="test-key", max_retries=0)

        with patch("httpx.AsyncClient", return_value=client_mock):
            chunks = [chunk async for chunk in provider.generate_stream("prompt")]

        assert chunks == ["Large ", "asteroids"]
        url = client_mock.stream.call_args[0][1]
        assert ":streamGenerateContent" in url
        assert "alt=sse" in url

    async def test_retries_transient_error_before_first_chunk(self) -> None:
        failed = _stream_context(_stream_response([], status=503))
        ok = _stream_context(_stream_response(_gemini_line("Hello")))
        client_mock = _mock_client([failed, ok])
        provider = GeminiLLM(api_key="test-key", base_delay=0.0, jitter=False)

        with patch("httpx.AsyncClient", return_value=client_mock):
            chunks = [chunk async for chunk in provider.generate_stream("prompt")]

        assert chunks == ["Hello"]
        assert client_mock.stream.call_count == 2

    async def test_does_not_retry_after_first_chunk(self) -> None:
        response = MagicMock(spec=httpx.Response)
        response.raise_for_status.return_value = None

        async def aiter_lines():
            yield 'data: {"candidates":[{"content":{"parts":[{"text":"Partial"}]}}]}'
            raise httpx.ReadError("connection dropped")

        response.aiter_lines = aiter_lines
        client_mock = _mock_client([_stream_context(response)])
        provider = GeminiLLM(api_key="test-key", base_delay=0.0, jitter=False)

        received: list[str] = []
        with patch("httpx.AsyncClient", return_value=client_mock):
            with pytest.raises(LLMError):
                async for chunk in provider.generate_stream("prompt"):
                    received.append(chunk)

        assert received == ["Partial"]
        assert client_mock.stream.call_count == 1

    async def test_non_retryable_error_raises_without_retry(self) -> None:
        client_mock = _mock_client([_stream_context(_stream_response([], status=401))])
        provider = GeminiLLM(api_key="test-key", base_delay=0.0, jitter=False)

        with patch("httpx.AsyncClient", return_value=client_mock):
            with pytest.raises(LLMError):
                [chunk async for chunk in provider.generate_stream("prompt")]

        assert client_mock.stream.call_count == 1


# ---------------------------------------------------------------------------
# OpenAI streaming
# ---------------------------------------------------------------------------


class TestOpenAIStream:
    async def test_streams_deltas_and_stops_at_done(self) -> None:
        lines = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            "",
            'data: {"choices":[{"delta":{"content":" world"}}]}',
            "",
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ]
        client_mock = _mock_client([_stream_context(_stream_response(lines))])
        provider = OpenAILLM(api_key="test-key", max_retries=0)

        with patch("httpx.AsyncClient", return_value=client_mock):
            chunks = [chunk async for chunk in provider.generate_stream("prompt")]

        assert chunks == ["Hello", " world"]
        _, kwargs = client_mock.stream.call_args
        assert kwargs["json"]["stream"] is True

    async def test_non_retryable_error_raises_immediately(self) -> None:
        client_mock = _mock_client([_stream_context(_stream_response([], status=401))])
        provider = OpenAILLM(api_key="test-key", base_delay=0.0, jitter=False)

        with patch("httpx.AsyncClient", return_value=client_mock):
            with pytest.raises(LLMError):
                [chunk async for chunk in provider.generate_stream("prompt")]

        assert client_mock.stream.call_count == 1


# ---------------------------------------------------------------------------
# POST /query/stream
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _deterministic_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the endpoint onto the deterministic dense-only, no-rerank path."""
    monkeypatch.setattr("app.retrieval.hybrid.get_bm25_encoder", lambda path=None: None)
    monkeypatch.setattr("app.retrieval.rerank.get_reranker", lambda: None)


def _patch_deps(
    search_results: list[SearchResult],
    llm: MagicMock,
) -> tuple[Any, Any, Any, Any]:
    """Start the standard dependency patches (call inside a `with` block)."""
    embed_patch = patch("app.main.get_embedding_provider")
    vs_patch = patch("app.main.QdrantVectorStore")
    llm_patch = patch("app.main.get_llm_provider")
    mock_embed_factory = embed_patch.start()
    mock_vs_cls = vs_patch.start()
    mock_llm_factory = llm_patch.start()

    mock_embedder = AsyncMock()
    mock_embedder.embed_query = AsyncMock(return_value=[0.1] * 768)
    mock_embed_factory.return_value = mock_embedder

    mock_vs = AsyncMock()
    mock_vs.search = AsyncMock(return_value=search_results)
    mock_vs_cls.return_value = mock_vs

    mock_llm_factory.return_value = llm

    return embed_patch, vs_patch, llm_patch, mock_vs


class TestQueryStreamEndpoint:
    def test_streams_meta_tokens_and_done(self) -> None:
        search_result = _make_search_result(
            chunk_id="asteroids_p3_c2",
            score=0.91,
            content="Large asteroids: 20 points each.",
            title="Asteroids Manual",
        )
        llm = _streaming_llm(["Large ", "asteroids ", "are worth 20 points [1]."])

        patches = _patch_deps([search_result], llm)
        try:
            response = client.post(
                "/query/stream",
                json={"question": "How do you score points in Asteroids?"},
            )
        finally:
            for patcher in patches[:3]:
                patcher.stop()

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        events = _parse_sse(response.text)
        names = [name for name, _ in events]
        assert names[0] == "meta"
        assert names[-1] == "done"
        assert set(names) == {"meta", "token", "done"}

        meta = events[0][1]
        assert len(meta["sources"]) == 1
        assert meta["sources"][0]["id"] == "asteroids_p3_c2"
        assert meta["sources"][0]["score"] == 0.91

        tokens = "".join(payload["text"] for name, payload in events if name == "token")
        assert tokens == "Large asteroids are worth 20 points [1]."

        done = events[-1][1]
        assert done["latency_ms"] >= 0.0

    def test_no_context_streams_single_dont_know_token(self) -> None:
        llm = _streaming_llm(["unused"])

        patches = _patch_deps([], llm)
        try:
            response = client.post("/query/stream", json={"question": "off topic?"})
        finally:
            for patcher in patches[:3]:
                patcher.stop()

        assert response.status_code == 200
        events = _parse_sse(response.text)
        assert events[0] == ("meta", {"sources": []})

        tokens = [payload["text"] for name, payload in events if name == "token"]
        assert len(tokens) == 1
        assert "I don't know" in tokens[0]
        assert events[-1][0] == "done"

    def test_embedding_error_returns_json_502_before_streaming(self) -> None:
        with patch("app.main.get_embedding_provider") as mock_embed_factory:
            mock_embedder = AsyncMock()
            mock_embedder.embed_query = AsyncMock(side_effect=EmbeddingError("no api key"))
            mock_embed_factory.return_value = mock_embedder

            response = client.post("/query/stream", json={"question": "q?"})

        assert response.status_code == 502
        assert response.headers["content-type"].startswith("application/json")
        assert "Embedding provider error" in response.json()["detail"]

    def test_vector_store_error_returns_json_503_before_streaming(self) -> None:
        with (
            patch("app.main.get_embedding_provider") as mock_embed_factory,
            patch("app.main.QdrantVectorStore") as mock_vs_cls,
        ):
            mock_embedder = AsyncMock()
            mock_embedder.embed_query = AsyncMock(return_value=[0.1] * 768)
            mock_embed_factory.return_value = mock_embedder

            mock_vs = AsyncMock()
            mock_vs.search = AsyncMock(side_effect=VectorStoreError("qdrant down"))
            mock_vs_cls.return_value = mock_vs

            response = client.post("/query/stream", json={"question": "q?"})

        assert response.status_code == 503
        assert response.headers["content-type"].startswith("application/json")

    def test_generation_error_emits_error_event_instead_of_done(self) -> None:
        search_result = _make_search_result()
        llm = _failing_stream_llm(["Partial "], "upstream exploded")

        patches = _patch_deps([search_result], llm)
        try:
            response = client.post("/query/stream", json={"question": "q?"})
        finally:
            for patcher in patches[:3]:
                patcher.stop()

        assert response.status_code == 200
        events = _parse_sse(response.text)
        names = [name for name, _ in events]
        assert "error" in names
        assert "done" not in names

        tokens = [payload["text"] for name, payload in events if name == "token"]
        assert tokens == ["Partial "]

        error_payload = next(payload for name, payload in events if name == "error")
        assert error_payload["code"] == "LLM_ERROR"
        assert "upstream exploded" in error_payload["message"]

    def test_query_log_includes_streamed_flag_and_chunk_count(self) -> None:
        search_result = _make_search_result()
        # Blank chunks are skipped and must not inflate chunk_count.
        llm = _streaming_llm(["a", "", "b"])

        patches = _patch_deps([search_result], llm)
        handler = _JsonCaptureHandler()
        main_logger = logging.getLogger("app.main")
        main_logger.addHandler(handler)
        try:
            response = client.post(
                "/query/stream",
                json={"question": "q?"},
                headers={"x-request-id": "trace-stream-1"},
            )
        finally:
            main_logger.removeHandler(handler)
            for patcher in patches[:3]:
                patcher.stop()

        assert response.status_code == 200
        event = next(p for p in handler.payloads if p["event"] == "query")
        assert event["streamed"] is True
        assert event["chunk_count"] == 2
        assert event["request_id"] == "trace-stream-1"
        assert set(event["stage_ms"]) == {"embed", "retrieve", "rerank", "generate"}

    def test_generation_error_logs_query_event_with_partial_chunk_count(self) -> None:
        search_result = _make_search_result()
        llm = _failing_stream_llm(["x", "y"], "boom")

        patches = _patch_deps([search_result], llm)
        handler = _JsonCaptureHandler()
        main_logger = logging.getLogger("app.main")
        main_logger.addHandler(handler)
        try:
            response = client.post("/query/stream", json={"question": "q?"})
        finally:
            main_logger.removeHandler(handler)
            for patcher in patches[:3]:
                patcher.stop()

        assert response.status_code == 200
        event = next(p for p in handler.payloads if p["event"] == "query")
        assert event["streamed"] is True
        assert event["chunk_count"] == 2

    def test_history_is_rendered_into_the_streamed_prompt(self) -> None:
        search_result = _make_search_result(chunk_id="asteroids_p4_c1")
        captured: dict[str, str] = {}

        llm = MagicMock()

        async def stream(prompt: str):
            captured["prompt"] = prompt
            yield "Saucers are worth 1000 points [1]."

        llm.generate_stream = stream
        llm.generate = AsyncMock(return_value="Saucers are worth 1000 points [1].")

        patches = _patch_deps([search_result], llm)
        try:
            response = client.post(
                "/query/stream",
                json={
                    "question": "What about the saucer?",
                    "history": [
                        {"role": "user", "content": "How do you score in Asteroids?"},
                        {"role": "assistant", "content": "Shoot asteroids [1]."},
                    ],
                },
            )
        finally:
            for patcher in patches[:3]:
                patcher.stop()

        assert response.status_code == 200
        assert "PRIOR CONVERSATION" in captured["prompt"]
        assert "USER: How do you score in Asteroids?" in captured["prompt"]
        assert "ASSISTANT: Shoot asteroids [1]." in captured["prompt"]
        assert "[1] (source: asteroids_p4_c1)" in captured["prompt"]

    def test_invalid_history_returns_422_before_streaming(self) -> None:
        response = client.post(
            "/query/stream",
            json={"question": "q?", "history": [{"role": "tool", "content": "x"}]},
        )
        assert response.status_code == 422
        assert response.headers["content-type"].startswith("application/json")
