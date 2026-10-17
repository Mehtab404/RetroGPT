"""Tests for the embeddings provider abstraction (Issue #8).

Covers:
- EmbeddingProvider ABC contract
- GeminiEmbedding: embed_documents, embed_query, batching, retry logic
- OpenAIEmbedding: embed_documents, embed_query, batching, retry logic
- get_embedding_provider factory
- EmbeddingError raised on bad response shapes
- Exponential backoff: correct sleep durations, retry counts
- Non-retryable HTTP errors raise EmbeddingError immediately

All external HTTP calls are mocked via pytest-mock / unittest.mock.
asyncio tests use pytest-asyncio.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.ingestion.embeddings import (
    _RETRYABLE_STATUS_CODES,
    EmbeddingError,
    EmbeddingProvider,
    GeminiEmbedding,
    OpenAIEmbedding,
    get_embedding_provider,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_gemini_response(texts: list[str], dims: int = 4) -> dict[str, Any]:
    """Build a minimal Gemini batchEmbedContents response."""
    return {
        "embeddings": [{"values": [float(i + j) for j in range(dims)]} for i, _ in enumerate(texts)]
    }


def _fake_openai_response(texts: list[str], dims: int = 4) -> dict[str, Any]:
    """Build a minimal OpenAI /v1/embeddings response."""
    return {
        "data": [
            {
                "index": i,
                "embedding": [float(i + j) for j in range(dims)],
                "object": "embedding",
            }
            for i, _ in enumerate(texts)
        ],
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": 5 * len(texts), "total_tokens": 5 * len(texts)},
    }


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


# ---------------------------------------------------------------------------
# EmbeddingProvider ABC
# ---------------------------------------------------------------------------


class TestEmbeddingProviderABC:
    """Verify the ABC cannot be instantiated directly and enforces interface."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            EmbeddingProvider(model="test")  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_all_methods(self):
        class Partial(EmbeddingProvider):
            async def embed_documents(self, texts):
                return []

            # Missing embed_query

        with pytest.raises(TypeError):
            Partial(model="test")  # type: ignore[abstract]

    def test_concrete_subclass_valid(self):
        class Full(EmbeddingProvider):
            async def embed_documents(self, texts):
                return [[0.0] * 4 for _ in texts]

            async def embed_query(self, text):
                return [0.0] * 4

        provider = Full(model="dummy")
        assert provider.model == "dummy"
        assert provider.max_retries == 3
        assert provider.base_delay == 1.0


# ---------------------------------------------------------------------------
# GeminiEmbedding — happy path
# ---------------------------------------------------------------------------


class TestGeminiEmbeddingHappyPath:
    """Gemini provider works correctly under normal conditions."""

    @pytest.mark.asyncio
    async def test_embed_documents_single_batch(self):
        texts = ["Pong is fun", "Asteroids!"]
        fake_resp = _mock_httpx_response(_fake_gemini_response(texts))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=fake_resp)
            mock_client_cls.return_value = mock_client

            provider = GeminiEmbedding(api_key="fake-key", model="text-embedding-004")
            result = await provider.embed_documents(texts)

        assert len(result) == 2
        assert all(isinstance(v, list) for v in result)
        assert all(isinstance(x, float) for x in result[0])

    @pytest.mark.asyncio
    async def test_embed_query_uses_retrieval_query_task_type(self):
        text = "How do I score in Asteroids?"
        fake_resp = _mock_httpx_response(_fake_gemini_response([text]))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=fake_resp)
            mock_client_cls.return_value = mock_client

            provider = GeminiEmbedding(api_key="fake-key", model="text-embedding-004")
            result = await provider.embed_query(text)

        assert isinstance(result, list)
        assert len(result) == 4  # dims from fake response

        # Verify task type was RETRIEVAL_QUERY
        call_kwargs = mock_client.post.call_args
        payload = (
            call_kwargs.kwargs["json"] if call_kwargs.kwargs.get("json") else call_kwargs.args[1]
        )
        for req in payload["requests"]:
            assert req["taskType"] == "RETRIEVAL_QUERY"

    @pytest.mark.asyncio
    async def test_embed_documents_uses_retrieval_document_task_type(self):
        texts = ["A doc"]
        fake_resp = _mock_httpx_response(_fake_gemini_response(texts))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=fake_resp)
            mock_client_cls.return_value = mock_client

            provider = GeminiEmbedding(api_key="fake-key")
            await provider.embed_documents(texts)

        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs["json"]
        for req in payload["requests"]:
            assert req["taskType"] == "RETRIEVAL_DOCUMENT"

    @pytest.mark.asyncio
    async def test_returns_correct_vector_count(self):
        texts = ["a", "b", "c"]
        fake_resp = _mock_httpx_response(_fake_gemini_response(texts, dims=8))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=fake_resp)
            mock_client_cls.return_value = mock_client

            provider = GeminiEmbedding(api_key="fake-key")
            result = await provider.embed_documents(texts)

        assert len(result) == 3
        assert all(len(v) == 8 for v in result)


# ---------------------------------------------------------------------------
# GeminiEmbedding — batching
# ---------------------------------------------------------------------------


class TestGeminiEmbeddingBatching:
    """Gemini provider sends multiple API calls when texts exceed batch_size."""

    @pytest.mark.asyncio
    async def test_large_input_split_into_batches(self):
        batch_size = 3
        texts = [f"text_{i}" for i in range(7)]  # 3 + 3 + 1 batches

        call_count = 0

        async def fake_post(url, **kwargs):
            nonlocal call_count
            batch = kwargs["json"]["requests"]
            call_count += 1
            return _mock_httpx_response(
                _fake_gemini_response([r["content"]["parts"][0]["text"] for r in batch])
            )

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = fake_post
            mock_client_cls.return_value = mock_client

            provider = GeminiEmbedding(api_key="fake-key", batch_size=batch_size)
            result = await provider.embed_documents(texts)

        assert call_count == 3  # ceil(7 / 3) = 3
        assert len(result) == 7

    @pytest.mark.asyncio
    async def test_batch_size_respects_provider_limit(self):
        """Each POST should receive at most batch_size texts."""
        batch_size = 2
        texts = ["t1", "t2", "t3"]

        batch_sizes_seen: list[int] = []

        async def fake_post(url, **kwargs):
            reqs = kwargs["json"]["requests"]
            batch_sizes_seen.append(len(reqs))
            batch_texts = [r["content"]["parts"][0]["text"] for r in reqs]
            return _mock_httpx_response(_fake_gemini_response(batch_texts))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = fake_post
            mock_client_cls.return_value = mock_client

            provider = GeminiEmbedding(api_key="fake-key", batch_size=batch_size)
            await provider.embed_documents(texts)

        assert all(s <= batch_size for s in batch_sizes_seen)
        assert sum(batch_sizes_seen) == 3

    @pytest.mark.asyncio
    async def test_preserves_order_across_batches(self):
        """Vectors must be returned in the same order as input texts."""
        batch_size = 2
        texts = ["a", "b", "c", "d"]

        async def fake_post(url, **kwargs):
            reqs = kwargs["json"]["requests"]
            batch_texts = [r["content"]["parts"][0]["text"] for r in reqs]
            # Use a unique sentinel float per text for order verification.
            values = [float(ord(t[0])) for t in batch_texts]
            return _mock_httpx_response({"embeddings": [{"values": [v]} for v in values]})

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = fake_post
            mock_client_cls.return_value = mock_client

            provider = GeminiEmbedding(api_key="fake-key", batch_size=batch_size)
            result = await provider.embed_documents(texts)

        expected = [[float(ord(t[0]))] for t in texts]
        assert result == expected


# ---------------------------------------------------------------------------
# GeminiEmbedding — error handling & retry
# ---------------------------------------------------------------------------


class TestGeminiEmbeddingRetry:
    """Verify retry behaviour for Gemini provider."""

    @pytest.mark.asyncio
    async def test_retries_on_429(self):
        texts = ["hello"]
        rate_limit_resp = _mock_httpx_response({}, status_code=429)
        success_resp = _mock_httpx_response(_fake_gemini_response(texts))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            # First call → 429, second call → 200
            mock_client.post = AsyncMock(side_effect=[rate_limit_resp, success_resp])
            mock_client_cls.return_value = mock_client

            provider = GeminiEmbedding(
                api_key="fake-key", max_retries=2, base_delay=0.0, jitter=False
            )
            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                result = await provider.embed_documents(texts)

        assert len(result) == 1
        assert mock_client.post.call_count == 2
        mock_sleep.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retries_on_500(self):
        texts = ["hello"]
        server_err = _mock_httpx_response({}, status_code=500)
        success = _mock_httpx_response(_fake_gemini_response(texts))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=[server_err, success])
            mock_client_cls.return_value = mock_client

            provider = GeminiEmbedding(
                api_key="fake-key", max_retries=3, base_delay=0.0, jitter=False
            )
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await provider.embed_documents(texts)

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_raises_embedding_error_after_all_retries_exhausted(self):
        texts = ["hello"]
        server_err = _mock_httpx_response({}, status_code=503)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            # Always fail
            mock_client.post = AsyncMock(return_value=server_err)
            mock_client_cls.return_value = mock_client

            provider = GeminiEmbedding(
                api_key="fake-key", max_retries=2, base_delay=0.0, jitter=False
            )
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(EmbeddingError, match="failed after"):
                    await provider.embed_documents(texts)

        assert mock_client.post.call_count == 3  # initial + 2 retries

    @pytest.mark.asyncio
    async def test_non_retryable_400_raises_immediately(self):
        texts = ["hello"]
        bad_req = _mock_httpx_response({}, status_code=400)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=bad_req)
            mock_client_cls.return_value = mock_client

            provider = GeminiEmbedding(api_key="fake-key", max_retries=3, jitter=False)
            with pytest.raises(EmbeddingError, match="Non-retryable HTTP error 400"):
                await provider.embed_documents(texts)

        # Should NOT retry — only one call
        assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_exponential_backoff_delay_doubles(self):
        """Each retry doubles the sleep delay (base_delay * 2^attempt)."""
        texts = ["hello"]
        err = _mock_httpx_response({}, status_code=503)
        ok = _mock_httpx_response(_fake_gemini_response(texts))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            # Fail twice, then succeed
            mock_client.post = AsyncMock(side_effect=[err, err, ok])
            mock_client_cls.return_value = mock_client

            sleep_args: list[float] = []

            async def mock_sleep(delay):
                sleep_args.append(delay)

            provider = GeminiEmbedding(
                api_key="fake-key", max_retries=3, base_delay=1.0, jitter=False
            )
            with patch("asyncio.sleep", side_effect=mock_sleep):
                await provider.embed_documents(texts)

        assert len(sleep_args) == 2
        # Without jitter: delay[0] = 1.0 * 2^0 = 1.0, delay[1] = 1.0 * 2^1 = 2.0
        assert sleep_args[0] == pytest.approx(1.0)
        assert sleep_args[1] == pytest.approx(2.0)

    @pytest.mark.asyncio
    async def test_request_error_triggers_retry(self):
        texts = ["hello"]
        ok = _mock_httpx_response(_fake_gemini_response(texts))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=[httpx.ConnectError("timeout"), ok])
            mock_client_cls.return_value = mock_client

            provider = GeminiEmbedding(
                api_key="fake-key", max_retries=2, base_delay=0.0, jitter=False
            )
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await provider.embed_documents(texts)

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_malformed_response_raises_embedding_error(self):
        texts = ["hello"]
        # Response has wrong number of embeddings
        bad_data = {"embeddings": []}
        resp = _mock_httpx_response(bad_data)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=resp)
            mock_client_cls.return_value = mock_client

            provider = GeminiEmbedding(api_key="fake-key", max_retries=0)
            with pytest.raises(EmbeddingError, match="Unexpected Gemini response shape"):
                await provider.embed_documents(texts)


# ---------------------------------------------------------------------------
# OpenAIEmbedding — happy path
# ---------------------------------------------------------------------------


class TestOpenAIEmbeddingHappyPath:
    """OpenAI provider works correctly under normal conditions."""

    @pytest.mark.asyncio
    async def test_embed_documents_basic(self):
        texts = ["Hello OpenAI", "Missile Command!"]
        fake_resp = _mock_httpx_response(_fake_openai_response(texts))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=fake_resp)
            mock_client_cls.return_value = mock_client

            provider = OpenAIEmbedding(api_key="sk-fake", model="text-embedding-3-small")
            result = await provider.embed_documents(texts)

        assert len(result) == 2
        assert all(isinstance(v, list) for v in result)

    @pytest.mark.asyncio
    async def test_embed_query_basic(self):
        text = "What is the highest score in Centipede?"
        fake_resp = _mock_httpx_response(_fake_openai_response([text]))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=fake_resp)
            mock_client_cls.return_value = mock_client

            provider = OpenAIEmbedding(api_key="sk-fake")
            result = await provider.embed_query(text)

        assert isinstance(result, list)
        assert len(result) == 4

    @pytest.mark.asyncio
    async def test_bearer_token_in_headers(self):
        texts = ["test"]
        fake_resp = _mock_httpx_response(_fake_openai_response(texts))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=fake_resp)
            mock_client_cls.return_value = mock_client

            provider = OpenAIEmbedding(api_key="sk-test-123")
            await provider.embed_documents(texts)

        call_kwargs = mock_client.post.call_args.kwargs
        headers = call_kwargs["headers"]
        assert headers["Authorization"] == "Bearer sk-test-123"

    @pytest.mark.asyncio
    async def test_correct_api_url(self):
        texts = ["test"]
        fake_resp = _mock_httpx_response(_fake_openai_response(texts))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=fake_resp)
            mock_client_cls.return_value = mock_client

            provider = OpenAIEmbedding(api_key="sk-fake")
            await provider.embed_documents(texts)

        url = mock_client.post.call_args.args[0]
        assert url == "https://api.openai.com/v1/embeddings"

    @pytest.mark.asyncio
    async def test_custom_base_url(self):
        texts = ["test"]
        fake_resp = _mock_httpx_response(_fake_openai_response(texts))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=fake_resp)
            mock_client_cls.return_value = mock_client

            provider = OpenAIEmbedding(
                api_key="sk-fake",
                base_url="https://my-proxy.example.com",
            )
            await provider.embed_documents(texts)

        url = mock_client.post.call_args.args[0]
        assert url == "https://my-proxy.example.com/v1/embeddings"

    @pytest.mark.asyncio
    async def test_results_sorted_by_index(self):
        """OpenAI results should be sorted by 'index' before returning."""
        texts = ["a", "b", "c"]
        # Deliberately return items out of order
        out_of_order_resp = {
            "data": [
                {"index": 2, "embedding": [2.0], "object": "embedding"},
                {"index": 0, "embedding": [0.0], "object": "embedding"},
                {"index": 1, "embedding": [1.0], "object": "embedding"},
            ]
        }
        fake_resp = _mock_httpx_response(out_of_order_resp)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=fake_resp)
            mock_client_cls.return_value = mock_client

            provider = OpenAIEmbedding(api_key="sk-fake")
            result = await provider.embed_documents(texts)

        assert result == [[0.0], [1.0], [2.0]]


# ---------------------------------------------------------------------------
# OpenAIEmbedding — batching
# ---------------------------------------------------------------------------


class TestOpenAIEmbeddingBatching:
    """OpenAI provider sends multiple API calls for large inputs."""

    @pytest.mark.asyncio
    async def test_large_input_split_into_batches(self):
        batch_size = 3
        texts = [f"text_{i}" for i in range(8)]  # 3 + 3 + 2 batches

        call_count = 0

        async def fake_post(url, **kwargs):
            nonlocal call_count
            batch = kwargs["json"]["input"]
            call_count += 1
            return _mock_httpx_response(_fake_openai_response(batch))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = fake_post
            mock_client_cls.return_value = mock_client

            provider = OpenAIEmbedding(api_key="sk-fake", batch_size=batch_size)
            result = await provider.embed_documents(texts)

        assert call_count == 3  # ceil(8 / 3) = 3
        assert len(result) == 8

    @pytest.mark.asyncio
    async def test_each_batch_within_size_limit(self):
        batch_size = 4
        texts = [f"t{i}" for i in range(9)]

        batch_sizes: list[int] = []

        async def fake_post(url, **kwargs):
            batch = kwargs["json"]["input"]
            batch_sizes.append(len(batch))
            return _mock_httpx_response(_fake_openai_response(batch))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = fake_post
            mock_client_cls.return_value = mock_client

            provider = OpenAIEmbedding(api_key="sk-fake", batch_size=batch_size)
            await provider.embed_documents(texts)

        assert all(s <= batch_size for s in batch_sizes)
        assert sum(batch_sizes) == 9


# ---------------------------------------------------------------------------
# OpenAIEmbedding — error handling & retry
# ---------------------------------------------------------------------------


class TestOpenAIEmbeddingRetry:
    """Verify retry behaviour for OpenAI provider."""

    @pytest.mark.asyncio
    async def test_retries_on_429_rate_limit(self):
        texts = ["hello"]
        rate_limit = _mock_httpx_response({}, status_code=429)
        success = _mock_httpx_response(_fake_openai_response(texts))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=[rate_limit, success])
            mock_client_cls.return_value = mock_client

            provider = OpenAIEmbedding(
                api_key="sk-fake", max_retries=2, base_delay=0.0, jitter=False
            )
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await provider.embed_documents(texts)

        assert len(result) == 1
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_raises_after_max_retries_exhausted(self):
        texts = ["hello"]
        err = _mock_httpx_response({}, status_code=503)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=err)
            mock_client_cls.return_value = mock_client

            provider = OpenAIEmbedding(
                api_key="sk-fake", max_retries=1, base_delay=0.0, jitter=False
            )
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(EmbeddingError, match="failed after"):
                    await provider.embed_documents(texts)

        assert mock_client.post.call_count == 2  # 1 initial + 1 retry

    @pytest.mark.asyncio
    async def test_non_retryable_401_raises_immediately(self):
        texts = ["hello"]
        unauth = _mock_httpx_response({}, status_code=401)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=unauth)
            mock_client_cls.return_value = mock_client

            provider = OpenAIEmbedding(api_key="sk-fake", max_retries=3, jitter=False)
            with pytest.raises(EmbeddingError, match="Non-retryable HTTP error 401"):
                await provider.embed_documents(texts)

        assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_malformed_response_raises_embedding_error(self):
        texts = ["hello", "world"]
        # 'data' has only 1 item instead of 2
        bad_data = {"data": [{"index": 0, "embedding": [1.0]}]}
        resp = _mock_httpx_response(bad_data)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=resp)
            mock_client_cls.return_value = mock_client

            provider = OpenAIEmbedding(api_key="sk-fake", max_retries=0)
            with pytest.raises(EmbeddingError, match="Unexpected OpenAI response shape"):
                await provider.embed_documents(texts)

    @pytest.mark.asyncio
    async def test_exponential_backoff_delay_doubles(self):
        texts = ["hello"]
        err = _mock_httpx_response({}, status_code=502)
        ok = _mock_httpx_response(_fake_openai_response(texts))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=[err, err, ok])
            mock_client_cls.return_value = mock_client

            sleep_args: list[float] = []

            async def mock_sleep(delay):
                sleep_args.append(delay)

            provider = OpenAIEmbedding(
                api_key="sk-fake", max_retries=3, base_delay=2.0, jitter=False
            )
            with patch("asyncio.sleep", side_effect=mock_sleep):
                await provider.embed_documents(texts)

        # base_delay=2.0, no jitter: delays = [2.0, 4.0]
        assert len(sleep_args) == 2
        assert sleep_args[0] == pytest.approx(2.0)
        assert sleep_args[1] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# get_embedding_provider factory
# ---------------------------------------------------------------------------


class TestGetEmbeddingProvider:
    """Factory function returns correct provider based on configuration."""

    def test_returns_gemini_for_gemini_string(self):
        provider = get_embedding_provider("gemini")
        assert isinstance(provider, GeminiEmbedding)

    def test_returns_openai_for_openai_string(self):
        provider = get_embedding_provider("openai")
        assert isinstance(provider, OpenAIEmbedding)

    def test_case_insensitive(self):
        assert isinstance(get_embedding_provider("Gemini"), GeminiEmbedding)
        assert isinstance(get_embedding_provider("OPENAI"), OpenAIEmbedding)

    def test_raises_for_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            get_embedding_provider("anthropic")

    def test_reads_from_settings_when_no_provider_given(self):
        with patch("app.ingestion.embeddings.settings") as mock_settings:
            mock_settings.embedding_provider = "gemini"
            mock_settings.embedding_model = "text-embedding-004"
            mock_settings.google_api_key = ""
            mock_settings.openai_api_key = ""
            provider = get_embedding_provider()
        assert isinstance(provider, GeminiEmbedding)

    def test_kwargs_forwarded_to_provider(self):
        provider = get_embedding_provider("gemini", max_retries=5, batch_size=50)
        assert isinstance(provider, GeminiEmbedding)
        assert provider.max_retries == 5
        assert provider.batch_size == 50

    def test_openai_kwargs_forwarded(self):
        provider = get_embedding_provider("openai", batch_size=100, base_delay=2.0)
        assert isinstance(provider, OpenAIEmbedding)
        assert provider.batch_size == 100
        assert provider.base_delay == 2.0


# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------


class TestProviderConfiguration:
    """Test that providers respect constructor arguments correctly."""

    def test_gemini_default_model_from_settings(self):
        with patch("app.ingestion.embeddings.settings") as mock_settings:
            mock_settings.embedding_model = "text-embedding-005"
            mock_settings.google_api_key = "key"
            provider = GeminiEmbedding()
        assert provider.model == "text-embedding-005"

    def test_openai_default_model_from_settings(self):
        with patch("app.ingestion.embeddings.settings") as mock_settings:
            mock_settings.embedding_model = "text-embedding-3-large"
            mock_settings.openai_api_key = "key"
            provider = OpenAIEmbedding()
        assert provider.model == "text-embedding-3-large"

    def test_gemini_explicit_model_overrides_settings(self):
        provider = GeminiEmbedding(model="text-embedding-004", api_key="k")
        assert provider.model == "text-embedding-004"

    def test_openai_explicit_model_overrides_settings(self):
        provider = OpenAIEmbedding(model="text-embedding-ada-002", api_key="k")
        assert provider.model == "text-embedding-ada-002"

    def test_gemini_is_embedding_provider_subclass(self):
        assert issubclass(GeminiEmbedding, EmbeddingProvider)

    def test_openai_is_embedding_provider_subclass(self):
        assert issubclass(OpenAIEmbedding, EmbeddingProvider)

    def test_retryable_status_codes_include_429(self):
        assert 429 in _RETRYABLE_STATUS_CODES

    def test_retryable_status_codes_include_5xx(self):
        for code in (500, 502, 503, 504):
            assert code in _RETRYABLE_STATUS_CODES

    def test_retryable_does_not_include_400(self):
        assert 400 not in _RETRYABLE_STATUS_CODES

    def test_retryable_does_not_include_401(self):
        assert 401 not in _RETRYABLE_STATUS_CODES
