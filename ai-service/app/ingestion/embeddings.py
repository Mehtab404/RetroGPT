"""Embeddings provider abstraction for the RAG ingestion pipeline.

Provides a uniform :class:`EmbeddingProvider` interface for producing
dense vector embeddings from text, with concrete implementations for:

* :class:`GeminiEmbedding`    — Google Generative AI ``text-embedding-004``
* :class:`OpenAIEmbedding`    — OpenAI ``text-embedding-3-small`` (or any model)
* :class:`FastEmbedEmbedding` — local ONNX embeddings via ``fastembed``
  (no API key required)

All implementations support:

* **Batching** — large lists of texts are split into provider-appropriate
  batches before being sent, reducing per-call overhead.
* **Retry with exponential backoff** — transient HTTP errors (429, 5xx) and
  network timeouts are retried up to *max_retries* times, doubling the wait
  time on each attempt, with optional jitter.  (Local fastembed inference
  needs no retries; the arguments are accepted for interface uniformity.)

Typical usage::

    from app.ingestion.embeddings import get_embedding_provider

    provider = get_embedding_provider()          # reads from settings
    vectors = await provider.embed_documents(["Hello world", "Asteroids!"])
    query_vec = await provider.embed_query("How do I score?")

The HTTP providers use ``httpx.AsyncClient`` internally; the fastembed
provider runs locally through ONNX Runtime.
"""

from __future__ import annotations

import asyncio
import logging
import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum texts per batch for each provider (empirical / documented limits).
_GEMINI_BATCH_SIZE: int = 100
_OPENAI_BATCH_SIZE: int = 2048

# Default HTTP timeout for embedding calls (seconds).
_DEFAULT_TIMEOUT: float = 30.0

# HTTP status codes considered transiently retryable.
_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})

#: Known output dimensionalities for common embedding models.  Used to size
#: the Qdrant collection automatically at ingestion time; for unknown models
#: callers fall back to ``settings.vector_size``.
_KNOWN_EMBEDDING_DIMS: dict[str, int] = {
    # Gemini
    "text-embedding-004": 768,
    "gemini-embedding-001": 3072,
    # OpenAI
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    # fastembed (local ONNX)
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    Subclasses must implement :meth:`embed_documents` and
    :meth:`embed_query`.  The contract for both is that the returned
    vectors are L2-normalised (unit-length), matching the expectation of
    most vector databases including Qdrant.

    Args:
        model:        Name of the embedding model to use.
        max_retries:  Number of retry attempts for transient errors.
        base_delay:   Initial retry delay in seconds (doubled each attempt).
        jitter:       If ``True``, add random jitter to each retry delay
                      to reduce thundering-herd effects.
        timeout:      HTTP timeout in seconds.
    """

    def __init__(
        self,
        model: str,
        max_retries: int = 3,
        base_delay: float = 1.0,
        jitter: bool = True,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.jitter = jitter
        self.timeout = timeout

    @property
    def dimension(self) -> int | None:
        """Output dimensionality of the model, when known.

        Returns ``None`` for models outside the built-in registry; callers
        should then fall back to ``settings.vector_size``.
        """
        return _KNOWN_EMBEDDING_DIMS.get(self.model)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a list of document texts.

        Args:
            texts: Non-empty sequence of strings to embed.

        Returns:
            A list of float vectors, one per input text, in the same order.

        Raises:
            EmbeddingError: If the API returns a non-retryable error after
                            all retry attempts are exhausted.
        """

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        Queries are typically encoded differently from documents in
        asymmetric models (e.g. Gemini uses ``RETRIEVAL_QUERY`` task type).

        Args:
            text: The query string to embed.

        Returns:
            A float vector for the query.

        Raises:
            EmbeddingError: On unrecoverable API errors.
        """

    # ------------------------------------------------------------------
    # Retry helper (shared by all subclasses)
    # ------------------------------------------------------------------

    async def _with_retry(self, coro_fn, *args, **kwargs) -> object:
        """Execute *coro_fn* with exponential-backoff retry.

        *coro_fn* should be an async callable that raises
        :class:`httpx.HTTPStatusError` or :class:`httpx.RequestError`
        on failure.  Only :data:`_RETRYABLE_STATUS_CODES` trigger a retry;
        4xx errors not in that set are re-raised immediately as
        :class:`EmbeddingError`.

        Args:
            coro_fn:  Async callable to invoke.
            *args:    Positional arguments forwarded to *coro_fn*.
            **kwargs: Keyword arguments forwarded to *coro_fn*.

        Returns:
            The return value of *coro_fn* on success.

        Raises:
            EmbeddingError: After all retries are exhausted, or on a
                            non-retryable HTTP error.
        """
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return await coro_fn(*args, **kwargs)
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status not in _RETRYABLE_STATUS_CODES:
                    raise EmbeddingError(
                        f"Non-retryable HTTP error {status}: {exc.response.text}"
                    ) from exc
                last_exc = exc
                logger.warning(
                    "Embedding API returned %s (attempt %d/%d).",
                    status,
                    attempt + 1,
                    self.max_retries + 1,
                )
            except httpx.RequestError as exc:
                last_exc = exc
                logger.warning(
                    "Embedding API request error (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )

            if attempt < self.max_retries:
                delay = self.base_delay * (2**attempt)
                if self.jitter:
                    delay *= 1 + random.random() * 0.2  # ±20% jitter
                logger.debug("Retrying in %.2f s…", delay)
                await asyncio.sleep(delay)

        raise EmbeddingError(
            f"Embedding API failed after {self.max_retries + 1} attempts."
        ) from last_exc


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class EmbeddingError(RuntimeError):
    """Raised when an embedding provider call fails permanently."""


# ---------------------------------------------------------------------------
# Gemini implementation
# ---------------------------------------------------------------------------


class GeminiEmbedding(EmbeddingProvider):
    """Embeddings via Google Generative AI (``text-embedding-004``).

    Uses the ``embedContent`` REST endpoint (v1beta).  Documents are
    embedded with task type ``RETRIEVAL_DOCUMENT``; queries use
    ``RETRIEVAL_QUERY`` so asymmetric retrieval works correctly.

    Args:
        model:       Gemini embedding model name.
                     Defaults to ``settings.embedding_model``.
        api_key:     Google API key.
                     Defaults to ``settings.google_api_key``.
        batch_size:  Number of texts per batch API call.
                     Gemini's ``batchEmbedContents`` supports up to 100.
        max_retries: Number of retry attempts (default 3).
        base_delay:  Initial retry back-off in seconds (default 1.0).
        jitter:      Add random jitter to delays (default ``True``).
        timeout:     HTTP timeout in seconds (default 30).

    Example::

        provider = GeminiEmbedding(api_key="...", model="text-embedding-004")
        vecs = await provider.embed_documents(["Pong rules", "Asteroids FAQ"])
    """

    _BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        batch_size: int = _GEMINI_BATCH_SIZE,
        max_retries: int = 3,
        base_delay: float = 1.0,
        jitter: bool = True,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        resolved_model = model or settings.embedding_model
        super().__init__(
            model=resolved_model,
            max_retries=max_retries,
            base_delay=base_delay,
            jitter=jitter,
            timeout=timeout,
        )
        self._api_key = api_key or settings.google_api_key
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed document texts using ``batchEmbedContents``.

        Splits *texts* into batches of :attr:`batch_size` and sends each
        batch to Gemini's ``batchEmbedContents`` endpoint.

        Args:
            texts: Sequence of document strings to embed.

        Returns:
            List of float vectors, one per input text.

        Raises:
            EmbeddingError: On API failure after retries.
        """
        return await self._embed_batched(list(texts), task_type="RETRIEVAL_DOCUMENT")

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        Args:
            text: Query to embed.

        Returns:
            Float vector for the query.

        Raises:
            EmbeddingError: On API failure after retries.
        """
        vectors = await self._embed_batched([text], task_type="RETRIEVAL_QUERY")
        return vectors[0]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _embed_batched(self, texts: list[str], task_type: str) -> list[list[float]]:
        """Send *texts* in batches and collect all embedding vectors."""
        all_vectors: list[list[float]] = []
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                vectors = await self._with_retry(self._call_batch_api, client, batch, task_type)
                all_vectors.extend(vectors)  # type: ignore[arg-type]
        return all_vectors

    async def _call_batch_api(
        self,
        client: httpx.AsyncClient,
        batch: list[str],
        task_type: str,
    ) -> list[list[float]]:
        """Call ``batchEmbedContents`` for a single batch.

        Args:
            client:    Shared ``httpx.AsyncClient`` instance.
            batch:     List of texts (≤ :attr:`batch_size`).
            task_type: Gemini task type (``RETRIEVAL_DOCUMENT`` or
                       ``RETRIEVAL_QUERY``).

        Returns:
            List of embedding vectors, one per text in *batch*.

        Raises:
            httpx.HTTPStatusError: On HTTP-level failures (triggers retry).
            EmbeddingError:        On malformed response.
        """
        url = f"{self._BASE_URL}/{self.model}:batchEmbedContents"
        payload = {
            "requests": [
                {
                    "model": f"models/{self.model}",
                    "content": {"parts": [{"text": text}]},
                    "taskType": task_type,
                }
                for text in batch
            ]
        }
        response = await client.post(
            url,
            json=payload,
            params={"key": self._api_key},
        )
        response.raise_for_status()

        data = response.json()
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(batch):
            raise EmbeddingError(
                f"Unexpected Gemini response shape: expected {len(batch)} embeddings, "
                f"got {len(embeddings) if isinstance(embeddings, list) else type(embeddings)}"
            )
        return [emb["values"] for emb in embeddings]


# ---------------------------------------------------------------------------
# OpenAI implementation
# ---------------------------------------------------------------------------


class OpenAIEmbedding(EmbeddingProvider):
    """Embeddings via the OpenAI API (``text-embedding-3-small`` by default).

    Uses the ``/v1/embeddings`` endpoint.  Unlike Gemini, OpenAI uses a
    single model for both document and query embedding, so
    :meth:`embed_documents` and :meth:`embed_query` both call the same
    endpoint.

    Args:
        model:       OpenAI embedding model name.
                     Defaults to ``settings.embedding_model``.
        api_key:     OpenAI API key.
                     Defaults to ``settings.openai_api_key``.
        base_url:    API base URL.  Override to point at a compatible proxy
                     (e.g. Azure OpenAI, local LiteLLM).
                     Defaults to ``"https://api.openai.com"``.
        batch_size:  Texts per batch (default 2048, matching OpenAI limit).
        max_retries: Number of retry attempts (default 3).
        base_delay:  Initial retry back-off in seconds (default 1.0).
        jitter:      Add random jitter to delays (default ``True``).
        timeout:     HTTP timeout in seconds (default 30).

    Example::

        provider = OpenAIEmbedding(api_key="sk-...", model="text-embedding-3-small")
        vecs = await provider.embed_documents(["Hello from OpenAI"])
    """

    _DEFAULT_BASE_URL = "https://api.openai.com"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int = _OPENAI_BATCH_SIZE,
        max_retries: int = 3,
        base_delay: float = 1.0,
        jitter: bool = True,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        resolved_model = model or settings.embedding_model
        super().__init__(
            model=resolved_model,
            max_retries=max_retries,
            base_delay=base_delay,
            jitter=jitter,
            timeout=timeout,
        )
        self._api_key = api_key or settings.openai_api_key
        self._base_url = (base_url or self._DEFAULT_BASE_URL).rstrip("/")
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a list of document texts.

        Args:
            texts: Sequence of document strings to embed.

        Returns:
            List of float vectors, one per input text.

        Raises:
            EmbeddingError: On API failure after retries.
        """
        return await self._embed_batched(list(texts))

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        Args:
            text: Query to embed.

        Returns:
            Float vector for the query.

        Raises:
            EmbeddingError: On API failure after retries.
        """
        vectors = await self._embed_batched([text])
        return vectors[0]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _embed_batched(self, texts: list[str]) -> list[list[float]]:
        """Send *texts* in batches and collect all embedding vectors."""
        all_vectors: list[list[float]] = []
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                vectors = await self._with_retry(self._call_api, client, batch)
                all_vectors.extend(vectors)  # type: ignore[arg-type]
        return all_vectors

    async def _call_api(
        self,
        client: httpx.AsyncClient,
        batch: list[str],
    ) -> list[list[float]]:
        """Call ``POST /v1/embeddings`` for a single batch.

        Args:
            client: Shared ``httpx.AsyncClient`` instance.
            batch:  List of texts to embed.

        Returns:
            List of embedding vectors, one per text in *batch*.

        Raises:
            httpx.HTTPStatusError: On HTTP-level failures (triggers retry).
            EmbeddingError:        On malformed response.
        """
        url = f"{self._base_url}/v1/embeddings"
        payload: dict[str, object] = {
            "model": self.model,
            "input": batch,
            "encoding_format": "float",
        }
        response = await client.post(
            url,
            json=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()

        data = response.json()
        embedding_items = data.get("data")
        if not isinstance(embedding_items, list) or len(embedding_items) != len(batch):
            raise EmbeddingError(
                f"Unexpected OpenAI response shape: expected {len(batch)} items in 'data', "
                f"got {len(embedding_items) if isinstance(embedding_items, list) else type(embedding_items)}"
            )
        # OpenAI returns items sorted by 'index'; sort to be safe.
        embedding_items.sort(key=lambda item: item["index"])
        return [item["embedding"] for item in embedding_items]


# ---------------------------------------------------------------------------
# fastembed implementation (local ONNX, no API key)
# ---------------------------------------------------------------------------


class FastEmbedEmbedding(EmbeddingProvider):
    """Local ONNX embeddings via ``fastembed`` — no API key required.

    Runs on CPU through the same ONNX Runtime stack as the cross-encoder
    re-ranker, so a Groq/OpenAI-style deployment needs no second hosted
    provider for embeddings.  The model is downloaded on first use and
    cached under *cache_dir* (mount ``/data`` in Docker to persist it).

    Args:
        model:          fastembed model id. Defaults to
                        ``settings.fastembed_model``
                        (``"BAAI/bge-small-en-v1.5"``, 384 dimensions).
        cache_dir:      Model download cache directory. Defaults to
                        ``settings.fastembed_cache_dir``.
        text_embedding: Optional pre-built ``fastembed.TextEmbedding``
                        instance (tests inject a stub to avoid downloads).
        max_retries:    Accepted for interface uniformity; local inference
                        has no HTTP calls to retry.
        base_delay:     Accepted for interface uniformity (unused).
        jitter:         Accepted for interface uniformity (unused).
        timeout:        Accepted for interface uniformity (unused).

    Example::

        provider = FastEmbedEmbedding()
        vecs = await provider.embed_documents(["Pong rules", "Asteroids FAQ"])
    """

    def __init__(
        self,
        model: str | None = None,
        cache_dir: str | None = None,
        text_embedding: Any | None = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
        jitter: bool = True,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        resolved_model = model or settings.fastembed_model
        super().__init__(
            model=resolved_model,
            max_retries=max_retries,
            base_delay=base_delay,
            jitter=jitter,
            timeout=timeout,
        )
        self._cache_dir = cache_dir or settings.fastembed_cache_dir
        self._text_embedding = text_embedding

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed document texts locally.

        Args:
            texts: Sequence of document strings to embed.

        Returns:
            List of float vectors, one per input text.

        Raises:
            EmbeddingError: If the model cannot be loaded or inference fails.
        """
        if not texts:
            return []
        return await asyncio.to_thread(self._embed_sync, list(texts), False)

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string locally.

        Uses fastembed's ``query_embed`` so models that rely on an
        asymmetric query prefix (the bge family) encode correctly.

        Args:
            text: Query to embed.

        Returns:
            Float vector for the query.

        Raises:
            EmbeddingError: If the model cannot be loaded or inference fails.
        """
        vectors = await asyncio.to_thread(self._embed_sync, [text], True)
        return vectors[0]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> Any:
        """Load the fastembed model lazily on first use."""
        if self._text_embedding is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:  # pragma: no cover - dependency is pinned
                raise EmbeddingError(
                    "The 'fastembed' package is not installed; install the "
                    "ai-service requirements or choose another EMBEDDING_PROVIDER."
                ) from exc
            try:
                self._text_embedding = TextEmbedding(
                    model_name=self.model, cache_dir=self._cache_dir
                )
            except Exception as exc:
                raise EmbeddingError(
                    f"Could not load fastembed model '{self.model}': {exc}"
                ) from exc
            logger.info(
                "Loaded fastembed model '%s' (cache: %s).",
                self.model,
                self._cache_dir,
            )
        return self._text_embedding

    def _embed_sync(self, texts: list[str], is_query: bool) -> list[list[float]]:
        """Run one synchronous fastembed batch and return plain float lists."""
        model = self._load_model()
        embed_fn = model.query_embed if is_query else model.embed
        try:
            return [[float(value) for value in vector] for vector in embed_fn(texts)]
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"fastembed inference failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_embedding_provider(
    provider: str | None = None,
    **kwargs: object,
) -> EmbeddingProvider:
    """Return an :class:`EmbeddingProvider` based on configuration.

    The *provider* argument (or ``settings.embedding_provider`` if omitted)
    selects the implementation:

    * ``"gemini"``    → :class:`GeminiEmbedding`
    * ``"openai"``    → :class:`OpenAIEmbedding`
    * ``"fastembed"`` → :class:`FastEmbedEmbedding` (local ONNX, no API key)

    Any extra *kwargs* are forwarded to the provider constructor (e.g.
    ``model``, ``batch_size``, ``max_retries``).

    Args:
        provider: Provider identifier (``"gemini"``, ``"openai"`` or
                  ``"fastembed"``). Defaults to ``settings.embedding_provider``.
        **kwargs: Additional keyword arguments passed to the provider.

    Returns:
        An initialised :class:`EmbeddingProvider` instance.

    Raises:
        ValueError: If *provider* is not a supported value.

    Example::

        provider = get_embedding_provider()
        # or force a specific one:
        provider = get_embedding_provider("openai", model="text-embedding-3-large")
    """
    resolved = (provider or settings.embedding_provider).lower()
    if resolved == "gemini":
        return GeminiEmbedding(**kwargs)  # type: ignore[arg-type]
    if resolved == "openai":
        return OpenAIEmbedding(**kwargs)  # type: ignore[arg-type]
    if resolved == "fastembed":
        return FastEmbedEmbedding(**kwargs)  # type: ignore[arg-type]
    raise ValueError(
        f"Unknown embedding provider '{resolved}'. "
        "Supported values: 'gemini', 'openai', 'fastembed'."
    )
