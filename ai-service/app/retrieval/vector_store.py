"""Qdrant vector store abstraction for the RetroGPT retrieval pipeline.

Wraps the ``qdrant-client`` library with a clean async interface that
handles collection creation, batch upsert, and similarity search.

Design decisions
----------------
* The ``QdrantClient`` is synchronous by default; all blocking calls are
  dispatched via :func:`asyncio.to_thread` so the FastAPI event loop is
  never blocked.
* Collections use Cosine distance because the embedding providers
  return L2-normalised vectors (cosine ≅ inner product for unit vecs).
* Sparse (BM25) indexes are left for Issue #10 (hybrid search); this
  module focuses on the dense-only path.
* Every upsert is idempotent: re-ingesting the same chunk_id silently
  overwrites the existing point (Qdrant's default ``upsert`` semantics).

Typical usage::

    store = QdrantVectorStore()
    await store.ensure_collection()
    await store.upsert(chunks, vectors)
    results = await store.search(query_vector, top_k=5)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import settings
from app.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

# Qdrant batch size for upsert calls.  Qdrant recommends ≤ 100 points per
# request to stay within default gRPC message limits.
_UPSERT_BATCH_SIZE: int = 100


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class VectorStoreError(RuntimeError):
    """Raised when a Qdrant operation fails permanently."""


# ---------------------------------------------------------------------------
# Search result model
# ---------------------------------------------------------------------------


class SearchResult:
    """A single hit from a vector similarity search.

    Attributes:
        chunk_id:  The chunk ID stored in Qdrant point payload.
        score:     Cosine similarity score (0–1, higher = more similar).
        payload:   Full metadata payload as stored in Qdrant.
        content:   The original chunk text (stored in payload as ``"content"``).
    """

    __slots__ = ("chunk_id", "score", "payload", "content")

    def __init__(
        self,
        chunk_id: str,
        score: float,
        payload: dict[str, Any],
        content: str,
    ) -> None:
        self.chunk_id = chunk_id
        self.score = score
        self.payload = payload
        self.content = content

    def __repr__(self) -> str:
        return (
            f"SearchResult(chunk_id={self.chunk_id!r}, "
            f"score={self.score:.4f}, "
            f"content={self.content[:60]!r}…)"
        )


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------


class QdrantVectorStore:
    """High-level async wrapper around Qdrant for dense-vector storage.

    Args:
        url:            Qdrant service URL.
                        Defaults to ``settings.qdrant_url``.
        collection:     Collection name.
                        Defaults to ``settings.qdrant_collection``.
        vector_size:    Dimensionality of the embedding vectors.  Must
                        match the embedding model (Gemini ``text-embedding-004``
                        produces 768-dimensional vectors; OpenAI
                        ``text-embedding-3-small`` produces 1536).
                        Defaults to 768.
        upsert_batch:   Number of points per Qdrant upsert request.
                        Defaults to 100.
        client:         Optional pre-constructed :class:`QdrantClient`.
                        Pass a mock in tests to avoid network calls.

    Example::

        store = QdrantVectorStore(url="http://localhost:6333")
        await store.ensure_collection()
        await store.upsert(chunks, embeddings)
        hits = await store.search(query_vec, top_k=5)
    """

    def __init__(
        self,
        url: str | None = None,
        collection: str | None = None,
        vector_size: int = 768,
        upsert_batch: int = _UPSERT_BATCH_SIZE,
        client: QdrantClient | None = None,
    ) -> None:
        self._url = url or settings.qdrant_url
        self.collection = collection or settings.qdrant_collection
        self.vector_size = vector_size
        self.upsert_batch = upsert_batch
        # Allow injecting a pre-built client (useful for testing).
        self._client: QdrantClient = client or QdrantClient(url=self._url)

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    async def ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not yet exist.

        Uses Cosine distance for similarity; the embedding providers already
        return L2-normalised vectors, so Cosine and dot-product are equivalent.

        If the collection already exists with compatible parameters, this
        is a no-op.

        Raises:
            VectorStoreError: If Qdrant returns an unexpected error.
        """
        try:
            exists = await asyncio.to_thread(self._collection_exists)
            if not exists:
                await asyncio.to_thread(self._create_collection)
                logger.info(
                    "Created Qdrant collection '%s' (dim=%d).",
                    self.collection,
                    self.vector_size,
                )
            else:
                logger.debug("Collection '%s' already exists.", self.collection)
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to ensure collection '{self.collection}': {exc}"
            ) from exc

    def _collection_exists(self) -> bool:
        """Return ``True`` if the collection already exists in Qdrant."""
        collections = self._client.get_collections().collections
        return any(c.name == self.collection for c in collections)

    def _create_collection(self) -> None:
        """Create the Qdrant collection (blocking call)."""
        self._client.create_collection(
            collection_name=self.collection,
            vectors_config=qdrant_models.VectorParams(
                size=self.vector_size,
                distance=qdrant_models.Distance.COSINE,
            ),
        )

    async def delete_collection(self) -> None:
        """Delete the collection if it exists.

        Useful for testing and fresh-restart ingestion.

        Raises:
            VectorStoreError: If Qdrant returns an unexpected error.
        """
        try:
            await asyncio.to_thread(self._client.delete_collection, self.collection)
            logger.info("Deleted Qdrant collection '%s'.", self.collection)
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to delete collection '{self.collection}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    async def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> int:
        """Upsert *chunks* and their *vectors* into Qdrant.

        Each chunk is stored as a Qdrant point where:

        * **id**      – the ``chunk_id`` from ``chunk.metadata["chunk_id"]``
        * **vector**  – the corresponding dense embedding
        * **payload** – all chunk metadata fields plus ``"content"`` for the
                        raw text

        The upsert is batched in groups of :attr:`upsert_batch` points to
        stay within Qdrant's recommended request sizes.

        Args:
            chunks:  Ordered sequence of :class:`~app.ingestion.chunker.Chunk`
                     objects to store.
            vectors: Corresponding embedding vectors; must be the same length
                     as *chunks*.

        Returns:
            Total number of points upserted.

        Raises:
            ValueError:        If ``len(chunks) != len(vectors)``.
            VectorStoreError:  On Qdrant API failure.
        """
        chunks_list = list(chunks)
        vectors_list = list(vectors)

        if len(chunks_list) != len(vectors_list):
            raise ValueError(
                f"chunks and vectors must have the same length; "
                f"got {len(chunks_list)} chunks and {len(vectors_list)} vectors."
            )

        if not chunks_list:
            return 0

        total = 0
        for batch_start in range(0, len(chunks_list), self.upsert_batch):
            batch_chunks = chunks_list[batch_start : batch_start + self.upsert_batch]
            batch_vectors = vectors_list[batch_start : batch_start + self.upsert_batch]

            points = [
                qdrant_models.PointStruct(
                    id=chunk.metadata.get("chunk_id") or f"chunk_{batch_start + i}",
                    vector=list(vec),
                    payload={**chunk.metadata, "content": chunk.content},
                )
                for i, (chunk, vec) in enumerate(zip(batch_chunks, batch_vectors, strict=True))
            ]

            try:
                await asyncio.to_thread(
                    self._client.upsert,
                    collection_name=self.collection,
                    points=points,
                    wait=True,
                )
                total += len(points)
                logger.debug(
                    "Upserted batch of %d points to '%s' (total so far: %d).",
                    len(points),
                    self.collection,
                    total,
                )
            except Exception as exc:
                raise VectorStoreError(
                    f"Qdrant upsert failed at batch offset {batch_start}: {exc}"
                ) from exc

        logger.info("Upserted %d points into '%s'.", total, self.collection)
        return total

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self,
        query_vector: Sequence[float],
        top_k: int | None = None,
        filter: qdrant_models.Filter | None = None,
    ) -> list[SearchResult]:
        """Perform a dense vector similarity search.

        Args:
            query_vector: Dense embedding for the query.
            top_k:        Maximum number of results to return.
                          Defaults to ``settings.retrieval_top_k``.
            filter:       Optional Qdrant ``Filter`` object to narrow results
                          (e.g. filter by game slug).

        Returns:
            Ordered list of :class:`SearchResult` objects, most similar first.

        Raises:
            VectorStoreError: On Qdrant API failure.
        """
        k = top_k if top_k is not None else settings.retrieval_top_k

        try:
            hits = await asyncio.to_thread(
                self._client.search,
                collection_name=self.collection,
                query_vector=list(query_vector),
                limit=k,
                query_filter=filter,
                with_payload=True,
            )
        except Exception as exc:
            raise VectorStoreError(f"Qdrant search failed in '{self.collection}': {exc}") from exc

        results: list[SearchResult] = []
        for hit in hits:
            payload: dict[str, Any] = hit.payload or {}
            results.append(
                SearchResult(
                    chunk_id=str(payload.get("chunk_id", hit.id)),
                    score=hit.score,
                    payload=payload,
                    content=payload.get("content", ""),
                )
            )
        return results

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    async def count(self) -> int:
        """Return the number of points currently stored in the collection.

        Returns:
            Point count, or 0 if the collection does not exist.

        Raises:
            VectorStoreError: On unexpected Qdrant errors.
        """
        try:
            result = await asyncio.to_thread(
                self._client.count,
                collection_name=self.collection,
                exact=True,
            )
            return result.count
        except UnexpectedResponse as exc:
            # 404 → collection not found → treat as 0 points.
            if exc.status_code == 404:
                return 0
            raise VectorStoreError(f"Failed to count points in '{self.collection}': {exc}") from exc
        except Exception as exc:
            raise VectorStoreError(f"Failed to count points in '{self.collection}': {exc}") from exc

    def close(self) -> None:
        """Close the underlying Qdrant client connection."""
        try:
            self._client.close()
        except Exception:
            pass  # Best-effort cleanup
