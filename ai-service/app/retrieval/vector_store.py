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
* Collections created with ``settings.hybrid_search_enabled`` also get a
  named ``"bm25"`` sparse-vector config.  Sparse vectors are computed
  client-side by :class:`~app.retrieval.bm25.Bm25Encoder` (Qdrant 1.7 has
  no server-side BM25) and attached to each point during upsert.
* Legacy dense-only collections are supported: :meth:`supports_sparse`
  detects a missing sparse config and upsert/search fall back to the
  dense path (Issue #20).
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
import uuid
from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import settings
from app.ingestion.chunker import Chunk
from app.retrieval.bm25 import Bm25Encoder

logger = logging.getLogger(__name__)

# Qdrant batch size for upsert calls.  Qdrant recommends ≤ 100 points per
# request to stay within default gRPC message limits.
_UPSERT_BATCH_SIZE: int = 100

#: Qdrant's default (unnamed) dense vector is addressed by the empty string.
DENSE_VECTOR_NAME: str = ""

#: Name of the named sparse (BM25) vector configured for hybrid search.
SPARSE_VECTOR_NAME: str = "bm25"


def _to_point_id(raw_id: str) -> int | str:
    """Convert a chunk ID into a valid Qdrant point ID.

    Qdrant point IDs must be unsigned 64-bit integers or UUIDs, while chunk
    IDs are 16-char SHA-256 fragments (e.g. ``"0f3a…"``).  Numeric strings
    are converted to integers; anything else is mapped to a deterministic
    UUIDv5, so re-ingesting the same chunk overwrites the same point
    (idempotent upserts) and citations keep the original short chunk ID.
    """
    try:
        return int(raw_id)
    except (TypeError, ValueError):
        pass
    try:
        return str(uuid.UUID(raw_id))
    except (AttributeError, ValueError):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"retrogpt:{raw_id}"))


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
        # BM25 encoder attached by the ingestion pipeline / hybrid search.
        self._bm25_encoder: Bm25Encoder | None = None
        # Cached tri-state: None = unknown, True/False = probed.
        self._sparse_supported: bool | None = None

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    async def ensure_collection(self, recreate_if_incompatible: bool = False) -> None:
        """Create the Qdrant collection if it does not yet exist.

        Uses Cosine distance for similarity; the embedding providers already
        return L2-normalised vectors, so Cosine and dot-product are equivalent.

        When ``settings.hybrid_search_enabled`` is true, new collections are
        also created with a named ``"bm25"`` sparse-vector config so they can
        store BM25 term weights for hybrid retrieval.

        If the collection already exists:

        * with a sparse config → no-op;
        * without a sparse config but hybrid search is enabled → a warning is
          logged (retrieval will fall back to dense) unless
          *recreate_if_incompatible* is true, in which case the collection is
          deleted and re-created.  **All existing points are dropped**;
          re-ingest afterwards.

        Args:
            recreate_if_incompatible: Whether to drop and re-create an
                existing collection that lacks the sparse-vector config.

        Raises:
            VectorStoreError: If Qdrant returns an unexpected error.
        """
        try:
            exists = await asyncio.to_thread(self._collection_exists)
            if not exists:
                await asyncio.to_thread(self._create_collection)
                logger.info(
                    "Created Qdrant collection '%s' (dim=%d, sparse=%s).",
                    self.collection,
                    self.vector_size,
                    settings.hybrid_search_enabled,
                )
                return

            if not settings.hybrid_search_enabled:
                logger.debug("Collection '%s' already exists.", self.collection)
                return

            if await asyncio.to_thread(self._collection_has_sparse):
                logger.debug(
                    "Collection '%s' already exists (hybrid search ready).",
                    self.collection,
                )
                return

            if recreate_if_incompatible:
                logger.warning(
                    "Recreating collection '%s' for hybrid search; "
                    "existing points are dropped (re-ingest afterwards).",
                    self.collection,
                )
                await asyncio.to_thread(self._client.delete_collection, self.collection)
                await asyncio.to_thread(self._create_collection)
                return

            logger.warning(
                "Collection '%s' exists without sparse vectors; hybrid search "
                "will fall back to dense. Delete and re-ingest the collection "
                "(or pass recreate_if_incompatible=True) to enable hybrid search.",
                self.collection,
            )
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

    def _collection_has_sparse(self) -> bool:
        """Probe the collection config for a sparse-vector section.

        Updates the cached :attr:`_sparse_supported` flag as a side effect.
        """
        info = self._client.get_collection(self.collection)
        sparse_vectors = getattr(info.config.params, "sparse_vectors", None)
        self._sparse_supported = bool(sparse_vectors)
        return self._sparse_supported

    def _create_collection(self) -> None:
        """Create the Qdrant collection (blocking call)."""
        sparse_vectors_config = None
        if settings.hybrid_search_enabled:
            sparse_vectors_config = {SPARSE_VECTOR_NAME: qdrant_models.SparseVectorParams()}
        self._client.create_collection(
            collection_name=self.collection,
            vectors_config=qdrant_models.VectorParams(
                size=self.vector_size,
                distance=qdrant_models.Distance.COSINE,
            ),
            sparse_vectors_config=sparse_vectors_config,
        )
        # We know the config we just created; avoids an extra round-trip.
        self._sparse_supported = sparse_vectors_config is not None

    def supports_sparse(self) -> bool:
        """Return whether the underlying collection can store sparse vectors.

        Probed from the collection config on first use and cached for the
        lifetime of the store.  Any probe error (e.g. collection missing)
        is treated as "no sparse support" so callers can fail open to the
        dense-only path.

        Note: the first call issues a blocking ``get_collection`` request.
        """
        if self._sparse_supported is None:
            try:
                self._collection_has_sparse()
            except Exception:
                self._sparse_supported = False
        return self._sparse_supported

    def set_bm25_encoder(self, encoder: Bm25Encoder | None) -> None:
        """Attach (or clear) the BM25 encoder used for sparse vectors."""
        self._bm25_encoder = encoder

    async def delete_collection(self) -> None:
        """Delete the collection if it exists.

        Useful for testing and fresh-restart ingestion.

        Raises:
            VectorStoreError: If Qdrant returns an unexpected error.
        """
        try:
            await asyncio.to_thread(self._client.delete_collection, self.collection)
            self._sparse_supported = None
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
        * **vector**  – the corresponding dense embedding; when a BM25
                        encoder is attached (see :meth:`set_bm25_encoder`)
                        and the collection supports sparse vectors, the
                        point stores ``{"": dense, "bm25": SparseVector}``
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

        # Decide once per call whether sparse vectors are attached.
        use_sparse = self._bm25_encoder is not None and await asyncio.to_thread(
            self.supports_sparse
        )
        encoder = self._bm25_encoder if use_sparse else None

        total = 0
        for batch_start in range(0, len(chunks_list), self.upsert_batch):
            batch_chunks = chunks_list[batch_start : batch_start + self.upsert_batch]
            batch_vectors = vectors_list[batch_start : batch_start + self.upsert_batch]

            points = []
            for i, (chunk, vec) in enumerate(zip(batch_chunks, batch_vectors, strict=True)):
                vector: Any
                if encoder is not None:
                    vector = {
                        DENSE_VECTOR_NAME: list(vec),
                        SPARSE_VECTOR_NAME: encoder.encode(chunk.content),
                    }
                else:
                    vector = list(vec)
                points.append(
                    qdrant_models.PointStruct(
                        id=_to_point_id(
                            str(chunk.metadata.get("chunk_id") or f"chunk_{batch_start + i}")
                        ),
                        vector=vector,
                        payload={**chunk.metadata, "content": chunk.content},
                    )
                )

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

        return [self._hit_to_result(hit) for hit in hits]

    async def search_sparse(
        self,
        query: str,
        top_k: int | None = None,
        filter: qdrant_models.Filter | None = None,
    ) -> list[SearchResult]:
        """Perform a BM25 sparse-vector search for the raw *query* text.

        The query is encoded client-side with the BM25 encoder attached via
        :meth:`set_bm25_encoder`, producing the same term indices/weights
        used when the points were ingested.

        Args:
            query:  Raw query text (not embedded).
            top_k:  Maximum number of results to return.
                    Defaults to ``settings.retrieval_top_k``.
            filter: Optional Qdrant ``Filter`` to narrow results.

        Returns:
            Ordered list of :class:`SearchResult`, best BM25 score first.

        Raises:
            VectorStoreError: If no encoder is attached, the collection has
                no sparse vectors, or Qdrant fails.
        """
        if self._bm25_encoder is None:
            raise VectorStoreError("sparse search requires an attached BM25 encoder")
        if not await asyncio.to_thread(self.supports_sparse):
            raise VectorStoreError(
                f"collection '{self.collection}' has no sparse vectors configured"
            )

        sparse_vector = self._bm25_encoder.encode(query)
        if not sparse_vector.indices:
            # No known terms in the query → nothing sparse can match.
            return []

        k = top_k if top_k is not None else settings.retrieval_top_k
        try:
            hits = await asyncio.to_thread(
                self._client.search,
                collection_name=self.collection,
                query_vector=qdrant_models.NamedSparseVector(
                    name=SPARSE_VECTOR_NAME, vector=sparse_vector
                ),
                limit=k,
                query_filter=filter,
                with_payload=True,
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Qdrant sparse search failed in '{self.collection}': {exc}"
            ) from exc

        return [self._hit_to_result(hit) for hit in hits]

    @staticmethod
    def _hit_to_result(hit: Any) -> SearchResult:
        """Convert a Qdrant scored point into a :class:`SearchResult`."""
        payload: dict[str, Any] = hit.payload or {}
        return SearchResult(
            chunk_id=str(payload.get("chunk_id", hit.id)),
            score=hit.score,
            payload=payload,
            content=payload.get("content", ""),
        )

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
