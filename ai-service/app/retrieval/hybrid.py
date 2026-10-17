"""Hybrid retrieval: dense + BM25 sparse vectors fused with RRF (Issue #20).

Why client-side BM25?
---------------------
``qdrant-client`` 1.7 has no server-side BM25/inference API and no
``Prefetch``/``query_points`` fusion support, so RetroGPT computes BM25 term
weights client-side with :class:`~app.retrieval.bm25.Bm25Encoder` and lets
Qdrant score them as sparse vectors (dot product).  The dense and sparse
result lists are then fused client-side with Reciprocal Rank Fusion
(`Cormack et al., 2009 <https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf>`_):

    score(doc) = Σ 1 / (k + rank(doc))

with the standard ``k = 60``.  Fused scores are normalised by the best hit
so the API/UI keeps a sensible 0–1 relevance display.

The whole module **fails open to dense-only retrieval**: a missing BM25
statistics file, a dense-only legacy collection, or a sparse-search error
never takes the query path down.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from app.config import settings
from app.retrieval.bm25 import Bm25Encoder
from app.retrieval.vector_store import SearchResult, VectorStoreError

if TYPE_CHECKING:
    from qdrant_client.http import models as qdrant_models

    from app.retrieval.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)

#: RRF rank constant (Cormack et al., 2009).
RRF_K = 60


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def rrf_fuse(
    dense_results: Sequence[SearchResult],
    sparse_results: Sequence[SearchResult],
    k: int = RRF_K,
    limit: int | None = None,
) -> list[SearchResult]:
    """Fuse two ranked result lists with Reciprocal Rank Fusion.

    Ranks are 1-based; each hit contributes ``1 / (k + rank)`` per list it
    appears in.  The fused scores are normalised by the best hit so the top
    result scores exactly ``1.0``.

    Args:
        dense_results:  Dense-vector hits, best first.
        sparse_results: BM25 sparse-vector hits, best first.
        k:              RRF rank constant (default 60).
        limit:          Optional maximum number of fused results to return.

    Returns:
        Fused :class:`SearchResult` list sorted by descending RRF score
        (ties broken by chunk ID for determinism).  When a chunk appears in
        both lists, the dense object's payload/content are kept.
    """
    scores: dict[str, float] = {}
    representatives: dict[str, SearchResult] = {}

    for results in (dense_results, sparse_results):
        for rank, result in enumerate(results, start=1):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (k + rank)
            representatives.setdefault(result.chunk_id, result)

    if not scores:
        return []

    best_score = max(scores.values())
    fused = [
        SearchResult(
            chunk_id=chunk_id,
            score=scores[chunk_id] / best_score,
            payload=representatives[chunk_id].payload,
            content=representatives[chunk_id].content,
        )
        for chunk_id in sorted(scores, key=lambda cid: (-scores[cid], cid))
    ]
    return fused[:limit] if limit is not None else fused


# ---------------------------------------------------------------------------
# BM25 statistics loading
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _load_bm25_stats_cached(path: str, _mtime: float) -> Bm25Encoder | None:
    """Load BM25 stats, cached by ``(path, mtime)``.

    The mtime participates in the cache key so re-ingestion (which rewrites
    the stats file) automatically invalidates stale entries.
    """
    try:
        return Bm25Encoder.load(Path(path))
    except (OSError, ValueError) as exc:
        # ValueError covers JSONDecodeError and schema/version mismatches.
        logger.warning("Could not load BM25 stats from '%s': %s", path, exc)
        return None


def get_bm25_encoder(path: str | Path | None = None) -> Bm25Encoder | None:
    """Return the fitted BM25 encoder from persisted statistics.

    The stats file is written by the ingestion pipeline
    (:meth:`app.ingestion.pipeline.IngestionPipeline.run`).  Returns
    ``None`` when the file does not exist or cannot be parsed — callers
    should fall back to dense-only retrieval.

    Args:
        path: Statistics location; defaults to ``settings.bm25_stats_path``.
    """
    stats_path = Path(path) if path is not None else Path(settings.bm25_stats_path)
    try:
        mtime = stats_path.stat().st_mtime
    except OSError:
        logger.debug("No BM25 stats at '%s'; hybrid search unavailable.", stats_path)
        return None
    return _load_bm25_stats_cached(str(stats_path), mtime)


# ---------------------------------------------------------------------------
# Hybrid search
# ---------------------------------------------------------------------------


async def hybrid_search(
    store: QdrantVectorStore,
    query: str,
    query_vector: Sequence[float],
    top_k: int | None = None,
    prefetch_k: int | None = None,
    filter: qdrant_models.Filter | None = None,
) -> list[SearchResult]:
    """Retrieve passages with dense + BM25 hybrid search, fused via RRF.

    When hybrid search is enabled and BM25 statistics are available, each
    retriever prefetches ``prefetch_k`` candidates which are fused and
    truncated to ``top_k``.  Otherwise — or if the sparse search fails —
    this degrades gracefully to the dense-only path.

    Args:
        store:        Vector store to query.
        query:        Raw query text (used for the BM25 branch).
        query_vector: Dense query embedding.
        top_k:        Number of results to return; defaults to
                      ``settings.retrieval_top_k``.
        prefetch_k:   Candidates per retriever before fusion; defaults to
                      ``settings.hybrid_prefetch_k``.
        filter:       Optional Qdrant payload filter (e.g. game slug).

    Returns:
        Fused (or dense-only) list of :class:`SearchResult`.

    Raises:
        VectorStoreError: If the dense search itself fails.
    """
    k = top_k if top_k is not None else settings.retrieval_top_k
    encoder = get_bm25_encoder() if settings.hybrid_search_enabled else None

    if encoder is None:
        # Dense-only path — also preserves the pre-hybrid behaviour exactly.
        return await store.search(query_vector, top_k=k, filter=filter)

    prefetch = prefetch_k if prefetch_k is not None else settings.hybrid_prefetch_k
    store.set_bm25_encoder(encoder)

    dense_results = await store.search(query_vector, top_k=prefetch, filter=filter)
    try:
        sparse_results = await store.search_sparse(query, top_k=prefetch, filter=filter)
    except VectorStoreError as exc:
        logger.warning("Sparse search failed; falling back to dense-only: %s", exc)
        return dense_results[:k]

    return rrf_fuse(dense_results, sparse_results, limit=k)
