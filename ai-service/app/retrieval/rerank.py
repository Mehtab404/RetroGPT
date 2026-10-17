"""Cross-encoder re-ranking of hybrid retrieval candidates (Issue #21).

Two-stage retrieval
-------------------
1. **Recall** — :func:`~app.retrieval.hybrid.hybrid_search` fetches a wide
   candidate pool (default: ``settings.retrieval_top_k`` = 20) with dense +
   BM25 sparse retrieval fused via RRF.
2. **Re-rank** — a cross-encoder scores each ``(query, passage)`` pair
   jointly and the top ``top_k`` passages are kept.  Cross-encoders are far
   more precise than bi-encoder similarity but too slow for a full corpus,
   hence the two-stage design.

Model runtime
-------------
Re-ranking uses :class:`fastembed.rerank.cross_encoder.TextCrossEncoder`
(ONNX Runtime, default ``Xenova/ms-marco-MiniLM-L-6-v2``).  This avoids the
multi-gigabyte PyTorch dependency of ``sentence-transformers`` while
providing the same MiniLM cross-encoder model family.  The model is
downloaded lazily on first use and cached under ``settings.rerank_cache_dir``.

Everything **fails open**: if ``fastembed`` is not installed, re-ranking is
disabled; if scoring raises, the hybrid RRF order is used unchanged.

Raw cross-encoder logits are unbounded (e.g. ``+9.8`` / ``-11.4``), so the
scores exposed on :class:`~app.retrieval.vector_store.SearchResult` are
squashed with a logistic sigmoid into ``(0, 1)``, preserving the ranking of
the frontend's percentage-match display.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from app.config import settings
from app.retrieval.hybrid import hybrid_search
from app.retrieval.vector_store import SearchResult

if TYPE_CHECKING:
    from qdrant_client.http import models as qdrant_models

    from app.retrieval.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RerankError(RuntimeError):
    """Raised when cross-encoder scoring fails."""


def _sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid (overflow-safe for large |x|)."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


# ---------------------------------------------------------------------------
# Cross-encoder reranker
# ---------------------------------------------------------------------------


class CrossEncoderReranker:
    """Lazily-loaded ONNX cross-encoder that scores query/passage pairs.

    Args:
        model_name: fastembed model identifier, e.g.
            ``"Xenova/ms-marco-MiniLM-L-6-v2"``.
        cache_dir:  Model download cache directory
            (``settings.rerank_cache_dir``).

    The underlying model is created on first :meth:`rerank` call, so simply
    constructing the reranker is cheap (no network, no model load).
    """

    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model: Any = None

    @staticmethod
    def available() -> bool:
        """Return whether the ``fastembed`` reranker runtime is importable."""
        try:
            import fastembed.rerank.cross_encoder  # noqa: F401
        except ImportError:
            return False
        return True

    def _ensure_model(self) -> Any:
        """Create the fastembed model on first use (runs in a worker thread)."""
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            logger.info("Loading cross-encoder model '%s'…", self.model_name)
            self._model = TextCrossEncoder(
                model_name=self.model_name,
                cache_dir=self.cache_dir,
            )
        return self._model

    async def rerank(
        self,
        query: str,
        results: Sequence[SearchResult],
        top_n: int | None = None,
    ) -> list[SearchResult]:
        """Re-order *results* by cross-encoder relevance to *query*.

        Args:
            query:   Raw query text.
            results: Candidate passages in recall order.
            top_n:   Optional cap on returned passages.

        Returns:
            New :class:`SearchResult` objects ordered by descending
            cross-encoder score, with logits squashed to ``(0, 1)`` via
            sigmoid.  Original payloads/content are preserved.

        Raises:
            RerankError: If model loading or scoring fails, or the model
                returns a different number of scores than documents.
        """
        if not results:
            return []

        documents = [r.content for r in results]
        try:
            model = await asyncio.to_thread(self._ensure_model)
            scores = await asyncio.to_thread(lambda: list(model.rerank(query, documents)))
        except Exception as exc:
            raise RerankError(f"cross-encoder re-ranking failed: {exc}") from exc

        if len(scores) != len(results):
            raise RerankError(
                f"cross-encoder returned {len(scores)} scores for {len(results)} documents"
            )

        ranked = sorted(
            zip(scores, results, strict=True),
            key=lambda pair: float(pair[0]),
            reverse=True,
        )
        reranked = [
            SearchResult(
                chunk_id=result.chunk_id,
                score=_sigmoid(float(score)),
                payload=result.payload,
                content=result.content,
            )
            for score, result in ranked
        ]
        return reranked[:top_n] if top_n is not None else reranked


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker | None:
    """Return the process-wide reranker, or ``None`` when unavailable.

    Returns ``None`` when re-ranking is disabled
    (``settings.rerank_enabled``) or ``fastembed`` is not installed.
    """
    if not settings.rerank_enabled:
        logger.debug("Cross-encoder re-ranking disabled via settings.")
        return None
    if not CrossEncoderReranker.available():
        logger.warning("fastembed is not installed; cross-encoder re-ranking disabled.")
        return None
    return CrossEncoderReranker(
        model_name=settings.rerank_model,
        cache_dir=settings.rerank_cache_dir,
    )


# ---------------------------------------------------------------------------
# Full retrieval pipeline
# ---------------------------------------------------------------------------


async def retrieve(
    store: QdrantVectorStore,
    query: str,
    query_vector: Sequence[float],
    top_k: int | None = None,
    filter: qdrant_models.Filter | None = None,
    timings: dict[str, float] | None = None,
) -> list[SearchResult]:
    """Run the two-stage retrieval pipeline: hybrid recall → re-rank.

    Fetches a candidate pool of at least ``settings.retrieval_top_k``
    passages via :func:`~app.retrieval.hybrid.hybrid_search` (which itself
    fails open to dense-only), then re-orders them with the cross-encoder
    and keeps the best *top_k*.

    Args:
        store:        Vector store to query.
        query:        Raw query text.
        query_vector: Dense query embedding.
        top_k:        Number of passages to return after re-ranking
                      (defaults to ``settings.rerank_top_n``).
        filter:       Optional Qdrant payload filter (e.g. game slug).
        timings:      Optional mapping populated in-place with the stage
                      timings in milliseconds (``hybrid_search_ms``,
                      ``rerank_ms``; the latter is ``0.0`` when no
                      re-ranker is configured) — Issue #24.

    Returns:
        Best passages, most relevant first.

    Raises:
        VectorStoreError: If the underlying hybrid/dense search fails.
    """
    final_k = top_k if top_k is not None else settings.rerank_top_n
    candidates_k = max(final_k, settings.retrieval_top_k)

    hybrid_start = time.monotonic()
    candidates = await hybrid_search(store, query, query_vector, top_k=candidates_k, filter=filter)
    if timings is not None:
        timings["hybrid_search_ms"] = (time.monotonic() - hybrid_start) * 1000.0

    reranker = get_reranker()
    if reranker is None:
        if timings is not None:
            timings["rerank_ms"] = 0.0
        return candidates[:final_k]

    rerank_start = time.monotonic()
    try:
        reranked = await reranker.rerank(query, candidates, top_n=final_k)
    except RerankError as exc:
        logger.warning("Re-ranking failed; using hybrid RRF order: %s", exc)
        reranked = candidates[:final_k]
    if timings is not None:
        timings["rerank_ms"] = (time.monotonic() - rerank_start) * 1000.0
    return reranked
