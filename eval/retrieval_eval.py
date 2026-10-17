#!/usr/bin/env python3
"""In-process retrieval evaluation harness (Issues #22 and #23).

Why this exists
---------------
``eval/run_eval.py`` measures the *service* end-to-end (HTTP + LLM answers),
which needs ``GOOGLE_API_KEY``/``OPENAI_API_KEY``.  Retrieval quality,
however, can be measured without any API key: this harness imports the
``ai-service`` code directly, ingests the corpus into a scratch Qdrant
collection, and scores the real two-stage retrieval pipeline
(hybrid search → optional cross-encoder re-ranking) against the same labeled
dataset.

Two use cases:

* **#22 — chunking experiments** — sweep chunk sizes and overlaps
  (``--sweep-sizes`` / ``--sweep-overlaps``) and compare Hit Rate/MRR to pick
  the best configuration.
* **#23 — CI regression guard** — run a small subset (``--limit 10``) and
  fail when Hit Rate drops more than 5 points below a committed baseline
  (``--baseline``).  ``HashingEmbedding`` keeps CI fully offline and
  deterministic.

Answers are never generated, so answer-quality metrics (refusal accuracy,
RAGAS) are ``n/a`` by design; the report says so explicitly.

Usage (inside the compose stack)::

    docker compose exec ai-service python /eval/retrieval_eval.py --limit 10

    docker compose exec ai-service python /eval/retrieval_eval.py \
        --sweep-sizes 400,800,1200 --sweep-overlaps 0,100 \
        --embedder fastembed --no-rerank --label chunking-sweep

Exit codes
----------
* ``0`` — evaluation completed (and any configured quality guards passed)
* ``1`` — a quality guard failed (``--min-hit-rate`` or ``--baseline``)
* ``2`` — fatal error (dataset/corpus missing, Qdrant unreachable, …)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import math
import re
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

import run_eval  # noqa: E402  (import after sys.path injection)

logger = logging.getLogger("retrogpt.retrieval_eval")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TOP_K = run_eval.DEFAULT_TOP_K  # 5
DEFAULT_COLLECTION = "retrogpt_eval"
DEFAULT_CORPUS = Path("/data/raw")
DEFAULT_EMBEDDER = "hashing"
DEFAULT_HASHING_DIM = 768
DEFAULT_FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_SWEEP_OVERLAPS = "0,100"
DEFAULT_MAX_HIT_RATE_DROP = run_eval.DEFAULT_MAX_HIT_RATE_DROP  # 0.05

#: Known output dimensionality for fastembed models we ship defaults for.
FASTEMBED_MODEL_DIMS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}

#: Default output dimensionality for the hosted embedding providers.
API_EMBEDDER_DIMS: dict[str, int] = {"gemini": 768, "openai": 1536}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


# ---------------------------------------------------------------------------
# ai-service import helpers
# ---------------------------------------------------------------------------


def _ensure_app_importable() -> None:
    """Put the ``ai-service`` package root on ``sys.path``.

    Works from the repository layout (``<repo>/ai-service``) and from inside
    the ai-service container (``/app``).  Raises :class:`RuntimeError` when
    the ``app`` package cannot be located anywhere.
    """
    candidates = [
        _EVAL_DIR.parent / "ai-service",
        Path("/app"),
        Path.cwd(),
    ]
    for candidate in candidates:
        if (candidate / "app" / "__init__.py").is_file():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return
    raise RuntimeError(
        "cannot locate the ai-service 'app' package; run this script inside "
        "the ai-service container or set PYTHONPATH to the ai-service directory"
    )


# ---------------------------------------------------------------------------
# Embedders (offline-safe by default)
# ---------------------------------------------------------------------------


class HashingEmbedding:
    """Deterministic feature-hashing embedder for offline evaluation.

    Token unigrams and bigrams are hashed (blake2b — stable across processes)
    into a *dim*-dimensional vector with signed accumulation; the result is
    L2-normalised.  Cosine similarity then approximates lexical overlap,
    which exercises the full retrieval stack without any embedding API.

    This is deliberately **not** a semantic model: use it for deterministic
    CI guards, not for judging absolute retrieval quality.
    """

    def __init__(self, dim: int = DEFAULT_HASHING_DIM, ngrams: int = 2) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.ngrams = max(1, ngrams)

    def _vector(self, text: str) -> list[float]:
        """Hash *text* into a unit-length vector (all-zeros if no tokens)."""
        tokens = _TOKEN_RE.findall(text.lower())
        features = list(tokens)
        for n in range(2, self.ngrams + 1):
            features.extend(" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1))

        vector = [0.0] * self.dim
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            sign = 1.0 if (value >> 63) & 1 else -1.0
            vector[value % self.dim] += sign

        norm = math.sqrt(sum(component * component for component in vector))
        if norm == 0.0:
            return vector
        return [component / norm for component in vector]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed document texts (one unit-length vector per text)."""
        return [self._vector(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self._vector(text)


class FastEmbedEmbedding:
    """Local ONNX embeddings via ``fastembed`` (no API key required).

    The model is loaded lazily on first use and cached by fastembed.  Output
    vectors are L2-normalised to match the app's embedding-provider contract.
    """

    def __init__(
        self, model_name: str = DEFAULT_FASTEMBED_MODEL, cache_dir: str | None = None
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model: Any = None

    @staticmethod
    def _normalize(vector: Sequence[float]) -> list[float]:
        norm = math.sqrt(sum(component * component for component in vector))
        if norm == 0.0:
            return list(vector)
        return [component / norm for component in vector]

    def _ensure_model(self) -> Any:
        if self._model is None:
            from fastembed import TextEmbedding

            logger.info("Loading fastembed model '%s'…", self.model_name)
            self._model = TextEmbedding(model_name=self.model_name, cache_dir=self.cache_dir)
        return self._model

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed document texts with the local ONNX model."""
        model = await asyncio.to_thread(self._ensure_model)
        raw = await asyncio.to_thread(lambda: [vec.tolist() for vec in model.embed(list(texts))])
        return [self._normalize(vector) for vector in raw]

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        vectors = await self.embed_documents([text])
        return vectors[0]


def build_embedder(
    name: str,
    *,
    dim: int | None = None,
    fastembed_model: str = DEFAULT_FASTEMBED_MODEL,
    fastembed_cache: str | None = None,
) -> tuple[Any, int]:
    """Build an embedding provider and resolve its vector dimensionality.

    Args:
        name:           ``hashing``, ``fastembed``, ``gemini`` or ``openai``.
        dim:            Explicit dimensionality override.
        fastembed_model: Model id for the ``fastembed`` provider.
        fastembed_cache: fastembed download cache directory.

    Returns:
        ``(embedder, dim)`` — an object with async ``embed_documents`` /
        ``embed_query`` methods and the vector size for the Qdrant collection.

    Raises:
        ValueError: On an unknown provider or missing dimensionality.
    """
    if name == "hashing":
        resolved_dim = dim or DEFAULT_HASHING_DIM
        return HashingEmbedding(dim=resolved_dim), resolved_dim
    if name == "fastembed":
        resolved_dim = dim or FASTEMBED_MODEL_DIMS.get(fastembed_model)
        if resolved_dim is None:
            raise ValueError(
                f"unknown dimensionality for fastembed model '{fastembed_model}'; "
                "pass --dim explicitly"
            )
        return (
            FastEmbedEmbedding(model_name=fastembed_model, cache_dir=fastembed_cache),
            resolved_dim,
        )
    if name in API_EMBEDDER_DIMS:
        from app.ingestion.embeddings import get_embedding_provider

        resolved_dim = dim or API_EMBEDDER_DIMS[name]
        return get_embedding_provider(name), resolved_dim
    raise ValueError(f"unknown embedder '{name}'")


# ---------------------------------------------------------------------------
# Sweep helpers
# ---------------------------------------------------------------------------


def parse_int_list(value: str, *, name: str) -> list[int]:
    """Parse ``"400, 800 ,1200"`` into ``[400, 800, 1200]``.

    Raises:
        ValueError: On empty input or non-integer / negative values.
    """
    numbers: list[int] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            number = int(raw)
        except ValueError as exc:
            raise ValueError(f"{name}: '{raw}' is not an integer") from exc
        if number < 0:
            raise ValueError(f"{name}: values must be >= 0 (got {number})")
        numbers.append(number)
    if not numbers:
        raise ValueError(f"{name}: expected at least one integer")
    return numbers


def build_sweep_grid(sizes: Sequence[int], overlaps: Sequence[int]) -> list[tuple[int, int]]:
    """Build the ``(chunk_size, chunk_overlap)`` combinations to evaluate.

    Invalid combinations (``overlap >= size`` or ``size <= 0``) are skipped
    because :class:`~app.ingestion.chunker.RecursiveCharacterTextSplitter`
    would reject them.

    Raises:
        ValueError: If no valid combination remains.
    """
    grid: list[tuple[int, int]] = []
    for size in sizes:
        for overlap in overlaps:
            if size <= 0 or overlap < 0 or overlap >= size:
                logger.warning(
                    "Skipping invalid chunk configuration (size=%s, overlap=%s).", size, overlap
                )
                continue
            grid.append((size, overlap))
    if not grid:
        raise ValueError("no valid (chunk_size, chunk_overlap) combinations to evaluate")
    return grid


def pick_best(
    summaries: Sequence[dict[str, Any]],
    *,
    preferred: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Return the best sweep entry by Hit Rate, with MRR as tie-breaker.

    When several configurations tie on both metrics and *preferred* is among
    them, it wins — a config should only change when there is evidence for
    it (the current default is passed as *preferred*).
    """

    def score(entry: dict[str, Any]) -> tuple[float, float]:
        metrics = entry["metrics"]
        return (float(metrics.get("hit_rate") or 0.0), float(metrics.get("mrr") or 0.0))

    best_score = max(score(entry) for entry in summaries)
    tied = [entry for entry in summaries if score(entry) == best_score]
    if preferred is not None:
        for entry in tied:
            if (entry["chunk_size"], entry["chunk_overlap"]) == preferred:
                return entry
    return tied[0]


def render_comparison_table(summaries: Sequence[dict[str, Any]], top_k: int) -> str:
    """Render the chunking sweep comparison as a Markdown table."""
    lines = [
        f"| Chunk size | Overlap | Chunks | Hit Rate@{top_k} | MRR | Mean ms | p95 ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for entry in summaries:
        metrics = entry["metrics"]
        latency = metrics["latency_ms"]
        lines.append(
            f"| {entry['chunk_size']} | {entry['chunk_overlap']} | {entry['chunks']} "
            f"| {run_eval._format_metric(metrics['hit_rate'], percent=True)} "
            f"| {run_eval._format_metric(metrics['mrr'])} "
            f"| {latency['mean']:.0f} | {latency['p95']:.0f} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@dataclass
class ConfigRun:
    """Outcome of evaluating one ``(chunk_size, chunk_overlap)`` configuration."""

    chunk_size: int
    chunk_overlap: int
    chunks: int
    metrics: dict[str, Any]
    markdown: str
    payload: dict[str, Any]


async def evaluate_config(
    *,
    chunk_size: int,
    chunk_overlap: int,
    embedder: Any,
    dim: int,
    items: Sequence[run_eval.EvalItem],
    args: argparse.Namespace,
    stats_path: Path,
    label: str,
) -> ConfigRun:
    """Ingest the corpus with one chunking config and evaluate retrieval.

    Uses the real ingestion pipeline and retrieval stack, mirroring the
    service's query path (minus answer generation).
    """
    from qdrant_client.http import models as qdrant_models

    from app.config import settings
    from app.ingestion.chunker import RecursiveCharacterTextSplitter
    from app.ingestion.pipeline import IngestionPipeline
    from app.retrieval.hybrid import _load_bm25_stats_cached
    from app.retrieval.rerank import get_reranker, retrieve
    from app.retrieval.vector_store import QdrantVectorStore, VectorStoreError

    # Point every configuration-dependent setting at this run.
    settings.chunk_size = chunk_size
    settings.chunk_overlap = chunk_overlap
    settings.bm25_stats_path = str(stats_path)
    if args.rerank is not None:
        settings.rerank_enabled = args.rerank
    get_reranker.cache_clear()
    _load_bm25_stats_cached.cache_clear()

    store = QdrantVectorStore(
        url=args.qdrant_url,
        collection=args.collection,
        vector_size=dim,
    )
    try:
        await store.delete_collection()
    except VectorStoreError:
        pass  # nothing to delete on a fresh Qdrant
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    pipeline = IngestionPipeline(vector_store=store, embedder=embedder, splitter=splitter)
    ingestion = await pipeline.run(args.corpus)
    if ingestion.errors:
        logger.warning("Ingestion reported %d error(s).", len(ingestion.errors))
    if ingestion.total_upserted == 0:
        raise RuntimeError(f"ingestion produced no points from '{args.corpus}'")
    # The stats file was just (re)written; drop any stale cached encoder.
    _load_bm25_stats_cached.cache_clear()

    rerank_enabled = settings.rerank_enabled
    results: list[run_eval.EvalResult] = []
    total = len(items)
    for index, item in enumerate(items, start=1):
        if not args.quiet:
            print(
                f"[{index}/{total}] {item.id}: {item.question[:60]}",
                file=sys.stderr,
            )
        game_slug = item.game if (args.game_filter and item.game and item.game != "multi") else None
        qdrant_filter = None
        if game_slug:
            qdrant_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="game",
                        match=qdrant_models.MatchValue(value=game_slug),
                    )
                ]
            )
        try:
            started = time.monotonic()
            query_vector = await embedder.embed_query(item.question)
            hits = await retrieve(
                store,
                item.question,
                query_vector,
                top_k=args.top_k,
                filter=qdrant_filter,
            )
            latency_ms = (time.monotonic() - started) * 1000.0
        except Exception as exc:  # noqa: BLE001 - recorded per item, like run_eval
            result = run_eval.EvalResult(item=item, error=f"{type(exc).__name__}: {exc}")
            result.latency_ms = (time.monotonic() - started) * 1000.0
            results.append(result)
            continue

        result = run_eval.EvalResult(item=item, latency_ms=latency_ms)
        result.sources = [
            {
                "id": hit.chunk_id,
                "source": str(hit.payload.get("source") or ""),
                "snippet": hit.content[:300],
                "score": round(hit.score, 4),
            }
            for hit in hits
        ]
        result.retrieved_source_ids = run_eval.retrieved_source_ids(result.sources)
        result.hit = run_eval.hit_at_k(
            result.retrieved_source_ids, item.expected_source_ids, args.top_k
        )
        result.reciprocal_rank = run_eval.reciprocal_rank(
            result.retrieved_source_ids, item.expected_source_ids, args.top_k
        )
        results.append(result)

    errors = sum(1 for result in results if result.error is not None)
    if errors == len(results):
        raise RuntimeError(f"every query failed; first error: {results[0].error}")

    metrics = run_eval.aggregate(results, top_k=args.top_k)
    # Retrieval-only run: answer generation never happens.
    metrics["refusal_accuracy"] = None

    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    markdown = run_eval.build_markdown_report(
        label,
        metrics,
        results,
        generated_at=generated_at,
        api_url=f"in-process (qdrant={args.qdrant_url or 'settings'}, collection={args.collection})",
        dataset_path=args.dataset,
    )
    markdown += "\n".join(
        [
            "",
            "## Retrieval configuration",
            "",
            "| Setting | Value |",
            "|---|---|",
            f"| Chunk size | {chunk_size} |",
            f"| Chunk overlap | {chunk_overlap} |",
            f"| Chunks ingested | {ingestion.total_chunks} |",
            f"| Embedder | {args.embedder} (dim={dim}) |",
            f"| Re-ranking | {'enabled' if rerank_enabled else 'disabled'} |",
            f"| Top-k | {args.top_k} |",
            "",
            "> Retrieval-only run: answers are not generated, so refusal",
            "> accuracy and RAGAS are not applicable (`n/a` above).",
            "",
        ]
    )
    payload = {
        "label": label,
        "generated_at": generated_at,
        "mode": "retrieval-only",
        "config": {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunks": ingestion.total_chunks,
            "embedder": args.embedder,
            "dim": dim,
            "rerank_enabled": rerank_enabled,
            "top_k": args.top_k,
            "collection": args.collection,
        },
        "dataset": str(args.dataset),
        "top_k": args.top_k,
        "metrics": metrics,
        "results": [result.to_dict() for result in results],
    }
    return ConfigRun(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunks=ingestion.total_chunks,
        metrics=metrics,
        markdown=markdown,
        payload=payload,
    )


# ---------------------------------------------------------------------------
# Reporting / orchestration
# ---------------------------------------------------------------------------


def _render_sweep_markdown(
    *,
    label: str,
    generated_at: str,
    summaries: Sequence[dict[str, Any]],
    best: dict[str, Any],
    args: argparse.Namespace,
    rerank_enabled: bool,
) -> str:
    lines = [
        f"# RetroGPT chunking sweep — {label}",
        "",
        f"- Generated: {generated_at}",
        f"- Dataset: `{args.dataset}`",
        f"- Embedder: `{args.embedder}`",
        f"- Re-ranking: {'enabled' if rerank_enabled else 'disabled'}",
        f"- Top-k: {args.top_k}",
        "",
        "## Comparison",
        "",
        render_comparison_table(summaries, args.top_k),
        "",
        "## Best configuration",
        "",
        f"`chunk_size={best['chunk_size']}, chunk_overlap={best['chunk_overlap']}` — "
        f"Hit Rate@{args.top_k} "
        f"{run_eval._format_metric(best['metrics']['hit_rate'], percent=True)}, "
        f"MRR {run_eval._format_metric(best['metrics']['mrr'])}.",
        "",
        "> Offline ranking (see docs/decisions.md); confirm with the credentialed",
        "> end-to-end evaluation before changing defaults.",
        "",
    ]
    return "\n".join(lines)


async def _run(args: argparse.Namespace, items: list[run_eval.EvalItem]) -> int:
    from app.config import settings

    # Capture the configured defaults before any evaluation mutates settings.
    default_chunk = (settings.chunk_size, settings.chunk_overlap)

    try:
        embedder, dim = build_embedder(
            args.embedder,
            dim=args.dim,
            fastembed_model=args.fastembed_model,
            fastembed_cache=args.fastembed_cache,
        )
    except ValueError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    rerank_state = args.rerank if args.rerank is not None else settings.rerank_enabled
    print(
        f"Retrieval evaluation: {len(items)} questions, embedder={args.embedder} (dim={dim}), "
        f"rerank={'on' if rerank_state else 'off'}, top_k={args.top_k}",
        file=sys.stderr,
    )

    with tempfile.TemporaryDirectory(prefix="retrogpt-eval-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        if args.sweep_sizes:
            try:
                sizes = parse_int_list(args.sweep_sizes, name="--sweep-sizes")
                overlaps = parse_int_list(
                    args.sweep_overlaps or DEFAULT_SWEEP_OVERLAPS, name="--sweep-overlaps"
                )
                grid = build_sweep_grid(sizes, overlaps)
            except ValueError as exc:
                print(f"FATAL: {exc}", file=sys.stderr)
                return 2

            if args.baseline is not None or args.min_hit_rate is not None:
                print(
                    "NOTE: quality guards apply to single-configuration runs only; "
                    "ignored in sweep mode.",
                    file=sys.stderr,
                )

            summaries: list[dict[str, Any]] = []
            for chunk_size, chunk_overlap in grid:
                print(
                    f"\n=== chunk_size={chunk_size}, chunk_overlap={chunk_overlap} ===",
                    file=sys.stderr,
                )
                run = await evaluate_config(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    embedder=embedder,
                    dim=dim,
                    items=items,
                    args=args,
                    stats_path=tmp_path / f"bm25_{chunk_size}_{chunk_overlap}.json",
                    label=f"{args.label}-{chunk_size}-{chunk_overlap}",
                )
                summaries.append(
                    {
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "chunks": run.chunks,
                        "metrics": run.metrics,
                    }
                )
                print(
                    f"chunk_size={chunk_size}, overlap={chunk_overlap}: "
                    f"Hit Rate@{args.top_k} "
                    f"{run_eval._format_metric(run.metrics['hit_rate'], percent=True)}, "
                    f"MRR {run_eval._format_metric(run.metrics['mrr'])}",
                    file=sys.stderr,
                )

            best = pick_best(summaries, preferred=default_chunk)
            generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
            markdown = _render_sweep_markdown(
                label=args.label,
                generated_at=generated_at,
                summaries=summaries,
                best=best,
                args=args,
                rerank_enabled=rerank_state,
            )
            payload = {
                "label": args.label,
                "generated_at": generated_at,
                "mode": "chunking-sweep",
                "dataset": str(args.dataset),
                "embedder": args.embedder,
                "top_k": args.top_k,
                "configs": summaries,
                "best": {
                    "chunk_size": best["chunk_size"],
                    "chunk_overlap": best["chunk_overlap"],
                    "metrics": best["metrics"],
                },
            }
            json_path, md_path = run_eval.write_reports(
                args.results_dir, args.label, payload, markdown
            )
            print("")
            print(render_comparison_table(summaries, args.top_k))
            print(
                f"\nBest: chunk_size={best['chunk_size']}, "
                f"chunk_overlap={best['chunk_overlap']}"
            )
            print(f"Reports written to {md_path} and {json_path}")
            return 0

        # Single-configuration run.
        chunk_size = args.chunk_size if args.chunk_size is not None else settings.chunk_size
        chunk_overlap = (
            args.chunk_overlap if args.chunk_overlap is not None else settings.chunk_overlap
        )
        if chunk_overlap >= chunk_size:
            print(
                f"FATAL: chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})",
                file=sys.stderr,
            )
            return 2

        run = await evaluate_config(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedder=embedder,
            dim=dim,
            items=items,
            args=args,
            stats_path=tmp_path / "bm25_stats.json",
            label=args.label,
        )

        json_path, md_path = run_eval.write_reports(
            args.results_dir, args.label, run.payload, run.markdown
        )
        metrics = run.metrics
        counts = metrics["counts"]
        print(
            f"Hit Rate@{args.top_k}: {run_eval._format_metric(metrics['hit_rate'], percent=True)}"
            f" | MRR: {run_eval._format_metric(metrics['mrr'])}"
        )
        print(f"Errors: {counts['errors']}/{counts['total']}")
        print(f"Reports written to {md_path} and {json_path}")

        # Quality guards (Issue #23 regression guard).
        guard_failures = run_eval.check_quality_guards(
            metrics,
            min_hit_rate=args.min_hit_rate,
            baseline_path=args.baseline,
            max_hit_rate_drop=args.max_hit_rate_drop,
        )
        if guard_failures:
            for failure in guard_failures:
                print(f"GUARD FAILED: {failure}", file=sys.stderr)
            return 1
        return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate retrieval quality in-process (no LLM, no API key) against "
            "a scratch Qdrant collection.  Supports chunking sweeps (#22) and "
            "CI regression guards (#23)."
        ),
    )
    parser.add_argument(
        "--corpus", type=Path, default=DEFAULT_CORPUS, help="Raw documents directory"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=run_eval.DEFAULT_DATASET,
        help="JSONL labeled dataset",
    )
    parser.add_argument(
        "--embedder",
        choices=["hashing", "fastembed", "gemini", "openai"],
        default=DEFAULT_EMBEDDER,
        help=(
            "Embedding provider: 'hashing' (deterministic, offline) or "
            "'fastembed' (local ONNX) or a hosted provider (needs API key)"
        ),
    )
    parser.add_argument("--dim", type=int, default=None, help="Vector size override")
    parser.add_argument(
        "--fastembed-model", default=DEFAULT_FASTEMBED_MODEL, help="fastembed model id"
    )
    parser.add_argument("--fastembed-cache", default=None, help="fastembed model cache directory")
    parser.add_argument("--chunk-size", type=int, default=None, help="Chunk size override")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="Chunk overlap override")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Retrieval cut-off")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N items")
    parser.add_argument(
        "--game-filter",
        action="store_true",
        help="Scope each query to the item's game via payload filter (off by default)",
    )
    parser.add_argument("--qdrant-url", default=None, help="Qdrant URL (default: settings)")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Scratch collection name")
    parser.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force cross-encoder re-ranking on/off (default: settings)",
    )
    parser.add_argument("--label", default="retrieval", help="Report label")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=run_eval.DEFAULT_RESULTS_DIR,
        help="Report output directory",
    )
    parser.add_argument(
        "--sweep-sizes",
        default=None,
        help="Comma-separated chunk sizes to sweep, e.g. '400,800,1200'",
    )
    parser.add_argument(
        "--sweep-overlaps",
        default=None,
        help=f"Comma-separated overlaps to sweep (default: {DEFAULT_SWEEP_OVERLAPS})",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-question progress")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    parser.add_argument(
        "--min-hit-rate",
        type=float,
        default=None,
        help="Exit 1 if Hit Rate is below this value (0–1)",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Previous JSON report to compare against (CI regression guard)",
    )
    parser.add_argument(
        "--max-hit-rate-drop",
        type=float,
        default=DEFAULT_MAX_HIT_RATE_DROP,
        help="Allowed Hit Rate drop vs. baseline before failing (default: 0.05)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stderr,
    )

    try:
        _ensure_app_importable()
    except RuntimeError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    corpus = args.corpus
    if not corpus.exists():
        fallback = _EVAL_DIR.parent / "data" / "raw"
        if fallback.is_dir():
            corpus = fallback
        else:
            print(f"FATAL: corpus directory not found: {corpus}", file=sys.stderr)
            return 2
    args.corpus = corpus

    try:
        items = run_eval.load_dataset(args.dataset)
    except (FileNotFoundError, ValueError) as exc:
        print(f"FATAL: cannot load dataset: {exc}", file=sys.stderr)
        return 2
    if args.limit is not None:
        items = items[: args.limit]
    if not items:
        print("FATAL: dataset selection is empty", file=sys.stderr)
        return 2

    try:
        return asyncio.run(_run(args, items))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - single top-level failure boundary
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
