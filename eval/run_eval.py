#!/usr/bin/env python3
"""RetroGPT evaluation harness (Issue #19).

Runs the labeled QA dataset in ``eval/dataset.jsonl`` against a running AI
service (or the API gateway), computes retrieval and answer-quality metrics,
and writes a JSON + Markdown report to ``eval/results/``.

Metrics
-------
* **Hit Rate@k** — share of answerable questions whose top-``k`` retrieved
  passages include at least one expected source document.
* **MRR** — mean reciprocal rank of the first expected source in the
  retrieved list (1.0 = always first, 0.0 = never retrieved).
* **Refusal accuracy** — share of unanswerable questions where the answer is
  a refusal ("I don't know …") instead of a hallucination.
* **RAGAS faithfulness / answer relevancy** — optional (``--with-ragas``);
  requires the ``ragas`` package and an LLM API key. Retrieved contexts are
  approximated with the 300-char snippets returned by the service.

The script is intentionally **stdlib-only** so it can run inside the
``ai-service`` container (``docker compose exec ai-service python
/eval/run_eval.py``) without extra dependencies.

Usage
-----
    python eval/run_eval.py --label baseline
    python eval/run_eval.py --subset-ci --with-ragas
    python eval/run_eval.py --baseline eval/results/2026-01-01-baseline.json

Exit codes
----------
* ``0`` — evaluation completed (and any configured quality guards passed)
* ``1`` — a quality guard failed (``--min-hit-rate`` or ``--baseline``)
* ``2`` — fatal error (dataset missing/invalid, API unreachable)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_ENDPOINT = "/query"
DEFAULT_TOP_K = 5
DEFAULT_LABEL = "baseline"
DEFAULT_MAX_HIT_RATE_DROP = 0.05  # 5 percentage points (Issue #23 guard)

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = EVAL_DIR / "dataset.jsonl"
DEFAULT_RESULTS_DIR = EVAL_DIR / "results"

#: Substrings (case-insensitive) that mark an answer as a refusal.
REFUSAL_MARKERS: tuple[str, ...] = (
    "i don't know",
    "i do not know",
    "not available in the provided sources",
    "unable to answer",
    "cannot answer",
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalItem:
    """One labeled question from the dataset."""

    id: str
    question: str
    expected_answer: str = ""
    expected_source_ids: tuple[str, ...] = ()
    category: str = "easy"
    game: str = ""

    @property
    def answerable(self) -> bool:
        """True when the dataset expects an answer (has expected sources)."""
        return bool(self.expected_source_ids)


@dataclass
class EvalResult:
    """Outcome of querying the service for a single :class:`EvalItem`."""

    item: EvalItem
    answer: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    retrieved_source_ids: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    error: str | None = None
    hit: bool = False
    reciprocal_rank: float = 0.0
    refused: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the JSON report (without the nested item object)."""
        return {
            "id": self.item.id,
            "game": self.item.game,
            "category": self.item.category,
            "question": self.item.question,
            "expected_source_ids": list(self.item.expected_source_ids),
            "retrieved_source_ids": self.retrieved_source_ids,
            "answer": self.answer,
            "hit": self.hit,
            "reciprocal_rank": round(self.reciprocal_rank, 4),
            "refused": self.refused,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
        }

    @property
    def contexts(self) -> list[str]:
        """Retrieved snippets, for RAGAS context metrics."""
        return [str(src.get("snippet", "")) for src in self.sources if src.get("snippet")]


class QueryClient(Protocol):
    """Minimal client interface used by :func:`evaluate_item`."""

    def query(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        game_slug: str | None = None,
    ) -> dict[str, Any]:
        """Send one question and return the service JSON response."""
        ...


class EvalClientError(RuntimeError):
    """Raised when the HTTP call to the service fails."""


class RagasUnavailable(RuntimeError):
    """Raised when RAGAS metrics are requested but cannot be computed."""


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------


class ApiClient:
    """Stdlib HTTP client for the AI service ``POST /query`` (or gateway)."""

    def __init__(
        self,
        base_url: str = DEFAULT_API_URL,
        endpoint: str = DEFAULT_ENDPOINT,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        self.api_key = api_key
        self.timeout = timeout

    @property
    def url(self) -> str:
        return f"{self.base_url}{self.endpoint}"

    def query(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        game_slug: str | None = None,
    ) -> dict[str, Any]:
        """POST one question and return the parsed JSON response."""
        payload: dict[str, Any] = {"question": question, "top_k": top_k}
        if game_slug:
            payload["game_slug"] = game_slug
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        if self.api_key:
            request.add_header("x-api-key", self.api_key)

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:200]
            raise EvalClientError(f"HTTP {exc.code} from {self.url}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise EvalClientError(f"request to {self.url} failed: {exc}") from exc

        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise EvalClientError(f"invalid JSON from {self.url}: {exc}") from exc
        if not isinstance(data, dict):
            raise EvalClientError(f"unexpected response type from {self.url}: {type(data)}")
        return data


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset(path: Path) -> list[EvalItem]:
    """Load and validate the JSONL evaluation dataset.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If a line is not a JSON object or lacks ``id``/``question``.
    """
    items: list[EvalItem] = []
    text = path.read_text(encoding="utf-8")
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        question = record.get("question")
        if not question or not isinstance(question, str):
            raise ValueError(f"{path}:{line_number}: missing 'question'")
        item_id = record.get("id") or f"line-{line_number}"
        expected = record.get("expected_source_ids") or []
        if not isinstance(expected, list):
            raise ValueError(f"{path}:{line_number}: 'expected_source_ids' must be a list")
        items.append(
            EvalItem(
                id=str(item_id),
                question=question,
                expected_answer=str(record.get("expected_answer", "")),
                expected_source_ids=tuple(str(src) for src in expected),
                category=str(record.get("category", "easy")),
                game=str(record.get("game", "")),
            )
        )
    if not items:
        raise ValueError(f"{path}: dataset is empty")
    return items


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def normalize_source_id(source: str) -> str:
    """Normalize a source path or identifier to a bare document stem.

    ``"/data/raw/asteroids/asteroids_manual.md"`` → ``"asteroids_manual"``.
    The dataset stores document slugs (file stems) so retrieval results must
    be normalized the same way before comparing.
    """
    cleaned = source.strip().replace("\\", "/")
    if not cleaned:
        return ""
    return Path(cleaned).stem


def retrieved_source_ids(sources: Sequence[dict[str, Any]]) -> list[str]:
    """Extract normalized document ids from response source entries, in order.

    Prefers the ``source`` path field (added to ``SourceDoc`` in Issue #19);
    falls back to the ``id`` field so older payloads still produce *some*
    signal. Entries with neither are kept as empty strings to preserve
    ranking positions.
    """
    ids: list[str] = []
    for source in sources:
        raw = str(source.get("source") or source.get("id") or "")
        ids.append(normalize_source_id(raw))
    return ids


def hit_at_k(stems: Sequence[str], expected: Sequence[str], k: int = DEFAULT_TOP_K) -> bool:
    """True if any expected document appears in the top *k* retrieved stems."""
    top = set(stems[:k])
    return any(doc in top for doc in expected)


def reciprocal_rank(
    stems: Sequence[str],
    expected: Sequence[str],
    k: int = DEFAULT_TOP_K,
) -> float:
    """Reciprocal rank (within top *k*) of the first expected document."""
    wanted = set(expected)
    for rank, stem in enumerate(stems[:k], start=1):
        if stem in wanted:
            return 1.0 / rank
    return 0.0


def is_refusal(answer: str) -> bool:
    """True when the answer is a refusal rather than an attempted answer."""
    normalized = " ".join(answer.lower().split())
    return any(marker in normalized for marker in REFUSAL_MARKERS)


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile (``pct`` in 0–100)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(pct / 100.0 * len(ordered)) - 1))
    return ordered[index]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_item(
    item: EvalItem,
    client: QueryClient,
    top_k: int = DEFAULT_TOP_K,
    game_filter: bool = False,
) -> EvalResult:
    """Query the service for a single item and score the outcome.

    By default no ``game_slug`` filter is applied — this measures the same
    retrieval path users hit. Pass ``game_filter=True`` to scope retrieval to
    the item's game (useful for debugging per-game retrieval).
    """
    result = EvalResult(item=item)
    game_slug = item.game if (game_filter and item.game and item.game != "multi") else None
    started = time.monotonic()
    try:
        response = client.query(item.question, top_k=top_k, game_slug=game_slug)
    except EvalClientError as exc:
        result.error = str(exc)
        result.latency_ms = (time.monotonic() - started) * 1000.0
        return result

    result.latency_ms = (time.monotonic() - started) * 1000.0
    result.answer = str(response.get("answer", ""))
    raw_sources = response.get("sources") or []
    result.sources = [src for src in raw_sources if isinstance(src, dict)]
    result.retrieved_source_ids = retrieved_source_ids(result.sources)
    result.hit = hit_at_k(result.retrieved_source_ids, item.expected_source_ids, top_k)
    result.reciprocal_rank = reciprocal_rank(
        result.retrieved_source_ids, item.expected_source_ids, top_k
    )
    result.refused = is_refusal(result.answer)
    return result


def evaluate_items(
    items: Sequence[EvalItem],
    client: QueryClient,
    top_k: int = DEFAULT_TOP_K,
    progress: bool = True,
    game_filter: bool = False,
) -> list[EvalResult]:
    """Evaluate every item sequentially, reporting progress to stderr."""
    results: list[EvalResult] = []
    total = len(items)
    for index, item in enumerate(items, start=1):
        if progress:
            print(f"[{index}/{total}] {item.id}: {item.question[:60]}", file=sys.stderr)
        results.append(evaluate_item(item, client, top_k, game_filter=game_filter))
    return results


def aggregate(results: Sequence[EvalResult], top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    """Compute aggregate metrics (retrieval, refusal, latency, per-category)."""
    errors = [r for r in results if r.error is not None]
    clean = [r for r in results if r.error is None]
    answerable = [r for r in clean if r.item.answerable]
    unanswerable = [r for r in clean if not r.item.answerable]

    hits = sum(1 for r in answerable if r.hit)
    refusals = sum(1 for r in unanswerable if r.refused)
    latencies = [r.latency_ms for r in clean]

    per_category: dict[str, dict[str, Any]] = {}
    for result in answerable:
        bucket = per_category.setdefault(result.item.category, {"n": 0, "hits": 0, "rr_sum": 0.0})
        bucket["n"] += 1
        bucket["hits"] += int(result.hit)
        bucket["rr_sum"] += result.reciprocal_rank
    for bucket in per_category.values():
        bucket["hit_rate"] = round(bucket["hits"] / bucket["n"], 4)
        bucket["mrr"] = round(bucket["rr_sum"] / bucket["n"], 4)
        del bucket["rr_sum"]

    return {
        "hit_rate": round(hits / len(answerable), 4) if answerable else None,
        "mrr": (
            round(sum(r.reciprocal_rank for r in answerable) / len(answerable), 4)
            if answerable
            else None
        ),
        "refusal_accuracy": (round(refusals / len(unanswerable), 4) if unanswerable else None),
        "counts": {
            "total": len(results),
            "evaluated": len(clean),
            "answerable": len(answerable),
            "unanswerable": len(unanswerable),
            "hits": hits,
            "refusals": refusals,
            "errors": len(errors),
            "top_k": top_k,
        },
        "latency_ms": {
            "mean": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "p50": round(percentile(latencies, 50), 2),
            "p95": round(percentile(latencies, 95), 2),
        },
        "per_category": per_category,
    }


# ---------------------------------------------------------------------------
# RAGAS (optional)
# ---------------------------------------------------------------------------


def compute_ragas_metrics(results: Sequence[EvalResult]) -> dict[str, float]:
    """Compute RAGAS faithfulness + answer relevancy for answerable results.

    Lazy-imports ``ragas`` so the baseline path has no heavy dependencies.
    Contexts are approximated with the service's returned snippets.

    Raises:
        RagasUnavailable: If ``ragas`` is missing or no usable results exist.
    """
    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate  # type: ignore
        from ragas.metrics import answer_relevancy, faithfulness  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised via mocks
        raise RagasUnavailable(f"ragas is not installed ({exc})") from exc

    samples = [
        SingleTurnSample(
            user_input=result.item.question,
            response=result.answer,
            retrieved_contexts=result.contexts,
            reference=result.item.expected_answer,
        )
        for result in results
        if result.error is None and result.item.answerable and result.answer
    ]
    if not samples:
        raise RagasUnavailable("no answerable results to score")

    report = evaluate(dataset=EvaluationDataset(samples=samples), metrics=[faithfulness, answer_relevancy])  # type: ignore

    scores: dict[str, float] = {}
    for name in ("faithfulness", "answer_relevancy"):
        try:
            value = report[name]
        except Exception:  # noqa: BLE001 - ragas result types vary by version
            value = getattr(report, name, None)
        if value is not None:
            scores[name] = round(float(value), 4)
    return scores


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _format_metric(value: float | None, percent: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%" if percent else f"{value:.3f}"


def build_markdown_report(
    label: str,
    metrics: dict[str, Any],
    results: Sequence[EvalResult],
    *,
    generated_at: str,
    api_url: str,
    dataset_path: Path,
    ragas: dict[str, float] | None = None,
    ragas_note: str | None = None,
) -> str:
    """Render the human-readable Markdown report."""
    counts = metrics["counts"]
    latency = metrics["latency_ms"]
    lines: list[str] = [
        f"# RetroGPT evaluation — {label}",
        "",
        f"- Generated: {generated_at}",
        f"- API: `{api_url}`",
        f"- Dataset: `{dataset_path}` ({counts['total']} questions)",
        f"- Top-k: {counts['top_k']}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Hit Rate@{counts['top_k']} | {_format_metric(metrics['hit_rate'], percent=True)}"
        f" ({counts['hits']}/{counts['answerable']}) |",
        f"| MRR | {_format_metric(metrics['mrr'])} |",
        f"| Refusal accuracy | {_format_metric(metrics['refusal_accuracy'], percent=True)}"
        f" ({counts['refusals']}/{counts['unanswerable']}) |",
    ]
    if ragas:
        lines.append(
            f"| Answer faithfulness (RAGAS) | {_format_metric(ragas.get('faithfulness'))} |"
        )
        lines.append(
            f"| Answer relevancy (RAGAS) | {_format_metric(ragas.get('answer_relevancy'))} |"
        )
    else:
        lines.append("| Answer faithfulness (RAGAS) | _not run_ |")
        lines.append("| Answer relevancy (RAGAS) | _not run_ |")
    lines += [
        "",
        "## Latency (uncached)",
        "",
        "| Stat | ms |",
        "|---|---|",
        f"| mean | {latency['mean']:.0f} |",
        f"| p50 | {latency['p50']:.0f} |",
        f"| p95 | {latency['p95']:.0f} |",
    ]
    if ragas_note:
        lines += ["", f"> RAGAS note: {ragas_note}"]

    if metrics["per_category"]:
        lines += [
            "",
            "## Retrieval by category",
            "",
            "| Category | n | Hit Rate | MRR |",
            "|---|---|---|---|",
        ]
        for category, bucket in sorted(metrics["per_category"].items()):
            lines.append(
                f"| {category} | {bucket['n']} | {_format_metric(bucket['hit_rate'], percent=True)}"
                f" | {_format_metric(bucket['mrr'])} |"
            )

    failures = [r for r in results if r.error is None and r.item.answerable and not r.hit]
    missed_refusals = [
        r for r in results if r.error is None and not r.item.answerable and not r.refused
    ]
    errors = [r for r in results if r.error is not None]
    if failures or missed_refusals or errors:
        lines += ["", "## Failures", ""]
        for result in failures[:15]:
            retrieved = ", ".join(f"`{stem}`" for stem in result.retrieved_source_ids) or "—"
            expected = ", ".join(f"`{stem}`" for stem in result.item.expected_source_ids)
            lines.append(
                f"- **{result.item.id}** (no hit) — expected {expected}; retrieved {retrieved}"
            )
        if len(failures) > 15:
            lines.append(f"- …and {len(failures) - 15} more retrieval failures (see JSON report).")
        for result in missed_refusals:
            lines.append(
                f"- **{result.item.id}** (should have refused) — answer: {result.answer[:160]!r}"
            )
        for result in errors[:10]:
            lines.append(f"- **{result.item.id}** (error) — {result.error}")
    lines.append("")
    return "\n".join(lines)


def write_reports(
    results_dir: Path,
    label: str,
    payload: dict[str, Any],
    markdown: str,
) -> tuple[Path, Path]:
    """Write ``<date>-<label>.json`` and ``.md``; time-suffix on collisions."""
    results_dir.mkdir(parents=True, exist_ok=True)
    date = datetime.now(UTC).strftime("%Y-%m-%d")
    safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in label) or "run"
    base = results_dir / f"{date}-{safe_label}"
    if base.with_suffix(".json").exists() or base.with_suffix(".md").exists():
        stamp = datetime.now(UTC).strftime("%H%M%S")
        base = results_dir / f"{date}-{safe_label}-{stamp}"

    json_path = base.with_suffix(".json")
    md_path = base.with_suffix(".md")
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    md_path.write_text(markdown, encoding="utf-8", newline="\n")
    return json_path, md_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the RetroGPT evaluation dataset and report metrics.",
    )
    parser.add_argument("--api-url", default=None, help="Service base URL (default: %(default)s)")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Query path (default: /query)")
    parser.add_argument("--api-key", default=None, help="x-api-key header (gateway runs only)")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="JSONL dataset path")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Retrieval cut-off")
    parser.add_argument("--label", default=DEFAULT_LABEL, help="Report label, e.g. 'baseline'")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N items")
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout (s)")
    parser.add_argument("--with-ragas", action="store_true", help="Also compute RAGAS metrics")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-question progress")
    parser.add_argument(
        "--use-game-filter",
        action="store_true",
        help="Scope each query to the item's game via game_slug (off by default)",
    )
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
        help="Previous JSON report to compare against (regression guard)",
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

    api_url = args.api_url or DEFAULT_API_URL
    try:
        items = load_dataset(args.dataset)
    except (FileNotFoundError, ValueError) as exc:
        print(f"FATAL: cannot load dataset: {exc}", file=sys.stderr)
        return 2
    if args.limit is not None:
        items = items[: args.limit]
    if not items:
        print("FATAL: dataset selection is empty", file=sys.stderr)
        return 2

    client = ApiClient(
        base_url=api_url,
        endpoint=args.endpoint,
        api_key=args.api_key,
        timeout=args.timeout,
    )
    print(f"Evaluating {len(items)} questions against {client.url} (top_k={args.top_k})…")

    results = evaluate_items(
        items,
        client,
        top_k=args.top_k,
        progress=not args.quiet,
        game_filter=args.use_game_filter,
    )
    errors = sum(1 for r in results if r.error is not None)
    if errors == len(results):
        print("FATAL: every request failed — is the service running?", file=sys.stderr)
        print(f"  first error: {results[0].error}", file=sys.stderr)
        return 2

    metrics = aggregate(results, top_k=args.top_k)

    ragas: dict[str, float] | None = None
    ragas_note: str | None = None
    if args.with_ragas:
        try:
            ragas = compute_ragas_metrics(results)
        except RagasUnavailable as exc:
            ragas_note = f"skipped — {exc}"
            print(f"RAGAS skipped: {exc}", file=sys.stderr)

    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    markdown = build_markdown_report(
        args.label,
        metrics,
        results,
        generated_at=generated_at,
        api_url=client.url,
        dataset_path=args.dataset,
        ragas=ragas,
        ragas_note=ragas_note,
    )
    payload = {
        "label": args.label,
        "generated_at": generated_at,
        "api_url": client.url,
        "dataset": str(args.dataset),
        "top_k": args.top_k,
        "metrics": metrics,
        "ragas": ragas,
        "ragas_note": ragas_note,
        "results": [result.to_dict() for result in results],
    }
    json_path, md_path = write_reports(args.results_dir, args.label, payload, markdown)

    counts = metrics["counts"]
    print(
        f"Hit Rate@{args.top_k}: {_format_metric(metrics['hit_rate'], percent=True)}"
        f" | MRR: {_format_metric(metrics['mrr'])}"
        f" | Refusal accuracy: {_format_metric(metrics['refusal_accuracy'], percent=True)}"
    )
    print(f"Errors: {counts['errors']}/{counts['total']}")
    print(f"Reports written to {md_path} and {json_path}")

    # ------------------------------------------------------------------
    # Quality guards (Issue #23 regression guard)
    # ------------------------------------------------------------------
    guard_failures: list[str] = []
    if args.min_hit_rate is not None and metrics["hit_rate"] is not None:
        if metrics["hit_rate"] < args.min_hit_rate:
            guard_failures.append(
                f"Hit Rate {metrics['hit_rate']:.3f} is below the required "
                f"{args.min_hit_rate:.3f}"
            )
    if args.baseline is not None and metrics["hit_rate"] is not None:
        try:
            baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
            baseline_rate = baseline.get("metrics", {}).get("hit_rate")
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARNING: cannot read baseline {args.baseline}: {exc}", file=sys.stderr)
            baseline_rate = None
        if baseline_rate is not None:
            drop = float(baseline_rate) - metrics["hit_rate"]
            if drop > args.max_hit_rate_drop:
                guard_failures.append(
                    f"Hit Rate dropped {drop:.3f} vs. baseline "
                    f"({baseline_rate:.3f} → {metrics['hit_rate']:.3f}), "
                    f"allowed {args.max_hit_rate_drop:.3f}"
                )
    if guard_failures:
        for failure in guard_failures:
            print(f"GUARD FAILED: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
