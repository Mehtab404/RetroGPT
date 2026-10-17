#!/usr/bin/env python3
"""RetroGPT agent-routing evaluation harness (Issue #35 / backlog #30).

Runs the labeled tool-routing cases in ``eval/agent_cases.jsonl`` against the
AI service ``POST /agent`` endpoint and measures whether the model picks the
expected tool with the expected arguments — the acceptance target for agent
mode is **≥ 8/10** cases routed correctly.

A case passes when **any** of its ``expected_calls`` matches **any** entry in
the response's ``tools_used`` list.  Matching compares the tool name exactly
and the ``args`` mapping as a subset (normalized, order-insensitive), so the
model may pass extra arguments or swap ``game_a``/``game_b`` while wrong-tool
and wrong-game routing still fail the case.

The routing run needs a real provider key (the model decides which tools to
call); the script itself is **stdlib-only** so it can run inside the
``ai-service`` container without extra dependencies.

Usage
-----
    python eval/agent_eval.py --label agent-routing
    python eval/agent_eval.py --min-pass-rate 0.8 --quiet

Exit codes
----------
* ``0`` — evaluation completed and any configured pass-rate guard passed
* ``1`` — pass rate below the ``--min-pass-rate`` guard
* ``2`` — fatal error (cases missing/invalid, service unreachable)
"""

from __future__ import annotations

import argparse
import json
import math
import re
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
DEFAULT_ENDPOINT = "/agent"
DEFAULT_LABEL = "agent-routing"
DEFAULT_MIN_PASS_RATE = 0.8  # acceptance target: 8/10 (Issue #35)
DEFAULT_TIMEOUT = 120.0

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_CASES = EVAL_DIR / "agent_cases.jsonl"
DEFAULT_RESULTS_DIR = EVAL_DIR / "results"

#: Tool names the AI service registers (mirrors app.agent.tools).
KNOWN_TOOLS: frozenset[str] = frozenset({"search_game_docs", "compare_games", "list_games_by_year"})


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpectedCall:
    """One acceptable tool invocation for a case."""

    tool: str
    args: dict[str, Any]


@dataclass(frozen=True)
class AgentCase:
    """One labeled routing case from ``agent_cases.jsonl``."""

    id: str
    question: str
    expected_calls: tuple[ExpectedCall, ...]


@dataclass
class CaseResult:
    """Outcome of asking the agent one case."""

    case: AgentCase
    passed: bool = False
    matched_tool: str = ""
    tools_used: list[dict[str, Any]] = field(default_factory=list)
    steps: int = 0
    latency_ms: float = 0.0
    answer: str = ""
    error: str | None = None

    @property
    def expected_tools(self) -> list[str]:
        """Distinct tool names the case accepts (for the report)."""
        seen: list[str] = []
        for call in self.case.expected_calls:
            if call.tool not in seen:
                seen.append(call.tool)
        return seen

    @property
    def called_tools(self) -> list[str]:
        """Tool names the agent actually called, in order of first use."""
        seen: list[str] = []
        for call in self.tools_used:
            name = str(call.get("name", ""))
            if name and name not in seen:
                seen.append(name)
        return seen

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the JSON report."""
        return {
            "id": self.case.id,
            "question": self.case.question,
            "expected_tools": self.expected_tools,
            "called_tools": self.called_tools,
            "matched_tool": self.matched_tool,
            "passed": self.passed,
            "steps": self.steps,
            "latency_ms": round(self.latency_ms, 2),
            "answer": self.answer,
            "error": self.error,
        }


class AgentClientProtocol(Protocol):
    """Minimal client interface used by :func:`evaluate_case`."""

    def ask(self, question: str, max_steps: int | None = None) -> dict[str, Any]:
        """Send one question to ``POST /agent`` and return the JSON response."""
        ...


class AgentClientError(RuntimeError):
    """Raised when the HTTP call to ``POST /agent`` fails."""


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------


class AgentClient:
    """Stdlib HTTP client for the AI service ``POST /agent`` endpoint."""

    def __init__(
        self,
        base_url: str = DEFAULT_API_URL,
        endpoint: str = DEFAULT_ENDPOINT,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        self.api_key = api_key
        self.timeout = timeout

    @property
    def url(self) -> str:
        return f"{self.base_url}{self.endpoint}"

    def ask(self, question: str, max_steps: int | None = None) -> dict[str, Any]:
        """POST one question and return the parsed JSON response."""
        payload: dict[str, Any] = {"question": question}
        if max_steps is not None:
            payload["max_steps"] = max_steps
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
            raise AgentClientError(f"HTTP {exc.code} from {self.url}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise AgentClientError(f"request to {self.url} failed: {exc}") from exc

        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise AgentClientError(f"invalid JSON from {self.url}: {exc}") from exc
        if not isinstance(data, dict):
            raise AgentClientError(f"unexpected response type from {self.url}: {type(data)}")
        return data


# ---------------------------------------------------------------------------
# Cases loading
# ---------------------------------------------------------------------------


def load_cases(path: Path) -> list[AgentCase]:
    """Load and validate the JSONL routing cases.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If a line is malformed or references an unknown tool.
    """
    cases: list[AgentCase] = []
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
        raw_calls = record.get("expected_calls")
        if not isinstance(raw_calls, list) or not raw_calls:
            raise ValueError(f"{path}:{line_number}: 'expected_calls' must be a non-empty list")
        expected_calls: list[ExpectedCall] = []
        for raw_call in raw_calls:
            if not isinstance(raw_call, dict):
                raise ValueError(f"{path}:{line_number}: each expected call must be an object")
            tool = str(raw_call.get("tool", ""))
            if tool not in KNOWN_TOOLS:
                raise ValueError(f"{path}:{line_number}: unknown tool '{tool}'")
            args = raw_call.get("args") or {}
            if not isinstance(args, dict):
                raise ValueError(f"{path}:{line_number}: 'args' must be an object")
            expected_calls.append(ExpectedCall(tool=tool, args=args))
        cases.append(
            AgentCase(
                id=str(record.get("id") or f"line-{line_number}"),
                question=question,
                expected_calls=tuple(expected_calls),
            )
        )
    if not cases:
        raise ValueError(f"{path}: cases file is empty")
    return cases


# ---------------------------------------------------------------------------
# Matching + scoring
# ---------------------------------------------------------------------------

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _normalize_value(value: Any) -> str:
    """Normalize a value for lenient comparison.

    Lower-cases, strips everything but ``[a-z0-9]`` and collapses separators,
    so ``"Pac-Man"``, ``"pacman"`` and ``"pac man"`` all compare equal, as do
    ``"space_invaders"`` and ``"Space Invaders"``.
    """
    return _NON_ALNUM.sub("", str(value).lower())


def args_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    """True when every *expected* key/value appears in *actual* (subset match).

    Integers (years) match their string form; everything else is compared
    after :func:`_normalize_value`.
    """
    for key, want in expected.items():
        if key not in actual:
            return False
        got = actual[key]
        if isinstance(want, int) and not isinstance(want, bool):
            try:
                if int(got) != want:
                    return False
            except (TypeError, ValueError):
                return False
        elif _normalize_value(got) != _normalize_value(want):
            return False
    return True


def score_case(case: AgentCase, tools_used: Sequence[dict[str, Any]]) -> tuple[bool, str]:
    """Score a response's tool calls against the case.

    Returns:
        ``(passed, matched_tool)`` — *matched_tool* is the expected tool name
        that matched (empty when the case failed).
    """
    for expected in case.expected_calls:
        for call in tools_used:
            if str(call.get("name", "")) != expected.tool:
                continue
            arguments = call.get("arguments")
            if isinstance(arguments, dict) and args_match(expected.args, arguments):
                return True, expected.tool
    return False, ""


def evaluate_case(
    case: AgentCase,
    client: AgentClientProtocol,
    max_steps: int | None = None,
) -> CaseResult:
    """Ask the agent one case and score the tool routing."""
    result = CaseResult(case=case)
    started = time.monotonic()
    try:
        response = client.ask(case.question, max_steps=max_steps)
    except AgentClientError as exc:
        result.error = str(exc)
        result.latency_ms = (time.monotonic() - started) * 1000.0
        return result

    result.latency_ms = (time.monotonic() - started) * 1000.0
    result.answer = str(response.get("answer", ""))
    raw_tools = response.get("tools_used") or []
    result.tools_used = [call for call in raw_tools if isinstance(call, dict)]
    try:
        result.steps = int(response.get("steps", 0))
    except (TypeError, ValueError):
        result.steps = 0
    result.passed, result.matched_tool = score_case(case, result.tools_used)
    return result


def evaluate_cases(
    cases: Sequence[AgentCase],
    client: AgentClientProtocol,
    progress: bool = True,
    max_steps: int | None = None,
) -> list[CaseResult]:
    """Evaluate every case sequentially, reporting progress to stderr."""
    results: list[CaseResult] = []
    total = len(cases)
    for index, case in enumerate(cases, start=1):
        if progress:
            print(f"[{index}/{total}] {case.id}: {case.question[:60]}", file=sys.stderr)
        results.append(evaluate_case(case, client, max_steps=max_steps))
    return results


# ---------------------------------------------------------------------------
# Aggregation + reporting
# ---------------------------------------------------------------------------


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile (``pct`` in 0–100)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(pct / 100.0 * len(ordered)) - 1))
    return ordered[index]


def aggregate(results: Sequence[CaseResult]) -> dict[str, Any]:
    """Compute aggregate routing metrics (pass rate, per-tool, steps, latency)."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    errors = sum(1 for r in results if r.error is not None)
    clean = [r for r in results if r.error is None]

    per_tool: dict[str, dict[str, int]] = {}
    for result in clean:
        for tool in result.expected_tools:
            bucket = per_tool.setdefault(tool, {"n": 0, "passed": 0})
            bucket["n"] += 1
            if result.passed and result.matched_tool == tool:
                bucket["passed"] += 1

    latencies = [r.latency_ms for r in clean]
    steps = [r.steps for r in clean]
    return {
        "pass_rate": round(passed / total, 4) if total else None,
        "counts": {
            "total": total,
            "passed": passed,
            "failed": total - passed - errors,
            "errors": errors,
        },
        "per_tool": per_tool,
        "steps": {
            "mean": round(sum(steps) / len(steps), 2) if steps else 0.0,
            "max": max(steps) if steps else 0,
        },
        "latency_ms": {
            "mean": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "p50": round(percentile(latencies, 50), 2),
            "p95": round(percentile(latencies, 95), 2),
        },
    }


def _format_call(call: dict[str, Any]) -> str:
    return json.dumps(call, ensure_ascii=False, sort_keys=True)


def build_markdown_report(
    label: str,
    metrics: dict[str, Any],
    results: Sequence[CaseResult],
    *,
    generated_at: str,
    api_url: str,
    cases_path: Path,
) -> str:
    """Render the human-readable Markdown report."""
    counts = metrics["counts"]
    lines: list[str] = [
        f"# RetroGPT agent-routing evaluation — {label}",
        "",
        f"- Generated: {generated_at}",
        f"- API: `{api_url}`",
        f"- Cases: `{cases_path}` ({counts['total']} prompts)",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        (
            f"| Routing pass rate | {metrics['pass_rate'] * 100:.1f}% "
            f"({counts['passed']}/{counts['total']}) |"
            if metrics["pass_rate"] is not None
            else "| Routing pass rate | n/a |"
        ),
        f"| Errors | {counts['errors']} |",
        f"| Mean model turns | {metrics['steps']['mean']:.2f} "
        f"(max {metrics['steps']['max']}) |",
        f"| Latency mean / p50 / p95 (ms) | {metrics['latency_ms']['mean']:.0f} / "
        f"{metrics['latency_ms']['p50']:.0f} / {metrics['latency_ms']['p95']:.0f} |",
    ]
    if metrics["per_tool"]:
        lines += [
            "",
            "## Routing by tool",
            "",
            "| Tool | n | Passed | Pass rate |",
            "|---|---|---|---|",
        ]
        for tool, bucket in sorted(metrics["per_tool"].items()):
            rate = bucket["passed"] / bucket["n"] if bucket["n"] else 0.0
            lines.append(f"| `{tool}` | {bucket['n']} | {bucket['passed']} | {rate * 100:.0f}% |")

    lines += [
        "",
        "## Cases",
        "",
        "| Case | Expected | Called | Pass |",
        "|---|---|---|---|",
    ]
    for result in results:
        expected = ", ".join(f"`{tool}`" for tool in result.expected_tools)
        called = ", ".join(f"`{tool}`" for tool in result.called_tools) or "—"
        status = "✅" if result.passed else ("⚠️" if result.error is not None else "❌")
        lines.append(f"| {result.case.id} | {expected} | {called} | {status} |")

    failures = [result for result in results if not result.passed and result.error is None]
    errors = [result for result in results if result.error is not None]
    if failures or errors:
        lines += ["", "## Details", ""]
        for result in failures:
            lines.append(f"- **{result.case.id}** (failed) — {result.case.question}")
            lines.append(
                f"  - expected: {', '.join(_format_call(c.__dict__) for c in result.case.expected_calls)}"
            )
            lines.append(
                "  - called: "
                + (", ".join(_format_call(call) for call in result.tools_used) or "no tools")
            )
        for result in errors:
            lines.append(f"- **{result.case.id}** (error) — {result.error}")
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
# Quality guard
# ---------------------------------------------------------------------------


def check_pass_rate_guard(
    metrics: dict[str, Any],
    min_pass_rate: float | None,
) -> list[str]:
    """Return guard failure messages (empty when the guard passes)."""
    failures: list[str] = []
    pass_rate = metrics.get("pass_rate")
    if pass_rate is None or min_pass_rate is None:
        return failures
    if pass_rate < min_pass_rate:
        failures.append(
            f"Routing pass rate {pass_rate:.3f} is below the required {min_pass_rate:.3f}"
        )
    return failures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the RetroGPT agent-routing cases and report metrics.",
    )
    parser.add_argument("--api-url", default=None, help="Service base URL (default: %(default)s)")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Agent path (default: /agent)")
    parser.add_argument("--api-key", default=None, help="x-api-key header (gateway runs only)")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES, help="JSONL cases path")
    parser.add_argument("--label", default=DEFAULT_LABEL, help="Report label")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N cases")
    parser.add_argument("--max-steps", type=int, default=None, help="Forwarded to POST /agent")
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT, help="Per-request timeout (s)"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-case progress")
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=DEFAULT_MIN_PASS_RATE,
        help="Exit 1 if the pass rate is below this value (0–1, default: 0.8)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    api_url = args.api_url or DEFAULT_API_URL
    try:
        cases = load_cases(args.cases)
    except (FileNotFoundError, ValueError) as exc:
        print(f"FATAL: cannot load cases: {exc}", file=sys.stderr)
        return 2
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        print("FATAL: case selection is empty", file=sys.stderr)
        return 2

    client = AgentClient(
        base_url=api_url,
        endpoint=args.endpoint,
        api_key=args.api_key,
        timeout=args.timeout,
    )
    print(f"Evaluating {len(cases)} agent cases against {client.url}…")

    results = evaluate_cases(
        cases,
        client,
        progress=not args.quiet,
        max_steps=args.max_steps,
    )
    errors = sum(1 for r in results if r.error is not None)
    if errors == len(results):
        print("FATAL: every request failed — is the service running?", file=sys.stderr)
        print(f"  first error: {results[0].error}", file=sys.stderr)
        return 2

    metrics = aggregate(results)
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    markdown = build_markdown_report(
        args.label,
        metrics,
        results,
        generated_at=generated_at,
        api_url=client.url,
        cases_path=args.cases,
    )
    payload = {
        "label": args.label,
        "generated_at": generated_at,
        "api_url": client.url,
        "cases": str(args.cases),
        "metrics": metrics,
        "results": [result.to_dict() for result in results],
    }
    json_path, md_path = write_reports(args.results_dir, args.label, payload, markdown)

    counts = metrics["counts"]
    rate = metrics["pass_rate"] or 0.0
    print(f"Routing pass rate: {rate * 100:.1f}% ({counts['passed']}/{counts['total']})")
    print(f"Errors: {counts['errors']}/{counts['total']}")
    print(f"Reports written to {md_path} and {json_path}")

    guard_failures = check_pass_rate_guard(metrics, args.min_pass_rate)
    if guard_failures:
        for failure in guard_failures:
            print(f"GUARD FAILED: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
