"""Tests for ``eval/run_eval.py`` (Issue #19).

The eval script is stdlib-only and injects its own HTTP client, so these
tests exercise it without a running service:

* metric helpers (normalization, Hit@k, MRR, refusal detection, percentiles)
* dataset loading / validation
* per-item and aggregate evaluation with fake clients
* RAGAS integration behind a mocked ``ragas`` module
* Markdown/JSON report rendering
* CLI orchestration including the quality guards (Issue #23 hooks)
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

_EVAL_DIR = Path(__file__).resolve().parents[1]
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

import run_eval  # noqa: E402  (import after sys.path injection)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_item(
    item_id: str = "q-1",
    question: str = "How many points is a large asteroid?",
    expected_sources: tuple[str, ...] = ("asteroids_manual",),
    category: str = "easy",
    game: str = "asteroids",
    expected_answer: str = "20 points",
) -> run_eval.EvalItem:
    return run_eval.EvalItem(
        id=item_id,
        question=question,
        expected_answer=expected_answer,
        expected_source_ids=expected_sources,
        category=category,
        game=game,
    )


class FakeClient:
    """Stand-in for :class:`run_eval.ApiClient` in tests."""

    url = "http://fake-service:8000/query"

    def __init__(
        self,
        responses: dict[str, dict[str, Any]] | None = None,
        error: str | None = None,
        answer: str = "Answer.",
    ) -> None:
        self.responses = responses or {}
        self.error = error
        self.answer = answer
        self.calls: list[tuple[str, int, str | None]] = []

    def query(
        self,
        question: str,
        top_k: int = 5,
        game_slug: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append((question, top_k, game_slug))
        if self.error is not None:
            raise run_eval.EvalClientError(self.error)
        if question in self.responses:
            return self.responses[question]
        return {
            "answer": self.answer,
            "sources": [
                {
                    "id": "abc123",
                    "title": "Asteroids Manual",
                    "source": "/data/raw/asteroids/asteroids_manual.md",
                    "score": 0.9,
                    "snippet": "Large asteroids are worth 20 points.",
                }
            ],
            "latency_ms": 5.0,
        }


def make_result(
    item: run_eval.EvalItem,
    retrieved: list[str],
    answer: str = "Answer.",
    error: str | None = None,
) -> run_eval.EvalResult:
    result = run_eval.EvalResult(item=item)
    result.answer = answer
    result.error = error
    result.latency_ms = 10.0
    if error is None:
        result.sources = [
            {"id": f"s{i}", "source": f"/data/raw/x/{stem}.md", "snippet": "ctx"}
            for i, stem in enumerate(retrieved)
        ]
        result.retrieved_source_ids = list(retrieved)
        result.hit = run_eval.hit_at_k(retrieved, item.expected_source_ids)
        result.reciprocal_rank = run_eval.reciprocal_rank(retrieved, item.expected_source_ids)
        result.refused = run_eval.is_refusal(answer)
    return result


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


class TestNormalizeSourceId:
    def test_posix_path_to_stem(self) -> None:
        assert (
            run_eval.normalize_source_id("/data/raw/asteroids/asteroids_manual.md")
            == "asteroids_manual"
        )

    def test_windows_path_to_stem(self) -> None:
        assert run_eval.normalize_source_id("D:\\data\\raw\\pong\\pong_faq.md") == "pong_faq"

    def test_path_without_extension(self) -> None:
        assert run_eval.normalize_source_id("/data/raw/pacman/pacman_manual") == "pacman_manual"

    def test_empty_and_whitespace(self) -> None:
        assert run_eval.normalize_source_id("") == ""
        assert run_eval.normalize_source_id("   ") == ""


class TestRetrievedSourceIds:
    def test_prefers_source_field(self) -> None:
        sources = [{"id": "deadbeef", "source": "/data/raw/pong/pong_manual.md"}]
        assert run_eval.retrieved_source_ids(sources) == ["pong_manual"]

    def test_falls_back_to_id(self) -> None:
        sources = [{"id": "/data/raw/breakout/breakout_manual.md"}]
        assert run_eval.retrieved_source_ids(sources) == ["breakout_manual"]

    def test_preserves_ranking_order_and_empties(self) -> None:
        sources = [
            {"source": "/data/a.md"},
            {"id": "unknown"},
            {"source": "/data/b.md"},
        ]
        assert run_eval.retrieved_source_ids(sources) == ["a", "unknown", "b"]

    def test_empty_list(self) -> None:
        assert run_eval.retrieved_source_ids([]) == []


class TestHitAtK:
    def test_hit_within_top_k(self) -> None:
        assert run_eval.hit_at_k(["a", "b", "asteroids_manual"], ("asteroids_manual",), k=3)

    def test_miss_outside_top_k(self) -> None:
        stems = ["a", "b", "c", "d", "e", "asteroids_manual"]
        assert not run_eval.hit_at_k(stems, ("asteroids_manual",), k=5)

    def test_multiple_expected_docs(self) -> None:
        stems = ["pong_manual", "x"]
        assert run_eval.hit_at_k(stems, ("a", "pong_manual"), k=2)

    def test_no_expected_docs_never_hits(self) -> None:
        assert not run_eval.hit_at_k(["a"], (), k=5)


class TestReciprocalRank:
    def test_first_position(self) -> None:
        assert run_eval.reciprocal_rank(["asteroids_manual"], ("asteroids_manual",)) == 1.0

    def test_third_position(self) -> None:
        stems = ["a", "b", "asteroids_manual"]
        assert run_eval.reciprocal_rank(stems, ("asteroids_manual",)) == pytest.approx(1 / 3)

    def test_not_found(self) -> None:
        assert run_eval.reciprocal_rank(["a", "b"], ("asteroids_manual",)) == 0.0

    def test_beyond_top_k(self) -> None:
        stems = ["a"] * 5 + ["asteroids_manual"]
        assert run_eval.reciprocal_rank(stems, ("asteroids_manual",), k=5) == 0.0

    def test_earliest_match_wins(self) -> None:
        stems = ["pong_manual", "asteroids_manual"]
        assert run_eval.reciprocal_rank(stems, ("asteroids_manual", "pong_manual")) == 1.0


class TestIsRefusal:
    def test_service_no_context_answer(self) -> None:
        answer = "I don't know — this information is not available in the provided sources."
        assert run_eval.is_refusal(answer)

    def test_extended_marker(self) -> None:
        assert run_eval.is_refusal("I am unable to answer that from the sources.")

    def test_normal_answer_is_not_refusal(self) -> None:
        assert not run_eval.is_refusal("Large asteroids are worth 20 points [1].")

    def test_case_and_whitespace_insensitive(self) -> None:
        assert run_eval.is_refusal("I  DON'T\nknow.")

    def test_empty_answer(self) -> None:
        assert not run_eval.is_refusal("")


class TestPercentile:
    def test_p50_p95(self) -> None:
        values = [float(i) for i in range(1, 11)]
        assert run_eval.percentile(values, 50) == 5.0
        assert run_eval.percentile(values, 95) == 10.0

    def test_single_value(self) -> None:
        assert run_eval.percentile([42.0], 95) == 42.0

    def test_empty(self) -> None:
        assert run_eval.percentile([], 95) == 0.0


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


class TestLoadDataset:
    def _write(self, tmp_path: Path, lines: list[str]) -> Path:
        path = tmp_path / "dataset.jsonl"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_parses_entries_and_skips_blank_lines(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            [
                '{"id": "a-1", "question": "Q1?", "expected_answer": "A1",'
                ' "expected_source_ids": ["asteroids_manual"], "category": "hard",'
                ' "game": "asteroids"}',
                "",
                '{"id": "a-2", "question": "Q2?", "expected_source_ids": []}',
            ],
        )
        items = run_eval.load_dataset(path)
        assert [item.id for item in items] == ["a-1", "a-2"]
        assert items[0].expected_source_ids == ("asteroids_manual",)
        assert items[0].category == "hard"
        assert items[1].expected_source_ids == ()
        assert not items[1].answerable
        assert items[0].answerable

    def test_missing_question_raises(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, ['{"id": "x"}'])
        with pytest.raises(ValueError, match="missing 'question'"):
            run_eval.load_dataset(path)

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, ["{not json}"])
        with pytest.raises(ValueError, match="invalid JSON"):
            run_eval.load_dataset(path)

    def test_non_object_raises(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, ['["a", "b"]'])
        with pytest.raises(ValueError, match="expected a JSON object"):
            run_eval.load_dataset(path)

    def test_expected_source_ids_must_be_list(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, ['{"id": "x", "question": "q", "expected_source_ids": "a"}'])
        with pytest.raises(ValueError, match="must be a list"):
            run_eval.load_dataset(path)

    def test_empty_dataset_raises(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, [""])
        with pytest.raises(ValueError, match="dataset is empty"):
            run_eval.load_dataset(path)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            run_eval.load_dataset(tmp_path / "nope.jsonl")

    def test_real_dataset_is_valid(self) -> None:
        items = run_eval.load_dataset(_EVAL_DIR / "dataset.jsonl")
        assert len(items) >= 50
        categories = {item.category for item in items}
        assert {"easy", "hard", "multi-hop", "unanswerable"} <= categories
        unanswerable = [item for item in items if item.category == "unanswerable"]
        assert unanswerable and all(not item.answerable for item in unanswerable)


# ---------------------------------------------------------------------------
# Per-item evaluation
# ---------------------------------------------------------------------------


class TestEvaluateItem:
    def test_hit_and_refusal_scoring(self) -> None:
        item = make_item()
        client = FakeClient()
        result = run_eval.evaluate_item(item, client)
        assert result.error is None
        assert result.hit
        assert result.reciprocal_rank == 1.0
        assert result.retrieved_source_ids == ["asteroids_manual"]
        assert not result.refused
        assert result.latency_ms >= 0
        # No game filter by default (matches the user-facing path).
        assert client.calls[0][2] is None
        assert client.calls[0][1] == 5

    def test_unanswerable_refusal_detected(self) -> None:
        item = make_item(expected_sources=(), category="unanswerable", game="multi")
        client = FakeClient(answer="I don't know — not available in the provided sources.")
        result = run_eval.evaluate_item(item, client)
        assert result.refused
        assert not result.hit
        assert result.reciprocal_rank == 0.0

    def test_game_filter_opt_in(self) -> None:
        item = make_item(game="pacman")
        client = FakeClient()
        run_eval.evaluate_item(item, client, game_filter=True)
        assert client.calls[0][2] == "pacman"

    def test_multi_game_items_never_filtered(self) -> None:
        item = make_item(game="multi", expected_sources=("a", "b"))
        client = FakeClient()
        run_eval.evaluate_item(item, client, game_filter=True)
        assert client.calls[0][2] is None

    def test_client_error_recorded_not_raised(self) -> None:
        item = make_item()
        client = FakeClient(error="connection refused")
        result = run_eval.evaluate_item(item, client)
        assert result.error == "connection refused"
        assert not result.hit

    def test_top_k_forwarded(self) -> None:
        item = make_item()
        client = FakeClient()
        run_eval.evaluate_item(item, client, top_k=10)
        assert client.calls[0][1] == 10


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class TestAggregate:
    def test_metrics_and_counts(self) -> None:
        item_a = make_item("a", expected_sources=("asteroids_manual",))
        item_b = make_item("b", expected_sources=("pong_manual",), game="pong")
        item_u = make_item("u", expected_sources=(), category="unanswerable")
        results = [
            make_result(item_a, ["asteroids_manual", "x"]),  # hit, rr=1
            make_result(item_b, ["x", "pong_manual"]),  # hit, rr=1/2
            make_result(item_u, [], answer="I don't know."),  # refusal
        ]
        metrics = run_eval.aggregate(results, top_k=5)
        assert metrics["hit_rate"] == 1.0
        assert metrics["mrr"] == pytest.approx(0.75)
        assert metrics["refusal_accuracy"] == 1.0
        assert metrics["counts"]["answerable"] == 2
        assert metrics["counts"]["unanswerable"] == 1
        assert metrics["counts"]["hits"] == 2
        assert metrics["counts"]["errors"] == 0
        assert metrics["per_category"]["easy"]["n"] == 2
        assert metrics["per_category"]["easy"]["hit_rate"] == 1.0

    def test_miss_lowers_hit_rate(self) -> None:
        item = make_item(expected_sources=("asteroids_manual",))
        metrics = run_eval.aggregate([make_result(item, ["x", "y"])])
        assert metrics["hit_rate"] == 0.0
        assert metrics["mrr"] == 0.0

    def test_none_when_bucket_empty(self) -> None:
        item = make_item()
        metrics = run_eval.aggregate([make_result(item, ["asteroids_manual"])])
        assert metrics["refusal_accuracy"] is None

        unanswered = make_item(expected_sources=(), category="unanswerable")
        metrics = run_eval.aggregate([make_result(unanswered, [], answer="I don't know.")])
        assert metrics["hit_rate"] is None
        assert metrics["mrr"] is None

    def test_errors_excluded_from_metrics(self) -> None:
        item = make_item()
        results = [
            make_result(item, ["asteroids_manual"]),
            make_result(item, [], error="HTTP 500"),
        ]
        metrics = run_eval.aggregate(results)
        assert metrics["counts"]["errors"] == 1
        assert metrics["counts"]["evaluated"] == 1
        assert metrics["hit_rate"] == 1.0

    def test_latency_stats(self) -> None:
        item = make_item()
        results = []
        for i in range(1, 11):
            result = make_result(item, ["asteroids_manual"])
            result.latency_ms = float(i * 10)
            results.append(result)
        metrics = run_eval.aggregate(results)
        assert metrics["latency_ms"]["mean"] == 55.0
        assert metrics["latency_ms"]["p50"] == 50.0
        assert metrics["latency_ms"]["p95"] == 100.0


# ---------------------------------------------------------------------------
# RAGAS integration (mocked)
# ---------------------------------------------------------------------------


class TestRagas:
    def _install_fake_ragas(
        self, monkeypatch: pytest.MonkeyPatch, scores: dict[str, float]
    ) -> None:
        fake_ragas = types.ModuleType("ragas")
        fake_metrics = types.ModuleType("ragas.metrics")

        class SingleTurnSample:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs

        class EvaluationDataset:
            def __init__(self, samples: list[Any]) -> None:
                self.samples = samples

        def evaluate(dataset: EvaluationDataset, metrics: list[Any]) -> dict[str, float]:
            return scores

        fake_ragas.EvaluationDataset = EvaluationDataset  # type: ignore[attr-defined]
        fake_ragas.SingleTurnSample = SingleTurnSample  # type: ignore[attr-defined]
        fake_ragas.evaluate = evaluate  # type: ignore[attr-defined]
        fake_metrics.faithfulness = "faithfulness"  # type: ignore[attr-defined]
        fake_metrics.answer_relevancy = "answer_relevancy"  # type: ignore[attr-defined]

        monkeypatch.setitem(sys.modules, "ragas", fake_ragas)
        monkeypatch.setitem(sys.modules, "ragas.metrics", fake_metrics)

    def test_scores_returned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._install_fake_ragas(monkeypatch, {"faithfulness": 0.91234, "answer_relevancy": 0.8})
        item = make_item()
        results = [make_result(item, ["asteroids_manual"], answer="20 points [1]")]
        scores = run_eval.compute_ragas_metrics(results)
        assert scores == {"faithfulness": 0.9123, "answer_relevancy": 0.8}

    def test_unavailable_when_import_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "ragas", None)
        item = make_item()
        results = [make_result(item, ["asteroids_manual"])]
        with pytest.raises(run_eval.RagasUnavailable):
            run_eval.compute_ragas_metrics(results)

    def test_unavailable_without_answerable_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._install_fake_ragas(monkeypatch, {})
        item = make_item(expected_sources=(), category="unanswerable")
        results = [make_result(item, [], answer="I don't know.")]
        with pytest.raises(run_eval.RagasUnavailable, match="no answerable results"):
            run_eval.compute_ragas_metrics(results)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


class TestBuildMarkdownReport:
    def _report(self, ragas: dict[str, float] | None = None) -> str:
        item = make_item("a-1")
        missed = make_item("a-2", question="Unanswerable?", expected_sources=())
        results = [
            make_result(item, ["asteroids_manual"]),
            make_result(missed, ["asteroids_manual"], answer="Actually it is 42."),
        ]
        metrics = run_eval.aggregate(results)
        return run_eval.build_markdown_report(
            "test-run",
            metrics,
            results,
            generated_at="2026-01-01 00:00:00 UTC",
            api_url="http://localhost:8000/query",
            dataset_path=Path("eval/dataset.jsonl"),
            ragas=ragas,
            ragas_note=None if ragas else "skipped — ragas is not installed",
        )

    def test_contains_metric_rows(self) -> None:
        report = self._report()
        assert "# RetroGPT evaluation — test-run" in report
        assert "| Hit Rate@5 |" in report
        assert "| MRR |" in report
        assert "| Refusal accuracy |" in report
        assert "_not run_" in report
        assert "skipped — ragas is not installed" in report

    def test_lists_failures(self) -> None:
        report = self._report()
        assert "- **a-2** (should have refused)" in report

    def test_ragas_values_rendered(self) -> None:
        report = self._report(ragas={"faithfulness": 0.91, "answer_relevancy": 0.5})
        assert "| Answer faithfulness (RAGAS) | 0.910 |" in report
        assert "_not run_" not in report

    def test_category_table(self) -> None:
        report = self._report()
        assert "## Retrieval by category" in report


class TestWriteReports:
    def test_writes_json_and_markdown(self, tmp_path: Path) -> None:
        json_path, md_path = run_eval.write_reports(
            tmp_path, "baseline", {"metrics": {}}, "# Report\n"
        )
        assert json_path.exists() and md_path.exists()
        assert json_path.suffix == ".json"
        assert md_path.read_text(encoding="utf-8") == "# Report\n"
        assert json.loads(json_path.read_text(encoding="utf-8")) == {"metrics": {}}
        assert md_path.name.endswith("-baseline.md")

    def test_collision_gets_time_suffix(self, tmp_path: Path) -> None:
        first_json, _ = run_eval.write_reports(tmp_path, "run", {}, "x")
        second_json, _ = run_eval.write_reports(tmp_path, "run", {}, "x")
        assert first_json != second_json


# ---------------------------------------------------------------------------
# CLI orchestration
# ---------------------------------------------------------------------------


def _write_dataset(path: Path) -> None:
    entries = [
        {
            "id": "a-1",
            "game": "asteroids",
            "category": "easy",
            "question": "How many points is a large asteroid?",
            "expected_answer": "20",
            "expected_source_ids": ["asteroids_manual"],
        },
        {
            "id": "u-1",
            "game": "multi",
            "category": "unanswerable",
            "question": "What is the cheat code for Pong 2?",
            "expected_answer": "",
            "expected_source_ids": [],
        },
    ]
    path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8")


class TestMain:
    def _fake_client(
        self,
        responses: dict[str, dict[str, Any]] | None = None,
        error: str | None = None,
    ) -> Any:
        def factory(**kwargs: Any) -> FakeClient:
            return FakeClient(responses=responses, error=error)

        return factory

    def _responses(self) -> dict[str, dict[str, Any]]:
        return {
            "How many points is a large asteroid?": {
                "answer": "20 points [1]",
                "sources": [
                    {
                        "id": "x",
                        "source": "/data/raw/asteroids/asteroids_manual.md",
                        "snippet": "20",
                    }
                ],
            },
            "What is the cheat code for Pong 2?": {
                "answer": "I don't know — not available in the provided sources.",
                "sources": [],
            },
        }

    def test_successful_run_writes_reports(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        dataset = tmp_path / "dataset.jsonl"
        _write_dataset(dataset)
        monkeypatch.setattr(run_eval, "ApiClient", self._fake_client(self._responses()))

        code = run_eval.main(
            [
                "--dataset",
                str(dataset),
                "--results-dir",
                str(tmp_path / "results"),
                "--label",
                "unit",
                "--quiet",
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "Hit Rate@5: 100.0%" in out
        assert "Refusal accuracy: 100.0%" in out
        assert list((tmp_path / "results").glob("*-unit.md"))
        assert list((tmp_path / "results").glob("*-unit.json"))

    def test_all_requests_failing_is_fatal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dataset = tmp_path / "dataset.jsonl"
        _write_dataset(dataset)
        monkeypatch.setattr(run_eval, "ApiClient", self._fake_client(error="down"))
        code = run_eval.main(["--dataset", str(dataset), "--results-dir", str(tmp_path), "--quiet"])
        assert code == 2

    def test_missing_dataset_is_fatal(self, tmp_path: Path) -> None:
        code = run_eval.main(
            ["--dataset", str(tmp_path / "missing.jsonl"), "--results-dir", str(tmp_path)]
        )
        assert code == 2

    def test_min_hit_rate_guard_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dataset = tmp_path / "dataset.jsonl"
        _write_dataset(dataset)
        responses = self._responses()
        responses["How many points is a large asteroid?"]["sources"] = []
        monkeypatch.setattr(run_eval, "ApiClient", self._fake_client(responses))
        code = run_eval.main(
            [
                "--dataset",
                str(dataset),
                "--results-dir",
                str(tmp_path / "results"),
                "--quiet",
                "--min-hit-rate",
                "0.5",
            ]
        )
        assert code == 1

    def test_baseline_regression_guard_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dataset = tmp_path / "dataset.jsonl"
        _write_dataset(dataset)
        responses = self._responses()
        responses["How many points is a large asteroid?"]["sources"] = []
        baseline = tmp_path / "baseline.json"
        baseline.write_text(json.dumps({"metrics": {"hit_rate": 0.9}}), encoding="utf-8")
        monkeypatch.setattr(run_eval, "ApiClient", self._fake_client(responses))
        code = run_eval.main(
            [
                "--dataset",
                str(dataset),
                "--results-dir",
                str(tmp_path / "results"),
                "--quiet",
                "--baseline",
                str(baseline),
            ]
        )
        assert code == 1

    def test_baseline_within_tolerance_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dataset = tmp_path / "dataset.jsonl"
        _write_dataset(dataset)
        baseline = tmp_path / "baseline.json"
        baseline.write_text(json.dumps({"metrics": {"hit_rate": 1.0}}), encoding="utf-8")
        monkeypatch.setattr(run_eval, "ApiClient", self._fake_client(self._responses()))
        code = run_eval.main(
            [
                "--dataset",
                str(dataset),
                "--results-dir",
                str(tmp_path / "results"),
                "--quiet",
                "--baseline",
                str(baseline),
            ]
        )
        assert code == 0

    def test_limit_applied(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        dataset = tmp_path / "dataset.jsonl"
        _write_dataset(dataset)
        client = FakeClient(responses=self._responses())
        monkeypatch.setattr(run_eval, "ApiClient", lambda **kwargs: client)
        code = run_eval.main(
            [
                "--dataset",
                str(dataset),
                "--results-dir",
                str(tmp_path / "results"),
                "--quiet",
                "--limit",
                "1",
            ]
        )
        assert code == 0
        assert len(client.calls) == 1

    def test_with_ragas_reports_skip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        dataset = tmp_path / "dataset.jsonl"
        _write_dataset(dataset)
        monkeypatch.setattr(run_eval, "ApiClient", self._fake_client(self._responses()))
        monkeypatch.setitem(sys.modules, "ragas", None)
        code = run_eval.main(
            [
                "--dataset",
                str(dataset),
                "--results-dir",
                str(tmp_path / "results"),
                "--quiet",
                "--with-ragas",
            ]
        )
        assert code == 0
        assert "RAGAS skipped" in capsys.readouterr().err
        report = next((tmp_path / "results").glob("*.json"))
        payload = json.loads(report.read_text(encoding="utf-8"))
        assert payload["ragas"] is None
        assert "skipped" in payload["ragas_note"]


class TestQualityGuards:
    def test_passes_without_guards(self) -> None:
        assert run_eval.check_quality_guards({"hit_rate": 0.5}) == []

    def test_min_hit_rate_failure_message(self) -> None:
        failures = run_eval.check_quality_guards({"hit_rate": 0.4}, min_hit_rate=0.8)
        assert len(failures) == 1
        assert "below the required 0.800" in failures[0]

    def test_min_hit_rate_met_passes(self) -> None:
        assert run_eval.check_quality_guards({"hit_rate": 0.9}, min_hit_rate=0.8) == []

    def test_baseline_drop_failure(self, tmp_path: Path) -> None:
        baseline = tmp_path / "baseline.json"
        baseline.write_text(json.dumps({"metrics": {"hit_rate": 0.9}}), encoding="utf-8")
        failures = run_eval.check_quality_guards(
            {"hit_rate": 0.8}, baseline_path=baseline, max_hit_rate_drop=0.05
        )
        assert len(failures) == 1
        assert "dropped 0.100" in failures[0]

    def test_baseline_within_tolerance_passes(self, tmp_path: Path) -> None:
        baseline = tmp_path / "baseline.json"
        baseline.write_text(json.dumps({"metrics": {"hit_rate": 0.83}}), encoding="utf-8")
        assert run_eval.check_quality_guards({"hit_rate": 0.8}, baseline_path=baseline) == []

    def test_unreadable_baseline_is_skipped(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        failures = run_eval.check_quality_guards(
            {"hit_rate": 0.1}, baseline_path=tmp_path / "missing.json"
        )
        assert failures == []
        assert "cannot read baseline" in capsys.readouterr().err

    def test_missing_hit_rate_is_skipped(self) -> None:
        assert run_eval.check_quality_guards({"hit_rate": None}, min_hit_rate=0.8) == []


class TestApiClient:
    def test_url_composition(self) -> None:
        client = run_eval.ApiClient(base_url="http://example.com:8000/", endpoint="query")
        assert client.url == "http://example.com:8000/query"

    def test_query_parses_json_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        class FakeResponse:
            def __enter__(self) -> FakeResponse:
                return self

            def __exit__(self, *args: Any) -> None:
                return None

            def read(self) -> bytes:
                return b'{"answer": "ok", "sources": []}'

        def fake_urlopen(request: Any, timeout: float | None = None) -> FakeResponse:
            captured["url"] = request.full_url
            captured["headers"] = dict(request.header_items())
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse()

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        client = run_eval.ApiClient(base_url="http://svc:8000", api_key="secret")
        response = client.query("q?", top_k=7)
        assert response["answer"] == "ok"
        assert captured["url"] == "http://svc:8000/query"
        assert captured["body"] == {"question": "q?", "top_k": 7}
        assert captured["headers"].get("X-api-key") == "secret"

    def test_http_error_raises_client_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import urllib.error

        def fake_urlopen(request: Any, timeout: float | None = None) -> None:
            raise urllib.error.HTTPError(request.full_url, 503, "down", {}, None)

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        client = run_eval.ApiClient()
        with pytest.raises(run_eval.EvalClientError, match="HTTP 503"):
            client.query("q?")

    def test_invalid_json_raises_client_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeResponse:
            def __enter__(self) -> FakeResponse:
                return self

            def __exit__(self, *args: Any) -> None:
                return None

            def read(self) -> bytes:
                return b"<html>nope</html>"

        monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout=None: FakeResponse())
        client = run_eval.ApiClient()
        with pytest.raises(run_eval.EvalClientError, match="invalid JSON"):
            client.query("q?")
