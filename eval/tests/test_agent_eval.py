"""Tests for ``eval/agent_eval.py`` (Issue #35 / backlog #30).

The script is stdlib-only and injects its own HTTP client, so these tests
exercise it without a running service or an LLM key:

* case loading / validation (including the committed ``agent_cases.jsonl``)
* value normalization + argument subset matching
* per-case scoring and aggregate metrics
* Markdown/JSON report rendering
* CLI orchestration including the pass-rate guard (exit codes 0/1/2)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

_EVAL_DIR = Path(__file__).resolve().parents[1]
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

import agent_eval  # noqa: E402  (import after sys.path injection)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_case(
    case_id: str = "agent-x",
    question: str = "How many points is a large asteroid worth?",
    expected_calls: list[tuple[str, dict[str, Any]]] | None = None,
) -> agent_eval.AgentCase:
    calls = expected_calls or [("search_game_docs", {"game": "asteroids"})]
    return agent_eval.AgentCase(
        id=case_id,
        question=question,
        expected_calls=tuple(agent_eval.ExpectedCall(tool=t, args=a) for t, a in calls),
    )


class FakeAgentClient:
    """Stand-in for :class:`agent_eval.AgentClient` in tests."""

    url = "http://fake-service:8000/agent"

    def __init__(
        self,
        responses: dict[str, dict[str, Any]] | None = None,
        error: str | None = None,
    ) -> None:
        self.responses = responses or {}
        self.error = error
        self.calls: list[tuple[str, int | None]] = []

    def ask(self, question: str, max_steps: int | None = None) -> dict[str, Any]:
        self.calls.append((question, max_steps))
        if self.error is not None:
            raise agent_eval.AgentClientError(self.error)
        if question in self.responses:
            return self.responses[question]
        return {"answer": "Answer [1].", "tools_used": [], "steps": 1, "latency_ms": 5.0}


def make_result(
    case: agent_eval.AgentCase,
    tools_used: list[dict[str, Any]],
    error: str | None = None,
) -> agent_eval.CaseResult:
    result = agent_eval.CaseResult(case=case)
    result.error = error
    result.latency_ms = 10.0
    if error is None:
        result.tools_used = tools_used
        result.steps = 2
        result.passed, result.matched_tool = agent_eval.score_case(case, tools_used)
    return result


def passing_responses(cases: list[agent_eval.AgentCase]) -> dict[str, dict[str, Any]]:
    """One correct agent response per case (first acceptable call)."""
    responses: dict[str, dict[str, Any]] = {}
    for case in cases:
        expected = case.expected_calls[0]
        responses[case.question] = {
            "answer": "Answer [1].",
            "tools_used": [{"name": expected.tool, "arguments": expected.args}],
            "steps": 2,
            "latency_ms": 10.0,
        }
    return responses


# ---------------------------------------------------------------------------
# Case loading
# ---------------------------------------------------------------------------


class TestLoadCases:
    def _write(self, tmp_path: Path, lines: list[str]) -> Path:
        path = tmp_path / "cases.jsonl"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def test_committed_cases_file_is_valid(self) -> None:
        cases = agent_eval.load_cases(agent_eval.DEFAULT_CASES)
        assert len(cases) == 10
        assert len({case.id for case in cases}) == len(cases)
        tools = {call.tool for case in cases for call in case.expected_calls}
        assert tools == set(agent_eval.KNOWN_TOOLS)
        for case in cases:
            assert case.question
            assert case.expected_calls

    def test_parses_entries_and_skips_blank_lines(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            [
                "",
                json.dumps(
                    {
                        "id": "a-1",
                        "question": "q?",
                        "expected_calls": [{"tool": "compare_games", "args": {"game_a": "pacman"}}],
                    }
                ),
                "",
            ],
        )
        cases = agent_eval.load_cases(path)
        assert len(cases) == 1
        assert cases[0].id == "a-1"
        assert cases[0].expected_calls[0].tool == "compare_games"

    def test_missing_question_raises(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, ['{"id": "x", "expected_calls": []}'])
        with pytest.raises(ValueError, match="missing 'question'"):
            agent_eval.load_cases(path)

    def test_empty_expected_calls_raises(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, ['{"id": "x", "question": "q", "expected_calls": []}'])
        with pytest.raises(ValueError, match="non-empty list"):
            agent_eval.load_cases(path)

    def test_unknown_tool_raises(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            ['{"id": "x", "question": "q", "expected_calls": [{"tool": "teleport", "args": {}}]}'],
        )
        with pytest.raises(ValueError, match="unknown tool"):
            agent_eval.load_cases(path)

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, ["{not json}"])
        with pytest.raises(ValueError, match="invalid JSON"):
            agent_eval.load_cases(path)

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, [""])
        with pytest.raises(ValueError, match="empty"):
            agent_eval.load_cases(path)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            agent_eval.load_cases(tmp_path / "nope.jsonl")


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


class TestValueNormalization:
    def test_pacman_variants_match(self) -> None:
        assert agent_eval._normalize_value("Pac-Man") == agent_eval._normalize_value("pacman")
        assert agent_eval._normalize_value("pac man") == "pacman"

    def test_underscore_variants_match(self) -> None:
        assert agent_eval._normalize_value("Space Invaders") == agent_eval._normalize_value(
            "space_invaders"
        )

    def test_case_insensitive(self) -> None:
        assert agent_eval._normalize_value("ASTEROIDS") == agent_eval._normalize_value("asteroids")


class TestArgsMatch:
    def test_subset_match_ignores_extra_arguments(self) -> None:
        assert agent_eval.args_match(
            {"game": "asteroids"}, {"game": "asteroids", "question": "scoring"}
        )

    def test_missing_key_fails(self) -> None:
        assert not agent_eval.args_match({"game": "asteroids"}, {"question": "scoring"})

    def test_wrong_value_fails(self) -> None:
        assert not agent_eval.args_match({"game": "pacman"}, {"game": "asteroids"})

    def test_year_matches_string_form(self) -> None:
        assert agent_eval.args_match({"year": 1980}, {"year": "1980"})

    def test_year_string_expected_matches_int_actual(self) -> None:
        assert agent_eval.args_match({"year": "1980"}, {"year": 1980})

    def test_year_mismatch_fails(self) -> None:
        assert not agent_eval.args_match({"year": 1980}, {"year": 1976})

    def test_non_numeric_year_fails(self) -> None:
        assert not agent_eval.args_match({"year": 1980}, {"year": "eighties"})

    def test_game_alias_forms_match(self) -> None:
        assert agent_eval.args_match({"game": "missile_command"}, {"game": "Missile Command"})


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class TestScoreCase:
    def test_exact_match_passes(self) -> None:
        case = make_case()
        passed, tool = agent_eval.score_case(
            case,
            [{"name": "search_game_docs", "arguments": {"game": "asteroids", "question": "q"}}],
        )
        assert passed is True
        assert tool == "search_game_docs"

    def test_wrong_tool_fails(self) -> None:
        case = make_case()
        passed, _ = agent_eval.score_case(case, [{"name": "list_games_by_year", "arguments": {}}])
        assert passed is False

    def test_wrong_game_fails(self) -> None:
        case = make_case()
        passed, _ = agent_eval.score_case(
            case, [{"name": "search_game_docs", "arguments": {"game": "pacman"}}]
        )
        assert passed is False

    def test_compare_order_is_interchangeable(self) -> None:
        case = make_case(
            expected_calls=[
                ("compare_games", {"game_a": "asteroids", "game_b": "breakout"}),
                ("compare_games", {"game_a": "breakout", "game_b": "asteroids"}),
            ]
        )
        passed, tool = agent_eval.score_case(
            case,
            [{"name": "compare_games", "arguments": {"game_a": "breakout", "game_b": "asteroids"}}],
        )
        assert passed is True
        assert tool == "compare_games"

    def test_second_call_can_pass(self) -> None:
        case = make_case()
        passed, _ = agent_eval.score_case(
            case,
            [
                {"name": "list_games_by_year", "arguments": {"year": 1980}},
                {"name": "search_game_docs", "arguments": {"game": "asteroids"}},
            ],
        )
        assert passed is True

    def test_no_tools_used_fails(self) -> None:
        passed, tool = agent_eval.score_case(make_case(), [])
        assert passed is False
        assert tool == ""

    def test_non_dict_arguments_fail(self) -> None:
        passed, _ = agent_eval.score_case(
            make_case(), [{"name": "search_game_docs", "arguments": "asteroids"}]
        )
        assert passed is False


# ---------------------------------------------------------------------------
# Per-case + aggregate evaluation
# ---------------------------------------------------------------------------


class TestEvaluateCase:
    def test_records_tools_steps_and_answer(self) -> None:
        case = make_case()
        client = FakeAgentClient(
            responses={
                case.question: {
                    "answer": "20 points [1].",
                    "tools_used": [
                        {"name": "search_game_docs", "arguments": {"game": "asteroids"}}
                    ],
                    "steps": 2,
                    "latency_ms": 5.0,
                }
            }
        )
        result = agent_eval.evaluate_case(case, client, max_steps=3)
        assert result.passed is True
        assert result.matched_tool == "search_game_docs"
        assert result.steps == 2
        assert result.answer == "20 points [1]."
        assert result.latency_ms >= 0
        assert client.calls == [(case.question, 3)]

    def test_client_error_recorded_not_raised(self) -> None:
        result = agent_eval.evaluate_case(make_case(), FakeAgentClient(error="down"))
        assert result.error == "down"
        assert result.passed is False

    def test_called_tools_deduplicated_in_order(self) -> None:
        case = make_case(
            expected_calls=[
                ("search_game_docs", {"game": "asteroids"}),
                ("compare_games", {"game_a": "asteroids", "game_b": "breakout"}),
            ]
        )
        client = FakeAgentClient(
            responses={
                case.question: {
                    "answer": "…",
                    "tools_used": [
                        {"name": "list_games_by_year", "arguments": {"year": 1980}},
                        {"name": "search_game_docs", "arguments": {"game": "asteroids"}},
                        {"name": "search_game_docs", "arguments": {"game": "breakout"}},
                    ],
                    "steps": 3,
                }
            }
        )
        result = agent_eval.evaluate_case(case, client)
        assert result.called_tools == ["list_games_by_year", "search_game_docs"]
        assert result.passed is True
        assert result.matched_tool == "search_game_docs"


class TestAggregate:
    def test_pass_rate_and_per_tool(self) -> None:
        search_case = make_case("c-1")
        list_case = make_case(
            "c-2",
            question="What came out in 1980?",
            expected_calls=[("list_games_by_year", {"year": 1980})],
        )
        results = [
            make_result(
                search_case,
                [{"name": "search_game_docs", "arguments": {"game": "asteroids"}}],
            ),
            make_result(list_case, [{"name": "list_games_by_year", "arguments": {"year": 1999}}]),
        ]
        metrics = agent_eval.aggregate(results)
        assert metrics["pass_rate"] == 0.5
        assert metrics["counts"] == {"total": 2, "passed": 1, "failed": 1, "errors": 0}
        assert metrics["per_tool"]["search_game_docs"] == {"n": 1, "passed": 1}
        assert metrics["per_tool"]["list_games_by_year"] == {"n": 1, "passed": 0}
        assert metrics["steps"]["mean"] == 2.0
        assert metrics["latency_ms"]["mean"] == 10.0

    def test_errors_excluded_from_metrics(self) -> None:
        case = make_case()
        results = [
            make_result(case, [], error="down"),
            make_result(case, [{"name": "search_game_docs", "arguments": {"game": "asteroids"}}]),
        ]
        metrics = agent_eval.aggregate(results)
        assert metrics["pass_rate"] == 0.5
        assert metrics["counts"]["errors"] == 1
        assert metrics["per_tool"]["search_game_docs"]["n"] == 1

    def test_empty_results(self) -> None:
        metrics = agent_eval.aggregate([])
        assert metrics["pass_rate"] is None
        assert metrics["counts"]["total"] == 0


class TestPassRateGuard:
    def test_below_minimum_fails(self) -> None:
        failures = agent_eval.check_pass_rate_guard({"pass_rate": 0.7}, 0.8)
        assert len(failures) == 1
        assert "0.700" in failures[0]

    def test_at_minimum_passes(self) -> None:
        assert agent_eval.check_pass_rate_guard({"pass_rate": 0.8}, 0.8) == []

    def test_none_values_skip(self) -> None:
        assert agent_eval.check_pass_rate_guard({"pass_rate": None}, 0.8) == []
        assert agent_eval.check_pass_rate_guard({"pass_rate": 0.1}, None) == []


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


class TestBuildMarkdownReport:
    def test_contains_summary_and_case_rows(self) -> None:
        case = make_case("c-1")
        results = [
            make_result(case, [{"name": "search_game_docs", "arguments": {"game": "asteroids"}}])
        ]
        metrics = agent_eval.aggregate(results)
        markdown = agent_eval.build_markdown_report(
            "agent-routing",
            metrics,
            results,
            generated_at="2026-01-01 00:00:00 UTC",
            api_url="http://service:8000/agent",
            cases_path=agent_eval.DEFAULT_CASES,
        )
        assert "Routing pass rate" in markdown
        assert "100.0%" in markdown
        assert "| c-1 |" in markdown
        assert "`search_game_docs`" in markdown

    def test_failures_and_errors_sections(self) -> None:
        failing = make_case("c-1")
        errored = make_case("c-2")
        results = [
            make_result(failing, [{"name": "compare_games", "arguments": {}}]),
            make_result(errored, [], error="boom"),
        ]
        metrics = agent_eval.aggregate(results)
        markdown = agent_eval.build_markdown_report(
            "agent-routing",
            metrics,
            results,
            generated_at="2026-01-01 00:00:00 UTC",
            api_url="http://service:8000/agent",
            cases_path=agent_eval.DEFAULT_CASES,
        )
        assert "(failed)" in markdown
        assert "(error) — boom" in markdown


class TestWriteReports:
    def test_writes_json_and_markdown(self, tmp_path: Path) -> None:
        json_path, md_path = agent_eval.write_reports(
            tmp_path, "agent-routing", {"metrics": {}}, "# report\n"
        )
        assert json_path.exists()
        assert md_path.exists()
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["metrics"] == {}

    def test_collision_gets_time_suffix(self, tmp_path: Path) -> None:
        agent_eval.write_reports(tmp_path, "agent-routing", {}, "# one\n")
        second_json, second_md = agent_eval.write_reports(tmp_path, "agent-routing", {}, "# two\n")
        first_json = next(tmp_path.glob("*-agent-routing.json"))
        assert second_json != first_json
        assert second_md.exists()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestMain:
    def test_exit_zero_when_guard_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cases = agent_eval.load_cases(agent_eval.DEFAULT_CASES)
        client = FakeAgentClient(responses=passing_responses(cases))
        monkeypatch.setattr(agent_eval, "AgentClient", lambda **kwargs: client)

        exit_code = agent_eval.main(
            ["--quiet", "--results-dir", str(tmp_path), "--label", "agent-test"]
        )

        assert exit_code == 0
        assert (tmp_path / "agent-test.json").exists() or list(tmp_path.glob("*-agent-test.json"))
        report = json.loads(next(tmp_path.glob("*-agent-test.json")).read_text(encoding="utf-8"))
        assert report["metrics"]["counts"]["passed"] == len(cases)

    def test_exit_one_when_guard_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = FakeAgentClient()  # default response calls no tools → all cases fail
        monkeypatch.setattr(agent_eval, "AgentClient", lambda **kwargs: client)

        exit_code = agent_eval.main(
            ["--quiet", "--results-dir", str(tmp_path), "--label", "agent-fail"]
        )

        assert exit_code == 1
        assert next(tmp_path.glob("*-agent-fail.json"))

    def test_exit_zero_when_guard_disabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = FakeAgentClient()
        monkeypatch.setattr(agent_eval, "AgentClient", lambda **kwargs: client)

        exit_code = agent_eval.main(
            [
                "--quiet",
                "--min-pass-rate",
                "0",
                "--results-dir",
                str(tmp_path),
                "--label",
                "agent-noguard",
            ]
        )

        assert exit_code == 0

    def test_exit_two_when_service_down(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = FakeAgentClient(error="connection refused")
        monkeypatch.setattr(agent_eval, "AgentClient", lambda **kwargs: client)

        exit_code = agent_eval.main(
            ["--quiet", "--results-dir", str(tmp_path), "--label", "agent-down"]
        )

        assert exit_code == 2
        assert not list(tmp_path.glob("*.json"))

    def test_exit_two_when_cases_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(agent_eval, "AgentClient", lambda **kwargs: FakeAgentClient())
        exit_code = agent_eval.main(
            ["--quiet", "--cases", str(tmp_path / "nope.jsonl"), "--results-dir", str(tmp_path)]
        )
        assert exit_code == 2

    def test_limit_and_max_steps_forwarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cases = agent_eval.load_cases(agent_eval.DEFAULT_CASES)
        client = FakeAgentClient(responses=passing_responses(cases))
        monkeypatch.setattr(agent_eval, "AgentClient", lambda **kwargs: client)

        exit_code = agent_eval.main(
            [
                "--quiet",
                "--limit",
                "2",
                "--max-steps",
                "4",
                "--min-pass-rate",
                "0",
                "--results-dir",
                str(tmp_path),
                "--label",
                "agent-limit",
            ]
        )

        assert exit_code == 0
        assert len(client.calls) == 2
        assert all(max_steps == 4 for _, max_steps in client.calls)
