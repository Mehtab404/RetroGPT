"""Tests for native tool-calling agent mode (Issue #35 / backlog #30).

Covers:

- GAME_CATALOG: years match the ``Release Date`` lines of the manuals
  (skipped automatically when the corpus is not mounted, e.g. in-container)
- GAME_SLUGS / slug aliases
- ToolRegistry: specs, dispatch, argument validation, evidence citation
  numbering, duplicate-evidence collapse, identical-call caching
- run_agent loop: tool-call round trip, neutral transcript shape, answer
  passthrough, step-budget fallback (grounded and no-evidence paths),
  empty-turn error
- GeminiLLM.generate_with_tools / OpenAILLM.generate_with_tools: payload
  translation and response parsing (httpx mocked)
- LLMProvider.generate_with_tools default: raises LLMError
- POST /agent: happy path, tools_used/sources/steps echo, max_steps
  override + exhausted-budget fallback, LLM error → 502, validation errors

All external HTTP calls and service dependencies are mocked via pytest-mock /
unittest.mock.  asyncio tests use pytest-asyncio.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from app.agent.loop import NO_CONTEXT_ANSWER, run_agent
from app.agent.tools import GAME_CATALOG, GAME_SLUGS, ToolRegistry
from app.generation.llm import (
    AgentTurn,
    GeminiLLM,
    GroqLLM,
    LLMError,
    LLMProvider,
    OpenAILLM,
    ToolCall,
)
from app.generation.prompts import AGENT_SYSTEM_PROMPT
from app.main import app
from app.retrieval.vector_store import SearchResult
from app.schemas import AGENT_MAX_STEPS_LIMIT, AgentRequest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_httpx_response(json_data: dict, status_code: int = 200) -> MagicMock:
    """Return a MagicMock shaped like an httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.text = str(json_data)
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status_code}",
            request=MagicMock(),
            response=resp,
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


def _make_search_result(
    chunk_id: str = "chunk_1",
    score: float = 0.9,
    content: str = "Sample content",
    title: str = "Test Manual",
    source: str = "/data/raw/asteroids/asteroids_manual.md",
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        score=score,
        payload={
            "chunk_id": chunk_id,
            "title": title,
            "source": source,
            "content": content,
        },
        content=content,
    )


class _SearchStub:
    """Deterministic async retrieval callable recording its invocations."""

    def __init__(self, results_by_game: dict[str | None, list[SearchResult]]) -> None:
        self.results_by_game = results_by_game
        self.calls: list[tuple[str, int, str | None]] = []

    async def __call__(
        self,
        question: str,
        top_k: int,
        game_slug: str | None,
    ) -> list[SearchResult]:
        self.calls.append((question, top_k, game_slug))
        return list(self.results_by_game.get(game_slug, []))


def _asteroids_results() -> list[SearchResult]:
    return [
        _make_search_result(chunk_id="asteroids_p1_c1", content="Large asteroids: 20 points."),
        _make_search_result(chunk_id="asteroids_p2_c1", content="Small asteroids: 100 points."),
        _make_search_result(chunk_id="asteroids_p3_c1", content="Saucers: 200 points."),
    ]


class ScriptedLLM(LLMProvider):
    """Fake provider returning scripted tool-calling turns (offline testing)."""

    def __init__(
        self,
        turns: list[AgentTurn],
        fallback: str = "Grounded fallback [1].",
    ) -> None:
        super().__init__(model="scripted")
        self._turns = list(turns)
        self._fallback = fallback
        self.transcripts: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
        self.generate_prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.generate_prompts.append(prompt)
        return self._fallback

    async def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentTurn:
        self.transcripts.append((json.loads(json.dumps(messages)), tools))
        if not self._turns:
            raise AssertionError("ScriptedLLM received an unexpected extra turn")
        return self._turns.pop(0)


@pytest.fixture(autouse=True)
def _deterministic_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force retrieval onto the deterministic dense-only, no-rerank path."""
    monkeypatch.setattr("app.retrieval.hybrid.get_bm25_encoder", lambda path=None: None)
    monkeypatch.setattr("app.retrieval.rerank.get_reranker", lambda: None)


# ---------------------------------------------------------------------------
# Game catalog (corpus-backed)
# ---------------------------------------------------------------------------

_CORPUS_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"


class TestGameCatalog:
    @pytest.mark.parametrize("entry", GAME_CATALOG, ids=lambda e: str(e["slug"]))
    def test_year_matches_manual_release_date(self, entry: dict[str, Any]) -> None:
        manual_dir = _CORPUS_RAW / str(entry["slug"])
        if not manual_dir.is_dir():
            pytest.skip("corpus data/raw not mounted in this environment")
        manuals = sorted(manual_dir.glob("*.md"))
        assert manuals, f"no manual found for {entry['slug']}"
        release_lines = [
            line
            for line in manuals[0].read_text(encoding="utf-8").splitlines()
            if "release date" in line.lower()
        ]
        assert release_lines, f"no Release Date line in {manuals[0].name}"
        assert str(entry["year"]) in release_lines[0]

    def test_slugs_are_unique_and_lowercase(self) -> None:
        assert len(set(GAME_SLUGS)) == len(GAME_SLUGS)
        assert all(slug == slug.lower() for slug in GAME_SLUGS)


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def test_specs_expose_three_tools(self) -> None:
        registry = ToolRegistry(search=_SearchStub({}))
        specs = registry.specs()
        assert [spec["name"] for spec in specs] == [
            "search_game_docs",
            "compare_games",
            "list_games_by_year",
        ]
        search_spec = specs[0]
        assert search_spec["parameters"]["properties"]["game"]["enum"] == list(GAME_SLUGS)
        assert search_spec["parameters"]["required"] == ["game", "question"]

    @pytest.mark.asyncio
    async def test_search_game_docs_returns_numbered_evidence(self) -> None:
        search = _SearchStub({"asteroids": _asteroids_results()})
        registry = ToolRegistry(search=search)

        payload = await registry.execute(
            "search_game_docs",
            {"game": "asteroids", "question": "How do you score?"},
        )

        assert search.calls == [("How do you score?", 3, "asteroids")]
        citations = [item["citation"] for item in payload["evidence"]]
        assert citations == [1, 2, 3]
        assert payload["evidence"][0]["content"] == "Large asteroids: 20 points."
        assert registry.evidence[0].chunk_id == "asteroids_p1_c1"

    @pytest.mark.asyncio
    async def test_citation_numbers_stay_stable_across_calls(self) -> None:
        search = _SearchStub({"asteroids": _asteroids_results()})
        registry = ToolRegistry(search=search)

        first = await registry.execute(
            "search_game_docs", {"game": "asteroids", "question": "scoring"}
        )
        second = await registry.execute(
            "search_game_docs", {"game": "asteroids", "question": "scoring"}
        )

        # Identical call is cached — no new retrieval, same payload.
        assert len(search.calls) == 1
        assert first == second

    @pytest.mark.asyncio
    async def test_duplicate_evidence_keeps_first_citation(self) -> None:
        shared = _make_search_result(chunk_id="asteroids_p1_c1")
        search = _SearchStub({"asteroids": [shared], "pacman": [shared]})
        registry = ToolRegistry(search=search)

        await registry.execute("search_game_docs", {"game": "asteroids", "question": "a"})
        pacman_payload = await registry.execute(
            "compare_games", {"game_a": "pacman", "game_b": "asteroids"}
        )

        assert pacman_payload["games"]["pacman"]["evidence"][0]["citation"] == 1
        assert len(registry.evidence) == 1

    @pytest.mark.asyncio
    async def test_compare_games_searches_both_games(self) -> None:
        search = _SearchStub(
            {
                "pacman": [_make_search_result(chunk_id="pacman_p1_c1")],
                "centipede": [_make_search_result(chunk_id="centipede_p1_c1")],
            }
        )
        registry = ToolRegistry(search=search)

        payload = await registry.execute(
            "compare_games", {"game_a": "pacman", "game_b": "centipede"}
        )

        assert sorted(payload["games"]) == ["centipede", "pacman"]
        assert [call[2] for call in search.calls] == ["pacman", "centipede"]
        assert payload["games"]["pacman"]["title"] == "Pac-Man"

    @pytest.mark.asyncio
    async def test_list_games_by_year(self) -> None:
        registry = ToolRegistry(search=_SearchStub({}))

        payload = await registry.execute("list_games_by_year", {"year": 1980})

        assert [game["slug"] for game in payload["games"]] == ["missile_command", "pacman"]

    @pytest.mark.asyncio
    async def test_list_games_by_year_unknown_year_returns_hint(self) -> None:
        registry = ToolRegistry(search=_SearchStub({}))

        payload = await registry.execute("list_games_by_year", {"year": 1999})

        assert payload["games"] == []
        assert "no knowledge-base games" in payload["note"]
        assert payload["catalog_years"] == [1976, 1978, 1979, 1980, 1981]

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_payload(self) -> None:
        registry = ToolRegistry(search=_SearchStub({}))

        payload = await registry.execute("teleport", {})

        assert "unknown tool" in payload["error"]
        assert "search_game_docs" in payload["available_tools"]

    @pytest.mark.asyncio
    async def test_unknown_game_returns_error_payload(self) -> None:
        registry = ToolRegistry(search=_SearchStub({}))

        payload = await registry.execute(
            "search_game_docs", {"game": "tetris", "question": "scoring"}
        )

        assert "invalid arguments" in payload["error"]
        assert "tetris" in payload["error"]

    @pytest.mark.asyncio
    async def test_alias_slugs_are_resolved(self) -> None:
        search = _SearchStub({"pacman": [_make_search_result(chunk_id="pacman_p1_c1")]})
        registry = ToolRegistry(search=search)

        payload = await registry.execute(
            "search_game_docs", {"game": "Pac-Man", "question": "ghosts"}
        )

        assert payload["game"] == "pacman"
        assert search.calls[0][2] == "pacman"

    @pytest.mark.asyncio
    async def test_compare_same_game_returns_error(self) -> None:
        registry = ToolRegistry(search=_SearchStub({}))

        payload = await registry.execute("compare_games", {"game_a": "pacman", "game_b": "pacman"})

        assert "different games" in payload["error"]

    @pytest.mark.asyncio
    async def test_empty_question_returns_error(self) -> None:
        registry = ToolRegistry(search=_SearchStub({}))

        payload = await registry.execute("search_game_docs", {"game": "pacman", "question": "   "})

        assert "non-empty" in payload["error"]

    @pytest.mark.asyncio
    async def test_no_results_includes_note(self) -> None:
        registry = ToolRegistry(search=_SearchStub({}))

        payload = await registry.execute(
            "search_game_docs", {"game": "pacman", "question": "anything"}
        )

        assert payload["evidence"] == []
        assert "no passages matched" in payload["note"]

    @pytest.mark.asyncio
    async def test_calls_echoed_in_order(self) -> None:
        search = _SearchStub({None: _asteroids_results()})
        registry = ToolRegistry(search=search)

        await registry.execute("list_games_by_year", {"year": 1980})
        await registry.execute("search_game_docs", {"game": "asteroids", "question": "a"})

        assert [call["name"] for call in registry.calls] == [
            "list_games_by_year",
            "search_game_docs",
        ]


# ---------------------------------------------------------------------------
# run_agent loop
# ---------------------------------------------------------------------------


class TestRunAgentLoop:
    @pytest.mark.asyncio
    async def test_tool_round_trip_returns_answer(self) -> None:
        search = _SearchStub({"asteroids": _asteroids_results()})
        registry = ToolRegistry(search=search)
        llm = ScriptedLLM(
            turns=[
                AgentTurn(
                    text=None,
                    tool_calls=[
                        ToolCall(
                            name="search_game_docs",
                            arguments={"game": "asteroids", "question": "scoring"},
                            call_id="call_1",
                        )
                    ],
                ),
                AgentTurn(text="Large asteroids are worth 20 points [1].", tool_calls=[]),
            ]
        )

        result = await run_agent(llm, "How do you score in Asteroids?", registry, max_steps=3)

        assert result.answer == "Large asteroids are worth 20 points [1]."
        assert result.steps == 2
        assert [source.chunk_id for source in result.sources][0] == "asteroids_p1_c1"
        assert result.tools_used == [
            {"name": "search_game_docs", "arguments": {"game": "asteroids", "question": "scoring"}}
        ]

    @pytest.mark.asyncio
    async def test_transcript_uses_neutral_tool_protocol(self) -> None:
        search = _SearchStub({"asteroids": _asteroids_results()})
        registry = ToolRegistry(search=search)
        llm = ScriptedLLM(
            turns=[
                AgentTurn(
                    text=None,
                    tool_calls=[
                        ToolCall(
                            name="search_game_docs",
                            arguments={"game": "asteroids", "question": "scoring"},
                            call_id="call_1",
                        )
                    ],
                ),
                AgentTurn(text="Answer [1].", tool_calls=[]),
            ]
        )

        await run_agent(llm, "question", registry, max_steps=3)

        messages, tools = llm.transcripts[1]
        assert messages[0] == {"role": "system", "content": AGENT_SYSTEM_PROMPT}
        assert messages[1] == {"role": "user", "content": "question"}
        assert messages[2]["role"] == "assistant"
        assert messages[2]["tool_calls"][0]["name"] == "search_game_docs"
        assert messages[2]["tool_calls"][0]["call_id"] == "call_1"
        assert messages[3]["role"] == "tool"
        assert messages[3]["name"] == "search_game_docs"
        assert messages[3]["tool_call_id"] == "call_1"
        payload = json.loads(messages[3]["content"])
        assert payload["evidence"][0]["citation"] == 1
        assert [tool["name"] for tool in tools] == [
            "search_game_docs",
            "compare_games",
            "list_games_by_year",
        ]

    @pytest.mark.asyncio
    async def test_budget_exhausted_falls_back_to_grounded_rag(self) -> None:
        search = _SearchStub({"asteroids": _asteroids_results()})
        registry = ToolRegistry(search=search)
        tool_turn = AgentTurn(
            text=None,
            tool_calls=[
                ToolCall(
                    name="search_game_docs",
                    arguments={"game": "asteroids", "question": "scoring"},
                )
            ],
        )
        llm = ScriptedLLM(turns=[tool_turn], fallback="Grounded fallback [1].")

        result = await run_agent(llm, "How do you score?", registry, max_steps=1)

        assert result.answer == "Grounded fallback [1]."
        assert result.steps == 1
        assert len(result.sources) == 3
        assert len(llm.generate_prompts) == 1
        assert "Large asteroids: 20 points." in llm.generate_prompts[0]

    @pytest.mark.asyncio
    async def test_budget_exhausted_without_evidence_refuses(self) -> None:
        registry = ToolRegistry(search=_SearchStub({}))
        llm = ScriptedLLM(
            turns=[
                AgentTurn(
                    text=None,
                    tool_calls=[ToolCall(name="list_games_by_year", arguments={"year": 1980})],
                )
            ]
        )

        result = await run_agent(llm, "What came out in 1980?", registry, max_steps=1)

        assert result.answer == NO_CONTEXT_ANSWER
        assert result.sources == []
        assert llm.generate_prompts == []

    @pytest.mark.asyncio
    async def test_empty_turn_raises_llm_error(self) -> None:
        registry = ToolRegistry(search=_SearchStub({}))
        llm = ScriptedLLM(turns=[AgentTurn(text=None, tool_calls=[])])

        with pytest.raises(LLMError, match="neither text nor tool calls"):
            await run_agent(llm, "question", registry, max_steps=3)


# ---------------------------------------------------------------------------
# Provider tool-calling translations
# ---------------------------------------------------------------------------


class TestBaseProviderToolCalling:
    @pytest.mark.asyncio
    async def test_default_implementation_raises(self) -> None:
        class Minimal(LLMProvider):
            async def generate(self, prompt: str) -> str:
                return "ok"

        with pytest.raises(LLMError, match="does not support tool calling"):
            await Minimal(model="minimal").generate_with_tools([], [])


class TestGeminiToolCalling:
    @pytest.mark.asyncio
    async def test_parses_text_and_function_call(self) -> None:
        provider = GeminiLLM(model="gemini-1.5-flash", api_key="test-key", jitter=False)
        resp = _mock_httpx_response(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "Let me look that up."},
                                {
                                    "functionCall": {
                                        "name": "search_game_docs",
                                        "args": {"game": "asteroids", "question": "scoring"},
                                    }
                                },
                            ]
                        }
                    }
                ]
            }
        )

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=resp)
            mock_client_cls.return_value = mock_client

            turn = await provider.generate_with_tools(
                [{"role": "user", "content": "How do you score?"}], provider_tools()
            )

        assert turn.text == "Let me look that up."
        assert len(turn.tool_calls) == 1
        assert turn.tool_calls[0].name == "search_game_docs"
        assert turn.tool_calls[0].arguments == {"game": "asteroids", "question": "scoring"}
        assert turn.tool_calls[0].call_id is None

    @pytest.mark.asyncio
    async def test_payload_translates_transcript_and_tools(self) -> None:
        provider = GeminiLLM(model="gemini-1.5-flash", api_key="key", jitter=False)
        resp = _mock_httpx_response({"candidates": [{"content": {"parts": [{"text": "done"}]}}]})
        messages = [
            {"role": "system", "content": "system rules"},
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"name": "search_game_docs", "arguments": {"game": "pacman"}, "call_id": None}
                ],
            },
            {
                "role": "tool",
                "name": "search_game_docs",
                "tool_call_id": None,
                "content": '{"evidence": [{"citation": 1}]}',
            },
        ]

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=resp)
            mock_client_cls.return_value = mock_client

            await provider.generate_with_tools(messages, provider_tools())

        payload = mock_client.post.call_args.kwargs["json"]
        assert payload["systemInstruction"] == {"parts": [{"text": "system rules"}]}
        assert payload["tools"][0]["functionDeclarations"][0]["name"] == "search_game_docs"
        contents = payload["contents"]
        assert contents[0] == {"role": "user", "parts": [{"text": "question"}]}
        assert contents[1]["role"] == "model"
        assert contents[1]["parts"][0]["functionCall"]["name"] == "search_game_docs"
        assert contents[2]["role"] == "user"
        function_response = contents[2]["parts"][0]["functionResponse"]
        assert function_response["name"] == "search_game_docs"
        assert function_response["response"] == {"evidence": [{"citation": 1}]}


class TestOpenAIToolCalling:
    @pytest.mark.asyncio
    async def test_parses_tool_calls_with_ids(self) -> None:
        provider = OpenAILLM(model="gpt-4o-mini", api_key="test-key", jitter=False)
        resp = _mock_httpx_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_abc",
                                    "type": "function",
                                    "function": {
                                        "name": "compare_games",
                                        "arguments": '{"game_a": "pacman", "game_b": "centipede"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        )

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=resp)
            mock_client_cls.return_value = mock_client

            turn = await provider.generate_with_tools(
                [{"role": "user", "content": "Compare them"}], provider_tools()
            )

        assert turn.text is None
        assert len(turn.tool_calls) == 1
        assert turn.tool_calls[0].name == "compare_games"
        assert turn.tool_calls[0].arguments == {"game_a": "pacman", "game_b": "centipede"}
        assert turn.tool_calls[0].call_id == "call_abc"

    @pytest.mark.asyncio
    async def test_payload_serializes_neutral_transcript(self) -> None:
        provider = OpenAILLM(model="gpt-4o-mini", api_key="key", jitter=False)
        resp = _mock_httpx_response(
            {"choices": [{"message": {"role": "assistant", "content": "done"}}]}
        )
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "name": "search_game_docs",
                        "arguments": {"game": "pacman"},
                        "call_id": "call_1",
                    }
                ],
            },
            {"role": "tool", "name": "search_game_docs", "tool_call_id": "call_1", "content": "{}"},
        ]

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=resp)
            mock_client_cls.return_value = mock_client

            await provider.generate_with_tools(messages, provider_tools())

        payload = mock_client.post.call_args.kwargs["json"]
        assert payload["tools"][0]["function"]["name"] == "search_game_docs"
        assistant = payload["messages"][0]
        assert assistant["tool_calls"][0]["id"] == "call_1"
        assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"game": "pacman"}
        assert payload["messages"][1] == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "{}",
        }


class TestGroqToolCalling:
    @pytest.mark.asyncio
    async def test_parses_tool_calls_with_ids(self) -> None:
        provider = GroqLLM(model="openai/gpt-oss-120b", api_key="test-key", jitter=False)
        resp = _mock_httpx_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_abc",
                                    "type": "function",
                                    "function": {
                                        "name": "compare_games",
                                        "arguments": '{"game_a": "pacman", "game_b": "centipede"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        )

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=resp)
            mock_client_cls.return_value = mock_client

            turn = await provider.generate_with_tools(
                [{"role": "user", "content": "Compare them"}], provider_tools()
            )

        assert turn.text is None
        assert len(turn.tool_calls) == 1
        assert turn.tool_calls[0].name == "compare_games"
        assert turn.tool_calls[0].arguments == {"game_a": "pacman", "game_b": "centipede"}
        assert turn.tool_calls[0].call_id == "call_abc"

    @pytest.mark.asyncio
    async def test_payload_omits_message_name_and_uses_max_completion_tokens(
        self,
    ) -> None:
        """Groq rejects messages[].name and requires max_completion_tokens."""
        provider = GroqLLM(model="openai/gpt-oss-120b", api_key="key", jitter=False)
        resp = _mock_httpx_response(
            {"choices": [{"message": {"role": "assistant", "content": "done"}}]}
        )
        messages = [
            {"role": "system", "content": "system rules"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "name": "search_game_docs",
                        "arguments": {"game": "pacman"},
                        "call_id": "call_1",
                    }
                ],
            },
            {"role": "tool", "name": "search_game_docs", "tool_call_id": "call_1", "content": "{}"},
        ]

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=resp)
            mock_client_cls.return_value = mock_client

            await provider.generate_with_tools(messages, provider_tools())

        payload = mock_client.post.call_args.kwargs["json"]
        # Groq returns 400 if any message carries the optional `name` field.
        assert all("name" not in message for message in payload["messages"])
        assert payload["max_completion_tokens"] == 1024
        assert "max_tokens" not in payload
        assert payload["tools"][0]["function"]["name"] == "search_game_docs"
        assert payload["messages"][-1] == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "{}",
        }


def provider_tools() -> list[dict[str, Any]]:
    """Minimal tool spec list for provider translation tests."""
    return [
        {
            "name": "search_game_docs",
            "description": "Search one game.",
            "parameters": {
                "type": "object",
                "properties": {"game": {"type": "string"}, "question": {"type": "string"}},
                "required": ["game", "question"],
            },
        }
    ]


# ---------------------------------------------------------------------------
# POST /agent endpoint
# ---------------------------------------------------------------------------


client = TestClient(app)


class TestAgentRequestSchema:
    def test_defaults(self) -> None:
        request = AgentRequest(question="Compare Asteroids and Breakout.")
        assert request.max_steps is None

    def test_max_steps_bounds(self) -> None:
        assert AgentRequest(question="q", max_steps=1).max_steps == 1
        assert (
            AgentRequest(question="q", max_steps=AGENT_MAX_STEPS_LIMIT).max_steps
            == AGENT_MAX_STEPS_LIMIT
        )
        with pytest.raises(ValueError):
            AgentRequest(question="q", max_steps=0)
        with pytest.raises(ValueError):
            AgentRequest(question="q", max_steps=AGENT_MAX_STEPS_LIMIT + 1)

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValueError):
            AgentRequest(question="q", surprise="field")


class TestAgentEndpoint:
    def test_happy_path_returns_answer_sources_and_tools(self) -> None:
        search_result = _make_search_result(
            chunk_id="asteroids_p1_c1",
            score=0.91,
            content="Large asteroids: 20 points each.",
            title="Asteroids Manual",
        )
        scripted = ScriptedLLM(
            turns=[
                AgentTurn(
                    text=None,
                    tool_calls=[
                        ToolCall(
                            name="search_game_docs",
                            arguments={"game": "asteroids", "question": "scoring"},
                            call_id="call_1",
                        )
                    ],
                ),
                AgentTurn(text="Large asteroids are worth 20 points [1].", tool_calls=[]),
            ]
        )

        with (
            patch("app.main.get_embedding_provider") as mock_embed_factory,
            patch("app.main.QdrantVectorStore") as mock_vs_cls,
            patch("app.main.get_llm_provider") as mock_llm_factory,
        ):
            mock_embedder = AsyncMock()
            mock_embedder.embed_query = AsyncMock(return_value=[0.1] * 768)
            mock_embed_factory.return_value = mock_embedder

            mock_vs = AsyncMock()
            mock_vs.search = AsyncMock(return_value=[search_result])
            mock_vs_cls.return_value = mock_vs

            mock_llm_factory.return_value = scripted

            response = client.post(
                "/agent",
                json={"question": "How do you score in Asteroids?"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Large asteroids are worth 20 points [1]."
        assert data["steps"] == 2
        assert data["cached"] is False
        assert data["latency_ms"] >= 0
        assert len(data["sources"]) == 1
        assert data["sources"][0]["id"] == "asteroids_p1_c1"
        assert data["tools_used"] == [
            {
                "name": "search_game_docs",
                "arguments": {"game": "asteroids", "question": "scoring"},
            }
        ]

    def test_answer_normalizes_full_width_citations(self) -> None:
        scripted = ScriptedLLM(
            turns=[
                AgentTurn(
                    text="Large asteroids are worth 20 points 【1】.",
                    tool_calls=[],
                )
            ]
        )

        with patch("app.main.get_llm_provider") as mock_llm_factory:
            mock_llm_factory.return_value = scripted

            response = client.post(
                "/agent",
                json={"question": "How do you score in Asteroids?"},
            )

        assert response.status_code == 200
        assert response.json()["answer"] == "Large asteroids are worth 20 points [1]."

    def test_request_max_steps_is_honoured(self) -> None:
        search_result = _make_search_result()
        scripted = ScriptedLLM(
            turns=[
                AgentTurn(
                    text=None,
                    tool_calls=[
                        ToolCall(
                            name="search_game_docs",
                            arguments={"game": "asteroids", "question": "scoring"},
                        )
                    ],
                )
            ]
        )

        with (
            patch("app.main.get_embedding_provider") as mock_embed_factory,
            patch("app.main.QdrantVectorStore") as mock_vs_cls,
            patch("app.main.get_llm_provider") as mock_llm_factory,
        ):
            mock_embedder = AsyncMock()
            mock_embedder.embed_query = AsyncMock(return_value=[0.1] * 768)
            mock_embed_factory.return_value = mock_embedder

            mock_vs = AsyncMock()
            mock_vs.search = AsyncMock(return_value=[search_result])
            mock_vs_cls.return_value = mock_vs

            mock_llm_factory.return_value = scripted

            response = client.post(
                "/agent",
                json={"question": "How do you score in Asteroids?", "max_steps": 1},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["steps"] == 1
        assert data["answer"] == "Grounded fallback [1]."
        assert len(scripted.generate_prompts) == 1

    def test_llm_failure_maps_to_502(self) -> None:
        with (
            patch("app.main.get_embedding_provider") as mock_embed_factory,
            patch("app.main.QdrantVectorStore") as mock_vs_cls,
            patch("app.main.get_llm_provider") as mock_llm_factory,
        ):
            mock_embedder = AsyncMock()
            mock_embedder.embed_query = AsyncMock(return_value=[0.1] * 768)
            mock_embed_factory.return_value = mock_embedder

            mock_vs = AsyncMock()
            mock_vs.search = AsyncMock(return_value=[])
            mock_vs_cls.return_value = mock_vs

            mock_llm = AsyncMock()
            mock_llm.generate_with_tools = AsyncMock(side_effect=LLMError("provider down"))
            mock_llm_factory.return_value = mock_llm

            response = client.post("/agent", json={"question": "anything"})

        assert response.status_code == 502
        assert "LLM provider error" in response.json()["detail"]

    def test_validation_errors(self) -> None:
        assert client.post("/agent", json={}).status_code == 422
        assert client.post("/agent", json={"question": "q", "max_steps": 0}).status_code == 422
        assert client.post("/agent", json={"question": "q", "max_steps": 99}).status_code == 422
        assert client.post("/agent", json={"question": "q", "extra": 1}).status_code == 422
