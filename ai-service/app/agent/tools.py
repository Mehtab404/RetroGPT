"""Agent tools for native tool-calling mode (Issue #35 / backlog #30).

Exposes a small registry of deterministic tools that the agent loop executes:

* :meth:`ToolRegistry.execute` dispatches by name, validates arguments, and
  turns recoverable argument errors (unknown game, malformed year) into
  ``{"error": ...}`` payloads the model can react to.  Retrieval/provider
  failures propagate to the caller and map to HTTP 502/503, exactly like the
  standard ``/query`` path.
* Tool payloads embed evidence passages with stable citation numbers so the
  final answer keeps the same ``[N]`` contract as the RAG path.
* Identical ``(name, arguments)`` calls are cached and duplicate evidence is
  registered once (first-seen order), keeping the transcript bounded.

The year catalog is sourced from the ``Release Date`` fields of the manuals
in ``data/raw/`` (checked by a corpus-audit test in ``tests/test_agent.py``);
it is static so the tool is deterministic and offline-testable.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from app.retrieval.vector_store import SearchResult

#: Signature of the retrieval callable injected by the service layer.
SearchFn = Callable[[str, int, str | None], Awaitable[list[SearchResult]]]

#: Game catalog for the knowledge base (slug, display title, release year).
GAME_CATALOG: tuple[dict[str, Any], ...] = (
    {"slug": "asteroids", "title": "Asteroids", "year": 1979},
    {"slug": "breakout", "title": "Breakout", "year": 1976},
    {"slug": "centipede", "title": "Centipede", "year": 1981},
    {"slug": "missile_command", "title": "Missile Command", "year": 1980},
    {"slug": "pacman", "title": "Pac-Man", "year": 1980},
    {"slug": "space_invaders", "title": "Space Invaders", "year": 1978},
)

#: Slugs accepted by game-addressed tools (also exposed as JSON-schema enums).
GAME_SLUGS: tuple[str, ...] = tuple(str(item["slug"]) for item in GAME_CATALOG)

_TITLES: dict[str, str] = {str(item["slug"]): str(item["title"]) for item in GAME_CATALOG}

#: Lenient slug aliases the model may emit (e.g. "Pac-Man").
_ALIASES: dict[str, str] = {
    "pac": "pacman",
    "pac_man": "pacman",
    "missilecommand": "missile_command",
    "spaceinvaders": "space_invaders",
}


class ToolRegistry:
    """Executes agent tools and tracks evidence + citation numbering.

    Args:
        search:       Async retrieval callable ``(question, top_k,
                      game_slug) -> list[SearchResult]`` (bound by the
                      service layer to the real hybrid-retrieval pipeline).
        max_evidence: Maximum evidence passages registered per tool call.
        top_k:        Retrieval depth requested per search.
    """

    def __init__(
        self,
        search: SearchFn,
        max_evidence: int = 5,
        top_k: int = 3,
    ) -> None:
        self._search = search
        self._max_evidence = max_evidence
        self._top_k = top_k
        self._evidence: list[SearchResult] = []
        self._citation_by_chunk: dict[str, int] = {}
        self._cache: dict[str, dict[str, Any]] = {}
        #: Tool invocations in call order (first occurrence; echoed in the
        #: ``tools_used`` response field and used by the routing eval).
        self.calls: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ specs
    def specs(self) -> list[dict[str, Any]]:
        """Return the JSON-schema tool specifications for the model."""
        return [
            {
                "name": "search_game_docs",
                "description": (
                    "Retrieve factual passages about one game from the "
                    "knowledge base. Returns numbered evidence passages."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "game": {"type": "string", "enum": list(GAME_SLUGS)},
                        "question": {
                            "type": "string",
                            "description": "What to look up for this game.",
                        },
                    },
                    "required": ["game", "question"],
                },
            },
            {
                "name": "compare_games",
                "description": (
                    "Retrieve evidence for two games side by side, for "
                    "comparison questions (difficulty, scoring, enemies, "
                    "controls, ...)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "game_a": {"type": "string", "enum": list(GAME_SLUGS)},
                        "game_b": {"type": "string", "enum": list(GAME_SLUGS)},
                    },
                    "required": ["game_a", "game_b"],
                },
            },
            {
                "name": "list_games_by_year",
                "description": ("List the knowledge-base games released in a given year."),
                "parameters": {
                    "type": "object",
                    "properties": {"year": {"type": "integer"}},
                    "required": ["year"],
                },
            },
        ]

    # --------------------------------------------------------------- evidence
    @property
    def evidence(self) -> list[SearchResult]:
        """Evidence passages registered so far (first-seen order)."""
        return list(self._evidence)

    # -------------------------------------------------------------- execution
    async def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute tool *name* with *arguments*; never raises for bad input.

        Unknown tools, unknown games and malformed arguments produce an
        ``{"error": ...}`` payload so the model can recover.  Retrieval and
        provider failures propagate to the caller (HTTP 502/503).
        """
        key = json.dumps({"name": name, "arguments": arguments}, sort_keys=True)
        self.calls.append({"name": name, "arguments": dict(arguments)})
        if key in self._cache:
            return self._cache[key]

        handler = {
            "search_game_docs": self._search_game_docs,
            "compare_games": self._compare_games,
            "list_games_by_year": self._list_games_by_year,
        }.get(name)

        try:
            if handler is None:
                payload: dict[str, Any] = {
                    "error": f"unknown tool '{name}'",
                    "available_tools": [
                        "search_game_docs",
                        "compare_games",
                        "list_games_by_year",
                    ],
                }
            else:
                payload = await handler(arguments)
        except (ValueError, TypeError, KeyError) as exc:
            payload = {"error": f"invalid arguments for '{name}': {exc}"}

        self._cache[key] = payload
        return payload

    # --------------------------------------------------------------- handlers
    async def _search_game_docs(self, arguments: dict[str, Any]) -> dict[str, Any]:
        game = self._validate_slug(arguments.get("game"), field="game")
        question = str(arguments.get("question", "")).strip()
        if not question:
            raise ValueError("'question' must be a non-empty string")

        results = await self._search(question, self._top_k, game)
        payload: dict[str, Any] = {
            "game": game,
            "question": question,
            "evidence": self._register(results),
        }
        if not payload["evidence"]:
            payload["note"] = "no passages matched this question"
        return payload

    async def _compare_games(self, arguments: dict[str, Any]) -> dict[str, Any]:
        game_a = self._validate_slug(arguments.get("game_a"), field="game_a")
        game_b = self._validate_slug(arguments.get("game_b"), field="game_b")
        if game_a == game_b:
            raise ValueError("'game_a' and 'game_b' must be different games")

        games: dict[str, Any] = {}
        for slug in (game_a, game_b):
            results = await self._search(
                f"core gameplay, rules and scoring of {_TITLES[slug]}",
                self._top_k,
                slug,
            )
            games[slug] = {
                "title": _TITLES[slug],
                "evidence": self._register(results),
            }
        return {"games": games}

    async def _list_games_by_year(self, arguments: dict[str, Any]) -> dict[str, Any]:
        year_raw = arguments.get("year")
        if isinstance(year_raw, bool) or year_raw is None:
            raise ValueError("'year' must be an integer year")
        try:
            year = int(year_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("'year' must be an integer year") from exc

        matches = [
            {"slug": item["slug"], "title": item["title"]}
            for item in GAME_CATALOG
            if item["year"] == year
        ]
        payload: dict[str, Any] = {"year": year, "games": matches}
        if not matches:
            payload["note"] = f"no knowledge-base games released in {year}"
            payload["catalog_years"] = sorted({int(item["year"]) for item in GAME_CATALOG})
        return payload

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _validate_slug(value: Any, field: str) -> str:
        slug = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
        slug = _ALIASES.get(slug, slug)
        if slug not in GAME_SLUGS:
            raise ValueError(f"unknown {field} '{value}'; known games: {', '.join(GAME_SLUGS)}")
        return slug

    def _register(self, results: list[SearchResult]) -> list[dict[str, Any]]:
        """Assign stable citation numbers to newly seen evidence passages."""
        items: list[dict[str, Any]] = []
        for result in results[: self._max_evidence]:
            if result.chunk_id not in self._citation_by_chunk:
                self._evidence.append(result)
                self._citation_by_chunk[result.chunk_id] = len(self._evidence)
            items.append(
                {
                    "citation": self._citation_by_chunk[result.chunk_id],
                    "source": str(result.payload.get("source", "")),
                    "content": result.content,
                }
            )
        return items
