"""Pydantic request and response models for the RetroGPT AI service.

Defines the wire-format schemas used by the ``POST /query`` endpoint:

* :class:`QueryRequest`  — incoming query from the API gateway.
* :class:`HistoryMessage` — one prior conversation turn (Issue #27).
* :class:`SourceDoc`     — a single retrieved source document / passage.
* :class:`QueryResponse` — the generated answer with sources and metadata.

All models use strict validation and forbid extra fields to catch
integration errors early.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# History bounds (mirrored by api-gateway/src/services/history.ts)
# ---------------------------------------------------------------------------

#: Maximum number of prior messages accepted (Issue #27).
HISTORY_MAX_MESSAGES: int = 6

#: Maximum total characters across the prior conversation (Issue #27).
HISTORY_MAX_CHARS: int = 4000

# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class HistoryMessage(BaseModel):
    """One prior conversation turn supplied by the API gateway (Issue #27).

    History is context for interpreting the current question only; it is
    never treated as factual evidence and never cited.

    Attributes:
        role:    Who authored the turn (``"user"`` or ``"assistant"``).
        content: Text of the turn.
    """

    model_config = {"extra": "forbid"}

    role: Annotated[
        Literal["user", "assistant"],
        Field(description="Author of this prior turn."),
    ]
    content: Annotated[
        str,
        Field(
            max_length=HISTORY_MAX_CHARS,
            description="Text of the prior turn.",
        ),
    ]


class QueryRequest(BaseModel):
    """Request body for ``POST /query``.

    Attributes:
        question:  The user's question about a classic video game.
                   Must be between 1 and 1000 characters.
        top_k:     Number of documents to retrieve from the vector store.
                   Defaults to 5; capped at 20 to avoid overly long prompts.
        game_slug: Optional filter to restrict retrieval to a specific game's
                   documents (e.g. ``"asteroids"``, ``"pong"``).
        history:   Optional bounded prior conversation (oldest first) used
                   only to interpret the current question (Issue #27).
    """

    model_config = {"extra": "forbid"}

    question: Annotated[
        str,
        Field(
            min_length=1,
            max_length=1000,
            description="The user's question about a classic video game.",
            examples=["How do you score points in Asteroids?"],
        ),
    ]
    top_k: Annotated[
        int,
        Field(
            default=5,
            ge=1,
            le=20,
            description="Number of documents to retrieve (1–20).",
        ),
    ] = 5
    game_slug: Annotated[
        str | None,
        Field(
            default=None,
            description="Optional game slug to restrict retrieval scope.",
            examples=["asteroids", "pong"],
        ),
    ] = None
    history: Annotated[
        list[HistoryMessage] | None,
        Field(
            default=None,
            max_length=HISTORY_MAX_MESSAGES,
            description=("Bounded prior conversation (oldest first) for follow-up context."),
        ),
    ] = None

    @model_validator(mode="after")
    def _validate_history_char_budget(self) -> QueryRequest:
        """Reject histories whose contents exceed ``HISTORY_MAX_CHARS``."""
        if self.history:
            total_chars = sum(len(message.content) for message in self.history)
            if total_chars > HISTORY_MAX_CHARS:
                raise ValueError(
                    f"history content must total at most {HISTORY_MAX_CHARS} characters"
                )
        return self


# ---------------------------------------------------------------------------
# Source document model
# ---------------------------------------------------------------------------


class SourceDoc(BaseModel):
    """A single retrieved source document included in the response.

    Attributes:
        id:       Unique chunk identifier (e.g. ``"asteroids_manual_p3_c2"``).
        title:    Human-readable document title, if available.
        source:   Source file path the passage came from (e.g.
                  ``"/data/raw/asteroids/asteroids_manual.md"``). Used by the
                  evaluation script to match expected source documents.
        score:    Cosine similarity score from the vector store (0–1).
        snippet:  Short excerpt from the retrieved passage (≤ 300 chars).
    """

    model_config = {"extra": "forbid"}

    id: Annotated[
        str,
        Field(description="Unique chunk identifier."),
    ]
    title: Annotated[
        str,
        Field(default="", description="Human-readable document title."),
    ] = ""
    source: Annotated[
        str,
        Field(
            default="",
            description="Source file path the passage was retrieved from.",
        ),
    ] = ""
    score: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Cosine similarity score (0–1, higher = more similar).",
        ),
    ]
    snippet: Annotated[
        str,
        Field(
            default="",
            description="Short excerpt from the retrieved passage.",
        ),
    ] = ""


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------


class QueryResponse(BaseModel):
    """Response body for ``POST /query``.

    Attributes:
        answer:     The LLM-generated answer, with inline ``[N]`` citations.
        sources:    Ordered list of source documents used to generate the answer.
        latency_ms: Total end-to-end latency in milliseconds.
        cached:     Whether this response was served from cache (always
                    ``False`` at this service layer; caching is handled by
                    the API gateway).
    """

    model_config = {"extra": "forbid"}

    answer: Annotated[
        str,
        Field(description="The LLM-generated answer with inline citations."),
    ]
    sources: Annotated[
        list[SourceDoc],
        Field(default_factory=list, description="Retrieved source documents."),
    ] = Field(default_factory=list)
    latency_ms: Annotated[
        float,
        Field(ge=0.0, description="Total end-to-end latency in milliseconds."),
    ]
    cached: Annotated[
        bool,
        Field(
            default=False,
            description="Whether this response was served from cache.",
        ),
    ] = False
