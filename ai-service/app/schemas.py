"""Pydantic request and response models for the RetroGPT AI service.

Defines the wire-format schemas used by the ``POST /query`` endpoint:

* :class:`QueryRequest`  — incoming query from the API gateway.
* :class:`SourceDoc`     — a single retrieved source document / passage.
* :class:`QueryResponse` — the generated answer with sources and metadata.

All models use strict validation and forbid extra fields to catch
integration errors early.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Request body for ``POST /query``.

    Attributes:
        question:  The user's question about a classic video game.
                   Must be between 1 and 1000 characters.
        top_k:     Number of documents to retrieve from the vector store.
                   Defaults to 5; capped at 20 to avoid overly long prompts.
        game_slug: Optional filter to restrict retrieval to a specific game's
                   documents (e.g. ``"asteroids"``, ``"pong"``).
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


# ---------------------------------------------------------------------------
# Source document model
# ---------------------------------------------------------------------------


class SourceDoc(BaseModel):
    """A single retrieved source document included in the response.

    Attributes:
        id:       Unique chunk identifier (e.g. ``"asteroids_manual_p3_c2"``).
        title:    Human-readable document title, if available.
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
