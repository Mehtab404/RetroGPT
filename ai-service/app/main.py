"""RetroGPT AI Service - FastAPI application."""

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from app.agent.loop import run_agent
from app.agent.tools import ToolRegistry
from app.config import settings
from app.generation.llm import LLMError, get_llm_provider
from app.generation.prompts import build_rag_prompt
from app.ingestion.embeddings import EmbeddingError, get_embedding_provider
from app.logging_config import configure_logging
from app.middleware import RequestIdMiddleware
from app.retrieval.rerank import retrieve
from app.retrieval.vector_store import QdrantVectorStore, SearchResult, VectorStoreError
from app.schemas import (
    AgentRequest,
    AgentResponse,
    AgentToolCall,
    QueryRequest,
    QueryResponse,
    SourceDoc,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RetroGPT AI Service",
    description="RAG-powered AI assistant for classic video games",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Issue #24: JSON logs in both services; x-request-id ties gateway call,
# retrieval and generation logs together under one trace id.
configure_logging()
app.add_middleware(RequestIdMiddleware)

# CORS middleware for API gateway
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to gateway URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Qdrant client (initialized on startup)
qdrant_client: QdrantClient | None = None


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize connections on startup."""
    global qdrant_client
    try:
        qdrant_client = QdrantClient(url=settings.qdrant_url)
        # Test connection
        qdrant_client.get_collections()
        logger.info("qdrant_connected", extra={"url": settings.qdrant_url})
    except Exception as e:
        logger.error("qdrant_connection_failed", extra={"error": str(e)})
        # Don't fail startup, allow health check to report status


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Clean up connections on shutdown."""
    global qdrant_client
    if qdrant_client:
        qdrant_client.close()
        logger.info("qdrant_connection_closed")


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """
    Health check endpoint.

    Returns service status and dependency health.
    """
    health_status = {"status": "ok", "service": "ai-service", "dependencies": {}}

    # Check Qdrant connection
    if qdrant_client is None:
        health_status["status"] = "degraded"
        health_status["dependencies"]["qdrant"] = "not_initialized"
    else:
        try:
            qdrant_client.get_collections()
            health_status["dependencies"]["qdrant"] = "ok"
        except UnexpectedResponse as e:
            health_status["status"] = "degraded"
            health_status["dependencies"]["qdrant"] = f"error: {str(e)}"
        except Exception as e:
            health_status["status"] = "degraded"
            health_status["dependencies"]["qdrant"] = f"error: {str(e)}"

    # Return 503 if degraded
    if health_status["status"] == "degraded":
        raise HTTPException(status_code=503, detail=health_status)

    return health_status


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "service": "RetroGPT AI Service",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Retrieve relevant passages and generate an answer for a question.

    Pipeline:
    1. Embed the question using the configured embedding provider.
    2. Retrieve the best passages with the two-stage pipeline: hybrid
       search (dense + BM25 sparse fused via RRF) over a wide candidate
       pool, then cross-encoder re-ranking to the top-K.  Each stage fails
       open (dense-only, then hybrid order) when unavailable.
    3. Build a RAG prompt from the retrieved context + question (plus the
       bounded prior conversation when supplied — Issue #27).
    4. Generate an answer with the configured LLM provider.
    5. Return the answer with source citations and latency.

    Args:
        request: Validated :class:`~app.schemas.QueryRequest` body.

    Returns:
        :class:`~app.schemas.QueryResponse` with answer, sources, and latency.

    Raises:
        HTTPException 422: On validation errors (handled by FastAPI).
        HTTPException 503: If Qdrant is unavailable.
        HTTPException 502: If the embedding or LLM provider call fails.
        HTTPException 500: On unexpected internal errors.
    """
    t_start = time.monotonic()
    stage_ms: dict[str, float] = {}
    retrieve_timings: dict[str, float] = {}

    search_results = await _embed_and_retrieve(request, stage_ms, retrieve_timings)

    # ------------------------------------------------------------------
    # 3. Build the RAG prompt
    # ------------------------------------------------------------------
    if not search_results:
        # No relevant passages found — return a graceful "no context" answer.
        total_ms = _elapsed_ms(t_start)
        _log_query(
            stage_ms=stage_ms,
            result_count=0,
            total_ms=total_ms,
            game_slug=request.game_slug,
            top_k=request.top_k,
        )
        return QueryResponse(
            answer="I don't know — this information is not available in the provided sources.",
            sources=[],
            latency_ms=total_ms,
        )

    prompt = build_rag_prompt(
        question=request.question,
        context_passages=[
            {"id": result.chunk_id, "content": result.content} for result in search_results
        ],
        history=_history_payload(request),
    )

    # ------------------------------------------------------------------
    # 4. Generate an answer
    # ------------------------------------------------------------------
    llm = get_llm_provider()
    t_stage = time.monotonic()
    try:
        answer = await llm.generate(prompt)
    except LLMError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LLM provider error: {exc}",
        ) from exc
    stage_ms["generate"] = _elapsed_ms(t_stage)

    # ------------------------------------------------------------------
    # 5. Build response with sources
    # ------------------------------------------------------------------
    total_ms = _elapsed_ms(t_start)

    _log_query(
        stage_ms=stage_ms,
        result_count=len(search_results),
        total_ms=total_ms,
        game_slug=request.game_slug,
        top_k=request.top_k,
    )

    return QueryResponse(
        answer=answer,
        sources=_build_sources(search_results),
        latency_ms=round(total_ms, 2),
    )


@app.post("/agent", response_model=AgentResponse)
async def agent_answer(request: AgentRequest) -> AgentResponse:
    """Answer a question with native tool calling (Issue #35 / backlog #30).

    Pipeline:

    1. Build a tool registry bound to the same retrieval pipeline as
       :func:`query` — evidence passages receive stable ``[N]`` citation
       numbers for the final answer.
    2. Run the bounded tool-calling loop with the configured LLM provider:
       the model may call ``search_game_docs``, ``compare_games`` and/or
       ``list_games_by_year`` before answering.
    3. Return the final answer plus the collected evidence and the tools used.

    Errors map exactly like :func:`query`:

    * ``502`` — embedding or LLM provider failure.
    * ``503`` — vector store unavailable.

    Args:
        request: Validated :class:`~app.schemas.AgentRequest` body.

    Returns:
        :class:`~app.schemas.AgentResponse` with answer, sources, tools used
        and the number of model turns taken.
    """
    t_start = time.monotonic()
    stage_ms: dict[str, float] = {}
    retrieve_timings: dict[str, float] = {}

    async def search(
        question: str,
        top_k: int,
        game_slug: str | None,
    ) -> list[SearchResult]:
        """Retrieval callable injected into the tool registry."""
        return await _retrieve_passages(
            question,
            top_k,
            game_slug,
            stage_ms,
            retrieve_timings,
        )

    registry = ToolRegistry(search=search)
    max_steps = request.max_steps or settings.agent_default_max_steps
    llm = get_llm_provider()
    try:
        result = await run_agent(
            llm=llm,
            question=request.question,
            registry=registry,
            max_steps=max_steps,
        )
    except LLMError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LLM provider error: {exc}",
        ) from exc

    total_ms = _elapsed_ms(t_start)
    _log_agent(
        stage_ms=stage_ms,
        tools_used=result.tools_used,
        steps=result.steps,
        result_count=len(result.sources),
        total_ms=total_ms,
    )

    return AgentResponse(
        answer=result.answer,
        sources=_build_sources(result.sources),
        tools_used=[
            AgentToolCall(name=str(call["name"]), arguments=call.get("arguments") or {})
            for call in result.tools_used
        ],
        steps=result.steps,
        latency_ms=round(total_ms, 2),
    )


@app.post("/query/stream")
async def query_stream(request: QueryRequest) -> StreamingResponse:
    """Stream the answer for a question as Server-Sent Events (Issue #25).

    Runs the same retrieval pipeline as :func:`query`, then streams the
    generated answer token by token so the UI can render it incrementally.

    Event protocol (``event:`` name + JSON ``data:`` payload):

    * ``meta``  — ``{"sources": [...]}`` sent once, before generation.
    * ``token`` — ``{"text": "..."}`` one per generated chunk.
    * ``done``  — ``{"latency_ms": 123.4}`` after the last chunk.
    * ``error`` — ``{"code": "...", "message": "..."}`` if generation fails
      mid-stream.

    Failures *before* the stream starts (embedding / retrieval) are regular
    JSON error responses with the same status codes as :func:`query`.

    Args:
        request: Validated :class:`~app.schemas.QueryRequest` body.

    Returns:
        :class:`~fastapi.responses.StreamingResponse` with the
        ``text/event-stream`` media type.
    """
    t_start = time.monotonic()
    stage_ms: dict[str, float] = {}
    retrieve_timings: dict[str, float] = {}

    search_results = await _embed_and_retrieve(request, stage_ms, retrieve_timings)

    if not search_results:
        return StreamingResponse(
            _stream_no_context(t_start, stage_ms, request),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

    prompt = build_rag_prompt(
        question=request.question,
        context_passages=[
            {"id": result.chunk_id, "content": result.content} for result in search_results
        ],
        history=_history_payload(request),
    )

    return StreamingResponse(
        _stream_answer(t_start, stage_ms, request, search_results, prompt),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


async def _embed_and_retrieve(
    request: QueryRequest,
    stage_ms: dict[str, float],
    retrieve_timings: dict[str, float],
) -> list[SearchResult]:
    """Run pipeline steps 1-2 for a ``/query`` request (thin wrapper).

    Kept for the ``/query`` and ``/query/stream`` endpoints; the actual
    work lives in :func:`_retrieve_passages`, which the agent tools also
    call (Issue #35).
    """
    return await _retrieve_passages(
        request.question,
        request.top_k,
        request.game_slug,
        stage_ms,
        retrieve_timings,
    )


async def _retrieve_passages(
    question: str,
    top_k: int,
    game_slug: str | None,
    stage_ms: dict[str, float],
    retrieve_timings: dict[str, float],
) -> list[SearchResult]:
    """Embed *question* and retrieve passages (shared by all endpoints).

    Shared by ``/query``, ``/query/stream`` and the agent tools (#35), which
    is why the inputs are plain values rather than a :class:`QueryRequest`.
    Per-stage latencies are recorded into *stage_ms*; *retrieve_timings*
    receives the hybrid-search / re-rank split from
    :func:`app.retrieval.rerank.retrieve`.

    Raises:
        HTTPException 502: If the embedding provider call fails.
        HTTPException 503: If the vector store is unavailable.
    """
    # ------------------------------------------------------------------
    # 1. Embed the question
    # ------------------------------------------------------------------
    embedder = get_embedding_provider()
    t_stage = time.monotonic()
    try:
        query_vector = await embedder.embed_query(question)
    except EmbeddingError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Embedding provider error: {exc}",
        ) from exc
    stage_ms["embed"] = _elapsed_ms(t_stage)

    # ------------------------------------------------------------------
    # 2. Retrieve relevant passages from the vector store
    # ------------------------------------------------------------------
    # Build an optional Qdrant payload filter for the game slug.
    # NOTE: ingested chunk metadata stores the slug under the ``game`` key
    # (see app/ingestion/loaders.py), so the filter must match ``game``.
    qdrant_filter = None
    if game_slug:
        from qdrant_client.http import models as qdrant_models

        qdrant_filter = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="game",
                    match=qdrant_models.MatchValue(value=game_slug),
                )
            ]
        )

    vector_store = QdrantVectorStore(client=qdrant_client)
    try:
        # ``retrieve`` reports the hybrid-search and re-rank split via
        # ``retrieve_timings`` (Issue #24 stage observability).
        search_results = await retrieve(
            vector_store,
            question,
            query_vector,
            top_k=top_k,
            filter=qdrant_filter,
            timings=retrieve_timings,
        )
    except VectorStoreError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Vector store error: {exc}",
        ) from exc

    stage_ms["retrieve"] = retrieve_timings.get("hybrid_search_ms", 0.0)
    stage_ms["rerank"] = retrieve_timings.get("rerank_ms", 0.0)
    return search_results


def _build_sources(search_results: list[SearchResult]) -> list[SourceDoc]:
    """Map retrieval results to the wire-format :class:`SourceDoc` list."""
    return [
        SourceDoc(
            id=result.chunk_id,
            title=str(result.payload.get("title", result.payload.get("source", ""))),
            source=str(result.payload.get("source", "")),
            score=round(result.score, 4),
            snippet=result.content[:300],
        )
        for result in search_results
    ]


def _history_payload(request: QueryRequest) -> list[dict[str, str]] | None:
    """Map validated history models to the prompt-builder payload (Issue #27)."""
    if not request.history:
        return None
    return [{"role": message.role, "content": message.content} for message in request.history]


#: Headers that keep SSE responses unbuffered end-to-end (nginx etc.).
_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _sse_event(event: str, data: dict[str, Any]) -> str:
    """Format one Server-Sent Event frame (``event:`` + single-line JSON)."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_no_context(
    t_start: float,
    stage_ms: dict[str, float],
    request: QueryRequest,
) -> AsyncIterator[str]:
    """SSE generator for a question with no retrieved context."""
    yield _sse_event("meta", {"sources": []})

    answer = "I don't know — this information is not available in the provided sources."
    yield _sse_event("token", {"text": answer})

    total_ms = _elapsed_ms(t_start)
    _log_query(
        stage_ms=stage_ms,
        result_count=0,
        total_ms=total_ms,
        game_slug=request.game_slug,
        top_k=request.top_k,
        streamed=True,
        chunk_count=1,
    )
    yield _sse_event("done", {"latency_ms": round(total_ms, 2)})


async def _stream_answer(
    t_start: float,
    stage_ms: dict[str, float],
    request: QueryRequest,
    search_results: list[SearchResult],
    prompt: str,
) -> AsyncIterator[str]:
    """SSE generator: metadata first, then answer tokens as they arrive."""
    sources = _build_sources(search_results)
    yield _sse_event("meta", {"sources": [source.model_dump() for source in sources]})

    llm = get_llm_provider()
    chunk_count = 0
    t_stage = time.monotonic()
    try:
        async for chunk in llm.generate_stream(prompt):
            if not chunk:
                continue
            chunk_count += 1
            yield _sse_event("token", {"text": chunk})
    except LLMError as exc:
        stage_ms["generate"] = _elapsed_ms(t_stage)
        _log_query(
            stage_ms=stage_ms,
            result_count=len(search_results),
            total_ms=_elapsed_ms(t_start),
            game_slug=request.game_slug,
            top_k=request.top_k,
            streamed=True,
            chunk_count=chunk_count,
        )
        logger.warning("query_stream_failed", extra={"error": str(exc)})
        yield _sse_event(
            "error",
            {"code": "LLM_ERROR", "message": f"LLM provider error: {exc}"},
        )
        return

    stage_ms["generate"] = _elapsed_ms(t_stage)
    total_ms = _elapsed_ms(t_start)
    _log_query(
        stage_ms=stage_ms,
        result_count=len(search_results),
        total_ms=total_ms,
        game_slug=request.game_slug,
        top_k=request.top_k,
        streamed=True,
        chunk_count=chunk_count,
    )
    yield _sse_event("done", {"latency_ms": round(total_ms, 2)})


def _elapsed_ms(start: float) -> float:
    """Milliseconds elapsed since a ``time.monotonic()`` reading."""
    return (time.monotonic() - start) * 1000.0


def _log_query(
    *,
    stage_ms: dict[str, float],
    result_count: int,
    total_ms: float,
    game_slug: str | None,
    top_k: int | None,
    streamed: bool = False,
    chunk_count: int | None = None,
) -> None:
    """Emit one structured ``query`` event with per-stage latencies (#24)."""
    extra: dict[str, Any] = {
        "stage_ms": {stage: round(value, 2) for stage, value in stage_ms.items()},
        "result_count": result_count,
        "total_ms": round(total_ms, 2),
        "game_slug": game_slug,
        "top_k": top_k,
        "streamed": streamed,
    }
    if chunk_count is not None:
        extra["chunk_count"] = chunk_count

    logger.info("query", extra=extra)


def _log_agent(
    *,
    stage_ms: dict[str, float],
    tools_used: list[dict[str, Any]],
    steps: int,
    result_count: int,
    total_ms: float,
) -> None:
    """Emit one structured ``agent`` event with per-stage latencies (#35)."""
    logger.info(
        "agent",
        extra={
            "stage_ms": {stage: round(value, 2) for stage, value in stage_ms.items()},
            "tools_used": [str(call.get("name", "")) for call in tools_used],
            "steps": steps,
            "result_count": result_count,
            "total_ms": round(total_ms, 2),
        },
    )
