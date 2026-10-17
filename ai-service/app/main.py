"""RetroGPT AI Service - FastAPI application."""

import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import settings
from app.generation.llm import LLMError, get_llm_provider
from app.generation.prompts import build_rag_prompt
from app.ingestion.embeddings import EmbeddingError, get_embedding_provider
from app.retrieval.hybrid import hybrid_search
from app.retrieval.vector_store import QdrantVectorStore, VectorStoreError
from app.schemas import QueryRequest, QueryResponse, SourceDoc

app = FastAPI(
    title="RetroGPT AI Service",
    description="RAG-powered AI assistant for classic video games",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

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
        print(f"✓ Connected to Qdrant at {settings.qdrant_url}")
    except Exception as e:
        print(f"✗ Failed to connect to Qdrant: {e}")
        # Don't fail startup, allow health check to report status


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Clean up connections on shutdown."""
    global qdrant_client
    if qdrant_client:
        qdrant_client.close()
        print("✓ Closed Qdrant connection")


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
    2. Retrieve the top-K passages with hybrid search (dense + BM25 sparse
       vectors fused via RRF); falls back to dense-only when BM25 stats or
       sparse vectors are unavailable.
    3. Build a RAG prompt from the retrieved context + question.
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

    # ------------------------------------------------------------------
    # 1. Embed the question
    # ------------------------------------------------------------------
    embedder = get_embedding_provider()
    try:
        query_vector = await embedder.embed_query(request.question)
    except EmbeddingError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Embedding provider error: {exc}",
        ) from exc

    # ------------------------------------------------------------------
    # 2. Retrieve relevant passages from the vector store
    # ------------------------------------------------------------------
    # Build an optional Qdrant payload filter for the game slug.
    # NOTE: ingested chunk metadata stores the slug under the ``game`` key
    # (see app/ingestion/loaders.py), so the filter must match ``game``.
    qdrant_filter = None
    if request.game_slug:
        from qdrant_client.http import models as qdrant_models

        qdrant_filter = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="game",
                    match=qdrant_models.MatchValue(value=request.game_slug),
                )
            ]
        )

    vector_store = QdrantVectorStore(client=qdrant_client)
    try:
        search_results = await hybrid_search(
            vector_store,
            request.question,
            query_vector,
            top_k=request.top_k,
            filter=qdrant_filter,
        )
    except VectorStoreError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Vector store error: {exc}",
        ) from exc

    # ------------------------------------------------------------------
    # 3. Build the RAG prompt
    # ------------------------------------------------------------------
    if not search_results:
        # No relevant passages found — return a graceful "no context" answer.
        t_end = time.monotonic()
        return QueryResponse(
            answer="I don't know — this information is not available in the provided sources.",
            sources=[],
            latency_ms=(t_end - t_start) * 1000,
        )

    context_passages = [
        {
            "id": result.chunk_id,
            "content": result.content,
        }
        for result in search_results
    ]
    prompt = build_rag_prompt(
        question=request.question,
        context_passages=context_passages,
    )

    # ------------------------------------------------------------------
    # 4. Generate an answer
    # ------------------------------------------------------------------
    llm = get_llm_provider()
    try:
        answer = await llm.generate(prompt)
    except LLMError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LLM provider error: {exc}",
        ) from exc

    # ------------------------------------------------------------------
    # 5. Build response with sources
    # ------------------------------------------------------------------
    t_end = time.monotonic()

    sources = [
        SourceDoc(
            id=result.chunk_id,
            title=str(result.payload.get("title", result.payload.get("source", ""))),
            source=str(result.payload.get("source", "")),
            score=round(result.score, 4),
            snippet=result.content[:300],
        )
        for result in search_results
    ]

    return QueryResponse(
        answer=answer,
        sources=sources,
        latency_ms=round((t_end - t_start) * 1000, 2),
    )
