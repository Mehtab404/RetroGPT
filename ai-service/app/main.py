"""RetroGPT AI Service - FastAPI application."""

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import settings

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
