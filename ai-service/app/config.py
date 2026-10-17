"""Configuration management using pydantic-settings."""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False
    )

    # LLM and Embeddings
    llm_provider: Literal["gemini", "openai"] = "gemini"
    llm_model: str = "gemini-1.5-flash"
    embedding_provider: Literal["gemini", "openai"] = "gemini"
    embedding_model: str = "text-embedding-004"
    google_api_key: str = ""
    openai_api_key: str = ""

    # Vector Database
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "retro_games"

    # Retrieval
    chunk_size: int = 800
    chunk_overlap: int = 120
    retrieval_top_k: int = 20
    rerank_top_n: int = 5

    # Hybrid search (dense + BM25 sparse, fused with RRF) — Issue #20
    hybrid_search_enabled: bool = True
    #: Candidates fetched from each retriever before RRF fusion.
    hybrid_prefetch_k: int = 20
    #: Persisted BM25 corpus statistics (written at ingestion time).
    bm25_stats_path: str = "/data/bm25_stats.json"

    # Cross-encoder re-ranking (Issue #21)
    rerank_enabled: bool = True
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rerank_cache_dir: str = "/data/fastembed_cache"


# Global settings instance
settings = Settings()
