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


# Global settings instance
settings = Settings()
