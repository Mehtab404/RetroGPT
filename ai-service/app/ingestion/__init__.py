"""Ingestion package: loaders, chunkers, embeddings, and ingestion pipeline."""

from app.ingestion.chunker import Chunk, RecursiveCharacterTextSplitter
from app.ingestion.embeddings import (
    EmbeddingError,
    EmbeddingProvider,
    FastEmbedEmbedding,
    GeminiEmbedding,
    OpenAIEmbedding,
    get_embedding_provider,
)
from app.ingestion.loaders import Document, MarkdownLoader, TextLoader, get_loader_for
from app.ingestion.pipeline import IngestionPipeline, IngestionResult

__all__ = [
    # Loaders
    "Document",
    "MarkdownLoader",
    "TextLoader",
    "get_loader_for",
    # Chunker
    "Chunk",
    "RecursiveCharacterTextSplitter",
    # Embeddings
    "EmbeddingProvider",
    "EmbeddingError",
    "GeminiEmbedding",
    "OpenAIEmbedding",
    "FastEmbedEmbedding",
    "get_embedding_provider",
    # Pipeline
    "IngestionPipeline",
    "IngestionResult",
]
