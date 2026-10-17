"""Ingestion pipeline for the RetroGPT RAG system.

Orchestrates the full load → chunk → embed → upsert workflow.

Pipeline stages
---------------
1. **Load** – :func:`~app.ingestion.loaders.get_loader_for` reads raw files
   from disk into :class:`~app.ingestion.loaders.Document` objects.
2. **Chunk** – :class:`~app.ingestion.chunker.RecursiveCharacterTextSplitter`
   splits each document into overlapping :class:`~app.ingestion.chunker.Chunk`
   objects that fit within the embedding model's context window.
3. **Embed** – :class:`~app.ingestion.embeddings.EmbeddingProvider` converts
   each chunk's text to a dense float vector (batched for efficiency).
4. **Upsert** – :class:`~app.retrieval.vector_store.QdrantVectorStore` stores
   ``(id, vector, payload)`` tuples in Qdrant.

The pipeline is **idempotent**: re-running with the same files will overwrite
existing Qdrant points with the same chunk IDs, producing no duplicates.

Typical usage::

    pipeline = IngestionPipeline()
    result = await pipeline.run(Path("/data/raw"))
    print(f"Ingested {result.total_chunks} chunks from {result.total_docs} docs")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from app.config import settings
from app.ingestion.chunker import Chunk, RecursiveCharacterTextSplitter
from app.ingestion.embeddings import EmbeddingProvider, get_embedding_provider
from app.ingestion.loaders import Document, get_loader_for
from app.retrieval.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)

# Supported file extensions for automatic discovery.
_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".md", ".markdown", ".txt"})


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass
class IngestionResult:
    """Summary of a completed ingestion run.

    Attributes:
        total_docs:    Number of source files loaded.
        total_chunks:  Number of chunks produced after splitting.
        total_upserted: Number of points successfully written to Qdrant.
        skipped_files: Files skipped due to unsupported extensions or errors.
        errors:        Per-file error messages (file path → error string).
    """

    total_docs: int = 0
    total_chunks: int = 0
    total_upserted: int = 0
    skipped_files: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class IngestionPipeline:
    """Orchestrates the full document ingestion pipeline.

    Args:
        vector_store:   Pre-built :class:`~app.retrieval.vector_store.QdrantVectorStore`.
                        If ``None``, a default store is created from settings.
        embedder:       Pre-built :class:`~app.ingestion.embeddings.EmbeddingProvider`.
                        If ``None``, the default provider is created from settings.
        splitter:       Pre-built :class:`~app.ingestion.chunker.RecursiveCharacterTextSplitter`.
                        If ``None``, a default splitter is created from settings.
        ensure_collection: If ``True`` (default), call
                           :meth:`~app.retrieval.vector_store.QdrantVectorStore.ensure_collection`
                           before upserting.  Set to ``False`` if the collection
                           is guaranteed to exist already.

    Example::

        pipeline = IngestionPipeline()
        result = await pipeline.run(Path("/data/raw"))
        print(result.total_upserted)
    """

    def __init__(
        self,
        vector_store: QdrantVectorStore | None = None,
        embedder: EmbeddingProvider | None = None,
        splitter: RecursiveCharacterTextSplitter | None = None,
        ensure_collection: bool = True,
    ) -> None:
        self._store = vector_store or QdrantVectorStore()
        self._embedder = embedder or get_embedding_provider()
        self._splitter = splitter or RecursiveCharacterTextSplitter()
        self._ensure_collection = ensure_collection

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, source: Path) -> IngestionResult:
        """Run the full ingestion pipeline on *source*.

        *source* may be:

        * A **directory** – all files with supported extensions are discovered
          recursively and processed in sorted order.
        * A **single file** – only that file is processed.

        Args:
            source: Path to a file or directory containing game documents.

        Returns:
            :class:`IngestionResult` summarising the run.

        Raises:
            FileNotFoundError: If *source* does not exist.
        """
        source = Path(source).resolve()
        if not source.exists():
            raise FileNotFoundError(f"Source path does not exist: {source}")

        result = IngestionResult()

        # Optionally create/verify the Qdrant collection up front.
        if self._ensure_collection:
            await self._store.ensure_collection()

        # Collect files to process.
        files = self._discover_files(source)
        logger.info("Discovered %d file(s) under '%s'.", len(files), source)

        # Load, chunk, embed, and upsert file by file.
        for file_path in sorted(files):
            try:
                docs = self._load(file_path)
                result.total_docs += len(docs)

                chunks = self._chunk(docs)
                if not chunks:
                    logger.warning("No chunks produced from '%s'; skipping.", file_path)
                    result.skipped_files.append(str(file_path))
                    continue

                vectors = await self._embed(chunks)
                upserted = await self._store.upsert(chunks, vectors)
                result.total_chunks += len(chunks)
                result.total_upserted += upserted

                logger.info(
                    "Processed '%s': %d doc(s) → %d chunk(s) → %d upserted.",
                    file_path.name,
                    len(docs),
                    len(chunks),
                    upserted,
                )
            except Exception as exc:
                logger.error("Failed to process '%s': %s", file_path, exc)
                result.errors[str(file_path)] = str(exc)

        logger.info(
            "Ingestion complete: %d docs, %d chunks, %d upserted, %d errors.",
            result.total_docs,
            result.total_chunks,
            result.total_upserted,
            len(result.errors),
        )
        return result

    async def ingest_documents(
        self, documents: list[Document]
    ) -> IngestionResult:
        """Ingest pre-loaded :class:`~app.ingestion.loaders.Document` objects.

        Useful when the caller has already loaded documents (e.g. from
        non-file sources like a database or API) and only needs the
        chunk → embed → upsert stages.

        Args:
            documents: List of :class:`~app.ingestion.loaders.Document` objects.

        Returns:
            :class:`IngestionResult` summarising the run.
        """
        result = IngestionResult(total_docs=len(documents))

        if self._ensure_collection:
            await self._store.ensure_collection()

        if not documents:
            return result

        chunks = self._chunk(documents)
        if not chunks:
            logger.warning("No chunks produced from provided documents.")
            return result

        vectors = await self._embed(chunks)
        upserted = await self._store.upsert(chunks, vectors)
        result.total_chunks = len(chunks)
        result.total_upserted = upserted

        logger.info(
            "Direct ingestion: %d docs → %d chunks → %d upserted.",
            len(documents),
            len(chunks),
            upserted,
        )
        return result

    # ------------------------------------------------------------------
    # Internal pipeline stages
    # ------------------------------------------------------------------

    def _discover_files(self, source: Path) -> list[Path]:
        """Return a list of files to process.

        Args:
            source: A file path or directory to scan recursively.

        Returns:
            Sorted list of :class:`Path` objects with supported extensions.
        """
        if source.is_file():
            if source.suffix.lower() in _SUPPORTED_EXTENSIONS:
                return [source]
            logger.warning(
                "File '%s' has unsupported extension '%s'; skipping.",
                source,
                source.suffix,
            )
            return []

        files: list[Path] = []
        for ext in _SUPPORTED_EXTENSIONS:
            files.extend(source.rglob(f"*{ext}"))
        return sorted(set(files))

    def _load(self, file_path: Path) -> list[Document]:
        """Load a single file into ``Document`` objects.

        Args:
            file_path: Path to the file to load.

        Returns:
            List of :class:`~app.ingestion.loaders.Document` objects.

        Raises:
            ValueError:       If no loader is registered for the extension.
            FileNotFoundError: If the file does not exist.
        """
        loader = get_loader_for(file_path)
        return loader.load(file_path)

    def _chunk(self, documents: list[Document]) -> list[Chunk]:
        """Split *documents* into chunks.

        Args:
            documents: List of :class:`~app.ingestion.loaders.Document` objects.

        Returns:
            Flat list of :class:`~app.ingestion.chunker.Chunk` objects.
        """
        return self._splitter.split_documents(documents)

    async def _embed(self, chunks: list[Chunk]) -> list[list[float]]:
        """Embed chunk texts using the configured embedding provider.

        Args:
            chunks: Chunks whose ``content`` field will be embedded.

        Returns:
            List of float vectors, one per chunk, in the same order.
        """
        texts = [chunk.content for chunk in chunks]
        return await self._embedder.embed_documents(texts)
