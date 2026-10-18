"""CLI ingestion tool for RetroGPT.

Loads game documents from disk into the Qdrant vector database using the
:class:`~app.ingestion.pipeline.IngestionPipeline`.

Usage::

    # From project root (Docker):
    docker compose exec ai-service python -m scripts.ingest /data/raw

    # Directly (with PYTHONPATH set):
    python -m scripts.ingest /data/raw
    python -m scripts.ingest /data/raw/asteroids/asteroids_manual.md

    # With options:
    python -m scripts.ingest /data/raw --collection retro_games --dry-run

Exit codes
----------
* ``0`` – Success (all files processed without errors)
* ``1`` – Partial failure (some files had errors, summary printed)
* ``2`` – Fatal error (source path missing, Qdrant unreachable, etc.)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger("retrogpt.ingest")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="scripts.ingest",
        description=(
            "Ingest game documents into the RetroGPT Qdrant vector database.\n"
            "Source can be a directory (scanned recursively) or a single file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m scripts.ingest /data/raw\n"
            "  python -m scripts.ingest /data/raw/asteroids --collection my_col\n"
            "  python -m scripts.ingest /data/raw --dry-run\n"
        ),
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Directory or file path to ingest.",
    )
    parser.add_argument(
        "--qdrant-url",
        type=str,
        default=None,
        metavar="URL",
        help=(
            "Qdrant service URL (default: from QDRANT_URL env / settings). "
            "E.g. http://localhost:6333"
        ),
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        metavar="NAME",
        help=("Qdrant collection name (default: from QDRANT_COLLECTION env / settings)."),
    )
    parser.add_argument(
        "--embedding-provider",
        type=str,
        default=None,
        choices=["gemini", "openai", "fastembed"],
        metavar="PROVIDER",
        help=(
            "Embedding provider to use: 'gemini', 'openai' or 'fastembed' "
            "(default: from settings)."
        ),
    )
    parser.add_argument(
        "--vector-size",
        type=int,
        default=None,
        metavar="DIM",
        help=(
            "Dense-vector dimensionality for new collections (default: derived "
            "from the embedding model when known, else from VECTOR_SIZE/settings)."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        metavar="CHARS",
        help="Maximum characters per chunk (default: from settings, typically 800).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        metavar="CHARS",
        help="Overlap characters between consecutive chunks (default: from settings).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Load and chunk documents but do NOT upsert into Qdrant. "
            "Useful for validating data without modifying the vector store."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


# ---------------------------------------------------------------------------
# Dry-run helper
# ---------------------------------------------------------------------------


async def _dry_run(source: Path, args: argparse.Namespace) -> int:
    """Simulate ingestion without writing to Qdrant.

    Loads and chunks documents, printing a summary.

    Returns:
        Exit code (0 = success, 1 = partial errors).
    """
    from app.ingestion.chunker import RecursiveCharacterTextSplitter
    from app.ingestion.loaders import get_loader_for
    from app.ingestion.pipeline import _SUPPORTED_EXTENSIONS

    # Build components with CLI overrides.
    chunk_kwargs: dict[str, int] = {}
    if args.chunk_size:
        chunk_kwargs["chunk_size"] = args.chunk_size
    if args.chunk_overlap:
        chunk_kwargs["chunk_overlap"] = args.chunk_overlap
    splitter = RecursiveCharacterTextSplitter(**chunk_kwargs)

    # Discover files.
    if source.is_file():
        files = [source] if source.suffix.lower() in _SUPPORTED_EXTENSIONS else []
    else:
        files = sorted(f for ext in _SUPPORTED_EXTENSIONS for f in source.rglob(f"*{ext}"))

    if not files:
        logger.warning("No supported files found under '%s'.", source)
        return 0

    total_docs = 0
    total_chunks = 0
    errors: dict[str, str] = {}

    for file_path in files:
        try:
            loader = get_loader_for(file_path)
            docs = loader.load(file_path)
            chunks = splitter.split_documents(docs)
            total_docs += len(docs)
            total_chunks += len(chunks)
            print(f"  [DRY RUN] {file_path.name}: " f"{len(docs)} doc(s), {len(chunks)} chunk(s)")
        except Exception as exc:
            logger.error("Error processing '%s': %s", file_path, exc)
            errors[str(file_path)] = str(exc)

    print(
        f"\n[DRY RUN SUMMARY] {total_docs} docs, {total_chunks} chunks "
        f"from {len(files)} file(s). "
        f"{len(errors)} error(s). (Nothing written to Qdrant)"
    )
    return 1 if errors else 0


# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------


async def _run_ingestion(source: Path, args: argparse.Namespace) -> int:
    """Execute the full ingestion pipeline and return an exit code.

    Args:
        source: Resolved source path.
        args:   Parsed CLI arguments.

    Returns:
        Exit code: 0 = success, 1 = partial errors, 2 = fatal error.
    """
    from app.config import settings
    from app.ingestion.chunker import RecursiveCharacterTextSplitter
    from app.ingestion.embeddings import get_embedding_provider
    from app.ingestion.pipeline import IngestionPipeline
    from app.retrieval.vector_store import QdrantVectorStore, VectorStoreError

    # Build the embedder first so the collection can be sized to match it.
    provider_name: str | None = args.embedding_provider
    embedder = get_embedding_provider(provider_name)

    # Explicit --vector-size wins; otherwise derive from the model when it is
    # in the built-in registry, falling back to VECTOR_SIZE / settings.
    vector_size = args.vector_size or embedder.dimension
    if vector_size is None:
        vector_size = settings.vector_size
        logger.warning(
            "Unknown dimensionality for embedding model '%s'; using "
            "VECTOR_SIZE=%d. Pass --vector-size if the model differs.",
            embedder.model,
            vector_size,
        )

    # Build component overrides from CLI args.
    store_kwargs: dict[str, object] = {"vector_size": vector_size}
    if args.qdrant_url:
        store_kwargs["url"] = args.qdrant_url
    if args.collection:
        store_kwargs["collection"] = args.collection
    store = QdrantVectorStore(**store_kwargs)  # type: ignore[arg-type]

    chunk_kwargs: dict[str, int] = {}
    if args.chunk_size:
        chunk_kwargs["chunk_size"] = args.chunk_size
    if args.chunk_overlap:
        chunk_kwargs["chunk_overlap"] = args.chunk_overlap
    splitter = RecursiveCharacterTextSplitter(**chunk_kwargs)

    pipeline = IngestionPipeline(
        vector_store=store,
        embedder=embedder,
        splitter=splitter,
    )

    t0 = time.monotonic()
    try:
        result = await pipeline.run(source)
    except FileNotFoundError as exc:
        logger.error("Source not found: %s", exc)
        return 2
    except VectorStoreError as exc:
        logger.error("Qdrant error: %s", exc)
        return 2
    except Exception as exc:
        logger.exception("Unexpected error during ingestion: %s", exc)
        return 2
    finally:
        store.close()

    elapsed = time.monotonic() - t0

    # Print human-readable summary.
    print(
        f"\n{'='*60}\n"
        f"  INGESTION SUMMARY\n"
        f"{'='*60}\n"
        f"  Source        : {source}\n"
        f"  Collection    : {store.collection}\n"
        f"  Documents     : {result.total_docs}\n"
        f"  Chunks        : {result.total_chunks}\n"
        f"  Upserted      : {result.total_upserted}\n"
        f"  Skipped files : {len(result.skipped_files)}\n"
        f"  Errors        : {len(result.errors)}\n"
        f"  Elapsed       : {elapsed:.1f}s\n"
        f"{'='*60}"
    )

    if result.errors:
        print("\nErrors:")
        for path, msg in result.errors.items():
            print(f"  {path}: {msg}")

    return 1 if result.errors else 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    source = args.source.resolve()
    if not source.exists():
        logger.error("Source path does not exist: %s", source)
        return 2

    logger.info("Starting ingestion from '%s'…", source)

    if args.dry_run:
        return asyncio.run(_dry_run(source, args))

    return asyncio.run(_run_ingestion(source, args))


if __name__ == "__main__":
    sys.exit(main())
