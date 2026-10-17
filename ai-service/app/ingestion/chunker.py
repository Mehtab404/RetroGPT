"""Text chunking utilities for the RAG ingestion pipeline.

Splits ``Document`` objects into smaller ``Chunk`` objects that fit within
the embedding model's context window.  Metadata is preserved and extended
with chunk-level fields (``chunk_id``, ``chunk_index``).

The default splitter mirrors LangChain's ``RecursiveCharacterTextSplitter``
behaviour: it tries to split on paragraph breaks first, then sentence
boundaries, then words, then individual characters — always producing chunks
of at most *chunk_size* characters.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Sequence

from app.config import settings
from app.ingestion.loaders import Document


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """A text chunk derived from a ``Document``, ready for embedding.

    Attributes:
        content:      The chunk text.
        metadata:     Copy of the parent document metadata, augmented with
                      ``chunk_id``, ``chunk_index``, and ``chunk_total``.
    """

    content: str
    metadata: dict[str, str | int | float | None] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str | None:
        """Shortcut for ``metadata["chunk_id"]``."""
        return self.metadata.get("chunk_id")  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_chunk_id(source: str, chunk_index: int) -> str:
    """Generate a deterministic, URL-safe chunk identifier.

    The ID is a SHA-256 digest (first 16 hex chars) of ``source:index``.
    Short enough to be used as a Qdrant point ID string while being
    collision-resistant within a single ingest run.

    Args:
        source:      Source file path (from document metadata).
        chunk_index: Zero-based index of the chunk within its document.

    Returns:
        A 16-character lowercase hex string, e.g. ``"3f4a1c9d2b8e7f60"``.

    Example::

        id1 = _make_chunk_id("data/raw/asteroids/asteroids_manual.md", 0)
        id2 = _make_chunk_id("data/raw/asteroids/asteroids_manual.md", 1)
        assert id1 != id2
    """
    raw = f"{source}:{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _merge_pieces_into_chunks(
    pieces: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Merge a list of fine-grained text *pieces* into chunks.

    Adjacent pieces are concatenated as long as the combined length stays
    within *chunk_size*.  When adding the next piece would exceed the limit
    the current accumulator is emitted, and the new chunk starts with the
    last ``chunk_overlap`` characters of the previous chunk (overlap window)
    followed by the new piece.  If the resulting chunk still exceeds
    *chunk_size* (because overlap + piece > chunk_size), the overlap is
    trimmed to fit.

    Args:
        pieces:        Non-empty text fragments to merge.
        chunk_size:    Hard upper bound on each emitted chunk's length.
        chunk_overlap: Characters of tail from the previous chunk prepended
                       to the next (must be < *chunk_size*).

    Returns:
        List of chunks, each ≤ *chunk_size* characters.
    """
    chunks: list[str] = []
    current = ""

    for piece in pieces:
        if not piece.strip():
            continue
        if not current:
            current = piece
        elif len(current) + len(piece) <= chunk_size:
            current += piece
        else:
            # Emit the current accumulator.
            chunks.append(current)
            # Build overlap prefix from the tail of the emitted chunk.
            if chunk_overlap > 0:
                overlap_text = current[-chunk_overlap:]
                candidate = overlap_text + piece
                # If overlap + piece still exceeds chunk_size, drop enough
                # overlap characters to fit (preserves the hard size limit).
                if len(candidate) > chunk_size:
                    trim = len(candidate) - chunk_size
                    overlap_text = overlap_text[trim:]
                current = overlap_text + piece
            else:
                current = piece

    if current.strip():
        chunks.append(current)

    return chunks


def _split_on_separators(
    text: str,
    separators: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Recursively split *text* on the first separator that produces chunks
    small enough, backing off to finer-grained separators as needed.

    The algorithm is intentionally similar to LangChain's implementation:

    1. Try the first separator in *separators*.
    2. If a resulting piece is still longer than *chunk_size*, recurse with
       the remaining separators.
    3. Merge small adjacent pieces into chunks up to *chunk_size*, applying
       *chunk_overlap* by re-including the tail of the previous chunk.

    Args:
        text:          Text to split.
        separators:    Ordered list of regex separator patterns to try.
        chunk_size:    Maximum character length of each chunk.
        chunk_overlap: Number of characters to repeat at the start of each
                       new chunk (must be < *chunk_size*).

    Returns:
        List of non-empty text chunks, each ≤ *chunk_size* characters
        (except when a single indivisible token exceeds *chunk_size*, in
        which case it is kept as-is rather than truncated).
    """
    if not text.strip():
        return []

    # If the whole text fits, nothing to do.
    if len(text) <= chunk_size:
        return [text]

    if not separators:
        # No more separators: yield fixed-size slices with overlap.
        slices: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            slices.append(text[start:end])
            if end == len(text):
                break
            start += chunk_size - chunk_overlap
        return slices

    separator = separators[0]
    remaining_seps = separators[1:]

    # Split using the current separator (keep separator at end of left piece).
    pieces = re.split(f"({separator})", text)
    # Re-attach the separator to the preceding piece.
    merged_pieces: list[str] = []
    i = 0
    while i < len(pieces):
        if i + 1 < len(pieces) and re.fullmatch(separator, pieces[i + 1]):
            merged_pieces.append(pieces[i] + pieces[i + 1])
            i += 2
        else:
            if pieces[i]:
                merged_pieces.append(pieces[i])
            i += 1

    # Recursively process pieces that are still too large.
    fine_pieces: list[str] = []
    for piece in merged_pieces:
        if len(piece) > chunk_size:
            fine_pieces.extend(
                _split_on_separators(piece, remaining_seps, chunk_size, chunk_overlap)
            )
        else:
            fine_pieces.append(piece)

    return _merge_pieces_into_chunks(fine_pieces, chunk_size, chunk_overlap)


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------

# Ordered from coarsest to finest — mirrors LangChain's default separators
# for Markdown/plain text.
_DEFAULT_SEPARATORS: list[str] = [
    r"\n\n",      # paragraph / section break
    r"\n",        # line break
    r"(?<=\.) ",  # sentence boundary (after period + space)
    r" ",         # word boundary
    r"",          # character-by-character fallback
]


class RecursiveCharacterTextSplitter:
    """Split documents into overlapping text chunks.

    Implements the same recursive strategy as LangChain's
    ``RecursiveCharacterTextSplitter``, without the LangChain dependency.
    Defaults are read from :mod:`app.config` so they stay in sync with the
    ``CHUNK_SIZE`` / ``CHUNK_OVERLAP`` environment variables.

    Args:
        chunk_size:    Maximum character length per chunk.
                       Defaults to ``settings.chunk_size`` (800).
        chunk_overlap: Characters of overlap between consecutive chunks.
                       Defaults to ``settings.chunk_overlap`` (120).
        separators:    Ordered list of regex patterns tried from coarsest
                       to finest.  Defaults to paragraph → line → sentence
                       → word → character.

    Example::

        splitter = RecursiveCharacterTextSplitter()
        doc = Document(content="Long text...", metadata={"source": "x.md"})
        chunks = splitter.split_document(doc)
        for chunk in chunks:
            print(chunk.metadata["chunk_id"], len(chunk.content))
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size: int = chunk_size if chunk_size is not None else settings.chunk_size
        self.chunk_overlap: int = (
            chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        )
        self.separators: list[str] = separators if separators is not None else _DEFAULT_SEPARATORS

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )

    def split_document(self, document: Document) -> list[Chunk]:
        """Split a single ``Document`` into ``Chunk`` objects.

        Each chunk inherits the document's metadata and gains three extra
        fields:

        * ``chunk_id``    – deterministic 16-char hex ID (SHA-256 of source + index)
        * ``chunk_index`` – zero-based position within this document
        * ``chunk_total`` – total number of chunks produced from this document

        Args:
            document: The document to split.

        Returns:
            Ordered list of ``Chunk`` objects.  Empty documents return ``[]``.
        """
        raw_chunks = _split_on_separators(
            document.content,
            self.separators,
            self.chunk_size,
            self.chunk_overlap,
        )

        source: str = document.metadata.get("source") or ""  # type: ignore[assignment]
        total = len(raw_chunks)

        chunks: list[Chunk] = []
        for idx, text in enumerate(raw_chunks):
            text = text.strip()
            if not text:
                continue
            chunk_metadata = dict(document.metadata)
            chunk_metadata["chunk_id"] = _make_chunk_id(source, idx)
            chunk_metadata["chunk_index"] = idx
            chunk_metadata["chunk_total"] = total
            chunks.append(Chunk(content=text, metadata=chunk_metadata))

        return chunks

    def split_documents(self, documents: Sequence[Document]) -> list[Chunk]:
        """Split multiple ``Document`` objects into ``Chunk`` objects.

        Args:
            documents: Iterable of ``Document`` objects.

        Returns:
            Flat list of all ``Chunk`` objects across all documents,
            preserving document order.
        """
        all_chunks: list[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.split_document(doc))
        return all_chunks
