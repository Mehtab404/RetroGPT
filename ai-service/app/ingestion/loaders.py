"""Document loaders for ingesting game manuals and reference documents.

Supports Markdown, plain text, HTML, and PDF formats.  Each loader
returns a list of ``Document`` objects ready for downstream chunking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Document:
    """A piece of raw text with associated metadata.

    Attributes:
        content: The full text of the document.
        metadata: Arbitrary key/value pairs (source path, title, game, etc.).
    """

    content: str
    metadata: dict[str, str | int | float | None] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loader protocol (open for extension)
# ---------------------------------------------------------------------------


@runtime_checkable
class DocumentLoader(Protocol):
    """Interface that all concrete loaders must satisfy."""

    def load(self, path: Path) -> list[Document]:
        """Load *path* and return one or more ``Document`` objects."""
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _infer_game_name(path: Path) -> str:
    """Derive a human-readable game name from a file path.

    Walks up the path components looking for a folder name that looks like a
    game slug (e.g. ``asteroids``, ``missile_command``).  Falls back to the
    stem of the file itself.

    >>> _infer_game_name(Path("data/raw/asteroids/asteroids_manual.md"))
    'asteroids'
    >>> _infer_game_name(Path("some_doc.md"))
    'some_doc'
    """
    # The parent directory is usually the game slug (e.g. data/raw/asteroids/)
    parent_name = path.parent.name
    if parent_name and parent_name not in {"raw", "data", ".", ""}:
        return parent_name
    return path.stem


def _extract_markdown_title(content: str) -> str:
    """Return the first ATX heading in *content*, or an empty string."""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return ""


# ---------------------------------------------------------------------------
# Markdown loader
# ---------------------------------------------------------------------------


class MarkdownLoader:
    """Load Markdown (``*.md``) files and return a single ``Document``.

    Metadata attached to every document:

    * ``source``   – absolute POSIX path to the file
    * ``title``    – first H1/H2/… heading found in the file (or the file stem)
    * ``game``     – inferred game slug from the directory name
    * ``file_type``– ``"markdown"``

    Example::

        loader = MarkdownLoader()
        docs = loader.load(Path("data/raw/asteroids/asteroids_manual.md"))
        assert docs[0].metadata["game"] == "asteroids"
    """

    # File extensions this loader handles.
    SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".md", ".markdown"})

    def load(self, path: Path) -> list[Document]:
        """Read *path* and return a list containing one ``Document``.

        Args:
            path: Path to the Markdown file.

        Returns:
            A single-element list with the loaded ``Document``.

        Raises:
            ValueError: If *path* does not have a supported extension.
            FileNotFoundError: If *path* does not exist.
        """
        path = path.resolve()

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"MarkdownLoader only handles {self.SUPPORTED_EXTENSIONS}; "
                f"got '{path.suffix}' for file '{path}'"
            )
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        content = path.read_text(encoding="utf-8")

        title = _extract_markdown_title(content) or path.stem
        game = _infer_game_name(path)

        metadata: dict[str, str | int | float | None] = {
            "source": path.as_posix(),
            "title": title,
            "game": game,
            "file_type": "markdown",
        }

        return [Document(content=content, metadata=metadata)]

    def load_directory(self, directory: Path) -> list[Document]:
        """Recursively load all Markdown files under *directory*.

        Args:
            directory: Root directory to scan.

        Returns:
            All ``Document`` objects found, in filesystem traversal order.

        Raises:
            NotADirectoryError: If *directory* is not a directory.
        """
        directory = directory.resolve()
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        documents: list[Document] = []
        for ext in self.SUPPORTED_EXTENSIONS:
            for md_file in sorted(directory.rglob(f"*{ext}")):
                documents.extend(self.load(md_file))
        return documents


# ---------------------------------------------------------------------------
# Plain-text loader (bonus, keeps things DRY for future use)
# ---------------------------------------------------------------------------


class TextLoader:
    """Load plain-text files (``*.txt``) as ``Document`` objects.

    Metadata fields are identical to :class:`MarkdownLoader` except
    ``file_type`` is ``"text"``.
    """

    SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".txt"})

    def load(self, path: Path) -> list[Document]:
        """Read *path* and return a list containing one ``Document``.

        Args:
            path: Path to the text file.

        Returns:
            A single-element list with the loaded ``Document``.

        Raises:
            ValueError: If *path* has an unsupported extension.
            FileNotFoundError: If *path* does not exist.
        """
        path = path.resolve()

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"TextLoader only handles {self.SUPPORTED_EXTENSIONS}; "
                f"got '{path.suffix}'"
            )
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        content = path.read_text(encoding="utf-8")
        title = path.stem
        game = _infer_game_name(path)

        metadata: dict[str, str | int | float | None] = {
            "source": path.as_posix(),
            "title": title,
            "game": game,
            "file_type": "text",
        }

        return [Document(content=content, metadata=metadata)]


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def get_loader_for(path: Path) -> DocumentLoader:
    """Return the appropriate loader for *path* based on its extension.

    Args:
        path: Path whose extension determines the loader type.

    Returns:
        A loader instance.

    Raises:
        ValueError: If no loader is registered for the file extension.
    """
    ext = path.suffix.lower()
    if ext in MarkdownLoader.SUPPORTED_EXTENSIONS:
        return MarkdownLoader()
    if ext in TextLoader.SUPPORTED_EXTENSIONS:
        return TextLoader()
    raise ValueError(
        f"No loader registered for extension '{ext}'. "
        f"Supported: {MarkdownLoader.SUPPORTED_EXTENSIONS | TextLoader.SUPPORTED_EXTENSIONS}"
    )
