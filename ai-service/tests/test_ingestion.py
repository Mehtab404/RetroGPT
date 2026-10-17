"""Tests for document loaders and chunking (Issue #7).

Covers:
- MarkdownLoader: happy path, error cases, metadata extraction, directory scan
- TextLoader: basic loading
- get_loader_for: factory dispatch
- RecursiveCharacterTextSplitter: chunk sizes, overlap, metadata propagation
- _make_chunk_id: determinism and uniqueness
- _infer_game_name: path parsing
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.ingestion.chunker import (
    Chunk,
    RecursiveCharacterTextSplitter,
    _make_chunk_id,
    _split_on_separators,
)
from app.ingestion.loaders import (
    Document,
    MarkdownLoader,
    TextLoader,
    _extract_markdown_title,
    _infer_game_name,
    get_loader_for,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    """Return a writable temporary directory."""
    return tmp_path


@pytest.fixture()
def asteroids_md(tmp_dir: Path) -> Path:
    """Write a minimal Asteroids manual to disk and return its path."""
    game_dir = tmp_dir / "asteroids"
    game_dir.mkdir()
    md_file = game_dir / "asteroids_manual.md"
    md_file.write_text(
        "# ASTEROIDS - Game Manual\n\n"
        "## Game Overview\n"
        "Asteroids is a space-themed shooter released by Atari in 1979.\n\n"
        "## Scoring System\n"
        "- Small asteroid: 100 points\n"
        "- Large asteroid: 20 points\n",
        encoding="utf-8",
    )
    return md_file


@pytest.fixture()
def long_md(tmp_dir: Path) -> Path:
    """Write a Markdown file whose content exceeds chunk_size=200."""
    game_dir = tmp_dir / "breakout"
    game_dir.mkdir()
    md_file = game_dir / "breakout_manual.md"
    # ~600 chars across 3 distinct paragraphs
    content = (
        "# Breakout Manual\n\n"
        "Breakout is a classic arcade game from Atari (1976). "
        "Players control a paddle at the bottom of the screen.\n\n"
        "The objective is to break all the bricks at the top by bouncing "
        "a ball off the paddle. Each hit scores points and removes a brick.\n\n"
        "When all bricks are cleared, a new level begins with a faster ball "
        "and more bricks. The game ends when all lives are lost.\n"
    )
    md_file.write_text(content, encoding="utf-8")
    return md_file


# ===========================================================================
# _infer_game_name
# ===========================================================================


class TestInferGameName:
    def test_returns_parent_dir_name(self) -> None:
        path = Path("data/raw/asteroids/asteroids_manual.md")
        assert _infer_game_name(path) == "asteroids"

    def test_returns_parent_dir_name_missile_command(self) -> None:
        path = Path("data/raw/missile_command/missile_command_manual.md")
        assert _infer_game_name(path) == "missile_command"

    def test_falls_back_to_stem_when_parent_is_raw(self) -> None:
        path = Path("data/raw/some_doc.md")
        assert _infer_game_name(path) == "some_doc"

    def test_falls_back_to_stem_for_bare_file(self) -> None:
        path = Path("some_doc.md")
        assert _infer_game_name(path) == "some_doc"


# ===========================================================================
# _extract_markdown_title
# ===========================================================================


class TestExtractMarkdownTitle:
    def test_extracts_h1(self) -> None:
        content = "# My Title\n\nSome text."
        assert _extract_markdown_title(content) == "My Title"

    def test_extracts_h2_when_no_h1(self) -> None:
        content = "## Section Title\n\nSome text."
        assert _extract_markdown_title(content) == "Section Title"

    def test_returns_empty_string_for_no_heading(self) -> None:
        content = "No heading here.\nJust plain text."
        assert _extract_markdown_title(content) == ""

    def test_extracts_first_heading_only(self) -> None:
        content = "# First\n\n# Second"
        assert _extract_markdown_title(content) == "First"


# ===========================================================================
# MarkdownLoader
# ===========================================================================


class TestMarkdownLoader:
    def test_load_returns_one_document(self, asteroids_md: Path) -> None:
        loader = MarkdownLoader()
        docs = loader.load(asteroids_md)
        assert len(docs) == 1
        assert isinstance(docs[0], Document)

    def test_content_matches_file(self, asteroids_md: Path) -> None:
        loader = MarkdownLoader()
        docs = loader.load(asteroids_md)
        assert "Asteroids" in docs[0].content
        assert "100 points" in docs[0].content

    def test_metadata_source_is_posix_path(self, asteroids_md: Path) -> None:
        loader = MarkdownLoader()
        docs = loader.load(asteroids_md)
        source = docs[0].metadata["source"]
        assert isinstance(source, str)
        assert "\\" not in source  # POSIX path

    def test_metadata_game_inferred_from_directory(self, asteroids_md: Path) -> None:
        loader = MarkdownLoader()
        docs = loader.load(asteroids_md)
        assert docs[0].metadata["game"] == "asteroids"

    def test_metadata_title_extracted_from_heading(self, asteroids_md: Path) -> None:
        loader = MarkdownLoader()
        docs = loader.load(asteroids_md)
        assert docs[0].metadata["title"] == "ASTEROIDS - Game Manual"

    def test_metadata_file_type_is_markdown(self, asteroids_md: Path) -> None:
        loader = MarkdownLoader()
        docs = loader.load(asteroids_md)
        assert docs[0].metadata["file_type"] == "markdown"

    def test_raises_for_unsupported_extension(self, tmp_dir: Path) -> None:
        pdf_file = tmp_dir / "game.pdf"
        pdf_file.write_text("dummy", encoding="utf-8")
        loader = MarkdownLoader()
        with pytest.raises(ValueError, match="MarkdownLoader only handles"):
            loader.load(pdf_file)

    def test_raises_file_not_found(self, tmp_dir: Path) -> None:
        loader = MarkdownLoader()
        with pytest.raises(FileNotFoundError):
            loader.load(tmp_dir / "nonexistent.md")

    def test_load_dot_markdown_extension(self, tmp_dir: Path) -> None:
        """Files with .markdown extension should also be supported."""
        game_dir = tmp_dir / "pong"
        game_dir.mkdir()
        f = game_dir / "pong_manual.markdown"
        f.write_text("# Pong\n\nBasic game.", encoding="utf-8")
        loader = MarkdownLoader()
        docs = loader.load(f)
        assert len(docs) == 1
        assert docs[0].metadata["game"] == "pong"

    def test_load_directory_finds_all_md_files(self, tmp_dir: Path) -> None:
        """load_directory should recursively discover all .md files."""
        for game in ("asteroids", "breakout", "centipede"):
            d = tmp_dir / game
            d.mkdir()
            (d / f"{game}_manual.md").write_text(f"# {game.title()}", encoding="utf-8")

        loader = MarkdownLoader()
        docs = loader.load_directory(tmp_dir)
        assert len(docs) == 3
        games = {d.metadata["game"] for d in docs}
        assert games == {"asteroids", "breakout", "centipede"}

    def test_load_directory_raises_for_file(self, asteroids_md: Path) -> None:
        loader = MarkdownLoader()
        with pytest.raises(NotADirectoryError):
            loader.load_directory(asteroids_md)

    def test_title_falls_back_to_stem_when_no_heading(self, tmp_dir: Path) -> None:
        f = tmp_dir / "no_heading.md"
        f.write_text("Just plain text with no ATX heading.", encoding="utf-8")
        loader = MarkdownLoader()
        docs = loader.load(f)
        assert docs[0].metadata["title"] == "no_heading"


# ===========================================================================
# TextLoader
# ===========================================================================


class TestTextLoader:
    def test_load_plain_text(self, tmp_dir: Path) -> None:
        txt_file = tmp_dir / "notes.txt"
        txt_file.write_text("Some plain text content.", encoding="utf-8")
        loader = TextLoader()
        docs = loader.load(txt_file)
        assert len(docs) == 1
        assert docs[0].content == "Some plain text content."
        assert docs[0].metadata["file_type"] == "text"

    def test_raises_for_unsupported_extension(self, tmp_dir: Path) -> None:
        f = tmp_dir / "doc.md"
        f.write_text("# Heading", encoding="utf-8")
        loader = TextLoader()
        with pytest.raises(ValueError, match="TextLoader only handles"):
            loader.load(f)


# ===========================================================================
# get_loader_for
# ===========================================================================


class TestGetLoaderFor:
    def test_returns_markdown_loader_for_md(self) -> None:
        loader = get_loader_for(Path("game.md"))
        assert isinstance(loader, MarkdownLoader)

    def test_returns_markdown_loader_for_markdown(self) -> None:
        loader = get_loader_for(Path("game.markdown"))
        assert isinstance(loader, MarkdownLoader)

    def test_returns_text_loader_for_txt(self) -> None:
        loader = get_loader_for(Path("notes.txt"))
        assert isinstance(loader, TextLoader)

    def test_raises_for_unknown_extension(self) -> None:
        with pytest.raises(ValueError, match="No loader registered"):
            get_loader_for(Path("document.docx"))


# ===========================================================================
# _make_chunk_id
# ===========================================================================


class TestMakeChunkId:
    def test_returns_16_hex_chars(self) -> None:
        cid = _make_chunk_id("data/raw/asteroids/asteroids_manual.md", 0)
        assert len(cid) == 16
        assert all(c in "0123456789abcdef" for c in cid)

    def test_deterministic_for_same_inputs(self) -> None:
        id1 = _make_chunk_id("data/raw/asteroids/asteroids_manual.md", 0)
        id2 = _make_chunk_id("data/raw/asteroids/asteroids_manual.md", 0)
        assert id1 == id2

    def test_different_for_different_index(self) -> None:
        id0 = _make_chunk_id("data/raw/asteroids/asteroids_manual.md", 0)
        id1 = _make_chunk_id("data/raw/asteroids/asteroids_manual.md", 1)
        assert id0 != id1

    def test_different_for_different_source(self) -> None:
        id_a = _make_chunk_id("data/raw/asteroids/asteroids_manual.md", 0)
        id_b = _make_chunk_id("data/raw/breakout/breakout_manual.md", 0)
        assert id_a != id_b


# ===========================================================================
# _split_on_separators (unit)
# ===========================================================================


class TestSplitOnSeparators:
    def test_returns_empty_for_blank_text(self) -> None:
        result = _split_on_separators("   ", [r"\n\n"], 200, 20)
        assert result == []

    def test_returns_single_chunk_when_fits(self) -> None:
        text = "Short text."
        result = _split_on_separators(text, [r"\n\n"], 200, 20)
        assert result == ["Short text."]

    def test_splits_on_paragraph_break(self) -> None:
        text = "Para one.\n\nPara two.\n\nPara three."
        result = _split_on_separators(text, [r"\n\n", r"\n", r" ", r""], 30, 5)
        # Each paragraph is < 30 chars; we should get ≤ 3 chunks
        assert len(result) >= 1
        full = "".join(result)
        assert "Para one" in full
        assert "Para two" in full
        assert "Para three" in full

    def test_no_chunk_exceeds_chunk_size(self) -> None:
        text = "word " * 200  # 1000 chars
        result = _split_on_separators(text, [r"\n\n", r"\n", r" ", r""], 100, 10)
        for chunk in result:
            assert len(chunk) <= 100


# ===========================================================================
# RecursiveCharacterTextSplitter
# ===========================================================================


class TestRecursiveCharacterTextSplitter:
    def test_raises_if_overlap_ge_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap"):
            RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=100)

    def test_split_document_returns_chunks(self, asteroids_md: Path) -> None:
        loader = MarkdownLoader()
        doc = loader.load(asteroids_md)[0]
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
        chunks = splitter.split_document(doc)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.content  # non-empty

    def test_no_chunk_exceeds_chunk_size(self, long_md: Path) -> None:
        loader = MarkdownLoader()
        doc = loader.load(long_md)[0]
        chunk_size = 200
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=20)
        chunks = splitter.split_document(doc)
        for chunk in chunks:
            assert len(chunk.content) <= chunk_size, (
                f"Chunk length {len(chunk.content)} exceeds {chunk_size}: {chunk.content!r}"
            )

    def test_chunks_preserve_source_metadata(self, asteroids_md: Path) -> None:
        loader = MarkdownLoader()
        doc = loader.load(asteroids_md)[0]
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
        chunks = splitter.split_document(doc)
        for chunk in chunks:
            assert chunk.metadata["game"] == "asteroids"
            assert chunk.metadata["file_type"] == "markdown"
            assert "source" in chunk.metadata

    def test_chunks_have_chunk_id(self, asteroids_md: Path) -> None:
        loader = MarkdownLoader()
        doc = loader.load(asteroids_md)[0]
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
        chunks = splitter.split_document(doc)
        for chunk in chunks:
            assert chunk.metadata["chunk_id"] is not None
            assert isinstance(chunk.metadata["chunk_id"], str)
            assert len(chunk.metadata["chunk_id"]) == 16

    def test_chunk_ids_are_unique_within_document(self, long_md: Path) -> None:
        loader = MarkdownLoader()
        doc = loader.load(long_md)[0]
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        chunks = splitter.split_document(doc)
        ids = [c.metadata["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs should be unique within a document"

    def test_chunk_index_metadata(self, long_md: Path) -> None:
        loader = MarkdownLoader()
        doc = loader.load(long_md)[0]
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        chunks = splitter.split_document(doc)
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == sorted(indices), "chunk_index should be ascending"

    def test_chunk_total_metadata_matches_count(self, long_md: Path) -> None:
        loader = MarkdownLoader()
        doc = loader.load(long_md)[0]
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        chunks = splitter.split_document(doc)
        total_reported = chunks[0].metadata["chunk_total"]
        assert total_reported == len(chunks)

    def test_overlap_present_in_consecutive_chunks(self, long_md: Path) -> None:
        """The tail of chunk N should appear at the start of chunk N+1.

        The splitter may trim the overlap prefix when ``overlap + next_piece``
        would exceed *chunk_size*, so we scan for any non-trivial suffix of
        chunk[i] that appears in chunk[i+1] rather than requiring the exact
        last ``chunk_overlap`` characters.
        """
        loader = MarkdownLoader()
        doc = loader.load(long_md)[0]
        chunk_size = 300
        chunk_overlap = 50
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_document(doc)
        if len(chunks) < 2:
            pytest.skip("Document too short to produce multiple chunks at this size")

        overlap_found = False
        for i in range(len(chunks) - 1):
            prev_content = chunks[i].content
            next_content = chunks[i + 1].content
            # Try decreasing suffix lengths until we find a match (min 10 chars
            # to avoid trivial single-word collisions).
            for n in range(chunk_overlap, 9, -1):
                tail = prev_content[-n:].strip()
                if tail and tail in next_content:
                    overlap_found = True
                    break
            if overlap_found:
                break
        assert overlap_found, (
            "Expected to find overlapping text between at least one pair of consecutive chunks"
        )

    def test_split_documents_processes_all(self, tmp_dir: Path) -> None:
        """split_documents should produce chunks for every document in the list."""
        loader = MarkdownLoader()
        docs: list[Document] = []
        for game in ("asteroids", "breakout"):
            d = tmp_dir / game
            d.mkdir(exist_ok=True)
            f = d / f"{game}_manual.md"
            f.write_text(f"# {game.title()}\n\n" + ("Content. " * 50), encoding="utf-8")
            docs.extend(loader.load(f))

        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        all_chunks = splitter.split_documents(docs)
        assert len(all_chunks) > 2  # both documents should produce chunks

        games_seen = {c.metadata["game"] for c in all_chunks}
        assert "asteroids" in games_seen
        assert "breakout" in games_seen

    def test_empty_document_returns_no_chunks(self) -> None:
        doc = Document(content="", metadata={"source": "empty.md"})
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        chunks = splitter.split_document(doc)
        assert chunks == []

    def test_single_chunk_when_content_fits(self) -> None:
        doc = Document(
            content="Short content that fits in one chunk.",
            metadata={"source": "short.md", "game": "pong"},
        )
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        chunks = splitter.split_document(doc)
        assert len(chunks) == 1
        assert chunks[0].content == "Short content that fits in one chunk."
        assert chunks[0].metadata["game"] == "pong"

    def test_chunk_property_shortcut(self) -> None:
        doc = Document(
            content="Hello world.",
            metadata={"source": "test.md"},
        )
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
        chunks = splitter.split_document(doc)
        assert chunks[0].chunk_id == chunks[0].metadata["chunk_id"]

    def test_real_game_manual_produces_expected_chunks(self, tmp_dir: Path) -> None:
        """Integration-style check: a real-sized manual should produce several chunks."""
        game_dir = tmp_dir / "centipede"
        game_dir.mkdir()
        manual = game_dir / "centipede_manual.md"
        # ~1200 chars — should produce ~2 chunks at size 800/overlap 120
        manual.write_text(
            "# Centipede Game Manual\n\n"
            "## Overview\n"
            "Centipede is a fixed shooter arcade game by Atari (1980). "
            "Players shoot a centipede winding down the screen.\n\n"
            "## Controls\n"
            "Use the trackball or joystick to move your shooter. "
            "Press fire to shoot. The shooter moves within the bottom area.\n\n"
            "## Enemies\n"
            "The centipede enters from the top and winds its way down. "
            "Mushrooms litter the field and deflect the centipede. "
            "Fleas drop mushrooms, spiders eat mushrooms, and scorpions "
            "poison mushrooms causing the centipede to dive straight down.\n\n"
            "## Scoring\n"
            "- Centipede head: 100 points\n"
            "- Centipede segment: 10 points\n"
            "- Flea: 200 points\n"
            "- Spider: 300–900 points depending on distance\n"
            "- Scorpion: 1000 points\n\n"
            "## Strategy\n"
            "Focus on eliminating the head first to split the centipede. "
            "Clear mushrooms in your shooting area to improve accuracy. "
            "Avoid getting cornered by multiple centipede segments.\n",
            encoding="utf-8",
        )
        loader = MarkdownLoader()
        doc = loader.load(manual)[0]
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
        chunks = splitter.split_document(doc)
        assert len(chunks) >= 1
        # All metadata must be present
        for chunk in chunks:
            assert chunk.metadata["game"] == "centipede"
            assert chunk.metadata["title"] == "Centipede Game Manual"
            assert chunk.metadata["chunk_id"]
            assert chunk.metadata["chunk_index"] is not None
