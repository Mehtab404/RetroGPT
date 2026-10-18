"""Tests for citation-marker normalization (full-width bracket handling).

Covers:

- ``normalize_citations``: ``【N】`` / ``［N］`` and mixed bracket pairs are
  rewritten to ASCII ``[N]``; ASCII markers and non-marker text are
  untouched; malformed groups (no digits, no closer, over-long digit runs)
  pass through unchanged.
- ``CitationStreamNormalizer``: markers split across chunk boundaries are
  normalized; held text is released by ``flush``; the concatenated output
  always equals the one-shot normalization of the concatenated input.

Pure unit tests — no network or service dependencies.
"""

from __future__ import annotations

from app.generation.citations import CitationStreamNormalizer, normalize_citations


class TestNormalizeCitations:
    def test_ascii_markers_pass_through_unchanged(self) -> None:
        text = "Large asteroids are worth 20 points [1]."
        assert normalize_citations(text) == text

    def test_cjk_lenticular_brackets_become_ascii(self) -> None:
        assert (
            normalize_citations("A power pellet is worth 50 points 【2】.")
            == "A power pellet is worth 50 points [2]."
        )

    def test_full_width_square_brackets_become_ascii(self) -> None:
        assert normalize_citations("Ghosts flee for 6-10 seconds ［1］.") == (
            "Ghosts flee for 6-10 seconds [1]."
        )

    def test_mixed_bracket_pairs_are_normalized(self) -> None:
        assert normalize_citations("A 【1] B ［2】 C [3］ D") == "A [1] B [2] C [3] D"

    def test_multiple_and_mixed_markers_in_one_answer(self) -> None:
        text = "Pellets are worth 50 points 【2】, ghosts give 200 【3] and dots [1]."
        assert normalize_citations(text) == (
            "Pellets are worth 50 points [2], ghosts give 200 [3] and dots [1]."
        )

    def test_non_marker_text_is_untouched(self) -> None:
        text = "Use 【abc】, [x], [] or 【12 notation — none are citations."
        assert normalize_citations(text) == text

    def test_index_zero_is_normalized_by_shape(self) -> None:
        # The server normalizes the bracket shape only; index validity is
        # enforced by consumers (e.g. the frontend citation parser).
        assert normalize_citations("Score 【0】.") == "Score [0]."

    def test_empty_string(self) -> None:
        assert normalize_citations("") == ""


class TestCitationStreamNormalizer:
    def test_marker_within_one_chunk(self) -> None:
        normalizer = CitationStreamNormalizer()
        assert normalizer.feed("Worth 50 points 【2】.") == "Worth 50 points [2]."
        assert normalizer.flush() == ""

    def test_marker_split_before_closer(self) -> None:
        normalizer = CitationStreamNormalizer()
        assert normalizer.feed("Worth 50 points 【2") == "Worth 50 points "
        assert normalizer.feed("】.") == "[2]."
        assert normalizer.flush() == ""

    def test_marker_split_across_three_chunks(self) -> None:
        normalizer = CitationStreamNormalizer()
        assert normalizer.feed("Worth 【") == "Worth "
        assert normalizer.feed("2") == ""
        assert normalizer.feed("】.") == "[2]."
        assert normalizer.flush() == ""

    def test_mixed_closer_variant_is_normalized(self) -> None:
        normalizer = CitationStreamNormalizer()
        assert normalizer.feed("See ［3") == "See "
        assert normalizer.feed("].") == "[3]."
        assert normalizer.flush() == ""

    def test_ascii_marker_split_across_chunks(self) -> None:
        normalizer = CitationStreamNormalizer()
        assert normalizer.feed("See [1") == "See "
        assert normalizer.feed("].") == "[1]."
        assert normalizer.flush() == ""

    def test_bare_opener_at_chunk_end_is_held_then_released(self) -> None:
        normalizer = CitationStreamNormalizer()
        assert normalizer.feed("array [") == "array "
        assert normalizer.feed("index]") == "[index]"
        assert normalizer.flush() == ""

    def test_flush_releases_incomplete_marker(self) -> None:
        normalizer = CitationStreamNormalizer()
        assert normalizer.feed("tail 【4") == "tail "
        assert normalizer.flush() == "【4"
        assert normalizer.flush() == ""

    def test_non_marker_full_width_text_passes_immediately(self) -> None:
        normalizer = CitationStreamNormalizer()
        assert normalizer.feed("【no digits】") == "【no digits】"
        assert normalizer.flush() == ""

    def test_overlong_digit_run_is_not_held(self) -> None:
        normalizer = CitationStreamNormalizer()
        chunk = "【1234567890"
        assert normalizer.feed(chunk) == chunk
        assert normalizer.flush() == ""

    def test_empty_chunks_are_ignored(self) -> None:
        normalizer = CitationStreamNormalizer()
        assert normalizer.feed("") == ""
        assert normalizer.feed("a") == "a"
        assert normalizer.flush() == ""

    def test_any_chunk_boundary_matches_one_shot_normalization(self) -> None:
        text = "Pellets 【2】, ghosts ［1］, dots [1] and stray 【text."
        expected = normalize_citations(text)
        for split in range(len(text) + 1):
            normalizer = CitationStreamNormalizer()
            out = normalizer.feed(text[:split]) + normalizer.feed(text[split:]) + normalizer.flush()
            assert out == expected, f"mismatch at split {split}"
