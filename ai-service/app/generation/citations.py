"""Citation-marker normalization for generated answers.

LLMs occasionally emit citation markers with full-width (CJK) brackets —
``【2】`` (U+3010/U+3011) or ``［2］`` (U+FF3B/U+FF3D) — or mixed pairs such
as ``【2]``.  This was observed in production with ``openai/gpt-oss-120b``
on Groq.  The wire contract, the frontend citation renderer and the
evaluation harness all expect ASCII ``[N]`` markers, so every answer path
(``POST /query``, ``POST /query/stream`` and ``POST /agent``) normalizes the
generated text through this module.

Provides:

* :func:`normalize_citations` — one-shot normalization of a complete
  answer.  Only the bracket characters are rewritten; digits and all other
  text are preserved, and ASCII markers pass through unchanged.
* :class:`CitationStreamNormalizer` — chunk-safe normalization for token
  streams.  A potential marker at the end of a chunk (e.g. ``"… 【2"``) is
  held back until its closer arrives, so markers split across chunks are
  normalized too.  :meth:`CitationStreamNormalizer.flush` releases any held
  text at the end of the stream (incomplete markers stay as-is).

Usage::

    from app.generation.citations import (
        CitationStreamNormalizer,
        normalize_citations,
    )

    answer = normalize_citations(await llm.generate(prompt))

    normalizer = CitationStreamNormalizer()
    async for chunk in llm.generate_stream(prompt):
        text = normalizer.feed(chunk)
        if text:
            emit(text)
    tail = normalizer.flush()
    if tail:
        emit(tail)
"""

from __future__ import annotations

import re

#: Opening brackets accepted around a citation index: ASCII ``[`` (U+005B),
#: CJK lenticular ``【`` (U+3010) and full-width square ``［`` (U+FF3B).
#: Closing brackets may come from any family, so mixed pairs also match.
_OPENERS = "[【［"
_CLOSERS = "]】］"

#: ASCII digits only — mirrors the JavaScript ``\d`` used by the frontend
#: parser, so both implementations accept exactly the same markers.
_DIGITS = "0123456789"

#: Longest digit run still treated as a potential citation index while
#: streaming.  Real answers cite small passage numbers; this only bounds how
#: much trailing text a chunk boundary can hold back.
_MAX_DIGITS = 6

_CITATION_RE = re.compile(r"[\[【［]([0-9]+)[\]】］]")


def normalize_citations(text: str) -> str:
    """Rewrite full-width citation markers (``【N】`` / ``［N］``) to ``[N]``.

    Any opener may combine with any closer (mixed pairs included).  ASCII
    markers and all non-marker text are returned unchanged.  Index validity
    is a consumer concern — this function only normalizes the bracket shape.

    Args:
        text: The generated answer text.

    Returns:
        The answer with every citation marker in ASCII ``[N]`` form.
    """
    return _CITATION_RE.sub(r"[\g<1>]", text)


class CitationStreamNormalizer:
    """Chunk-safe :func:`normalize_citations` for streamed token chunks.

    Feed chunks in order and emit the returned text; call :meth:`flush` after
    the last chunk.  The concatenation of all returned strings equals
    :func:`normalize_citations` applied to the concatenated input, even when
    a marker is split across chunk boundaries (holds are bounded to an opener
    plus at most ``_MAX_DIGITS`` digits; over-long digit runs are treated as
    plain text).
    """

    def __init__(self) -> None:
        self._pending = ""

    def feed(self, chunk: str) -> str:
        """Consume the next chunk; return the safely-normalized prefix.

        The return value may be empty when the entire chunk is a prefix of a
        possible marker and is being held until more text arrives.
        """
        if not chunk:
            return ""
        buf = self._pending + chunk
        self._pending = ""
        out: list[str] = []
        i = 0
        n = len(buf)
        while i < n:
            char = buf[i]
            if char not in _OPENERS:
                out.append(char)
                i += 1
                continue
            j = i + 1
            while j < n and buf[j] in _DIGITS:
                j += 1
            digits = buf[i + 1 : j]
            if j == n:
                # The marker may still complete in a later chunk: hold the
                # tail (bounded: opener + at most _MAX_DIGITS digits).
                if len(digits) <= _MAX_DIGITS:
                    self._pending = buf[i:]
                    break
                out.append(char)
                i += 1
                continue
            if digits and len(digits) <= _MAX_DIGITS and buf[j] in _CLOSERS:
                out.append(f"[{digits}]")
                i = j + 1
                continue
            out.append(char)
            i += 1
        return "".join(out)

    def flush(self) -> str:
        """Release text held back at the end of the stream.

        Incomplete markers (no closing bracket yet) are returned as-is.
        """
        tail = self._pending
        self._pending = ""
        return tail
