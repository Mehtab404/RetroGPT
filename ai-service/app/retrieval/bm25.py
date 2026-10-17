"""Lightweight BM25 sparse-vector encoder for hybrid retrieval (Issue #20).

Qdrant 1.7 has no server-side BM25, so sparse vectors are computed
client-side:

* At **ingestion** time, :meth:`Bm25Encoder.fit` is fitted over all chunk
  texts, producing IDF weights, a deterministic term → index vocabulary, and
  corpus length statistics.
* The fitted encoder is persisted to JSON (``BM25_STATS_PATH``) so the query
  path can encode questions with *exactly* the same statistics.
* :meth:`Bm25Encoder.encode` produces a :class:`~qdrant_client.http.models.SparseVector`
  of ``(token index, BM25 term weight)`` pairs, which Qdrant stores and scores
  by dot product.

The implementation follows the classic BM25 formula (``k1`` / ``b``
parameters)::

    weight(term, doc) = idf(term) * tf * (k1 + 1)
                        / (tf + k1 * (1 - b + b * dl / avgdl))
    idf(term)         = ln(1 + (N - df + 0.5) / (df + 0.5))

Tokenization is intentionally simple and deterministic
(lowercased ``[a-z0-9]+`` runs) — the game manuals are plain technical
English, so stemming/stop-words add cost without measurable benefit.

Typical usage::

    encoder = Bm25Encoder().fit([chunk.content for chunk in chunks])
    encoder.save(Path("/data/bm25_stats.json"))
    ...
    encoder = Bm25Encoder.load(Path("/data/bm25_stats.json"))
    sparse = encoder.encode("How many points is a large asteroid?")
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from qdrant_client.http import models as qdrant_models

logger = logging.getLogger(__name__)

#: Current persisted-stats format version.
STATS_VERSION = 1

#: Token pattern — lowercased alphanumeric runs.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split *text* into alphanumeric tokens."""
    return _TOKEN_RE.findall(text.lower())


class Bm25Encoder:
    """Fit/encode BM25 sparse vectors with persisted corpus statistics.

    Args:
        k1: Term-frequency saturation parameter (default 1.2).
        b:  Length-normalization parameter (default 0.75).

    Example::

        encoder = Bm25Encoder().fit(["a manual", "another doc"])
        vector = encoder.encode("manual")
        assert vector.indices and vector.values
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._idf: dict[str, float] = {}
        self._index: dict[str, int] = {}
        self._avgdl: float = 0.0
        self._n_docs: int = 0

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """True once :meth:`fit` has seen at least one document."""
        return self._n_docs > 0

    @property
    def vocabulary_size(self) -> int:
        """Number of indexed terms."""
        return len(self._index)

    @property
    def n_docs(self) -> int:
        """Number of documents the encoder was fitted on."""
        return self._n_docs

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, corpus: Sequence[str]) -> Bm25Encoder:
        """Fit IDF weights and vocabulary over *corpus*.

        Args:
            corpus: One text per document (here: per chunk).

        Returns:
            ``self``, to allow chaining.

        Raises:
            ValueError: If *corpus* is empty.
        """
        doc_count = len(corpus)
        if doc_count == 0:
            raise ValueError("cannot fit BM25 on an empty corpus")

        document_freq: Counter[str] = Counter()
        total_length = 0
        for text in corpus:
            tokens = tokenize(text)
            total_length += len(tokens)
            document_freq.update(set(tokens))

        self._n_docs = doc_count
        self._avgdl = total_length / doc_count if total_length else 1.0
        # Deterministic vocabulary: sorted term → contiguous index.
        self._index = {term: i for i, term in enumerate(sorted(document_freq))}
        self._idf = {
            term: math.log(1.0 + (doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in document_freq.items()
        }
        logger.debug(
            "Fitted BM25: %d docs, %d terms, avgdl=%.1f.",
            self._n_docs,
            len(self._index),
            self._avgdl,
        )
        return self

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, text: str) -> qdrant_models.SparseVector:
        """Encode *text* as a BM25 sparse vector.

        Terms not seen during fitting are skipped. Returns an empty vector
        (no indices) when no term matches.

        Raises:
            ValueError: If the encoder has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError("BM25 encoder must be fitted before calling encode()")

        tokens = tokenize(text)
        if not tokens:
            return qdrant_models.SparseVector(indices=[], values=[])

        term_freq = Counter(tokens)
        doc_length = len(tokens)
        indices: list[int] = []
        values: list[float] = []
        for term, freq in sorted(term_freq.items()):
            term_index = self._index.get(term)
            idf = self._idf.get(term)
            if term_index is None or idf is None or idf <= 0.0:
                continue
            denominator = freq + self.k1 * (1.0 - self.b + self.b * doc_length / self._avgdl)
            indices.append(term_index)
            values.append(idf * (freq * (self.k1 + 1.0)) / denominator)

        return qdrant_models.SparseVector(indices=indices, values=values)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the fitted statistics to a JSON-compatible dict."""
        return {
            "version": STATS_VERSION,
            "k1": self.k1,
            "b": self.b,
            "n_docs": self._n_docs,
            "avgdl": self._avgdl,
            "idf": self._idf,
            "index": self._index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Bm25Encoder:
        """Reconstruct an encoder from :meth:`to_dict` output.

        Raises:
            ValueError: If the payload is not a valid stats dict.
        """
        version = data.get("version")
        if version != STATS_VERSION:
            raise ValueError(
                f"unsupported BM25 stats version: {version!r} (expected {STATS_VERSION})"
            )
        encoder = cls(k1=float(data["k1"]), b=float(data["b"]))
        encoder._n_docs = int(data["n_docs"])
        encoder._avgdl = float(data["avgdl"])
        encoder._idf = {str(term): float(value) for term, value in data["idf"].items()}
        encoder._index = {str(term): int(value) for term, value in data["index"].items()}
        if not encoder._index or encoder._n_docs <= 0:
            raise ValueError("BM25 stats are empty")
        return encoder

    def save(self, path: Path) -> None:
        """Persist statistics to *path* (creating parent directories)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.debug("Saved BM25 stats to '%s'.", path)

    @classmethod
    def load(cls, path: Path) -> Bm25Encoder:
        """Load statistics from *path*.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError:         If the file contents are invalid.
        """
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"BM25 stats at '{path}' must be a JSON object")
        return cls.from_dict(data)
