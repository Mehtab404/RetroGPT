"""Tests for ``eval/retrieval_eval.py`` (Issues #22 and #23).

Only the app-free parts are unit-tested here (hashing embedder, sweep grid,
comparison rendering, guard wiring); the in-process retrieval run itself is
exercised against a real scratch Qdrant collection by the CI job.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

_EVAL_DIR = Path(__file__).resolve().parents[1]
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

import retrieval_eval  # noqa: E402  (import after sys.path injection)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def cosine(left: list[float], right: list[float]) -> float:
    """Cosine similarity; the embedder returns unit vectors so this is a dot product."""
    return sum(a * b for a, b in zip(left, right, strict=True))


def embed(text: str, **kwargs: object) -> list[float]:
    return asyncio.run(retrieval_eval.HashingEmbedding(**kwargs).embed_query(text))


def make_summary(
    chunk_size: int,
    chunk_overlap: int,
    hit_rate: float | None,
    mrr: float | None = None,
    chunks: int = 10,
) -> dict:
    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunks": chunks,
        "metrics": {
            "hit_rate": hit_rate,
            "mrr": mrr,
            "latency_ms": {"mean": 12.0, "p50": 10.0, "p95": 25.0},
        },
    }


# ---------------------------------------------------------------------------
# HashingEmbedding
# ---------------------------------------------------------------------------


class TestHashingEmbedding:
    def test_dimension_matches_configured_dim(self) -> None:
        assert len(embed("asteroids scoring 20 points", dim=64)) == 64

    def test_vectors_are_unit_length(self) -> None:
        vector = embed("extra lives are awarded every 10000 points")
        norm = sum(component * component for component in vector) ** 0.5
        assert norm == pytest.approx(1.0)

    def test_deterministic_across_instances(self) -> None:
        assert embed("pacman power pellet") == embed("pacman power pellet", dim=768)

    def test_case_and_punctuation_insensitive(self) -> None:
        assert embed("Asteroids: 20 Points!") == embed("asteroids 20 points")

    def test_lexical_similarity_ordering(self) -> None:
        query = embed("how many points is a large asteroid worth")
        related = embed("large asteroids are worth 20 points each; small ones 100")
        unrelated = embed("the saucer drifts across the top of the screen")
        assert cosine(query, related) > cosine(query, unrelated)

    def test_empty_text_returns_zero_vector_without_crashing(self) -> None:
        vector = embed("   ... !!!   ", dim=32)
        assert vector == [0.0] * 32

    def test_invalid_dim_raises(self) -> None:
        with pytest.raises(ValueError):
            retrieval_eval.HashingEmbedding(dim=0)

    def test_embed_documents_matches_embed_query(self) -> None:
        embedder = retrieval_eval.HashingEmbedding(dim=64)
        documents = asyncio.run(embedder.embed_documents(["centipede segments", "mushrooms"]))
        assert documents[0] == asyncio.run(embedder.embed_query("centipede segments"))
        assert len(documents) == 2


# ---------------------------------------------------------------------------
# build_embedder
# ---------------------------------------------------------------------------


class TestBuildEmbedder:
    def test_hashing_default_dim(self) -> None:
        embedder, dim = retrieval_eval.build_embedder("hashing")
        assert isinstance(embedder, retrieval_eval.HashingEmbedding)
        assert dim == retrieval_eval.DEFAULT_HASHING_DIM

    def test_hashing_dim_override(self) -> None:
        _, dim = retrieval_eval.build_embedder("hashing", dim=128)
        assert dim == 128

    def test_fastembed_known_model_dim(self) -> None:
        embedder, dim = retrieval_eval.build_embedder(
            "fastembed", fastembed_model="BAAI/bge-small-en-v1.5"
        )
        assert isinstance(embedder, retrieval_eval.FastEmbedEmbedding)
        assert dim == 384

    def test_fastembed_unknown_model_requires_dim(self) -> None:
        with pytest.raises(ValueError):
            retrieval_eval.build_embedder("fastembed", fastembed_model="no/such-model")

    def test_unknown_embedder_raises(self) -> None:
        with pytest.raises(ValueError):
            retrieval_eval.build_embedder("telepathy")


# ---------------------------------------------------------------------------
# Sweep helpers
# ---------------------------------------------------------------------------


class TestParseIntList:
    def test_parses_comma_separated_values(self) -> None:
        assert retrieval_eval.parse_int_list("400,800,1200", name="sizes") == [400, 800, 1200]

    def test_ignores_whitespace_and_trailing_commas(self) -> None:
        assert retrieval_eval.parse_int_list(" 0 , 100 ,", name="overlaps") == [0, 100]

    def test_rejects_non_integers(self) -> None:
        with pytest.raises(ValueError):
            retrieval_eval.parse_int_list("400,fast", name="sizes")

    def test_rejects_negative_values(self) -> None:
        with pytest.raises(ValueError):
            retrieval_eval.parse_int_list("-1", name="sizes")

    def test_rejects_empty_input(self) -> None:
        with pytest.raises(ValueError):
            retrieval_eval.parse_int_list(" , ", name="sizes")


class TestBuildSweepGrid:
    def test_cross_product(self) -> None:
        assert retrieval_eval.build_sweep_grid([400, 800], [0, 100]) == [
            (400, 0),
            (400, 100),
            (800, 0),
            (800, 100),
        ]

    def test_skips_invalid_overlap(self) -> None:
        # 400/400 is invalid (overlap >= size); only 0 survives.
        grid = retrieval_eval.build_sweep_grid([400], [0, 400])
        assert grid == [(400, 0)]

    def test_skips_non_positive_size(self) -> None:
        with pytest.raises(ValueError):
            retrieval_eval.build_sweep_grid([0], [0])

    def test_empty_grid_raises(self) -> None:
        with pytest.raises(ValueError):
            retrieval_eval.build_sweep_grid([400], [400, 500])


class TestPickBest:
    def test_highest_hit_rate_wins(self) -> None:
        summaries = [
            make_summary(400, 0, 0.8, 0.6),
            make_summary(800, 100, 0.9, 0.7),
            make_summary(1200, 100, 0.85, 0.8),
        ]
        assert retrieval_eval.pick_best(summaries)["chunk_size"] == 800

    def test_mrr_breaks_hit_rate_ties(self) -> None:
        summaries = [
            make_summary(400, 0, 0.9, 0.6),
            make_summary(800, 100, 0.9, 0.75),
        ]
        assert retrieval_eval.pick_best(summaries)["chunk_size"] == 800

    def test_preferred_config_wins_full_ties(self) -> None:
        summaries = [
            make_summary(400, 0, 0.9, 0.7),
            make_summary(800, 120, 0.9, 0.7),
        ]
        best = retrieval_eval.pick_best(summaries, preferred=(800, 120))
        assert (best["chunk_size"], best["chunk_overlap"]) == (800, 120)

    def test_first_entry_wins_when_no_preference(self) -> None:
        summaries = [
            make_summary(400, 0, 0.9, 0.7),
            make_summary(800, 120, 0.9, 0.7),
        ]
        assert retrieval_eval.pick_best(summaries)["chunk_size"] == 400


class TestRenderComparisonTable:
    def test_renders_rows_and_headers(self) -> None:
        table = retrieval_eval.render_comparison_table(
            [make_summary(400, 0, 1.0, 0.9, chunks=120), make_summary(800, 100, 0.5, 0.4)],
            top_k=5,
        )
        assert "| Chunk size | Overlap | Chunks | Hit Rate@5 | MRR | Mean ms | p95 ms |" in table
        assert "| 400 | 0 | 120 | 100.0% | 0.900 | 12 | 25 |" in table
        assert "| 800 | 100 | 10 | 50.0% | 0.400 |" in table

    def test_missing_metrics_render_as_na(self) -> None:
        table = retrieval_eval.render_comparison_table([make_summary(800, 100, None)], top_k=5)
        assert "| n/a | n/a |" in table


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


class TestParser:
    def test_defaults(self) -> None:
        args = retrieval_eval.build_parser().parse_args([])
        assert args.embedder == "hashing"
        assert args.collection == "retrogpt_eval"
        assert args.top_k == 5
        assert args.rerank is None
        assert args.sweep_sizes is None
        assert args.baseline is None
        assert args.max_hit_rate_drop == pytest.approx(0.05)

    def test_sweep_and_guard_flags(self) -> None:
        args = retrieval_eval.build_parser().parse_args(
            [
                "--sweep-sizes",
                "400,800,1200",
                "--sweep-overlaps",
                "0,100",
                "--no-rerank",
                "--baseline",
                "eval/baselines/ci-offline.json",
                "--min-hit-rate",
                "0.8",
            ]
        )
        assert args.sweep_sizes == "400,800,1200"
        assert args.sweep_overlaps == "0,100"
        assert args.rerank is False
        assert args.baseline == Path("eval/baselines/ci-offline.json")
        assert args.min_hit_rate == pytest.approx(0.8)
