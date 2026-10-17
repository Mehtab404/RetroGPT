"""Tests for Qdrant vector store and ingestion pipeline (Issue #9).

Covers:
- QdrantVectorStore: ensure_collection, upsert, search, count, delete_collection
- IngestionPipeline: run (directory + single file), ingest_documents, error handling
- VectorStoreError propagation
- CLI ingest.py argument parsing and main() entry point

All Qdrant calls are mocked so no real Qdrant instance is required.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.ingestion.chunker import Chunk, RecursiveCharacterTextSplitter
from app.ingestion.loaders import Document
from app.ingestion.pipeline import IngestionPipeline, IngestionResult
from app.retrieval.vector_store import QdrantVectorStore, SearchResult, VectorStoreError


# ===========================================================================
# Helpers / Fixtures
# ===========================================================================


def _make_chunk(content: str = "Some text", source: str = "test.md", idx: int = 0) -> Chunk:
    """Create a minimal Chunk for testing."""
    return Chunk(
        content=content,
        metadata={
            "chunk_id": f"deadbeef{idx:08x}",
            "chunk_index": idx,
            "chunk_total": 1,
            "source": source,
            "game": "asteroids",
            "title": "Test",
            "file_type": "markdown",
        },
    )


def _make_mock_client(
    *,
    collections: list[str] | None = None,
    search_hits: list | None = None,
    count: int = 0,
) -> MagicMock:
    """Return a MagicMock QdrantClient with sensible defaults."""
    client = MagicMock()

    # get_collections
    col_names = collections or []
    mock_col = MagicMock()
    mock_col.name = "retro_games" if "retro_games" in col_names else "__none__"
    collections_response = MagicMock()
    collections_response.collections = [
        _named_collection(n) for n in col_names
    ]
    client.get_collections.return_value = collections_response

    # create_collection
    client.create_collection.return_value = True

    # delete_collection
    client.delete_collection.return_value = True

    # upsert
    upsert_result = MagicMock()
    client.upsert.return_value = upsert_result

    # search
    client.search.return_value = search_hits or []

    # count
    count_result = MagicMock()
    count_result.count = count
    client.count.return_value = count_result

    return client


def _named_collection(name: str) -> MagicMock:
    """Return a mock collection object with a .name attribute."""
    c = MagicMock()
    c.name = name
    return c


def _make_fake_embedder(dim: int = 768, num: int | None = None) -> MagicMock:
    """Return a mock EmbeddingProvider."""
    embedder = MagicMock()

    async def embed_documents(texts):
        n = num if num is not None else len(texts)
        return [[0.1] * dim for _ in range(n)]

    embedder.embed_documents = embed_documents
    return embedder


# ===========================================================================
# VectorStoreError
# ===========================================================================


class TestVectorStoreError:
    def test_is_runtime_error(self):
        err = VectorStoreError("oops")
        assert isinstance(err, RuntimeError)
        assert str(err) == "oops"

    def test_can_wrap_cause(self):
        cause = ValueError("root cause")
        err = VectorStoreError("wrapper") 
        try:
            raise err from cause
        except VectorStoreError as e:
            assert e.__cause__ is cause


# ===========================================================================
# SearchResult
# ===========================================================================


class TestSearchResult:
    def test_attributes(self):
        r = SearchResult(
            chunk_id="abc123",
            score=0.87,
            payload={"game": "pong", "content": "Ball bounces off paddle."},
            content="Ball bounces off paddle.",
        )
        assert r.chunk_id == "abc123"
        assert r.score == pytest.approx(0.87)
        assert r.content == "Ball bounces off paddle."
        assert r.payload["game"] == "pong"

    def test_repr_contains_chunk_id(self):
        r = SearchResult(chunk_id="myid", score=0.5, payload={}, content="hello")
        assert "myid" in repr(r)


# ===========================================================================
# QdrantVectorStore — ensure_collection
# ===========================================================================


class TestEnsureCollection:
    @pytest.mark.asyncio
    async def test_creates_collection_when_absent(self):
        client = _make_mock_client(collections=[])  # no collections exist
        store = QdrantVectorStore(client=client, collection="retro_games")
        await store.ensure_collection()
        client.create_collection.assert_called_once()
        call_kwargs = client.create_collection.call_args[1]
        assert call_kwargs["collection_name"] == "retro_games"

    @pytest.mark.asyncio
    async def test_skips_creation_when_exists(self):
        client = _make_mock_client(collections=["retro_games"])
        store = QdrantVectorStore(client=client, collection="retro_games")
        await store.ensure_collection()
        client.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_vector_store_error_on_failure(self):
        client = MagicMock()
        client.get_collections.side_effect = Exception("network down")
        store = QdrantVectorStore(client=client)
        with pytest.raises(VectorStoreError, match="Failed to ensure collection"):
            await store.ensure_collection()

    @pytest.mark.asyncio
    async def test_vector_params_use_cosine_distance(self):
        from qdrant_client.http import models as m
        client = _make_mock_client(collections=[])
        store = QdrantVectorStore(client=client, vector_size=768)
        await store.ensure_collection()
        call_kwargs = client.create_collection.call_args[1]
        assert call_kwargs["vectors_config"].distance == m.Distance.COSINE

    @pytest.mark.asyncio
    async def test_vector_size_passed_to_qdrant(self):
        client = _make_mock_client(collections=[])
        store = QdrantVectorStore(client=client, vector_size=1536)
        await store.ensure_collection()
        call_kwargs = client.create_collection.call_args[1]
        assert call_kwargs["vectors_config"].size == 1536


# ===========================================================================
# QdrantVectorStore — delete_collection
# ===========================================================================


class TestDeleteCollection:
    @pytest.mark.asyncio
    async def test_calls_delete_on_client(self):
        client = _make_mock_client()
        store = QdrantVectorStore(client=client, collection="retro_games")
        await store.delete_collection()
        client.delete_collection.assert_called_once_with("retro_games")

    @pytest.mark.asyncio
    async def test_raises_vector_store_error_on_failure(self):
        client = MagicMock()
        client.delete_collection.side_effect = Exception("boom")
        store = QdrantVectorStore(client=client)
        with pytest.raises(VectorStoreError, match="Failed to delete collection"):
            await store.delete_collection()


# ===========================================================================
# QdrantVectorStore — upsert
# ===========================================================================


class TestUpsert:
    @pytest.mark.asyncio
    async def test_empty_input_returns_zero(self):
        client = _make_mock_client()
        store = QdrantVectorStore(client=client)
        result = await store.upsert([], [])
        assert result == 0
        client.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_chunk_upserted(self):
        client = _make_mock_client()
        store = QdrantVectorStore(client=client)
        chunk = _make_chunk("Asteroid scoring", idx=0)
        vectors = [[0.1] * 768]
        result = await store.upsert([chunk], vectors)
        assert result == 1
        client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_payload_includes_content(self):
        client = _make_mock_client()
        store = QdrantVectorStore(client=client)
        chunk = _make_chunk("Ball bounces", idx=0)
        await store.upsert([chunk], [[0.0] * 768])
        call_kwargs = client.upsert.call_args[1]
        point = call_kwargs["points"][0]
        assert point.payload["content"] == "Ball bounces"

    @pytest.mark.asyncio
    async def test_chunk_id_used_as_point_id(self):
        client = _make_mock_client()
        store = QdrantVectorStore(client=client)
        chunk = _make_chunk(idx=5)
        await store.upsert([chunk], [[0.0] * 768])
        call_kwargs = client.upsert.call_args[1]
        point = call_kwargs["points"][0]
        assert point.id == chunk.metadata["chunk_id"]

    @pytest.mark.asyncio
    async def test_raises_on_length_mismatch(self):
        client = _make_mock_client()
        store = QdrantVectorStore(client=client)
        chunks = [_make_chunk(idx=i) for i in range(3)]
        vectors = [[0.0] * 768, [0.0] * 768]  # only 2 vectors for 3 chunks
        with pytest.raises(ValueError, match="same length"):
            await store.upsert(chunks, vectors)

    @pytest.mark.asyncio
    async def test_batching_respects_batch_size(self):
        client = _make_mock_client()
        store = QdrantVectorStore(client=client, upsert_batch=3)
        chunks = [_make_chunk(idx=i) for i in range(7)]
        vectors = [[float(i)] * 768 for i in range(7)]
        result = await store.upsert(chunks, vectors)
        assert result == 7
        # 7 chunks with batch_size=3 → 3 batches (3+3+1)
        assert client.upsert.call_count == 3

    @pytest.mark.asyncio
    async def test_all_batches_use_wait_true(self):
        client = _make_mock_client()
        store = QdrantVectorStore(client=client, upsert_batch=2)
        chunks = [_make_chunk(idx=i) for i in range(4)]
        vectors = [[0.0] * 768] * 4
        await store.upsert(chunks, vectors)
        for call in client.upsert.call_args_list:
            assert call[1]["wait"] is True

    @pytest.mark.asyncio
    async def test_raises_vector_store_error_on_qdrant_failure(self):
        client = MagicMock()
        client.upsert.side_effect = Exception("disk full")
        store = QdrantVectorStore(client=client)
        with pytest.raises(VectorStoreError, match="Qdrant upsert failed"):
            await store.upsert([_make_chunk()], [[0.0] * 768])

    @pytest.mark.asyncio
    async def test_metadata_preserved_in_payload(self):
        client = _make_mock_client()
        store = QdrantVectorStore(client=client)
        chunk = _make_chunk("text", source="data/asteroids/manual.md")
        await store.upsert([chunk], [[0.0] * 768])
        call_kwargs = client.upsert.call_args[1]
        point = call_kwargs["points"][0]
        assert point.payload["source"] == "data/asteroids/manual.md"
        assert point.payload["game"] == "asteroids"


# ===========================================================================
# QdrantVectorStore — search
# ===========================================================================


def _make_qdrant_hit(chunk_id: str, score: float, content: str) -> MagicMock:
    """Create a mock Qdrant scored point."""
    hit = MagicMock()
    hit.id = chunk_id
    hit.score = score
    hit.payload = {"chunk_id": chunk_id, "content": content, "game": "asteroids"}
    return hit


class TestSearch:
    @pytest.mark.asyncio
    async def test_returns_search_results(self):
        hit = _make_qdrant_hit("abc", 0.91, "Asteroids scoring rules.")
        client = _make_mock_client(search_hits=[hit])
        store = QdrantVectorStore(client=client)
        results = await store.search([0.1] * 768, top_k=5)
        assert len(results) == 1
        assert results[0].chunk_id == "abc"
        assert results[0].score == pytest.approx(0.91)
        assert results[0].content == "Asteroids scoring rules."

    @pytest.mark.asyncio
    async def test_empty_results_returned_as_empty_list(self):
        client = _make_mock_client(search_hits=[])
        store = QdrantVectorStore(client=client)
        results = await store.search([0.1] * 768)
        assert results == []

    @pytest.mark.asyncio
    async def test_top_k_passed_to_client(self):
        client = _make_mock_client(search_hits=[])
        store = QdrantVectorStore(client=client)
        await store.search([0.1] * 768, top_k=7)
        call_kwargs = client.search.call_args[1]
        assert call_kwargs["limit"] == 7

    @pytest.mark.asyncio
    async def test_default_top_k_from_settings(self):
        client = _make_mock_client(search_hits=[])
        store = QdrantVectorStore(client=client)
        with patch("app.retrieval.vector_store.settings") as mock_settings:
            mock_settings.retrieval_top_k = 13
            mock_settings.qdrant_url = "http://qdrant:6333"
            mock_settings.qdrant_collection = "retro_games"
            await store.search([0.1] * 768)
        call_kwargs = client.search.call_args[1]
        assert call_kwargs["limit"] == 13

    @pytest.mark.asyncio
    async def test_filter_forwarded_to_client(self):
        from qdrant_client.http import models as m
        client = _make_mock_client(search_hits=[])
        store = QdrantVectorStore(client=client)
        f = m.Filter(
            must=[m.FieldCondition(key="game", match=m.MatchValue(value="pong"))]
        )
        await store.search([0.1] * 768, filter=f)
        call_kwargs = client.search.call_args[1]
        assert call_kwargs["query_filter"] is f

    @pytest.mark.asyncio
    async def test_raises_vector_store_error_on_failure(self):
        client = MagicMock()
        client.search.side_effect = Exception("timeout")
        store = QdrantVectorStore(client=client)
        with pytest.raises(VectorStoreError, match="Qdrant search failed"):
            await store.search([0.1] * 768)

    @pytest.mark.asyncio
    async def test_multiple_hits_returned_in_order(self):
        hits = [
            _make_qdrant_hit("h1", 0.95, "First"),
            _make_qdrant_hit("h2", 0.80, "Second"),
            _make_qdrant_hit("h3", 0.70, "Third"),
        ]
        client = _make_mock_client(search_hits=hits)
        store = QdrantVectorStore(client=client)
        results = await store.search([0.1] * 768)
        assert [r.chunk_id for r in results] == ["h1", "h2", "h3"]


# ===========================================================================
# QdrantVectorStore — count
# ===========================================================================


class TestCount:
    @pytest.mark.asyncio
    async def test_returns_point_count(self):
        client = _make_mock_client(count=42)
        store = QdrantVectorStore(client=client)
        n = await store.count()
        assert n == 42

    @pytest.mark.asyncio
    async def test_returns_zero_for_missing_collection(self):
        from qdrant_client.http.exceptions import UnexpectedResponse
        client = MagicMock()
        resp = MagicMock()
        resp.status_code = 404
        client.count.side_effect = UnexpectedResponse(
            status_code=404, reason_phrase="Not Found", content=b"", headers={}
        )
        store = QdrantVectorStore(client=client)
        n = await store.count()
        assert n == 0

    @pytest.mark.asyncio
    async def test_raises_on_unexpected_error(self):
        client = MagicMock()
        client.count.side_effect = Exception("db error")
        store = QdrantVectorStore(client=client)
        with pytest.raises(VectorStoreError, match="Failed to count"):
            await store.count()


# ===========================================================================
# QdrantVectorStore — close
# ===========================================================================


class TestClose:
    def test_close_calls_client_close(self):
        client = MagicMock()
        store = QdrantVectorStore(client=client)
        store.close()
        client.close.assert_called_once()

    def test_close_does_not_raise_on_error(self):
        client = MagicMock()
        client.close.side_effect = Exception("already closed")
        store = QdrantVectorStore(client=client)
        store.close()  # should not raise


# ===========================================================================
# IngestionPipeline — ingest_documents
# ===========================================================================


class TestIngestDocuments:
    @pytest.mark.asyncio
    async def test_empty_documents_returns_zero_upserted(self):
        mock_store = MagicMock()
        mock_store.ensure_collection = AsyncMock()
        mock_store.upsert = AsyncMock(return_value=0)
        pipeline = IngestionPipeline(
            vector_store=mock_store,
            embedder=_make_fake_embedder(),
        )
        result = await pipeline.ingest_documents([])
        assert result.total_upserted == 0
        mock_store.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_document_ingested(self):
        mock_store = MagicMock()
        mock_store.ensure_collection = AsyncMock()
        mock_store.upsert = AsyncMock(return_value=2)
        doc = Document(
            content="## Asteroids\n\nShooting rocks for fun and profit.\n\n"
                    "Large rock: 20 pts. Small rock: 100 pts.",
            metadata={"source": "asteroids_manual.md", "game": "asteroids"},
        )
        pipeline = IngestionPipeline(
            vector_store=mock_store,
            embedder=_make_fake_embedder(),
            splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10),
        )
        result = await pipeline.ingest_documents([doc])
        assert result.total_docs == 1
        assert result.total_chunks > 0
        assert result.total_upserted == 2

    @pytest.mark.asyncio
    async def test_ensure_collection_called_by_default(self):
        mock_store = MagicMock()
        mock_store.ensure_collection = AsyncMock()
        mock_store.upsert = AsyncMock(return_value=1)
        doc = Document(content="x" * 50, metadata={"source": "f.md"})
        pipeline = IngestionPipeline(
            vector_store=mock_store,
            embedder=_make_fake_embedder(),
        )
        await pipeline.ingest_documents([doc])
        mock_store.ensure_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_collection_skipped_when_disabled(self):
        mock_store = MagicMock()
        mock_store.ensure_collection = AsyncMock()
        mock_store.upsert = AsyncMock(return_value=1)
        doc = Document(content="x" * 50, metadata={"source": "f.md"})
        pipeline = IngestionPipeline(
            vector_store=mock_store,
            embedder=_make_fake_embedder(),
            ensure_collection=False,
        )
        await pipeline.ingest_documents([doc])
        mock_store.ensure_collection.assert_not_called()


# ===========================================================================
# IngestionPipeline — run (directory)
# ===========================================================================


class TestPipelineRunDirectory:
    @pytest.mark.asyncio
    async def test_run_on_empty_directory(self, tmp_path: Path):
        mock_store = MagicMock()
        mock_store.ensure_collection = AsyncMock()
        mock_store.upsert = AsyncMock(return_value=0)
        pipeline = IngestionPipeline(
            vector_store=mock_store,
            embedder=_make_fake_embedder(),
        )
        result = await pipeline.run(tmp_path)
        assert result.total_docs == 0
        assert result.total_chunks == 0
        assert result.total_upserted == 0
        mock_store.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_discovers_markdown_files(self, tmp_path: Path):
        (tmp_path / "asteroids").mkdir()
        (tmp_path / "asteroids" / "manual.md").write_text(
            "# Asteroids\n\nA classic arcade game from 1979.\n\n"
            "Players pilot a spacecraft through a field of asteroids.",
            encoding="utf-8",
        )
        mock_store = MagicMock()
        mock_store.ensure_collection = AsyncMock()
        mock_store.upsert = AsyncMock(return_value=1)
        pipeline = IngestionPipeline(
            vector_store=mock_store,
            embedder=_make_fake_embedder(),
            splitter=RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20),
        )
        result = await pipeline.run(tmp_path)
        assert result.total_docs >= 1
        assert result.total_chunks >= 1
        assert result.total_upserted >= 1
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_run_discovers_txt_files(self, tmp_path: Path):
        (tmp_path / "pong_faq.txt").write_text(
            "What is Pong?\n\nPong is a table tennis game.\nIt was released in 1972.\n",
            encoding="utf-8",
        )
        mock_store = MagicMock()
        mock_store.ensure_collection = AsyncMock()
        mock_store.upsert = AsyncMock(return_value=1)
        pipeline = IngestionPipeline(
            vector_store=mock_store,
            embedder=_make_fake_embedder(),
        )
        result = await pipeline.run(tmp_path)
        assert result.total_docs == 1

    @pytest.mark.asyncio
    async def test_run_raises_for_missing_source(self, tmp_path: Path):
        mock_store = MagicMock()
        mock_store.ensure_collection = AsyncMock()
        pipeline = IngestionPipeline(vector_store=mock_store, embedder=_make_fake_embedder())
        with pytest.raises(FileNotFoundError):
            await pipeline.run(tmp_path / "nonexistent")

    @pytest.mark.asyncio
    async def test_run_ignores_unsupported_extensions(self, tmp_path: Path):
        (tmp_path / "notes.pdf").write_bytes(b"%PDF-1.4 fake")
        (tmp_path / "data.csv").write_text("a,b,c\n1,2,3\n", encoding="utf-8")
        mock_store = MagicMock()
        mock_store.ensure_collection = AsyncMock()
        mock_store.upsert = AsyncMock(return_value=0)
        pipeline = IngestionPipeline(
            vector_store=mock_store,
            embedder=_make_fake_embedder(),
        )
        result = await pipeline.run(tmp_path)
        assert result.total_docs == 0
        mock_store.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_processes_multiple_files(self, tmp_path: Path):
        for name in ("asteroids", "pong", "breakout"):
            d = tmp_path / name
            d.mkdir()
            (d / f"{name}_manual.md").write_text(
                f"# {name.capitalize()}\n\nA classic arcade game.\n\n"
                f"This is the {name} manual with enough text to chunk.\n",
                encoding="utf-8",
            )
        mock_store = MagicMock()
        mock_store.ensure_collection = AsyncMock()
        mock_store.upsert = AsyncMock(side_effect=lambda **kwargs: len(kwargs.get("points", [])))
        # Patch upsert to return correct count
        async def fake_upsert(chunks, vectors):
            return len(chunks)
        mock_store.upsert = AsyncMock(side_effect=fake_upsert)
        pipeline = IngestionPipeline(
            vector_store=mock_store,
            embedder=_make_fake_embedder(),
            splitter=RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20),
        )
        result = await pipeline.run(tmp_path)
        assert result.total_docs == 3

    @pytest.mark.asyncio
    async def test_run_records_errors_and_continues(self, tmp_path: Path):
        """Even if one file fails, others are still processed."""
        (tmp_path / "good.md").write_text(
            "# Good\n\nThis file is fine and has enough text.", encoding="utf-8"
        )
        (tmp_path / "bad.md").write_text(
            "# Bad\n\nThis file will cause an embedding error.", encoding="utf-8"
        )

        call_count = 0

        async def flaky_embed(texts):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Embedding service down")
            return [[0.1] * 768 for _ in texts]

        mock_store = MagicMock()
        mock_store.ensure_collection = AsyncMock()
        mock_store.upsert = AsyncMock(return_value=1)
        embedder = MagicMock()
        embedder.embed_documents = flaky_embed

        pipeline = IngestionPipeline(
            vector_store=mock_store,
            embedder=embedder,
        )
        result = await pipeline.run(tmp_path)
        # One error, one success
        assert len(result.errors) == 1
        assert result.total_upserted >= 1


# ===========================================================================
# IngestionPipeline — run (single file)
# ===========================================================================


class TestPipelineRunSingleFile:
    @pytest.mark.asyncio
    async def test_run_on_single_md_file(self, tmp_path: Path):
        md = tmp_path / "manual.md"
        md.write_text(
            "# Centipede\n\nShoot the centipede to score points.\n\n"
            "Each segment destroyed scores differently.",
            encoding="utf-8",
        )
        mock_store = MagicMock()
        mock_store.ensure_collection = AsyncMock()

        async def fake_upsert(chunks, vectors):
            return len(chunks)
        mock_store.upsert = AsyncMock(side_effect=fake_upsert)

        pipeline = IngestionPipeline(
            vector_store=mock_store,
            embedder=_make_fake_embedder(),
            splitter=RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20),
        )
        result = await pipeline.run(md)
        assert result.total_docs == 1
        assert result.total_chunks >= 1
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_run_on_unsupported_extension_is_skipped(self, tmp_path: Path):
        bad = tmp_path / "notes.pdf"
        bad.write_bytes(b"%PDF fake content")
        mock_store = MagicMock()
        mock_store.ensure_collection = AsyncMock()
        mock_store.upsert = AsyncMock(return_value=0)
        pipeline = IngestionPipeline(
            vector_store=mock_store,
            embedder=_make_fake_embedder(),
        )
        result = await pipeline.run(bad)
        assert result.total_docs == 0
        mock_store.upsert.assert_not_called()


# ===========================================================================
# IngestionPipeline — embedding error propagates to per-file error dict
# ===========================================================================


class TestPipelineEmbeddingErrors:
    @pytest.mark.asyncio
    async def test_embedding_error_captured_per_file(self, tmp_path: Path):
        from app.ingestion.embeddings import EmbeddingError
        md = tmp_path / "fail.md"
        md.write_text("# Fail\n\nThis will fail on embedding.", encoding="utf-8")

        async def always_fail(texts):
            raise EmbeddingError("API down")

        mock_store = MagicMock()
        mock_store.ensure_collection = AsyncMock()
        mock_store.upsert = AsyncMock(return_value=0)
        embedder = MagicMock()
        embedder.embed_documents = always_fail

        pipeline = IngestionPipeline(vector_store=mock_store, embedder=embedder)
        result = await pipeline.run(tmp_path)
        assert len(result.errors) == 1
        assert "API down" in list(result.errors.values())[0]

    @pytest.mark.asyncio
    async def test_upsert_error_captured_per_file(self, tmp_path: Path):
        from app.retrieval.vector_store import VectorStoreError
        md = tmp_path / "doc.md"
        md.write_text("# Doc\n\nContent here.", encoding="utf-8")

        mock_store = MagicMock()
        mock_store.ensure_collection = AsyncMock()
        mock_store.upsert = AsyncMock(side_effect=VectorStoreError("upsert failed"))

        pipeline = IngestionPipeline(
            vector_store=mock_store,
            embedder=_make_fake_embedder(),
        )
        result = await pipeline.run(tmp_path)
        assert len(result.errors) == 1


# ===========================================================================
# IngestionResult model
# ===========================================================================


class TestIngestionResult:
    def test_default_values(self):
        r = IngestionResult()
        assert r.total_docs == 0
        assert r.total_chunks == 0
        assert r.total_upserted == 0
        assert r.skipped_files == []
        assert r.errors == {}

    def test_fields_are_mutable(self):
        r = IngestionResult()
        r.total_docs = 5
        r.errors["x.md"] = "some error"
        assert r.total_docs == 5
        assert "x.md" in r.errors


# ===========================================================================
# CLI — argument parsing
# ===========================================================================


class TestCLIArgumentParsing:
    def test_source_required(self):
        from scripts.ingest import _build_parser
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_source_parsed(self, tmp_path: Path):
        from scripts.ingest import _build_parser
        parser = _build_parser()
        args = parser.parse_args([str(tmp_path)])
        assert args.source == tmp_path

    def test_dry_run_flag(self, tmp_path: Path):
        from scripts.ingest import _build_parser
        parser = _build_parser()
        args = parser.parse_args([str(tmp_path), "--dry-run"])
        assert args.dry_run is True

    def test_collection_option(self, tmp_path: Path):
        from scripts.ingest import _build_parser
        parser = _build_parser()
        args = parser.parse_args([str(tmp_path), "--collection", "my_col"])
        assert args.collection == "my_col"

    def test_qdrant_url_option(self, tmp_path: Path):
        from scripts.ingest import _build_parser
        parser = _build_parser()
        args = parser.parse_args([str(tmp_path), "--qdrant-url", "http://myhost:6333"])
        assert args.qdrant_url == "http://myhost:6333"

    def test_embedding_provider_option(self, tmp_path: Path):
        from scripts.ingest import _build_parser
        parser = _build_parser()
        args = parser.parse_args([str(tmp_path), "--embedding-provider", "openai"])
        assert args.embedding_provider == "openai"

    def test_chunk_size_option(self, tmp_path: Path):
        from scripts.ingest import _build_parser
        parser = _build_parser()
        args = parser.parse_args([str(tmp_path), "--chunk-size", "500"])
        assert args.chunk_size == 500

    def test_verbose_flag(self, tmp_path: Path):
        from scripts.ingest import _build_parser
        parser = _build_parser()
        args = parser.parse_args([str(tmp_path), "--verbose"])
        assert args.verbose is True

    def test_invalid_embedding_provider_rejected(self, tmp_path: Path):
        from scripts.ingest import _build_parser
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([str(tmp_path), "--embedding-provider", "anthropic"])


# ===========================================================================
# CLI — main() exit codes
# ===========================================================================


class TestCLIMain:
    def test_main_returns_2_for_missing_source(self, tmp_path: Path):
        from scripts.ingest import main
        exit_code = main([str(tmp_path / "nonexistent_path")])
        assert exit_code == 2

    def test_main_dry_run_returns_0_on_empty_dir(self, tmp_path: Path):
        from scripts.ingest import main
        exit_code = main([str(tmp_path), "--dry-run"])
        assert exit_code == 0

    def test_main_dry_run_returns_0_with_md_files(self, tmp_path: Path):
        (tmp_path / "doc.md").write_text(
            "# Test\n\nSome content here.", encoding="utf-8"
        )
        from scripts.ingest import main
        exit_code = main([str(tmp_path), "--dry-run"])
        assert exit_code == 0

    def test_main_dry_run_with_chunk_size_override(self, tmp_path: Path):
        (tmp_path / "doc.md").write_text("# Doc\n\nContent.", encoding="utf-8")
        from scripts.ingest import main
        exit_code = main([str(tmp_path), "--dry-run", "--chunk-size", "400"])
        assert exit_code == 0

    def test_main_full_run_with_mocked_dependencies(self, tmp_path: Path):
        """Full pipeline run with mocked Qdrant and embeddings."""
        (tmp_path / "pong.md").write_text(
            "# Pong\n\nPong is a classic game.\n\nScore points by beating the opponent.",
            encoding="utf-8",
        )
        from scripts.ingest import main
        with (
            patch("app.retrieval.vector_store.QdrantVectorStore.ensure_collection", new_callable=AsyncMock),
            patch("app.retrieval.vector_store.QdrantVectorStore.upsert", new_callable=AsyncMock, return_value=2),
            patch("app.ingestion.embeddings.get_embedding_provider") as mock_factory,
        ):
            mock_embedder = MagicMock()
            mock_embedder.embed_documents = AsyncMock(return_value=[[0.1] * 768, [0.2] * 768])
            mock_factory.return_value = mock_embedder
            exit_code = main([str(tmp_path)])
        assert exit_code == 0
