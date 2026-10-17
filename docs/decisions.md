# Architecture Decision Records (ADR)

This document tracks key architectural decisions, tradeoffs, and rationale for RetroGPT.

---

## ADR 0001: System Architecture and Tech Stack Selection

- **Status:** Accepted
- **Date:** 2026-10-17

### Context
RetroGPT is designed as a production-grade Retrieval-Augmented Generation (RAG) assistant for classic games. The system requires high retrieval accuracy, sub-second cached responses, verifiable citations, and full observability.

### Decision
1. **Multi-tier Architecture:**
   - **Nginx:** Reverse proxy, rate limiting, and routing.
   - **Frontend:** React + Vite + TypeScript for lightweight, responsive UI.
   - **API Gateway:** Node.js 20 LTS + TypeScript + Express for authentication, input validation (zod), and session management.
   - **AI Service:** Python 3.12 + FastAPI + LangChain for data ingestion, embeddings, hybrid search, and LLM orchestration.
   - **Storage & Vector Store:** Qdrant (hybrid vector search), MongoDB (chat history/sessions), Redis (response caching & rate limiting).

2. **Retrieval Pipeline:**
   - Recursive character chunking (default chunk size: 800, overlap: 120).
   - Hybrid retrieval (vector embeddings + BM25 sparse search with Reciprocal Rank Fusion).
   - Cross-encoder re-ranking for top candidates.

3. **Evaluation Framework:**
   - RAGAS + automated Hit Rate@5 / MRR metrics evaluated against curated game QA datasets.

### Consequences
- Clean separation of concerns between API orchestration (Node.js) and GenAI/RAG workloads (Python).
- Local-to-production parity using Docker Compose.

---

## ADR 0002: Hybrid Search via Client-Side BM25 + Reciprocal Rank Fusion

- **Status:** Accepted
- **Date:** 2026-10-17
- **Issue:** #20

### Context
ADR 0001 committed RetroGPT to hybrid retrieval (dense embeddings + BM25 sparse search fused with RRF). Qdrant supports sparse vectors natively, but the pinned `qdrant-client==1.7.3` has no server-side BM25 encoding/inference and no `query_points`/`Prefetch` fusion API.

### Decision
1. **Client-side BM25 encoder** (`app/retrieval/bm25.py`): plain-Python BM25 (`k1=1.2`, `b=0.75`), deterministic sorted vocabulary, simple lowercase alphanumeric tokenisation. Corpus statistics are persisted to `/data/bm25_stats.json` after ingestion; the query path loads the same statistics so term indices/weights match the indexed sparse vectors exactly.
2. **Sparse vectors stored in Qdrant**: collections created with hybrid enabled carry a named `"bm25"` sparse-vector config; each point stores `{"": dense, "bm25": sparse}`.
3. **Client-side RRF fusion** (`app/retrieval/hybrid.py`): dense and sparse prefetch (default 20 candidates each) are fused with `score = Σ 1/(60 + rank)`; the fused best hit is normalised to `1.0` so the UI's percentage-match display stays meaningful.
4. **Fail-open**: a missing stats file, a dense-only legacy collection, or a sparse-search error logs a warning and silently degrades to dense-only retrieval.
5. **Legacy collections**: `ensure_collection(recreate_if_incompatible=True)` drops and re-creates a dense-only collection; by default a warning is logged and hybrid falls back to dense until the collection is re-created and re-ingested.

### Consequences
- Full hybrid retrieval on Qdrant 1.7 without upgrading the server or client.
- Ingestion is two-phase: load + chunk *all* files, fit BM25 over the whole run's corpus, then embed + upsert per file.
- Collection and stats file must be rebuilt together; stale stats are rejected via the persisted format version.
- **Pending:** end-to-end hybrid evaluation numbers (README metrics table) require a valid `GOOGLE_API_KEY` for embeddings; the dense baseline and hybrid rerun are queued until credentials are available.

---

## ADR 0003: Cross-Encoder Re-Ranking with fastembed (ONNX)

- **Status:** Accepted
- **Date:** 2026-10-17
- **Issue:** #21

### Context
ADR 0001 includes cross-encoder re-ranking after hybrid retrieval. The standard implementation (`sentence-transformers` `CrossEncoder`) drags in PyTorch (~1 GB in CI images and slower container builds) for a model that runs on CPU just as well through ONNX Runtime.

### Decision
1. **Runtime:** `fastembed` (`TextCrossEncoder`, ONNX Runtime) with the `Xenova/ms-marco-MiniLM-L-6-v2` model — same MiniLM cross-encoder family used by `sentence-transformers`, without the PyTorch dependency. Added to `requirements.txt` (pinned).
2. **Two-stage retrieval** (`app/retrieval/rerank.py: retrieve()`): hybrid search recalls a wide candidate pool (`max(top_k, RETRIEVAL_TOP_K)` = 20 by default), the cross-encoder scores every `(query, passage)` pair, and the best `top_k` (default `RERANK_TOP_N` = 5) are kept for generation.
3. **Score display:** cross-encoder logits are unbounded (e.g. `+9.8` / `-11.4`), so they are squashed with a logistic sigmoid into `(0, 1)` to keep the frontend's percentage-match display and `SourceDoc.score` validation meaningful.
4. **Lazy + fail-open:** the model is loaded on first use and cached under `/data/fastembed_cache`; if `fastembed` is missing or scoring fails, retrieval silently falls back to the hybrid RRF order. No query ever fails because of re-ranking.
5. Re-ranking can be disabled entirely via `RERANK_ENABLED=false` (e.g. for latency comparisons in #22).

### Consequences
- Re-ranking adds one CPU inference pass over ~20 candidates (expected low-hundreds of ms; exact latency to be recorded with the eval rerun).
- Container/CI images stay lightweight (ONNX Runtime ~24 MB vs PyTorch ~1 GB).
- First request after a restart downloads the model (~90 MB) unless the cache volume is warm.
- **Pending:** eval rerun for accuracy/latency numbers requires a valid `GOOGLE_API_KEY` (same blocker as ADR 0002).
