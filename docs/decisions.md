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
