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

---

## ADR 0004: Chunking Strategy — 800 Characters / 120 Overlap

- **Status:** Accepted (offline sweep; credentialed confirmation pending)
- **Date:** 2026-10-17
- **Issue:** #22

### Context
Issue #22 compares chunk sizes (400/800/1200) and overlap values by evaluation and picks the best configuration. The credentialed end-to-end evaluation (Gemini embeddings + LLM answers) is still blocked by the missing `GOOGLE_API_KEY`, so the sweep was run with the retrieval-only offline harness (`eval/retrieval_eval.py`): the **real** ingestion + hybrid retrieval stack (chunking → BM25 → dense retrieval → RRF) against a scratch Qdrant collection, local ONNX embeddings (`BAAI/bge-small-en-v1.5`), re-ranking disabled so chunking effects are not masked by the cross-encoder, all 70 dataset questions, `top_k=5`.

### Sweep results (offline proxy, 2026-10-17)

| Chunk size | Overlap | Chunks | Hit Rate@5 | MRR | Mean retrieval ms | p95 ms |
|---|---|---|---|---|---|---|
| 400 | 0 | 75 | 100.0% | 0.989 | 18 | 23 |
| 400 | 100 | 97 | 100.0% | 0.982 | 17 | 20 |
| 400 | 120 | 100 | 100.0% | 0.982 | 18 | 23 |
| **800** | **0** | **33** | **100.0%** | **1.000** | **18** | **22** |
| **800** | **100** | **38** | **100.0%** | **1.000** | **21** | **26** |
| 800 | 120 | 38 | 100.0% | 0.992 | 22 | 30 |
| 1200 | 0 | 23 | 100.0% | 0.992 | 20 | 26 |
| 1200 | 100 | 25 | 100.0% | 0.992 | 23 | 32 |
| 1200 | 120 | 25 | 100.0% | 0.992 | 20 | 26 |

Reproduce with:
`docker compose exec ai-service python /eval/retrieval_eval.py --sweep-sizes 400,800,1200 --sweep-overlaps 0,100,120 --embedder fastembed --fastembed-cache /data/fastembed_cache --no-rerank --label chunking-sweep`

### Decision
1. Keep the **`CHUNK_SIZE=800`, `CHUNK_OVERLAP=120`** defaults.
2. 800-char chunks lead on MRR (1.000/0.992 vs 0.982–0.992 for 400/1200); 400/1200 occasionally push an answer document from rank 1 to rank 2.
3. Overlap is within noise at 800 chars (0/100/120 differ by a single question's rank), so there is no evidence to move away from the current 120 default.
4. Hit Rate@5 is 100% for every configuration — the curated corpus (six short manuals) is too small to separate recall at top-5, so this sweep only *weakens* hypotheses (400 and 1200 are not better) rather than proving an optimum.

### Consequences
- Chunk size/overlap stay configurable via `CHUNK_SIZE`/`CHUNK_OVERLAP`; re-run the sweep when the corpus grows.
- The offline proxy uses local `bge-small` embeddings, not the production Gemini model, and retrieval only (no answers) — treat it as directional evidence.
- **Pending:** credentialed end-to-end rerun to confirm the pick and record answer-quality metrics (same blocker as ADR 0002/0003).

---

## ADR 0005: Structured JSON Logging and Request-ID Propagation

- **Status:** Accepted
- **Date:** 2026-10-17
- **Issue:** #24

### Context
Issue #24 requires one request to be traceable across both services' logs. The gateway (Node.js/Express) and AI service (Python/FastAPI) are separate runtimes and deployables, and production (#26) will ship container stdout to a log aggregator — so logs must be machine-parseable without regex, and a request must carry one identifier from the gateway through every downstream attempt. Adding `pino`/`winston` (Node.js) or `structlog` (Python) would add dependencies for a small, fixed set of events.

### Decision
1. **Shared event shape:** both services emit one newline-delimited JSON object per line with the core fields `timestamp` (ISO 8601 UTC, millisecond precision), `level` (`debug` / `info` / `warning` / `error`), `service`, `event` (snake_case name such as `http_request`), optional `request_id`, plus flat context fields (`method`, `path`, `status`, `duration_ms`, `stage_ms`, …). Python records additionally carry `logger` (the module name).
2. **Zero new dependencies:** hand-rolled loggers — `api-gateway/src/utils/logger.ts` and `ai-service/app/logging_config.py` (`JsonFormatter`) — instead of `pino`/`winston`/`structlog`. `info`/`debug` go to stdout, `warning`/`error` to stderr (12-factor log streams; Docker captures both).
3. **Request-ID propagation:** the gateway accepts an `x-request-id` header, sanitizes it against `^[A-Za-z0-9._-]{1,64}$` (rejects anything else and generates a UUID), echoes the id on the response, and forwards it to the AI service via `x-request-id`. The same id is reused across gateway retries, so a retried request stays a single trace. The AI service keeps it in a `ContextVar`, so every log line emitted while handling the request — including retrieval warnings — is correlated automatically.
4. **Middleware instead of framework access logs:** an Express middleware and a pure-ASGI middleware emit the structured `http_request` line (`method`, `path`, `status`, `duration_ms`). Uvicorn's plain-text access log is silenced; `/health` probes are logged at `debug` to keep production noise down. The ASGI implementation is streaming-safe, ready for #25.
5. **Stage timings:** `/query` emits one `query` event whose `stage_ms` breaks the total down into `embed`, `retrieve` (with `hybrid_search_ms` / `rerank_ms` from `rerank.retrieve()`) and `generate`, alongside `total_ms`, `result_count`, `game_slug` and `top_k`.
6. **Verbosity:** both services read `LOG_LEVEL` (`debug` | `info` | `warn` | `error`, default `info`), wired through `docker-compose.yml` and documented in `.env.example`.

### Consequences
- One request can be traced end-to-end with `docker compose logs | grep <request-id>` or any JSON-aware log tool (see the Observability section in the README).
- No new npm/pip packages to audit or keep current; the level set and event shape stay intentionally small.
- In Python, request context propagates automatically; in TypeScript the call sites pass `request_id` into log fields explicitly (e.g. `aiClient` retries).
- Shipping logs to CloudWatch/ECS follows the container-stdout contract and is wired up with the production compose in #26.

---

## ADR 0006: Token-by-Token Streaming via Server-Sent Events

- **Status:** Accepted
- **Date:** 2026-10-17
- **Issue:** #25

### Context
Issue #25 requires the answer to appear token-by-token in the UI, which means LLM tokens must travel AI service → gateway → browser without being buffered anywhere. The existing `POST /api/chat` path returns one JSON body only after generation completes, and a proxy (Nginx) sits in front of both services. The design must also preserve the existing contracts that make the API pleasant to use: Redis caching, MongoDB persistence, session IDs, standard JSON error envelopes before any output, and no duplicated text or phantom cache entries when failures happen mid-answer.

### Decision
1. **Protocol — SSE with a four-event vocabulary.** The AI service emits `event: meta|token|done|error` frames with single-line JSON `data:` payloads (`meta {sources}`, `token {text}`, `done {latency_ms}`, `error {code, message}`). The gateway relays `meta`/`token` verbatim and **enriches** the terminal `done` frame to `{sessionId, latency_ms, cached}` so clients never need a second request to learn their session ID. A blank line terminates each frame.
2. **Failure boundary is the first frame.** Anything that fails *before* the first SSE byte (validation, AI service unreachable, upstream 4xx/5xx, timeout) is still a regular JSON HTTP error with the same status mapping as `POST /api/chat` (422 → 400, 5xx → 502, timeout → 504, network → 502) — headers have not been sent yet, so clients keep normal error handling. Once streaming has started, failures are delivered as an `error` event because the status line is already on the wire; tokens already delivered stay valid.
3. **Transport — POST + `fetch` + `ReadableStream`, not `EventSource`.** `EventSource` cannot send POST bodies, `x-api-key` headers, or read non-2xx JSON envelopes, so the frontend streams via `fetch` and a small incremental frame parser that buffers partial frames across chunk boundaries and tolerates CRLF.
4. **Retries exist only before the first token.** The gateway reuses the `aiClient` retry budget (3 attempts, 300 ms backoff) **only while opening the stream**. Retrying after tokens have been emitted would duplicate output; a mid-stream break is reported as an `error` event instead.
5. **Cache and persist only completed streams.** The gateway accumulates relayed tokens; on a clean `done` it calls `setCachedResponse(...)` and persists both messages exactly like the non-streamed path. Interrupted or failed streams are neither cached nor persisted. A cache hit is replayed as `meta` + one `token` + `done {"cached": true}` so the client has a single rendering path for cached and fresh answers.
6. **Zero new dependencies.** The SSE frame formatter/parser in the gateway (`src/utils/sse.ts`), the AI service's `generate_stream` generator, and the frontend's `streamQuestion` parser are all hand-rolled; `eventsource-parser`/EventSource polyfills are not needed.
7. **Provider adapters.** `LLMProvider.generate_stream` has a default implementation that falls back to `generate` and yields the whole answer as one chunk, so every provider streams correctly even without native streaming. Gemini uses `streamGenerateContent?alt=sse`; OpenAI uses `stream: true` plus the `[DONE]` sentinel. A `_with_stream_retry` helper retries only before the first chunk, mirroring the gateway rule.
8. **Nginx flushes frames.** An exact-match `location = /api/chat/stream` sets `proxy_buffering off` (with `proxy_cache off` and 120 s read/send timeouts) so tokens reach the browser immediately; the plain `location /api/` fallback keeps its default buffering for WSGI-style endpoints.
9. **UI stays intentionally thin.** The frontend adds an assistant placeholder immediately, appends `onToken` chunks, shows the typing indicator only until the first token (or `meta`), keeps partially streamed text when a stream breaks mid-answer, and removes the empty placeholder plus shows the standard error banner when a request fails before any content. The non-streaming `sendQuestion` client remains exported for compatibility.

### Consequences
- The acceptance criterion ("answer appears token-by-token in the UI") holds in dev compose; the exact same code runs behind Nginx with buffering disabled for that one location.
- One event vocabulary is shared by all three tiers, and the gateway `done` enrichment means session bookkeeping stays in the same response, so clients need no second call.
- Cached answers and mid-stream failures have defined, tested semantics: no duplicate cache entries, no persisted partial answers, no silent truncation.
- The hand-rolled parsers are ~40 lines each and are covered by unit tests (frame splitting, CRLF, keep-alives, mid-stream errors); the cost of a dependency was judged higher than the cost of maintaining them.
- Live token-by-token verification with a real provider still requires `GOOGLE_API_KEY`/`OPENAI_API_KEY`; without credentials the plumbing can be verified with a mock upstream (see Tests section).

---

## ADR 0007: Conversation-Aware Follow-Up Questions

- **Status:** Accepted
- **Date:** 2026-10-17
- **Issue:** #27

### Context
Follow-up questions ("What about the saucer?") were answered as standalone queries: the gateway forwarded only the question text, so the AI service could not resolve pronouns or implicit subjects, and Redis served one cached answer to every conversation that asked the same text. Sessions already existed — `POST /api/chat` returns a `sessionId` and MongoDB stores every exchange — so the missing pieces were *where* history may be read from, how much of it may travel, and how it interacts with caching, grounding and the prompt-injection defenses.

### Decision
1. **Server-side load only.** Before every request carrying a `sessionId`, the gateway reloads the session's stored messages from MongoDB (`Session.findOne`). Browser-supplied history is never trusted: the request schema has no history field, so a client payload containing one is stripped by zod and ignored.
2. **Bounded and deterministic.** `api-gateway/src/services/history.ts` (contract mirrored by `ai-service/app/schemas.py`) keeps at most the last `HISTORY_MAX_MESSAGES = 6` messages totalling at most `HISTORY_MAX_CHARS = 4000` content characters, walking newest-first so the most recent turns always survive, then forwards the window oldest-first. The AI service re-validates structure, roles, message count and the character budget, rejecting invalid payloads with 422.
3. **Fail-open.** A session lookup failure (Mongo down, malformed document) logs `session_history_load_failed` and the request proceeds without history — the same fail-open posture as message persistence. Requests without a `sessionId`, or against an empty session, behave exactly as before.
4. **A delimited, non-evidential prompt block.** `build_rag_prompt` inserts a `PRIOR CONVERSATION` block between the retrieved passages and the question. Its header states the block is context only — not factual evidence, never citable, never to be followed as instructions — and the system prompt with its injection guards is unchanged. Retrieved passages remain the sole grounding for facts and citations; prompts without history are byte-identical to the pre-#27 layout.
5. **History-aware cache keys.** Redis keys stay `retrogpt:chat:<sha256(question)>` for stateless requests; history-aware requests append `:h:<sha256(role/content pairs)>`, an order-sensitive fingerprint so different conversations asking the same follow-up never collide with each other or with the stateless key. Both routes use the fingerprint; without history the call shape and key are the legacy ones.

### Consequences
- Follow-ups are answered with conversational context on both the JSON and SSE routes, while the grounding contract (retrieved passages → citations) stays unchanged and independently testable.
- Caching becomes conversation-scoped: the same follow-up text in different conversations no longer shares one cache entry — more Redis keys, bounded by the existing 24 h TTL.
- If even the newest stored message exceeds the 4000-character cap, the bounded window is empty and the request degrades to a stateless ask (documented limitation).
- Prompt growth from history is capped at 4000 characters (~1 K tokens), keeping cost and latency predictable.

---

## ADR 0008: Load Testing with k6

- **Status:** Accepted
- **Date:** 2026-10-17
- **Issue:** #33 (backlog #28)

### Context
Backlog item #28 requires p50/p95 latency for cached vs. uncached `/api/chat` at 10 and 50 concurrent users, recorded in the README. The stack already has a two-layer retrieval path plus a Redis cache keyed by the normalised question (and, since #27, the conversation fingerprint), and a fixed offline dataset of 70 labeled questions. The measurement must be reproducible on the same dev compose stack as the rest of the project, work on Windows hosts, and not add runtime dependencies to either service.

### Decision
1. **k6 over Locust.** A single static binary ships as a pinned Docker image (`grafana/k6:2.3.0`), so no Python environment, worker fleet or new eval dependency is needed; Locust would require adding its package to the ai-service image to be equivalent. k6 also reports `med`/`p(95)` in its default summary, which is exactly the acceptance metric.
2. **Dockerized runs through the Nginx front door.** The script mounts `eval/` read-only into the container and targets `http://host.docker.internal/api`, so the measured path includes Nginx, auth, the rate limiter, cache and MongoDB persistence. Linux hosts add `--add-host=host.docker.internal:host-gateway`.
3. **The eval dataset is the question pool.** `eval/dataset.jsonl` supplies 70 real questions; because the cache key is `sha256(question)`, one cold pass is exactly 70 cache misses and the warm pass replays the identical set. Each response's `cached` flag is asserted (`cold ⇒ false`, `warm ⇒ true`), so the load test validates the cache contract instead of trusting status codes alone.
4. **Cold and warm are separate modes with different executors.** `shared-iterations` guarantees each pooled question is asked once (uncached distribution); `constant-vus` sustains cached throughput for a fixed duration. VU count, duration and pool size are environment inputs, so the 10/50-VU matrix is four commands, not four scripts.
5. **Thresholds catch misconfiguration, not performance regressions.** Cold runs fail above 5% error responses, warm runs above 1%; `--no-thresholds` bypasses them. The harness is never run in CI because uncached requests cost provider tokens and credentials; a `k6 inspect` syntax check runs on eval-touching PRs instead.
6. **Operational prerequisites stay explicit.** Uncached runs need `GOOGLE_API_KEY`; the default 30 req/min IP rate limit must be raised via `RATE_LIMIT_PER_MIN` (and restored) for the run; the cache must be flushed before each cold pass. `eval/loadtest/README.md` captures the exact sequence and the troubleshooting table.

### Consequences
- The 10/50 × cached/uncached matrix is reproducible from the repo alone using the pinned image; numbers describe a local dev stack and are presented as relative measurements.
- Uncached p95 interpolates 70 samples per flush and is LLM-bound (provider latency dominates), both stated in the README to prevent over-reading.
- The warm path is verifiable without credentials: seeding one Redis key with a valid `CachedResponse` exercises the full cached flow (`cached: true`, thresholds green); full cold numbers remain pending a provider key.
- Streaming latency (time-to-first-token on `/api/chat/stream`) is explicitly out of scope and left as future work.

---

## ADR 0009: Agent Mode — Native Tool Calling on the Existing Provider Layer

- **Status:** Accepted
- **Date:** 2026-10-18
- **Issue:** #35 (backlog #30)

### Context
Backlog item #30 (stretch) asks for an *agent mode*: the model should decide what to retrieve instead of always running one fixed RAG pass, using tools such as search, cross-game comparison and release-year lookup. The obvious library choice — LangChain agents (which the README incorrectly claimed as part of the stack) — has never been a dependency. The AI service is deliberately dependency-light: hand-rolled `httpx` providers for Gemini and OpenAI (`app/generation/llm.py`) and a plain prompt builder. Adding an agent framework would pull in a large transitive tree for what is, here, a bounded loop over two REST APIs that already support function calling natively. The acceptance criterion is behavioural: the agent must choose the correct tool for ≥ 8/10 labeled prompts (`eval/agent_cases.jsonl`).

### Decision
1. **Native tool calling, zero new dependencies.** The provider abstraction gains `generate_with_tools(messages, tools) -> AgentTurn`, where `AgentTurn` carries optional text plus `ToolCall(name, arguments, call_id)` items. Gemini maps the neutral transcript to `functionDeclarations` / `functionCall` / `functionResponse`; OpenAI maps it to `tools` / `tool_calls` / `tool` messages. The base-class default raises `LLMError`, so a future provider must opt in explicitly.
2. **One provider-neutral transcript.** The loop speaks a small message format (`system` / `user` / `assistant` with `tool_calls` / `tool` with a JSON payload); adapters translate it (`_gemini_contents` merges consecutive same-role turns, as Gemini requires alternating roles).
3. **Three deterministic tools with registry-owned citations.** `search_game_docs(game, question)`, `compare_games(game_a, game_b)` and `list_games_by_year(year)` live in `app/agent/tools.py`. The registry injects the same hybrid-retrieval callable as `/query` (`_retrieve_passages`, refactored out of `_embed_and_retrieve`), assigns stable `[N]` citation numbers across all tool calls, registers each evidence passage once (first-seen order) and caches identical calls, so both the transcript and the final `sources` list stay bounded and duplicate-free. The year catalog is static, sourced from the manuals' `Release Date` lines and covered by a corpus-audit test.
4. **Bounded loop with a grounded fallback.** `run_agent` (`app/agent/loop.py`) allows at most `max_steps` model turns (request 1–6; service default 3 via `agent_default_max_steps`). If the budget is exhausted while the model keeps calling tools, the loop closes deterministically: with evidence it answers through the standard `build_rag_prompt`; without evidence it returns the exact RAG refusal. A tool-happy model can neither loop forever nor answer ungrounded.
5. **`POST /agent` on the AI service only.** The endpoint returns `answer`, the collected `sources`, `tools_used` (name + arguments in call order), `steps` and `latency_ms`; errors map exactly like `/query` (502 embedding/LLM, 503 vector store). Gateway routing, UI and caching are out of scope for this item, mirroring how previous features landed service-first.
6. **Routing is evaluated offline-first.** `eval/agent_eval.py` scores the 10 labeled prompts against `POST /agent` (subset argument matching tolerates `game_a`/`game_b` order and alias forms; the guard fails below 80%). `tests/test_agent.py` scripts the model with a fake provider, so the loop, registry and both provider adapters are covered without an API key; the live ≥ 8/10 run remains pending a provider key (same blocker as ADR 0002–0004).
7. **The README's LangChain claim is removed.** The tech-stack line and acknowledgments now describe the actual stack (custom httpx providers, fastembed).

### Consequences
- Agent mode is dependency-free and reviewable: one new package (`app/agent/`), one tool-calling method per vendor, one endpoint, and deterministic tests for every path except live model routing.
- The neutral transcript is the new contract; adding a provider or a tool is additive, and tools must go through the registry to preserve citation/source invariants.
- Latency is strictly higher than `/query`: each tool round-trip adds a retrieval plus one model call (bounded at `max_steps` turns plus the final answer). `_log_agent` emits `tools_used`, `steps` and per-stage timings per request.
- Live routing quality (≥ 8/10) and fallback answer quality still need a credentialed run; the offline harness makes that a one-command follow-up when a key is available.

---

## ADR 0010: Groq as an OpenAI-Compatible LLM Provider and Key-Free Local Embeddings

- **Status:** Accepted
- **Date:** 2026-10-18
- **Issue:** owner request (provider portability)

### Context
The owner plans to run RetroGPT on the Groq API (`https://api.groq.com/openai/v1`) instead of Gemini/OpenAI. Groq serves open-weight text models behind an OpenAI-compatible REST API — `chat/completions` with streaming (SSE + `[DONE]`), function calling with the same `tools` / `tool_calls` / `role: tool` message shapes, and Bearer auth — so the existing hand-rolled `httpx` provider layer needs no new dependency and no protocol work. Two concrete differences matter: Groq deprecates `max_tokens` in favour of `max_completion_tokens`, and it rejects the optional `messages[].name` field. Separately, Groq does not offer an embeddings API, and the credentialed-eval backlog (ADR 0002–0004, README metrics) has been blocked on key availability; embeddings are the one layer that cannot move to Groq.

### Decision
1. **Groq is a first-class provider via configuration.** `LLM_PROVIDER=groq` + `GROQ_API_KEY` + `LLM_MODEL` (e.g. `openai/gpt-oss-120b`) selects it; Gemini stays the default. `GroqLLM` subclasses `OpenAILLM`; the parent gained a `base_url` parameter, a `_TOKEN_LIMIT_FIELD` class attribute and a `_PROVIDER_LABEL` used in error messages, so payload, streaming and tool-calling logic are shared and each vendor divergence is a small override. The provider-neutral agent transcript is untouched — the OpenAI translation already omits `messages[].name`.
2. **Embeddings gain a local fastembed option instead of a Groq one.** `EMBEDDING_PROVIDER=fastembed` runs `BAAI/bge-small-en-v1.5` (384-dim) on CPU through the fastembed/ONNX stack already shipped for the cross-encoder re-ranker — no API key, no second hosted vendor. The provider contract is unchanged (`embed_documents` / `embed_query`, async via `asyncio.to_thread`); Gemini and OpenAI embeddings remain available and unchanged.
3. **Collection sizing becomes explicit and mostly automatic.** Settings gain `VECTOR_SIZE` (default 768); the ingestion CLI derives the dense-vector size from a small known-model registry (`embedding.dimension`) when possible, accepts `--vector-size` as an explicit override, and falls back to `VECTOR_SIZE`. `ensure_collection` verifies the stored dense-vector size and raises a clear `VectorStoreError` on mismatch (unless the caller explicitly opts into recreation), so switching embedding models can never mix incompatible vectors silently; doing so means re-ingesting into a fresh collection, which the README now states.
4. **Tests never touch the network.** Groq providers are covered with mocked `httpx` (base URL, Bearer key from settings, `max_completion_tokens` payload, streaming deltas + `[DONE]`, tool-call round-trip without `name`); fastembed is covered with an injected stub model, so CI downloads nothing and spends no credits.

### Consequences
- The stack runs end-to-end on Groq + local embeddings with a single free-tier-or-paid key, unblocking the credentialed eval and load-test numbers that ADR 0002–0004 left pending.
- Cost is negligible: `openai/gpt-oss-120b` is $0.15/$0.60 per M tokens, so the full eval + load-test validation suite is a few cents; the free tier (8 K TPM binding, ~200 K TPD documented — see ADR 0011) still covers light runs.
- The OpenAI-compatible seam is now a documented extension point: any further OpenAI-compatible vendor is a small subclass, and `_TOKEN_LIMIT_FIELD` / `_PROVIDER_LABEL` capture the known divergences.
- `fastembed` embeddings are English-centric and CPU-bound; for multilingual corpora or hosted quality, `EMBEDDING_PROVIDER=gemini`/`openai` remain one env var away (with a re-ingest and matching `VECTOR_SIZE`).
- Groq's gpt-oss models do not support parallel tool calls; the agent loop already copes with one call per turn, so routing behaviour is unaffected.

---

