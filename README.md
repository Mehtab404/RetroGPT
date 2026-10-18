# RetroGPT — AI Assistant for Classic Games 🎮

> A production-grade Retrieval-Augmented Generation (RAG) chatbot that answers questions about classic video games (Pong, Asteroids, Centipede, Missile Command, Breakout, etc.) with verifiable citations from game manuals, wikis, and FAQs.

[![CI](https://github.com/Mehtab404/RetroGPT/actions/workflows/ci.yml/badge.svg)](https://github.com/Mehtab404/RetroGPT/actions/workflows/ci.yml)
[![Deploy](https://github.com/Mehtab404/RetroGPT/actions/workflows/deploy.yml/badge.svg)](https://github.com/Mehtab404/RetroGPT/actions/workflows/deploy.yml)

---

## 🚀 Features

- **Accurate retrieval:** Hybrid search (vector + BM25) with cross-encoder re-ranking
- **Source citations:** Every answer includes references to original documents
- **Token-by-token streaming:** Answers stream over Server-Sent Events (SSE) from the LLM to the UI
- **Production-ready:** Docker Compose, CI/CD, automated evaluation, structured JSON logs with request IDs
- **Fast responses:** Redis caching for sub-200ms repeated queries
- **Persistent chat:** MongoDB-backed session history
- **Secure:** API key authentication with rate limiting

---

## 📊 Metrics (v1.0)

Measured 2026-10-18 on the local Groq + fastembed stack
(`openai/gpt-oss-120b`, `BAAI/bge-small-en-v1.5`, Qdrant `retro_games`,
top-k 5, re-ranking on):

| Metric | Target | Actual |
|--------|--------|--------|
| Retrieval Hit Rate @5 | ≥ 85% | **100% (63/63 answerable)** |
| MRR (Mean Reciprocal Rank) | ≥ 0.70 | **1.000** |
| Refusal accuracy | — | **100% (6/6)** |
| Answer faithfulness (RAGAS) | ≥ 0.85 | not run¹ |
| p95 latency (uncached) | < 4 s | 11.7 s² |
| p95 latency (cached) | < 200 ms | **45 ms @ 10 / 145 ms @ 50 VUs** |
| Test coverage (AI service) | ≥ 70% | ≈96% (pytest-cov) |

> ¹ RAGAS needs an OpenAI judge key plus the optional `ragas` package; neither
> is configured on this stack, and the harness records the skip in its report
> instead of inventing numbers.
> ² Quota-throttled: the Groq free tier caps at 8K tokens/min, so the eval's
> 70 generations were paced by retried `429 Too Many Requests` responses (288
> logged; one question exhausted its retries → HTTP 502). Even the unthrottled
> single-request smoke test took 4.97 s, so the < 4 s target is not
> demonstrated on this hardware. Reports, raw numbers and conditions:
> [`eval/results/`](./eval/results) and [ADR 0011](./docs/decisions.md).

---

## 🏗️ Architecture

```
                ┌──────────────┐
   Browser ───► │    Nginx     │  (reverse proxy, rate limit)
                └──────┬───────┘
          ┌────────────┴─────────────┐
          ▼                          ▼
   ┌─────────────┐           ┌────────────────┐
   │  Frontend   │           │  API Gateway   │  Node.js + TypeScript
   │ React+Vite  │           │  - auth        │
   └─────────────┘           │  - validation  │──────► MongoDB (chat)
                             │  - caching     │──────► Redis (cache)
                             └───────┬────────┘
                                     │ HTTP + SSE
                                     ▼
                             ┌────────────────┐
                             │   AI Service   │  Python + FastAPI
                             │ - ingestion    │
                             │ - retrieval    │──────► Qdrant (vectors)
                             │ - re-ranking   │
                             │ - generation   │──────► Gemini / Groq / OpenAI
                             └────────────────┘
```

### Tech Stack

- **AI Service:** Python 3.12, FastAPI, httpx (native Gemini / OpenAI / Groq providers), fastembed (ONNX embeddings + re-ranking)
- **API Gateway:** Node.js 20 LTS, TypeScript, Express, zod
- **Frontend:** React, Vite, TypeScript
- **Vector DB:** Qdrant (hybrid search support)
- **Cache/DB:** Redis 7, MongoDB 7
- **Proxy:** Nginx
- **LLM:** Google Gemini (default), Groq or OpenAI (optional)
- **Evaluation:** RAGAS + custom metrics
- **Infrastructure:** Docker Compose, GitHub Actions CI/CD, AWS EC2/ECS

---

## 🛠️ Quick Start (Local Development)

### Prerequisites

- Docker Desktop (WSL2 backend for Windows)
- Git
- Node.js 20 LTS
- Python 3.12
- An LLM API key: [Gemini](https://ai.google.dev/) (default), [Groq](https://console.groq.com/keys) (has a free tier) or OpenAI

### Setup

```powershell
# Clone the repository
git clone https://github.com/Mehtab404/RetroGPT.git
cd RetroGPT

# Configure environment variables
copy .env.example .env
# Edit .env: add your LLM key (GOOGLE_API_KEY, or GROQ_API_KEY with LLM_PROVIDER=groq)

# Start all services
docker compose up -d --build

# Check service health
docker compose ps

# Ingest game documents into vector database
docker compose exec ai-service python -m scripts.ingest /data/raw

# Open the app
start http://localhost
```

### Service Endpoints (Local)

| Service | URL |
|---------|-----|
| App (via Nginx) | http://localhost |
| API Gateway | http://localhost:4000 |
| AI Service (docs) | http://localhost:8000/docs |
| Qdrant Dashboard | http://localhost:6333/dashboard |

### Development Commands

```powershell
# View logs
docker compose logs -f ai-service
docker compose logs -f api-gateway

# Run tests
docker compose exec ai-service pytest
docker compose exec api-gateway npm test

# Run evaluation suite
docker compose exec ai-service python /eval/run_eval.py

# Stop services
docker compose down

# Stop and remove volumes (fresh start)
docker compose down -v
```

### Switching the LLM provider (e.g. Groq)

The LLM provider and model are plain environment variables — no code changes
needed. Gemini stays the default; Groq and OpenAI are OpenAI-compatible
add-ons:

```dotenv
# Groq (https://console.groq.com/keys) — free tier available
LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-120b
GROQ_API_KEY=gsk_...

# Local embeddings (no API key) if you do not have a Google/OpenAI key;
# re-ingest and pick a fresh collection when switching embedding models —
# the ingestion CLI refuses to mix vector dimensions and errors clearly
# if the existing collection was built with a different-size model.
EMBEDDING_PROVIDER=fastembed
FASTEMBED_MODEL=BAAI/bge-small-en-v1.5
VECTOR_SIZE=384
```

Restart and verify with `docker compose up -d` then `docker compose exec
ai-service python -m scripts.ingest /data/raw` (first run downloads the
fastembed model into `/data/fastembed_cache`). See
[ADR 0010](./docs/decisions.md) for the design rationale.

### Observability

Both services emit one newline-delimited JSON object per log line with a shared
shape (`timestamp`, `level`, `service`, `event`, optional `request_id` plus
context fields), so logs can be piped straight into `jq` or any log shipper:

```json
{"timestamp":"2026-10-17T12:30:36.561Z","level":"info","service":"api-gateway","event":"http_request","request_id":"demo-24-trace-live","method":"POST","path":"/api/chat","status":502,"duration_ms":1980.42}
{"timestamp":"2026-10-17T12:30:36.553Z","level":"info","service":"ai-service","logger":"app.request","event":"http_request","request_id":"demo-24-trace-live","method":"POST","path":"/query","status":502,"duration_ms":158.27}
```

The gateway accepts an `x-request-id` header (sanitized, max 64 chars),
generates one when absent, echoes it on the response and forwards it to the
AI service — reusing the same id across retries — so every line belonging to
one request is correlated across both services. The AI service additionally
logs per-stage latencies (`embed`, `retrieve` with `hybrid_search_ms` /
`rerank_ms`, `generate`) in a single `query` event; streamed requests add a
`streamed` flag and `chunk_count` to the same events.

Set `LOG_LEVEL` (`debug` | `info` | `warn` | `error`, default `info`) in `.env`
to control verbosity; `debug` also logs `/health` probes. See
[ADR 0005](./docs/decisions.md) for the design rationale.

---

## 📁 Project Structure

```
retrogpt/
├── ai-service/          # Python FastAPI service
│   ├── app/
│   │   ├── ingestion/   # Document loaders, chunking, pipeline
│   │   ├── retrieval/   # Vector store, hybrid search, re-ranking
│   │   ├── generation/  # LLM abstraction, prompts
│   │   └── main.py
│   ├── scripts/
│   │   └── ingest.py    # CLI ingestion tool
│   └── tests/
├── api-gateway/         # Node.js + TypeScript gateway
│   ├── src/
│   │   ├── routes/      # Chat, sessions endpoints
│   │   ├── middleware/  # Auth, rate limiting, errors
│   │   └── services/    # AI client, cache, DB
│   └── tests/
├── frontend/            # React + Vite UI
│   └── src/
├── eval/                # Evaluation framework
│   ├── dataset.jsonl    # Test questions + expected results
│   ├── run_eval.py      # End-to-end eval: hit rate, MRR, RAGAS metrics
│   ├── retrieval_eval.py # Offline retrieval eval: chunking sweeps + CI guard
│   └── baselines/       # Committed baselines for the CI regression guard
├── data/
│   ├── raw/             # Source game manuals (committed)
│   └── README.md        # Data sources & licenses
├── nginx/
│   ├── nginx.conf       # Dev reverse proxy
│   └── prod/            # Production edge templates (HTTP/HTTPS)
├── deploy/              # ECR build-push, remote deploy, Let's Encrypt scripts
├── .github/
│   └── workflows/       # CI/CD pipelines
├── docker-compose.yml
└── docker-compose.prod.yml   # Production stack (ECR images, secrets via env)
```

---

## 🧪 Testing

### Unit & Integration Tests

```powershell
# Python tests (ai-service)
docker compose exec ai-service pytest -v --cov=app

# Node tests (api-gateway)
docker compose exec api-gateway npm test -- --coverage

# End-to-end tests (optional)
docker compose exec frontend npm run test:e2e
```

### Evaluation Suite

```powershell
# End-to-end evaluation (needs an LLM key — GOOGLE_API_KEY / GROQ_API_KEY / OPENAI_API_KEY — for answers)
docker compose exec ai-service python /eval/run_eval.py --label baseline

# Retrieval-only evaluation — no API key needed (writes eval/results/)
docker compose exec ai-service python /eval/retrieval_eval.py --limit 10

# Chunking sweep (chunk sizes x overlaps, comparison table)
docker compose exec ai-service python /eval/retrieval_eval.py `
  --sweep-sizes 400,800,1200 --sweep-overlaps 0,100,120 `
  --embedder fastembed --no-rerank --label chunking-sweep

# CI regression guard: fail if Hit Rate drops > 5 points vs. baseline
docker compose exec ai-service python /eval/retrieval_eval.py `
  --limit 10 --embedder hashing --no-rerank `
  --baseline /eval/baselines/ci-offline.json
```

---

## 🔐 Security

- API key authentication via `x-api-key` header
- Rate limiting: 30 requests/minute per IP (configurable)
- Input validation with zod
- No secrets in code or logs
- CORS restricted to frontend origin
- Prompt injection guards in system prompts

---

## 📈 Evaluation & Quality

RetroGPT is measured against a 70-question labeled dataset (`eval/dataset.jsonl`):

- **Hit Rate @5:** Percentage of queries where the correct document appears in top 5 results
- **MRR (Mean Reciprocal Rank):** Average of 1/rank for first relevant result
- **Refusal accuracy:** Unanswerable questions correctly answered with "I don't know"
- **RAGAS Faithfulness / Answer Relevancy:** Optional LLM-judged answer quality

```powershell
# Run the evaluation against the local stack (writes eval/results/)
docker compose exec ai-service python /eval/run_eval.py --label baseline
```

Every run writes a JSON + Markdown report to `eval/results/`. Retrieval-only
runs (`eval/retrieval_eval.py`) additionally sweep chunking configurations
([ADR 0004](./docs/decisions.md)) and power the CI regression guard: on PRs
touching `ai-service/`/`eval/`, the first 10 questions are re-run against a
throwaway Qdrant with deterministic offline embeddings and fail the build if
Hit Rate drops more than 5 points below the committed baseline — no API keys
required. See [`eval/README.md`](./eval/README.md) for the dataset schema,
metric definitions, flags, and CI details.

### Load Testing

[`eval/loadtest/`](./eval/loadtest) contains a [k6](https://k6.io) harness that
drives `POST /api/chat` at 10 and 50 concurrent users, cached and uncached.
Measured 2026-10-18 (60 s per run, 100% `cached: true`, 0% failed requests):

| Concurrent users | Uncached p50 / p95 | Cached p50 / p95 |
|---|---|---|
| 10 | not run¹ | 25 / 45 ms |
| 50 | not run¹ | 113 / 145 ms |

> ¹ Uncached runs were not executed: one pass generates 70 answers (~110K
> tokens) and the remaining Groq free-tier budget (8K tokens/min, ~200K
> tokens/day) was already mostly consumed by the credentialed evaluation, so
> the runs would either exhaust the daily quota or measure `429` retries. The
> cached runs were warmed with a 10-question subset (`POOL=10`, documented
> deviation — the cache-hit path is identical). Throughput: 358 / 438 req/s;
> max latency 303 / 317 ms. Runbook and caveats:
> [`eval/loadtest/README.md`](./eval/loadtest/README.md); conditions and
> limitations: [ADR 0011](./docs/decisions.md).

---

## 🚢 Deployment

### Production Deployment (AWS)

A complete single-host production setup lives in
[`docker-compose.prod.yml`](./docker-compose.prod.yml) (images from ECR,
password-protected Redis/MongoDB, size-capped container logs, only Nginx
exposed) with helper scripts in [`deploy/`](./deploy):

```bash
# 1. Build and push images to AWS ECR (tags: git SHA + latest)
API_KEY=<public-demo-key> \
ECR_REGISTRY=<account>.dkr.ecr.<region>.amazonaws.com \
AWS_REGION=<region> \
  ./deploy/build-push.sh

# 2. On the EC2 host (repo checkout with .env.production + ./data):
IMAGE_TAG=<sha> ./deploy/remote-deploy.sh   # pull, apply, smoke-test /api/health

# 3. Seed the vector database (once)
docker compose --env-file .env.production -f docker-compose.prod.yml \
  exec ai-service python -m scripts.ingest /data/raw

# 4. Optional: HTTPS via Let's Encrypt (requires a domain)
DOMAIN=demo.example.com EMAIL=you@example.com ./deploy/init-letsencrypt.sh
```

Local dry run of the exact same stack (no AWS account needed):

```bash
cp .env.production.example .env.production
docker compose --env-file .env.production -f docker-compose.prod.yml up -d --build
curl http://localhost/api/health
```

See [`docs/deployment.md`](./docs/deployment.md) for the full AWS walkthrough
(ECR, security group, SSM secrets, ingestion, renewals, rollback).

> **Status:** the production stack, scripts, docs and CD workflow are complete
> and validated locally; the public demo URL is **pending an AWS account +
> credentials + domain**. Merging to `main` deploys automatically through
> [`.github/workflows/deploy.yml`](./.github/workflows/deploy.yml) — it skips
> with a notice until the repository secrets in the deployment guide are set.

### Environment Variables (Production)

Store secrets in AWS Systems Manager Parameter Store or Secrets Manager. Never commit `.env` files. Start from [`.env.production.example`](./.env.production.example).

Required variables:
- LLM provider key: `GOOGLE_API_KEY` (default LLM + embeddings), or `GROQ_API_KEY` with `LLM_PROVIDER=groq`, or `OPENAI_API_KEY`
- `API_KEY` (gateway authentication; also baked into the frontend bundle at build time)
- `DOMAIN`, `ECR_REGISTRY`
- `REDIS_PASSWORD`, `MONGO_USER`, `MONGO_PASSWORD`
- optional: `PUBLIC_ORIGIN`, `AI_WORKERS`, `EMBEDDING_PROVIDER=fastembed` (key-free local embeddings), tuning vars from the example file

---

## 📝 API Documentation

### Chat Endpoint

```bash
POST /api/chat
Headers: x-api-key: <your-api-key>
Body:
{
  "sessionId": "optional-session-uuid",
  "question": "How do you score points in Asteroids?"
}

Response:
{
  "sessionId": "uuid",
  "answer": "You score points by destroying asteroids... [1]",
  "sources": [
    {
      "id": "asteroids_manual_p3_c2",
      "title": "Asteroids Manual",
      "source": "/data/raw/asteroids/asteroids_manual.md",
      "score": 0.91,
      "snippet": "Large asteroids: 20 points..."
    }
  ],
  "latency_ms": 1820,
  "cached": false
}
```

### Streaming Chat (SSE)

`POST /api/chat/stream` is the streaming twin of `POST /api/chat`: instead of
one JSON response it relays the answer token-by-token as Server-Sent Events.
The UI uses it so answers appear as they are generated.

```bash
POST /api/chat/stream
Headers: x-api-key: <your-api-key>
         Accept: text/event-stream
Body:
{
  "sessionId": "optional-session-uuid",
  "question": "How do you score points in Asteroids?"
}
```

The response is an SSE stream of `event:` / `data:` frames:

```
event: meta
data: {"sources":[{"id":"asteroids_manual_p3_c2","title":"Asteroids Manual","score":0.91,"snippet":"..."}]}

event: token
data: {"text":"You score points "}

event: token
data: {"text":"by destroying asteroids [1]."}

event: done
data: {"sessionId":"uuid","latency_ms":1820,"cached":false}
```

- `meta` arrives once retrieval finishes and carries the sources for the answer.
- `token` carries one generated chunk (`text`).
- `done` closes the stream with the `sessionId`, upstream latency and whether
  the answer was served from the Redis cache.
- `error` (`{"code","message"}`) is sent if generation fails mid-stream; tokens
  already delivered remain valid.

Failures **before** the first frame (validation, AI service down, timeout) are
regular JSON error responses with the same status codes as `POST /api/chat`
(400 / 502 / 504). Completed streams are cached and persisted to MongoDB
exactly like non-streamed answers, and a cache hit is replayed as SSE with
`"cached": true` so clients use one code path.

### Agent Mode (Tool Calling)

`POST /agent` on the AI service (port 8000) answers with **native tool
calling** instead of a single retrieval pass: the model chooses between three
deterministic tools, the service executes them, and the final answer cites
the evidence they returned.

```bash
POST /agent
Body:
{
  "question": "Compare the scoring systems in Asteroids and Breakout.",
  "max_steps": 3
}

Response:
{
  "answer": "Asteroids awards 20 points per large asteroid [1], while Breakout ... [2]",
  "sources": [ ... ],
  "tools_used": [
    {"name": "compare_games", "arguments": {"game_a": "asteroids", "game_b": "breakout"}}
  ],
  "steps": 2,
  "latency_ms": 4310,
  "cached": false
}
```

| Tool | Purpose |
|---|---|
| `search_game_docs(game, question)` | Passages about one game |
| `compare_games(game_a, game_b)` | Side-by-side evidence for two games |
| `list_games_by_year(year)` | Knowledge-base games released in a year |

- `max_steps` caps model turns (1–6; service default 3). If the budget runs
  out mid-tool-use, the service closes with a deterministic grounded answer
  built from the evidence already collected.
- Errors mirror `POST /query`: 502 (embedding/LLM provider), 503 (vector store).
- Tool routing is measured by [`eval/agent_eval.py`](./eval/agent_eval.py)
  against 10 labeled prompts (target: ≥ 8/10). The route is currently
  AI-service-only — the API gateway and UI still use the standard chat path.

### Session History

```bash
GET /api/sessions/:sessionId
Headers: x-api-key: <your-api-key>

Response:
{
  "sessionId": "uuid",
  "messages": [
    {
      "role": "user",
      "content": "How do you score points?",
      "createdAt": "2026-10-17T12:00:00Z"
    },
    {
      "role": "assistant",
      "content": "You score by destroying asteroids... [1]",
      "sources": [...],
      "createdAt": "2026-10-17T12:00:02Z"
    }
  ]
}
```

Sessions are also the source of conversational context: `POST /api/chat` and
`POST /api/chat/stream` requests that carry a `sessionId` reload a bounded
window of the stored messages (the last 6 messages, at most 4000 characters
total) **server-side** so follow-up questions are interpreted against the real
conversation. History is context only — answers stay grounded in the retrieved
passages — and any history sent by a browser/client is ignored. Requests
without a `sessionId` behave exactly as before.

---

## 🤝 Contributing

This is a portfolio/learning project. Contributions are welcome!

### Development Workflow

1. Create a branch: `git checkout -b feat/issue-number-description`
2. Make changes following coding standards (see `RetroGPT-PROJECT.md`)
3. Write/update tests
4. Ensure CI passes: `docker compose exec ai-service pytest && docker compose exec api-gateway npm test`
5. Commit using [Conventional Commits](https://www.conventionalcommits.org/): `feat: add hybrid search (#20)`
6. Push and open a PR with `Closes #<issue-number>` in description

### Coding Standards

- **Python:** Type hints, ruff + black, pydantic models, 70%+ test coverage
- **TypeScript:** `strict: true`, no `any`, zod validation, proper error handling
- **Tests:** Mock external APIs in unit tests; use real services in integration tests

---

## 📚 Documentation

- [`RetroGPT-PROJECT.md`](./RetroGPT-PROJECT.md) — Complete project specification
- [`docs/decisions.md`](./docs/decisions.md) — Architecture decision records
- [`docs/deployment.md`](./docs/deployment.md) — AWS deployment guide (EC2 + ECR + HTTPS)
- [`data/README.md`](./data/README.md) — Data sources and licenses
- [AI Service API docs](http://localhost:8000/docs) — Swagger/OpenAPI (when running)

---

## 🎯 Roadmap

- [x] M0: Repository setup and infrastructure services
- [x] M1: RAG core (ingestion, retrieval, generation)
- [x] M2: API gateway (auth, caching, sessions)
- [x] M3: Frontend UI
- [x] M4: Evaluation suite and quality improvements
- [ ] M5: Production deployment (AWS + CI/CD) — structured logging, production compose and the CD workflow (`deploy.yml`) done; public URL pending AWS credentials
- [ ] M6: Polish (load tests, documentation) — streaming done (#25); agent mode with native tool calling done (#35); k6 load-test harness ready (numbers pending a provider key)

See [GitHub Issues](https://github.com/Mehtab404/RetroGPT/issues) for detailed task breakdown.

---

## 🐛 Known Issues

_None yet — this section will be updated as issues are discovered._

---

## 📄 License

This project is for educational and portfolio purposes. Game manuals and documentation are sourced from public archives and used under fair use for research and evaluation.

See [`data/README.md`](./data/README.md) for detailed source attributions and licenses.

---

## 💡 What I Learned / Challenges

_This section will be filled after M4-M6 completion._

### Key Takeaways
- Production RAG system design patterns
- Hybrid search and re-ranking techniques
- Evaluation-driven development for GenAI systems
- Docker multi-service orchestration
- End-to-end CI/CD with automated testing

### Challenges Overcome
- Retrieval accuracy: hybrid BM25 + vector search with RRF and cross-encoder
  re-ranking reached Hit Rate@5 = 100% and MRR = 1.000 on the 70-question
  evaluation ([ADR 0011](./docs/decisions.md))
- Latency vs. quality: Redis caching brings repeated queries to a 45–145 ms
  p95 (10–50 VUs) versus multi-second uncached generation, measured in
  [`eval/results/`](./eval/results)
- Chunking strategy: picked 800 chars / 120 overlap from an offline sweep
  ([ADR 0004](./docs/decisions.md)), pinned by the CI regression guard

---

## 🙏 Acknowledgments

- Classic game manuals from [Internet Archive](https://archive.org/)
- [fastembed](https://github.com/qdrant/fastembed) for local ONNX embeddings and cross-encoder re-ranking
- [Qdrant](https://qdrant.tech/) for hybrid vector search
- [RAGAS](https://github.com/explodinggradients/ragas) for evaluation framework

---

**Built with ❤️ as a portfolio project to demonstrate production-level RAG system development.**
