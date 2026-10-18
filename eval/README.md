# Evaluation

Measures RetroGPT's retrieval and answer quality against a labeled dataset.

## Files

| Path | Purpose |
|---|---|
| `dataset.jsonl` | 70 labeled questions (Issue #18) |
| `run_eval.py` | Stdlib-only evaluation harness (Issue #19) |
| `retrieval_eval.py` | In-process retrieval evaluation: chunking sweeps + CI regression guard (Issues #22/#23) |
| `agent_cases.jsonl` | 10 labeled tool-routing prompts for agent mode (Issue #35) |
| `agent_eval.py` | Agent-routing harness: measures tool choice against `POST /agent` (Issue #35) |
| `baselines/` | Committed baselines for the CI regression guard (Issue #23) |
| `results/` | Saved runs — `<date>-<label>.json` + `.md` |
| `requirements-eval.txt` | Optional RAGAS dependencies (`--with-ragas`) |
| `tests/` | Unit tests for the harness (no services needed) |
| `loadtest/` | k6 load-test harness: cached vs. uncached p50/p95 at 10/50 concurrent users (Issue #33) |

## Dataset schema

One JSON object per line:

```json
{
  "id": "ast-01",
  "game": "asteroids",
  "category": "easy",
  "question": "How many points is a large asteroid worth in Asteroids?",
  "expected_answer": "20 points.",
  "expected_source_ids": ["asteroids_manual"]
}
```

- `category` — `easy` | `hard` | `multi-hop` | `unanswerable`
- `expected_source_ids` — **document slugs** (file stems of `data/raw/<game>/<file>.md`,
  e.g. `asteroids_manual`), not chunk ids. The harness normalizes retrieved
  `source` paths to stems before comparing. Unanswerable items have no expected
  sources and expect a refusal.
- `game` — slug, or `multi` for cross-game questions.

## Running

Inside the compose stack (the AI service container mounts `./eval` at `/eval`):

```powershell
docker compose exec ai-service python /eval/run_eval.py --label baseline
```

Or standalone against any running service:

```powershell
python eval/run_eval.py --api-url http://localhost:8000 --label baseline
python eval/run_eval.py --api-url http://localhost/api --endpoint /api/chat --api-key change-me
```

Requires a populated Qdrant collection (`python -m scripts.ingest /data/raw`)
and a real LLM provider key in the AI service for answers — `GOOGLE_API_KEY`
(default), or `GROQ_API_KEY` with `LLM_PROVIDER=groq`/`LLM_MODEL` set.

### Useful flags

| Flag | Meaning |
|---|---|
| `--label NAME` | Report label, e.g. `baseline`, `hybrid-search` |
| `--limit N` | Evaluate only the first N questions |
| `--top-k K` | Retrieval cut-off for Hit@k (default 5) |
| `--use-game-filter` | Scope each query by `game_slug` (off by default) |
| `--with-ragas` | Also compute RAGAS faithfulness + answer relevancy |
| `--min-hit-rate 0.85` | Exit 1 if Hit Rate falls below the value |
| `--baseline results/….json` | Exit 1 if Hit Rate drops > 5 points vs. that run |
| `--quiet` | Suppress per-question progress |

Exit codes: `0` ok · `1` quality guard failed · `2` fatal (bad dataset, service down).

## Retrieval-only evaluation (`retrieval_eval.py`, #22/#23)

`run_eval.py` measures the *service* (HTTP + generated answers) and needs API
keys.  `retrieval_eval.py` instead imports the ai-service code directly and
scores the **real retrieval stack** (chunking → BM25 → hybrid search →
optional re-ranking) against a scratch Qdrant collection — no LLM, no API
key required.  Answers are never generated, so refusal accuracy/RAGAS are
`n/a` by design.

Run it inside the ai-service container (the only place with the app + deps):

```powershell
# Quick single-configuration run (10 questions)
docker compose exec ai-service python /eval/retrieval_eval.py --limit 10

# Chunking sweep (#22): sizes x overlaps, full dataset
docker compose exec ai-service python /eval/retrieval_eval.py `
  --sweep-sizes 400,800,1200 --sweep-overlaps 0,100,120 `
  --embedder fastembed --fastembed-cache /data/fastembed_cache --no-rerank `
  --label chunking-sweep

# CI regression guard (#23): fail if Hit Rate drops > 5 points vs. baseline
docker compose exec ai-service python /eval/retrieval_eval.py `
  --limit 10 --embedder hashing --no-rerank `
  --baseline /eval/baselines/ci-offline.json
```

Embedders:

| `--embedder` | Notes |
|---|---|
| `hashing` (default) | Deterministic feature hashing, fully offline — for CI guards, not quality claims |
| `fastembed` | Local ONNX model (`BAAI/bge-small-en-v1.5` by default); `--fastembed-cache` persists the download |
| `gemini` / `openai` | Hosted embeddings; needs `GOOGLE_API_KEY` / `OPENAI_API_KEY` in the container |

The service itself also accepts `EMBEDDING_PROVIDER=fastembed` (local ONNX, no
API key) for key-free ingestion and queries.

Notes:

- The scratch collection (`--collection`, default `retrogpt_eval`) is deleted
  and re-created on every configuration run; it never touches `retro_games`.
- Re-ranking follows `RERANK_ENABLED` unless `--rerank`/`--no-rerank` is
  passed; sweeps usually pass `--no-rerank` so chunking is measured on its own.
- Guards (`--baseline`, `--min-hit-rate`, `--max-hit-rate-drop`) apply to
  single-configuration runs; baseline JSON is the file written by any earlier
  run of `retrieval_eval.py` or `run_eval.py`.

## Metrics

- **Hit Rate@k** — share of answerable questions whose top-k retrieved
  documents include an expected source. Target: ≥ 85%.
- **MRR** — mean reciprocal rank of the first expected source. Target: ≥ 0.70.
- **Refusal accuracy** — share of unanswerable questions answered with
  "I don't know …" instead of a hallucination.
- **Latency** — mean / p50 / p95 per uncached request.
- **RAGAS** (optional) — answer faithfulness and relevancy; contexts are
  approximated with the service's returned snippets.

Latest credentialed run (2026-10-18, Groq `openai/gpt-oss-120b` + fastembed
`BAAI/bge-small-en-v1.5`): Hit Rate@5 **100% (63/63)**, MRR **1.000**, refusal
accuracy **100% (6/6)**, one provider `429` error — report:
[`results/2026-10-18-groq-e2e.md`](./results/2026-10-18-groq-e2e.md); full
conditions in [ADR 0011](../docs/decisions.md).

## RAGAS setup

RAGAS needs its own judge LLM. Install and run:

```powershell
docker compose exec ai-service pip install -r /eval/requirements-eval.txt
docker compose exec -e OPENAI_API_KEY=$env:OPENAI_API_KEY ai-service python /eval/run_eval.py --with-ragas
```

Without the package or key, the run continues and the report notes why RAGAS
was skipped.

## CI regression guard (#23)

`.github/workflows/ci.yml` runs `eval/tests` on every PR. On top of that,
`.github/workflows/eval.yml` guards retrieval quality: on PRs touching
`ai-service/`, `eval/` or `data/raw/` it runs the first 10 dataset questions
through the **real retrieval stack** (chunking → BM25 → hybrid RRF,
re-ranking disabled) against a throwaway Qdrant, using deterministic hashing
embeddings — no API keys required, so it runs on forks and costs nothing.
The job fails when Hit Rate drops more than 5 points below the committed
baseline `eval/baselines/ci-offline.json`.

Reproduce the exact check locally:

```powershell
docker compose exec ai-service python /eval/retrieval_eval.py `
  --corpus /data/raw --dataset /eval/dataset.jsonl --limit 10 `
  --embedder hashing --no-rerank --label ci-offline `
  --baseline /eval/baselines/ci-offline.json
```

Exit codes: `0` passed · `1` guard failed · `2` fatal. The run's JSON /
Markdown report is uploaded as the `retrieval-eval-report` CI artifact.

The credentialed end-to-end eval (`run_eval.py` with Gemini, Groq or OpenAI)
remains the source of truth for answer-quality metrics; the offline guard only
protects retrieval from regressions.

## Load testing (`loadtest/`, #33)

[`loadtest/chat.js`](./loadtest/chat.js) is a k6 script that drives
`POST /api/chat` at 10 and 50 concurrent users, cached and uncached, against
the running compose stack (Nginx → gateway → Redis/AI service). Uncached runs
need a provider key (they reach the LLM); cached runs replay a warmed cache
and work without one. The full runbook — rate-limit raise, cache flush,
reading p50/p95, caveats — lives in
[`loadtest/README.md`](./loadtest/README.md). Latest run (2026-10-18):
cached-only, p95 45 ms @ 10 VUs / 145 ms @ 50 VUs — uncached runs skipped on
quota (see that README).

## Agent routing evaluation (`agent_eval.py`, #35)

`agent_eval.py` measures whether **agent mode** (`POST /agent` on the AI
service) picks the right tool with the right arguments. It replays the 10
labeled prompts in `agent_cases.jsonl` (5 × `search_game_docs`,
3 × `compare_games`, 2 × `list_games_by_year`) and fails when the pass rate
drops below `--min-pass-rate` (default 0.8 — the #35 acceptance target is
≥ 8/10).

Matching is deliberately lenient about *how* a correct route is phrased —
extra arguments are ignored, `game_a`/`game_b` may be swapped, and
`Pac-Man`/`pacman`, `space_invaders`/`Space Invaders` compare equal — but a
wrong tool or wrong game fails the case.

Needs a real provider key (the model chooses the calls); the harness itself
is stdlib-only:

```powershell
docker compose exec ai-service python /eval/agent_eval.py --label agent-routing
```

Reports land in `results/<date>-agent-routing.{json,md}`. Exit codes:
`0` ok · `1` pass rate below the guard · `2` fatal (missing cases, service
down). Latest credentialed run (2026-10-18): **90% (9/10)**, above the ≥ 8/10
target; the single miss was a provider `429` error, not a routing mistake
([`results/2026-10-18-groq-agent-routing.md`](./results/2026-10-18-groq-agent-routing.md)).

The unit tests for the harness (`tests/test_agent_eval.py`) and for the
agent loop itself (`ai-service/tests/test_agent.py`) run in CI like any
other test — no key required, because they script the model.
