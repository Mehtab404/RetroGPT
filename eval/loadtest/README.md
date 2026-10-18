# Load Testing (k6)

Measures `/api/chat` latency — **cached vs. uncached at 10 and 50 concurrent
users** — for RetroGPT-PROJECT.md M6 [#28] (issue [#33]). The harness is
[`chat.js`](./chat.js), a [k6](https://k6.io) script run from the pinned
`grafana/k6:2.3.0` Docker image, so nothing needs to be installed on the host.

Design rationale (k6 vs. Locust, question pool, cold/warm modes):
[ADR 0008](../../docs/decisions.md).

## What is measured

`POST /api/chat` through the full local path: Nginx → API gateway
(auth → rate limiter → cache → AI service → MongoDB persistence).
Requests carry no `sessionId`, so every question maps to one cache key
(`sha256(question)` — see [`cache.ts`](../../api-gateway/src/services/cache.ts)).

| Mode | Scenario | k6 executor | Meaning |
|---|---|---|---|
| `cold` | one pass over the question pool after a cache flush | `shared-iterations` (one unique question per iteration) | uncached latency — includes embedding, retrieval, re-ranking and the LLM round-trip |
| `warm` | replays the same questions for `DURATION` | `constant-vus` | cached latency — Redis hit plus the same response framing; each response must report `"cached": true` |

The question pool is [`../dataset.jsonl`](../dataset.jsonl): all 70 labeled
questions are valid `/api/chat` requests, so a cold pass produces exactly one
cache miss per question. Uncached percentiles therefore come from 70 samples
per run — enough to read p50/p95, but not a large-sample benchmark.

## Prerequisites

- Local stack up and healthy (`docker compose ps`) with data ingested —
  see the Quick Start in the [root README](../../README.md).
- An LLM provider key (`GOOGLE_API_KEY`, or `GROQ_API_KEY` with
  `LLM_PROVIDER=groq`, or `OPENAI_API_KEY` with `LLM_PROVIDER=openai`) set in
  `.env`, otherwise every **uncached** request returns HTTP 502 from the AI
  service. **Cached** runs never reach the LLM and work without a key.
- Docker (the k6 container reaches the stack via `host.docker.internal`).

## Runbook (PowerShell)

```powershell
# 1. Raise the gateway rate limit (default 30 req/min would throttle the run)
$env:RATE_LIMIT_PER_MIN = "100000"
docker compose up -d api-gateway

# 2. Uncached @ 10 concurrent users — flush, then ask each question once
docker compose exec -T redis redis-cli FLUSHDB
docker run --rm -v "${PWD}/eval:/eval:ro" grafana/k6:2.3.0 run `
  -e BASE_URL=http://host.docker.internal/api -e API_KEY=change-me `
  -e MODE=cold -e VUS=10 `
  /eval/loadtest/chat.js

# 3. Cached @ 10 concurrent users — same questions, served by Redis
docker run --rm -v "${PWD}/eval:/eval:ro" grafana/k6:2.3.0 run `
  -e BASE_URL=http://host.docker.internal/api -e API_KEY=change-me `
  -e MODE=warm -e VUS=10 -e DURATION=60s `
  /eval/loadtest/chat.js

# 4. Repeat steps 2-3 with -e VUS=50 (flush first: cold must start cold)

# 5. Restore the default rate limit
Remove-Item Env:RATE_LIMIT_PER_MIN
docker compose up -d api-gateway
```

`API_KEY` must match the gateway key (`API_KEY` in `.env`, default
`change-me`). Bash equivalents: `$PWD` instead of `${PWD}`,
`export RATE_LIMIT_PER_MIN=100000`, and on Linux add
`--add-host=host.docker.internal:host-gateway` to `docker run`.

## Reading the results

k6 prints both numbers in its summary:

```
http_req_duration..............: avg=... min=... med=10.36ms p(90)=... p(95)=17.47ms max=...
```

- `med` is **p50**, `p(95)` is **p95**.
- `checks` must be 100%: `status is 200` and (warm) `cached flag is true`.
- A warm run that reports `cached flag is false` was not warmed: the cache was
  flushed after the cold pass or the 24 h TTL (`CACHE_TTL_SECONDS`) expired.

Measured 2026-10-18 (Groq + fastembed stack, `warm` mode, 60 s per run, rate
limit raised to 100,000 req/min for the runs and restored afterwards;
artifacts: `eval/results/2026-10-18-k6-warm-{10,50}vu.{json,log}`):

| Concurrent users | Uncached p50 / p95 | Cached p50 / p95 |
|---|---|---|
| 10 | not run* | 25 / 45 ms |
| 50 | not run* | 113 / 145 ms |

- Warm-run health: 100% checks (`status is 200` + `cached flag is true`),
  0.00% `http_req_failed`, 21,496 / 26,328 requests (358 / 438 req/s).
- *Uncached runs were skipped on purpose: each pass generates 70 answers
  (~110K tokens) and the remaining Groq free-tier budget (8K TPM / ~200K TPD)
  was already mostly spent by the credentialed evaluation; a 50-VU burst
  would mostly record `429` retries. Run them after a quota reset with the
  runbook above.
- Deviation: the cache was warmed with the first 10 dataset questions
  (`POOL=10`) instead of all 70 to stay inside the same budget; the cache-hit
  path is identical, only question variety changes.
- The table is mirrored in the root README's **Load Testing** section.

## Caveats

- **Local dev stack.** Numbers describe one developer machine running the
  `development` Docker targets; treat them as relative (cached vs. uncached,
  10 vs. 50 VUs), not as production SLOs.
- **Uncached runs are LLM-bound.** Most uncached latency is the provider
  round-trip; provider quota or throttling can add 429/502s at 50 VUs.
- **70-sample percentiles.** p95 of an uncached run interpolates 70 unique
  requests (the pool size); run-to-run jitter is expected.
- **Cached hits still write to MongoDB** (the gateway persists the exchange on
  cache hits by design), so cached latency includes two inserts.
- **Streaming endpoint not covered.** `/api/chat/stream` (time-to-first-token)
  is future work.
- **k6 thresholds** fail cold runs at >5% errors and warm runs at >1% — that
  is intentional when the rate limit or API key is misconfigured.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `status is 200` fails with 401 | `API_KEY` does not match the gateway |
| 429 responses | rate limit not raised, or gateway not recreated after `$env:RATE_LIMIT_PER_MIN` |
| 502 on every cold request | LLM provider key missing (`GOOGLE_API_KEY` / `GROQ_API_KEY` / `OPENAI_API_KEY`) or provider error (check `docker compose logs ai-service`) |
| Warm run reports `cached: false` | cache flushed after warming, or TTL expired |
| `dial tcp: lookup host.docker.internal` | Linux host: add `--add-host=host.docker.internal:host-gateway` |
