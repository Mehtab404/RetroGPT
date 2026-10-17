# Evaluation

Measures RetroGPT's retrieval and answer quality against a labeled dataset.

## Files

| Path | Purpose |
|---|---|
| `dataset.jsonl` | 70 labeled questions (Issue #18) |
| `run_eval.py` | Stdlib-only evaluation harness (Issue #19) |
| `results/` | Saved runs — `<date>-<label>.json` + `.md` |
| `requirements-eval.txt` | Optional RAGAS dependencies (`--with-ragas`) |
| `tests/` | Unit tests for the harness (no services needed) |

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
and a real `GOOGLE_API_KEY`/`OPENAI_API_KEY` in the AI service for LLM answers.

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

## Metrics

- **Hit Rate@k** — share of answerable questions whose top-k retrieved
  documents include an expected source. Target: ≥ 85%.
- **MRR** — mean reciprocal rank of the first expected source. Target: ≥ 0.70.
- **Refusal accuracy** — share of unanswerable questions answered with
  "I don't know …" instead of a hallucination.
- **Latency** — mean / p50 / p95 per uncached request.
- **RAGAS** (optional) — answer faithfulness and relevancy; contexts are
  approximated with the service's returned snippets.

## RAGAS setup

RAGAS needs its own judge LLM. Install and run:

```powershell
docker compose exec ai-service pip install -r /eval/requirements-eval.txt
docker compose exec -e OPENAI_API_KEY=$env:OPENAI_API_KEY ai-service python /eval/run_eval.py --with-ragas
```

Without the package or key, the run continues and the report notes why RAGAS
was skipped.

## CI regression guard (#23)

`.github/workflows/ci.yml` runs `eval/tests` on every PR. When a live stack
with credentials is available, a small subset can guard quality:

```powershell
docker compose exec ai-service python /eval/run_eval.py `
  --limit 10 --label ci --min-hit-rate 0.80 `
  --baseline /eval/results/baseline.json
```
