# RetroGPT evaluation — local-pre-push

- Generated: 2026-10-17 15:43:19 UTC
- API: `in-process (qdrant=http://host.docker.internal:6339, collection=retrogpt_eval)`
- Dataset: `eval/dataset.jsonl` (10 questions)
- Top-k: 5

## Metrics

| Metric | Value |
|---|---|
| Hit Rate@5 | 100.0% (9/9) |
| MRR | 1.000 |
| Refusal accuracy | n/a (0/1) |
| Answer faithfulness (RAGAS) | _not run_ |
| Answer relevancy (RAGAS) | _not run_ |

## Latency (uncached)

| Stat | ms |
|---|---|
| mean | 20 |
| p50 | 16 |
| p95 | 36 |

## Retrieval by category

| Category | n | Hit Rate | MRR |
|---|---|---|---|
| easy | 5 | 100.0% | 1.000 |
| hard | 3 | 100.0% | 1.000 |
| multi-hop | 1 | 100.0% | 1.000 |

## Failures

- **ast-09** (should have refused) — answer: ''

## Retrieval configuration

| Setting | Value |
|---|---|
| Chunk size | 800 |
| Chunk overlap | 120 |
| Chunks ingested | 38 |
| Embedder | hashing (dim=768) |
| Re-ranking | disabled |
| Top-k | 5 |

> Retrieval-only run: answers are not generated, so refusal
> accuracy and RAGAS are not applicable (`n/a` above).
