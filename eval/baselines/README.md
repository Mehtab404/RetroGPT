# Committed evaluation baselines

Baselines used by the CI regression guard (Issue #23). Each file is a JSON
report produced by an eval harness run; the guard compares the current run's
`metrics.hit_rate` against the baseline and fails when the drop exceeds
5 points (`--max-hit-rate-drop 0.05`).

| File | Generated with | Notes |
|---|---|---|
| `ci-offline.json` | `python eval/retrieval_eval.py --corpus data/raw --dataset eval/dataset.jsonl --limit 10 --embedder hashing --no-rerank --label ci-offline` (against Qdrant v1.7.4) | Deterministic offline retrieval baseline for the first 10 dataset questions: Hit Rate@5 = 100% (9 answerable), MRR 1.000. Re-generate only when the dataset subset, corpus, or guard configuration intentionally changes. |

The credentialed end-to-end baseline (hosted embeddings + LLM answers) will be
added to `eval/results/` once an LLM provider key is available; it is not used
by CI.
