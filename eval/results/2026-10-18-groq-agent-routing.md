# RetroGPT agent-routing evaluation — groq-agent-routing

- Generated: 2026-10-18 11:26:07 UTC
- API: `http://localhost:8000/agent`
- Cases: `/eval/agent_cases.jsonl` (10 prompts)

## Summary

| Metric | Value |
|---|---|
| Routing pass rate | 90.0% (9/10) |
| Errors | 1 |
| Mean model turns | 2.00 (max 2) |
| Latency mean / p50 / p95 (ms) | 10961 / 14844 / 21249 |

## Routing by tool

| Tool | n | Passed | Pass rate |
|---|---|---|---|
| `compare_games` | 2 | 2 | 100% |
| `list_games_by_year` | 2 | 2 | 100% |
| `search_game_docs` | 5 | 5 | 100% |

## Cases

| Case | Expected | Called | Pass |
|---|---|---|---|
| agent-01 | `search_game_docs` | `search_game_docs` | ✅ |
| agent-02 | `search_game_docs` | `search_game_docs` | ✅ |
| agent-03 | `search_game_docs` | `search_game_docs` | ✅ |
| agent-04 | `search_game_docs` | `search_game_docs` | ✅ |
| agent-05 | `search_game_docs` | `search_game_docs` | ✅ |
| agent-06 | `compare_games` | `compare_games` | ✅ |
| agent-07 | `compare_games` | `compare_games` | ✅ |
| agent-08 | `compare_games` | — | ⚠️ |
| agent-09 | `list_games_by_year` | `list_games_by_year` | ✅ |
| agent-10 | `list_games_by_year` | `list_games_by_year` | ✅ |

## Details

- **agent-08** (error) — HTTP 502 from http://localhost:8000/agent: {"detail":"LLM provider error: LLM API failed after 4 attempts: Client error '429 Too Many Requests' for url 'https://api.groq.com/openai/v1/chat/completions'\nFor more information check: https://deve
