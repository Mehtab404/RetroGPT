/**
 * RetroGPT load test — POST /api/chat (non-streaming JSON route).
 *
 * Implements the harness for RetroGPT-PROJECT.md M6 #28:
 *   "Use k6 or Locust; record p50/p95 latency cached vs. uncached at
 *    10/50 concurrent users."
 *
 * Modes (env MODE):
 *   cold — one unique pass over the question pool against a flushed cache.
 *          Every response is a cache miss, so the aggregated
 *          `http_req_duration` is the uncached latency distribution.
 *   warm — constant load replaying the same questions after a cold pass
 *          (or an equivalent warm-up), so every response is a cache hit.
 *          The `cached` flag in each response body is asserted.
 *
 * Full runbook (rate-limit raise, cache flush, 10/50 VU commands):
 *   eval/loadtest/README.md
 *
 * Environment:
 *   BASE_URL      Gateway base URL             (default http://localhost/api)
 *   API_KEY       x-api-key value              (default change-me)
 *   MODE          cold | warm                  (default cold)
 *   VUS           concurrent users             (default 10)
 *   DURATION      warm-mode duration           (default 30s)
 *   ITERATIONS    cold-mode request budget     (default: pool size)
 *   POOL          cap the question pool        (default: all dataset entries)
 *   MAX_DURATION  cold-mode safety cap         (default 15m)
 */

import http from 'k6/http';
import { check } from 'k6';
import exec from 'k6/execution';

const BASE_URL = (__ENV.BASE_URL || 'http://localhost/api').replace(/\/+$/, '');
const API_KEY = __ENV.API_KEY || 'change-me';
const MODE = (__ENV.MODE || 'cold').toLowerCase();
const VUS = Number(__ENV.VUS || 10);
const DURATION = __ENV.DURATION || '30s';
const MAX_DURATION = __ENV.MAX_DURATION || '15m';

if (MODE !== 'cold' && MODE !== 'warm') {
  throw new Error(`MODE must be "cold" or "warm", got "${MODE}"`);
}
if (!Number.isInteger(VUS) || VUS < 1) {
  throw new Error(`VUS must be a positive integer, got "${__ENV.VUS}"`);
}

// The eval dataset supplies the question pool: every entry is a valid
// 3-500 character request and maps to exactly one cache key
// (sha256 of the normalised text), so one cold pass = one miss per entry.
const ALL_QUESTIONS = open('../dataset.jsonl')
  .split('\n')
  .map((line) => line.trim())
  .filter((line) => line.length > 0)
  .map((line) => JSON.parse(line))
  .map((item) => item.question)
  .filter((q) => typeof q === 'string' && q.length >= 3 && q.length <= 500);

const POOL = Number(__ENV.POOL || ALL_QUESTIONS.length);
if (!Number.isInteger(POOL) || POOL < 1) {
  throw new Error(`POOL must be a positive integer, got "${__ENV.POOL}"`);
}
const QUESTIONS = ALL_QUESTIONS.slice(0, Math.min(POOL, ALL_QUESTIONS.length));
if (QUESTIONS.length === 0) {
  throw new Error('Question pool is empty — check eval/dataset.jsonl and POOL');
}

const ITERATIONS = Math.min(Number(__ENV.ITERATIONS || QUESTIONS.length), QUESTIONS.length);
if (!Number.isInteger(ITERATIONS) || ITERATIONS < 1) {
  throw new Error(`ITERATIONS must be a positive integer, got "${__ENV.ITERATIONS}"`);
}

export const options =
  MODE === 'cold'
    ? {
        scenarios: {
          cold: {
            executor: 'shared-iterations',
            vus: VUS,
            iterations: ITERATIONS,
            maxDuration: MAX_DURATION,
          },
        },
        thresholds: {
          http_req_failed: ['rate<0.05'],
          checks: ['rate>0.95'],
        },
        summaryTrendStats: ['avg', 'min', 'med', 'p(90)', 'p(95)', 'max'],
      }
    : {
        scenarios: {
          warm: {
            executor: 'constant-vus',
            vus: VUS,
            duration: DURATION,
          },
        },
        thresholds: {
          http_req_failed: ['rate<0.01'],
          checks: ['rate>0.99'],
        },
        summaryTrendStats: ['avg', 'min', 'med', 'p(90)', 'p(95)', 'max'],
      };

export default function () {
  // shared-iterations hands every iteration a unique index, so a cold run
  // asks each pooled question exactly once; warm runs cycle the pool.
  const index = exec.scenario.iterationInTest % QUESTIONS.length;

  const res = http.post(
    `${BASE_URL}/chat`,
    JSON.stringify({ question: QUESTIONS[index] }),
    {
      headers: { 'Content-Type': 'application/json', 'x-api-key': API_KEY },
      tags: { mode: MODE },
    },
  );

  const body = res.json();

  check(res, {
    'status is 200': (r) => r.status === 200,
    [`cached flag is ${MODE === 'warm'}`]: () =>
      body !== null && typeof body === 'object' && body.cached === (MODE === 'warm'),
  });
}
