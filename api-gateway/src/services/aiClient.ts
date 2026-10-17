/**
 * HTTP client for communicating with the Python AI service.
 *
 * Features:
 * - 15 s request timeout
 * - Automatic retry with exponential back-off (up to 3 attempts) for
 *   transient network errors and 5xx responses.
 * - Typed request / response models matching the AI service wire format.
 */

import axios, { AxiosError, AxiosInstance } from 'axios';
import { config } from '../config';

// ---------------------------------------------------------------------------
// Wire-format types (mirrors ai-service/app/schemas.py)
// ---------------------------------------------------------------------------

export interface AiQueryRequest {
  question: string;
  top_k?: number;
  game_slug?: string | null;
}

export interface SourceDoc {
  id: string;
  title: string;
  score: number;
  snippet: string;
}

export interface AiQueryResponse {
  answer: string;
  sources: SourceDoc[];
  latency_ms: number;
  cached: boolean;
}

// ---------------------------------------------------------------------------
// Retry configuration
// ---------------------------------------------------------------------------

const MAX_RETRIES = 3;
const INITIAL_RETRY_DELAY_MS = 300;

/** Returns true for errors that are worth retrying. */
function isRetryable(error: AxiosError): boolean {
  // Network-level failures (no response at all)
  if (!error.response) return true;

  // 5xx server errors
  const status = error.response.status;
  return status >= 500 && status <= 599;
}

/** Sleep for `ms` milliseconds. */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ---------------------------------------------------------------------------
// Client factory
// ---------------------------------------------------------------------------

function createAxiosInstance(): AxiosInstance {
  return axios.create({
    baseURL: config.aiServiceUrl,
    timeout: 15_000, // 15 s
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
  });
}

// Singleton instance — can be swapped in tests via `setAxiosInstance`.
let axiosInstance: AxiosInstance = createAxiosInstance();

/**
 * Override the underlying Axios instance.
 * Intended for unit tests only.
 */
export function setAxiosInstance(instance: AxiosInstance): void {
  axiosInstance = instance;
}

/**
 * Reset to the default Axios instance.
 * Call this in `afterEach` / `afterAll` when using `setAxiosInstance`.
 */
export function resetAxiosInstance(): void {
  axiosInstance = createAxiosInstance();
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Send a query to the AI service `POST /query` endpoint.
 *
 * Retries up to {@link MAX_RETRIES} times on transient / 5xx errors using
 * exponential back-off, then re-throws the last error.
 */
export async function queryAiService(request: AiQueryRequest): Promise<AiQueryResponse> {
  let lastError: AxiosError | null = null;

  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    try {
      const response = await axiosInstance.post<AiQueryResponse>('/query', request);
      return response.data;
    } catch (err) {
      const axiosErr = err as AxiosError;
      lastError = axiosErr;

      const shouldRetry = isRetryable(axiosErr) && attempt < MAX_RETRIES;
      if (!shouldRetry) {
        break;
      }

      // Exponential back-off: 300 ms, 600 ms, …
      const delay = INITIAL_RETRY_DELAY_MS * 2 ** (attempt - 1);
      await sleep(delay);
    }
  }

  // Re-throw the last error so callers can handle it appropriately.
  throw lastError;
}
