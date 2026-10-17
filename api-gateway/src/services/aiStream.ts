/**
 * Streaming HTTP client for the AI service `POST /query/stream` endpoint
 * (Issue #25).
 *
 * Uses Node's global `fetch` (undici) instead of axios because the answer
 * arrives as a chunked `text/event-stream` body that must be consumed
 * incrementally.  Behaviour mirrors `aiClient.queryAiService` where it
 * matters:
 *
 * - transient failures while *opening* the stream (connection errors and
 *   5xx responses) are retried up to 3 attempts with exponential back-off;
 * - non-retryable HTTP errors surface as {@link AiStreamHttpError} carrying
 *   the upstream status so the route can map them to its own envelope;
 * - once the stream is open, mid-stream failures propagate to the caller
 *   (there is no meaningful retry after tokens have been emitted).
 */

import type { ReadableStream as WebReadableStream } from 'stream/web';
import { config } from '../config';
import type { HistoryMessage } from './history';
import { createSseParser } from '../utils/sse';

// ---------------------------------------------------------------------------
// Wire-format types (mirrors ai-service/app/main.py streaming protocol)
// ---------------------------------------------------------------------------

export interface AiQueryRequest {
  question: string;
  top_k?: number;
  game_slug?: string | null;
  /** Bounded prior conversation for follow-up questions (Issue #27). */
  history?: HistoryMessage[];
}

export interface AiStreamEvent {
  event: string;
  data: Record<string, unknown>;
}

export interface AiStream {
  events: AsyncGenerator<AiStreamEvent>;
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/** The AI service answered with a non-2xx HTTP status before streaming. */
export class AiStreamHttpError extends Error {
  public readonly status: number;
  public readonly data: unknown;

  constructor(status: number, message: string, data: unknown = {}) {
    super(message);
    this.name = 'AiStreamHttpError';
    this.status = status;
    this.data = data;
  }
}

/** The AI service could not be reached (or timed out) before streaming. */
export class AiStreamNetworkError extends Error {
  public readonly timedOut: boolean;

  constructor(message: string, timedOut = false) {
    super(message);
    this.name = 'AiStreamNetworkError';
    this.timedOut = timedOut;
  }
}

// ---------------------------------------------------------------------------
// Connection handling
// ---------------------------------------------------------------------------

const MAX_ATTEMPTS = 3;
const INITIAL_RETRY_DELAY_MS = 300;
/** Max time to wait for response *headers* (generation happens after). */
const HEADER_TIMEOUT_MS = 15_000;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isRetryableStatus(status: number): boolean {
  return status >= 500 && status <= 599;
}

/**
 * Open the SSE connection, retrying transient failures.
 *
 * @throws AiStreamHttpError    Non-2xx response (after retries when retryable).
 * @throws AiStreamNetworkError Connection failure or header timeout.
 */
async function openConnection(request: AiQueryRequest, requestId?: string): Promise<Response> {
  let lastError: Error | null = null;

  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    };
    if (requestId) headers['x-request-id'] = requestId;

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), HEADER_TIMEOUT_MS);

    let response: Response | null = null;
    try {
      response = await fetch(`${config.aiServiceUrl}/query/stream`, {
        method: 'POST',
        headers,
        body: JSON.stringify(request),
        signal: controller.signal,
      });
    } catch {
      lastError = controller.signal.aborted
        ? new AiStreamNetworkError('Timed out waiting for the AI service', true)
        : new AiStreamNetworkError('Unable to reach the AI service');
    } finally {
      clearTimeout(timer);
    }

    if (response !== null) {
      if (response.ok) return response;

      let payload: unknown = {};
      try {
        payload = await response.json();
      } catch {
        // Non-JSON error body — keep the empty payload.
      }
      const error = new AiStreamHttpError(
        response.status,
        `AI service responded with HTTP ${response.status}`,
        payload,
      );
      if (!isRetryableStatus(response.status)) throw error;
      lastError = error;
    }

    if (attempt < MAX_ATTEMPTS) {
      await sleep(INITIAL_RETRY_DELAY_MS * 2 ** (attempt - 1));
    }
  }

  throw lastError ?? new AiStreamNetworkError('Unable to reach the AI service');
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Open a streaming query against the AI service.
 *
 * Resolves once the upstream responded 2xx; iterate `events` to consume the
 * SSE frames.  Mid-stream failures are thrown from the iterator.
 *
 * @param request   - Query payload (question + optional retrieval options).
 * @param requestId - Trace id forwarded as `x-request-id` (Issue #24).
 */
export async function openAiStream(request: AiQueryRequest, requestId?: string): Promise<AiStream> {
  const response = await openConnection(request, requestId);
  return { events: parseEventStream(response.body) };
}

/** Parse a web `ReadableStream` body into typed SSE events. */
async function* parseEventStream(
  body: WebReadableStream<Uint8Array> | null,
): AsyncGenerator<AiStreamEvent> {
  if (body === null) {
    throw new AiStreamNetworkError('AI service returned an empty stream');
  }

  const reader = body.getReader();
  const decoder = new TextDecoder();
  const parse = createSseParser();

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;

      for (const frame of parse(decoder.decode(value, { stream: true }))) {
        const data = safeJsonParse(frame.data);
        if (data !== null) yield { event: frame.event, data };
      }
    }

    for (const frame of parse(decoder.decode())) {
      const data = safeJsonParse(frame.data);
      if (data !== null) yield { event: frame.event, data };
    }
  } finally {
    // Releasing is a no-op after completion and frees the reader on abort.
    reader.releaseLock();
  }
}

function safeJsonParse(raw: string): Record<string, unknown> | null {
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
    return null;
  } catch {
    return null;
  }
}
