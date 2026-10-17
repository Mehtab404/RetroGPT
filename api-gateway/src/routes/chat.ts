/**
 * Chat route — POST /api/chat
 *
 * Request lifecycle:
 *  1. Validate incoming body with zod.
 *  2. Load the bounded prior conversation from the stored session
 *     (server-side only; failures are logged and treated as "no history" —
 *     Issue #27).
 *  3. Check Redis cache — return immediately on a hit (cached: true).
 *  4. Forward the question (plus the bounded history when present) to the
 *     AI service.
 *  5. Store the AI response in Redis for future requests.
 *  6. Persist both messages to MongoDB (non-blocking; DB errors don't fail
 *     the response).
 *  7. Return a structured response that includes an auto-generated (or
 *     echoed) sessionId and `cached: false`.
 */

import { randomUUID } from 'crypto';
import { Router, Request, Response, NextFunction } from 'express';
import { z } from 'zod';
import { queryAiService } from '../services/aiClient';
import {
  openAiStream,
  AiStreamHttpError,
  AiStreamNetworkError,
  type AiStream,
} from '../services/aiStream';
import { getCachedResponse, setCachedResponse, type CachedResponse } from '../services/cache';
import { boundHistory, type HistoryMessage } from '../services/history';
import { AppError } from '../middleware/errorHandler';
import { AxiosError } from 'axios';
import { Session } from '../models/session';
import { logger } from '../utils/logger';
import { formatSseEvent } from '../utils/sse';

// ---------------------------------------------------------------------------
// Validation schema
// ---------------------------------------------------------------------------

const chatRequestSchema = z.object({
  /** The user's question — 3 to 500 characters. */
  question: z
    .string()
    .min(3, 'question must be at least 3 characters')
    .max(500, 'question must be at most 500 characters'),

  /** Optional session identifier (UUID v4). */
  sessionId: z.string().uuid('sessionId must be a valid UUID v4').optional(),
});

export type ChatRequest = z.infer<typeof chatRequestSchema>;

/**
 * Validate the request body against {@link chatRequestSchema}.
 *
 * On failure responds with the standard 400 envelope and returns `null`.
 */
function validateChatRequest(req: Request, res: Response): ChatRequest | null {
  const parseResult = chatRequestSchema.safeParse(req.body);
  if (!parseResult.success) {
    res.status(400).json({
      error: {
        code: 'VALIDATION_ERROR',
        message: 'Invalid request data',
        details: parseResult.error.errors,
      },
    });
    return null;
  }
  return parseResult.data;
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

const router = Router();

/**
 * POST /api/chat
 *
 * Request body:
 *   { "question": "...", "sessionId": "optional-uuid" }
 *
 * Response:
 *   { "sessionId": "uuid", "answer": "...", "sources": [...],
 *     "latency_ms": 1234, "cached": true | false }
 */
router.post('/chat', async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  // 1. Validate request body
  const parsed = validateChatRequest(req, res);
  if (parsed === null) return;

  const { question, sessionId } = parsed;
  const resolvedSessionId = sessionId ?? randomUUID();

  // 2. Load the bounded prior conversation server-side (fail-open)
  const history = sessionId ? await loadRecentHistory(sessionId, req.requestId) : [];

  // 3. Check Redis cache (history-aware key when history is present)
  const cached = await getCachedAnswer(question, history);
  if (cached !== null) {
    // Cache hit — persist the exchange and return immediately
    await persistMessages(
      resolvedSessionId,
      question,
      cached.answer,
      cached.sources,
      req.requestId,
    );

    logger.info('chat_completed', {
      request_id: req.requestId,
      cached: true,
      latency_ms: cached.latency_ms,
      sources: cached.sources.length,
      question_chars: question.length,
    });

    res.status(200).json({
      sessionId: resolvedSessionId,
      answer: cached.answer,
      sources: cached.sources,
      latency_ms: cached.latency_ms,
      cached: true,
    });
    return;
  }

  // 3. Cache miss — proxy to AI service
  try {
    const upstreamStart = Date.now();
    const aiResponse = await queryAiService(
      history.length > 0 ? { question, history } : { question },
      req.requestId,
    );
    const upstreamMs = Date.now() - upstreamStart;

    // 4. Store in Redis (fire-and-forget; errors are swallowed inside)
    await cacheAnswer(
      question,
      {
        answer: aiResponse.answer,
        sources: aiResponse.sources,
        latency_ms: aiResponse.latency_ms,
      },
      history,
    );

    // 5. Persist messages to MongoDB (non-blocking to response)
    await persistMessages(
      resolvedSessionId,
      question,
      aiResponse.answer,
      aiResponse.sources,
      req.requestId,
    );

    logger.info('chat_completed', {
      request_id: req.requestId,
      cached: false,
      latency_ms: aiResponse.latency_ms,
      upstream_ms: upstreamMs,
      sources: aiResponse.sources.length,
      question_chars: question.length,
    });

    // 6. Respond
    res.status(200).json({
      sessionId: resolvedSessionId,
      answer: aiResponse.answer,
      sources: aiResponse.sources,
      latency_ms: aiResponse.latency_ms,
      cached: false,
    });
  } catch (err) {
    const axiosErr = err as AxiosError;

    if (axiosErr.isAxiosError) {
      // AI service returned an HTTP error
      if (axiosErr.response) {
        const status = axiosErr.response.status;

        if (status === 422) {
          // Validation error from AI service — pass through as 400
          logger.warn('chat_upstream_rejected', {
            request_id: req.requestId,
            upstream_status: status,
          });
          res.status(400).json({
            error: {
              code: 'VALIDATION_ERROR',
              message: 'AI service rejected the request',
              details: axiosErr.response.data,
            },
          });
          return;
        }

        // 5xx or other upstream errors
        const upstreamError: AppError = new Error(
          'AI service is unavailable. Please try again later.',
        );
        upstreamError.statusCode = 502;
        upstreamError.code = 'AI_SERVICE_ERROR';
        return next(upstreamError);
      }

      // Network / timeout error
      if (axiosErr.code === 'ECONNABORTED' || axiosErr.code === 'ETIMEDOUT') {
        const timeoutError: AppError = new Error(
          'Request to AI service timed out. Please try again.',
        );
        timeoutError.statusCode = 504;
        timeoutError.code = 'AI_SERVICE_TIMEOUT';
        return next(timeoutError);
      }

      // Connection refused / DNS failure
      const connError: AppError = new Error('Unable to reach AI service. Please try again later.');
      connError.statusCode = 502;
      connError.code = 'AI_SERVICE_UNREACHABLE';
      return next(connError);
    }

    // Unknown error — hand off to global error handler
    next(err);
  }
});

// ---------------------------------------------------------------------------
// Streaming route
// ---------------------------------------------------------------------------

/**
 * POST /api/chat/stream
 *
 * Streaming variant of `POST /api/chat` (Issue #25): relays the AI
 * service's Server-Sent Events to the client and accumulates the answer so
 * completed streams are cached and persisted exactly like non-streamed
 * requests.
 *
 * SSE events sent to the client:
 *   meta  — `{ sources: [...] }` once retrieval finished
 *   token — `{ text: "..." }` per generated chunk
 *   done  — `{ sessionId, latency_ms, cached }`
 *   error — `{ code, message }` if generation fails mid-stream
 *
 * Failures *before* the first SSE event are regular JSON responses with the
 * same status codes as `POST /api/chat` (400/502/504).
 */
router.post(
  '/chat/stream',
  async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    // 1. Validate request body
    const parsed = validateChatRequest(req, res);
    if (parsed === null) return;

    const { question, sessionId } = parsed;
    const resolvedSessionId = sessionId ?? randomUUID();

    // 2. Load the bounded prior conversation server-side (fail-open)
    const history = sessionId ? await loadRecentHistory(sessionId, req.requestId) : [];

    // 3. Cache hit — replay the stored answer as SSE so the client code
    //    path is identical for cached and fresh responses.
    const cached = await getCachedAnswer(question, history);
    if (cached !== null) {
      await persistMessages(
        resolvedSessionId,
        question,
        cached.answer,
        cached.sources,
        req.requestId,
      );

      logger.info('chat_completed', {
        request_id: req.requestId,
        cached: true,
        streamed: true,
        latency_ms: cached.latency_ms,
        sources: cached.sources.length,
        question_chars: question.length,
      });

      writeSseHeaders(res);
      writeSse(res, 'meta', { sources: cached.sources });
      writeSse(res, 'token', { text: cached.answer });
      writeSse(res, 'done', {
        sessionId: resolvedSessionId,
        latency_ms: cached.latency_ms,
        cached: true,
      });
      res.end();
      return;
    }

    // 3. Open the upstream stream. Failures here happen before any SSE byte
    //    is sent, so they can still become regular JSON error responses.
    let stream: AiStream;
    try {
      stream = await openAiStream(
        history.length > 0 ? { question, history } : { question },
        req.requestId,
      );
    } catch (err) {
      handleStreamOpenError(err, req, res, next);
      return;
    }

    writeSseHeaders(res);

    let clientClosed = false;
    res.on('close', () => {
      if (!res.writableEnded) clientClosed = true;
    });
    // Swallow late socket errors (e.g. client disconnects) so a dropped
    // connection never crashes the process.
    res.on('error', () => {
      clientClosed = true;
    });

    const startMs = Date.now();
    let answer = '';
    let sources: CachedResponse['sources'] = [];
    let upstreamLatencyMs: number | null = null;
    let chunkCount = 0;
    let failed = false;

    // 4. Relay events, accumulating the answer for cache/persistence.
    try {
      for await (const event of stream.events) {
        if (clientClosed) break;

        switch (event.event) {
          case 'meta': {
            sources = Array.isArray(event.data.sources)
              ? (event.data.sources as CachedResponse['sources'])
              : [];
            writeSse(res, 'meta', { sources });
            break;
          }
          case 'token': {
            const text = typeof event.data.text === 'string' ? event.data.text : '';
            if (text.length > 0) {
              answer += text;
              chunkCount += 1;
              writeSse(res, 'token', { text });
            }
            break;
          }
          case 'done': {
            upstreamLatencyMs =
              typeof event.data.latency_ms === 'number' ? event.data.latency_ms : null;
            break;
          }
          case 'error': {
            failed = true;
            const code = typeof event.data.code === 'string' ? event.data.code : 'AI_SERVICE_ERROR';
            const message =
              typeof event.data.message === 'string'
                ? event.data.message
                : 'The AI service failed while generating the answer.';
            logger.warn('chat_stream_failed', {
              request_id: req.requestId,
              code,
              chunk_count: chunkCount,
            });
            writeSse(res, 'error', { code, message });
            break;
          }
          default:
            break;
        }
      }
    } catch (err) {
      failed = true;
      logger.error('chat_stream_failed', {
        request_id: req.requestId,
        chunk_count: chunkCount,
        error: err,
      });
      writeSse(res, 'error', {
        code: 'AI_SERVICE_ERROR',
        message: 'The connection to the AI service was interrupted.',
      });
    }

    if (clientClosed) {
      logger.info('chat_stream_client_closed', {
        request_id: req.requestId,
        chunk_count: chunkCount,
      });
      res.end();
      return;
    }

    if (failed) {
      // Partial answers are not cached or persisted, mirroring the
      // non-streaming route where a failed generation is never stored.
      res.end();
      return;
    }

    const latencyMs = upstreamLatencyMs ?? Date.now() - startMs;

    // 5. Cache + persist the completed answer (same as the non-streaming path).
    await cacheAnswer(question, { answer, sources, latency_ms: latencyMs }, history);
    await persistMessages(resolvedSessionId, question, answer, sources, req.requestId);

    logger.info('chat_completed', {
      request_id: req.requestId,
      cached: false,
      streamed: true,
      latency_ms: latencyMs,
      chunk_count: chunkCount,
      sources: sources.length,
      question_chars: question.length,
    });

    writeSse(res, 'done', {
      sessionId: resolvedSessionId,
      latency_ms: latencyMs,
      cached: false,
    });
    res.end();
  },
);

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Load the bounded pre-request conversation for a stored session (Issue #27).
 *
 * Called only when the request carries a sessionId.  Messages are read
 * server-side from MongoDB — browser-supplied history is never trusted — and
 * bounded by {@link boundHistory}.  Any failure (DB outage, malformed
 * document) is logged and treated as "no history" so a lookup problem can
 * never fail the chat response (fail-open, mirroring `persistMessages`).
 */
async function loadRecentHistory(sessionId: string, requestId: string): Promise<HistoryMessage[]> {
  try {
    const session = await Session.findOne({ sessionId });
    const messages = session?.messages ?? [];
    return boundHistory(
      messages.map((message) => ({ role: message.role, content: message.content })),
    );
  } catch (err) {
    logger.warn('session_history_load_failed', {
      request_id: requestId,
      error: err,
      session_id: sessionId,
    });
    return [];
  }
}

/**
 * Cache lookup that preserves the legacy call shape for stateless requests:
 * requests without history resolve to exactly the pre-#27 key and arguments.
 */
function getCachedAnswer(
  question: string,
  history: HistoryMessage[],
): Promise<CachedResponse | null> {
  return history.length === 0 ? getCachedResponse(question) : getCachedResponse(question, history);
}

/**
 * Cache store that preserves the legacy call shape for stateless requests
 * (two-argument `setCachedResponse`); history-aware requests pass the bounded
 * conversation after the TTL parameter.
 */
function cacheAnswer(
  question: string,
  payload: CachedResponse,
  history: HistoryMessage[],
): Promise<void> {
  return history.length === 0
    ? setCachedResponse(question, payload)
    : setCachedResponse(question, payload, undefined, history);
}

/**
 * Upsert the session document by appending the user message and the
 * assistant reply.  If the session doesn't exist yet, MongoDB creates it.
 *
 * Errors are logged but NOT re-thrown so a DB outage does not fail the chat
 * response to the user.
 */
async function persistMessages(
  sessionId: string,
  question: string,
  answer: string,
  sources: { id: string; title: string; score: number; snippet: string }[],
  requestId: string,
): Promise<void> {
  try {
    const now = new Date();
    await Session.findOneAndUpdate(
      { sessionId },
      {
        $push: {
          messages: {
            $each: [
              { role: 'user', content: question, createdAt: now },
              { role: 'assistant', content: answer, sources, createdAt: now },
            ],
          },
        },
      },
      { upsert: true, new: true },
    );
  } catch (err) {
    logger.error('session_persist_failed', {
      request_id: requestId,
      error: err,
      session_id: sessionId,
    });
  }
}

export { persistMessages };

/**
 * Write SSE response headers.  `X-Accel-Buffering: no` disables nginx
 * proxy buffering so tokens reach the browser as they are produced.
 */
function writeSseHeaders(res: Response): void {
  res.status(200);
  res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
  res.setHeader('Cache-Control', 'no-cache, no-transform');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');
}

/** Write one SSE frame unless the response already ended/closed. */
function writeSse(res: Response, event: string, data: unknown): void {
  if (res.writableEnded || res.destroyed) return;
  res.write(formatSseEvent(event, data));
}

/** Map `openAiStream` opening failures to the standard error envelope. */
function handleStreamOpenError(
  err: unknown,
  req: Request,
  res: Response,
  next: NextFunction,
): void {
  if (err instanceof AiStreamHttpError) {
    if (err.status === 422) {
      // Validation error from AI service — pass through as 400
      logger.warn('chat_upstream_rejected', {
        request_id: req.requestId,
        upstream_status: err.status,
      });
      res.status(400).json({
        error: {
          code: 'VALIDATION_ERROR',
          message: 'AI service rejected the request',
          details: err.data,
        },
      });
      return;
    }

    logger.error('chat_stream_upstream_error', {
      request_id: req.requestId,
      upstream_status: err.status,
    });
    const unavailable: AppError = new Error('AI service is unavailable. Please try again later.');
    unavailable.statusCode = 502;
    unavailable.code = 'AI_SERVICE_ERROR';
    next(unavailable);
    return;
  }

  if (err instanceof AiStreamNetworkError) {
    if (err.timedOut) {
      const timeoutError: AppError = new Error(
        'Request to AI service timed out. Please try again.',
      );
      timeoutError.statusCode = 504;
      timeoutError.code = 'AI_SERVICE_TIMEOUT';
      next(timeoutError);
      return;
    }

    const connError: AppError = new Error('Unable to reach AI service. Please try again later.');
    connError.statusCode = 502;
    connError.code = 'AI_SERVICE_UNREACHABLE';
    next(connError);
    return;
  }

  // Unknown error — hand off to the global error handler
  next(err);
}

export default router;
