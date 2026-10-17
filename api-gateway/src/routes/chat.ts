/**
 * Chat route — POST /api/chat
 *
 * Request lifecycle:
 *  1. Validate incoming body with zod.
 *  2. Check Redis cache — return immediately on a hit (cached: true).
 *  3. Forward the question to the AI service.
 *  4. Store the AI response in Redis for future requests.
 *  5. Persist both messages to MongoDB (non-blocking; DB errors don't fail
 *     the response).
 *  6. Return a structured response that includes an auto-generated (or
 *     echoed) sessionId and `cached: false`.
 */

import { randomUUID } from 'crypto';
import { Router, Request, Response, NextFunction } from 'express';
import { z } from 'zod';
import { queryAiService } from '../services/aiClient';
import { getCachedResponse, setCachedResponse } from '../services/cache';
import { AppError } from '../middleware/errorHandler';
import { AxiosError } from 'axios';
import { Session } from '../models/session';

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
  sessionId: z
    .string()
    .uuid('sessionId must be a valid UUID v4')
    .optional(),
});

export type ChatRequest = z.infer<typeof chatRequestSchema>;

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
router.post(
  '/chat',
  async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    // 1. Validate request body
    const parseResult = chatRequestSchema.safeParse(req.body);
    if (!parseResult.success) {
      res.status(400).json({
        error: {
          code: 'VALIDATION_ERROR',
          message: 'Invalid request data',
          details: parseResult.error.errors,
        },
      });
      return;
    }

    const { question, sessionId } = parseResult.data;
    const resolvedSessionId = sessionId ?? randomUUID();

    // 2. Check Redis cache
    const cached = await getCachedResponse(question);
    if (cached !== null) {
      // Cache hit — persist the exchange and return immediately
      await persistMessages(resolvedSessionId, question, cached.answer, cached.sources);

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
      const aiResponse = await queryAiService({ question });

      // 4. Store in Redis (fire-and-forget; errors are swallowed inside)
      await setCachedResponse(question, {
        answer: aiResponse.answer,
        sources: aiResponse.sources,
        latency_ms: aiResponse.latency_ms,
      });

      // 5. Persist messages to MongoDB (non-blocking to response)
      await persistMessages(resolvedSessionId, question, aiResponse.answer, aiResponse.sources);

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
        const connError: AppError = new Error(
          'Unable to reach AI service. Please try again later.',
        );
        connError.statusCode = 502;
        connError.code = 'AI_SERVICE_UNREACHABLE';
        return next(connError);
      }

      // Unknown error — hand off to global error handler
      next(err);
    }
  },
);

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

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
    console.error('[chat] failed to persist session messages:', err);
  }
}

export { persistMessages };

export default router;
