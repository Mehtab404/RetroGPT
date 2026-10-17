/**
 * Chat route — POST /api/chat
 *
 * Validates the incoming request, proxies it to the AI service, and returns
 * a structured response that includes an auto-generated (or echoed) sessionId.
 */

import { randomUUID } from 'crypto';
import { Router, Request, Response, NextFunction } from 'express';
import { z } from 'zod';
import { queryAiService } from '../services/aiClient';
import { AppError } from '../middleware/errorHandler';
import { AxiosError } from 'axios';

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
 *     "latency_ms": 1234, "cached": false }
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

    // 2. Proxy to AI service
    try {
      const aiResponse = await queryAiService({ question });

      res.status(200).json({
        sessionId: resolvedSessionId,
        answer: aiResponse.answer,
        sources: aiResponse.sources,
        latency_ms: aiResponse.latency_ms,
        cached: aiResponse.cached,
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

export default router;
