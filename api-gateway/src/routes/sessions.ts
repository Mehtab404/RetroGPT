/**
 * Sessions route — GET /api/sessions/:id
 *
 * Returns the full message history for a given session.
 */

import { Router, Request, Response, NextFunction } from 'express';
import { z } from 'zod';
import { Session } from '../models/session';
import { AppError } from '../middleware/errorHandler';

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

const sessionIdSchema = z.string().uuid('sessionId must be a valid UUID v4');

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

const router = Router();

/**
 * GET /api/sessions/:id
 *
 * Returns:
 *   {
 *     "sessionId": "uuid",
 *     "messages": [
 *       { "role": "user",      "content": "...", "createdAt": "ISO8601" },
 *       { "role": "assistant", "content": "...", "sources": [...], "createdAt": "ISO8601" }
 *     ]
 *   }
 *
 * 400  — id is not a valid UUID v4
 * 404  — no session found with that id
 */
router.get(
  '/sessions/:id',
  async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    // 1. Validate param
    const parseResult = sessionIdSchema.safeParse(req.params.id);
    if (!parseResult.success) {
      res.status(400).json({
        error: {
          code: 'VALIDATION_ERROR',
          message: 'Invalid session ID',
          details: parseResult.error.errors,
        },
      });
      return;
    }

    const sessionId = parseResult.data;

    // 2. Fetch from MongoDB
    try {
      const session = await Session.findOne({ sessionId }).lean();

      if (!session) {
        const notFound: AppError = new Error(`Session '${sessionId}' not found`);
        notFound.statusCode = 404;
        notFound.code = 'NOT_FOUND';
        return next(notFound);
      }

      // 3. Shape the response
      const messages = (session.messages ?? []).map((msg) => {
        const out: {
          role: string;
          content: string;
          createdAt: string;
          sources?: typeof msg.sources;
        } = {
          role: msg.role,
          content: msg.content,
          createdAt: msg.createdAt instanceof Date
            ? msg.createdAt.toISOString()
            : String(msg.createdAt),
        };

        if (msg.sources && msg.sources.length > 0) {
          out.sources = msg.sources;
        }

        return out;
      });

      res.status(200).json({ sessionId, messages });
    } catch (err) {
      next(err);
    }
  },
);

export default router;
