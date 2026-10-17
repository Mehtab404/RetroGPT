/**
 * Redis-backed rate limiting middleware.
 *
 * Implements a fixed-window counter per client IP.  Each request increments
 * a window-scoped Redis key (`ratelimit:<ip>:<windowStartMs>`) and the key
 * receives a TTL equal to the window size on first increment.
 *
 * Design choices:
 * - **Fail open:** if Redis is unavailable the request is allowed (with a
 *   logged warning).  Rate limiting protects against abuse but must not take
 *   the whole API down during a cache outage; availability wins.
 * - **Standard headers:** responses carry `X-RateLimit-Limit`,
 *   `X-RateLimit-Remaining`, and `X-RateLimit-Reset` so clients can back off
 *   proactively.
 * - **429 shape:** exceeded requests produce the project-standard error body
 *   `{ "error": { "code": "RATE_LIMIT_EXCEEDED", "message": "..." } }` via
 *   the central error handler.
 *
 * Usage:
 *   app.use('/api', rateLimit());                    // config defaults
 *   app.use('/api', rateLimit({ limit: 5, windowSeconds: 10 }));
 */

import { Request, Response, NextFunction } from 'express';
import { getRedisClient } from '../services/cache';
import { config } from '../config';
import { RateLimitError } from './errorHandler';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface RateLimitOptions {
  /** Maximum requests allowed per window (defaults to `RATE_LIMIT_PER_MIN`). */
  limit?: number;
  /** Window duration in seconds (default 60 — i.e. requests per minute). */
  windowSeconds?: number;
  /** Redis key namespace (default `ratelimit`). */
  keyPrefix?: string;
}

// ---------------------------------------------------------------------------
// Middleware factory
// ---------------------------------------------------------------------------

/**
 * Build an Express middleware enforcing a per-IP fixed-window rate limit.
 *
 * @param options - Optional overrides for limit, window size, and key prefix.
 * @returns Express middleware; calls `next(RateLimitError)` once the limit is
 *          exceeded, otherwise `next()`.
 */
export function rateLimit(options: RateLimitOptions = {}) {
  const limit = options.limit ?? config.rateLimitPerMin;
  const windowSeconds = options.windowSeconds ?? 60;
  const keyPrefix = options.keyPrefix ?? 'ratelimit';

  return async function rateLimitMiddleware(
    req: Request,
    res: Response,
    next: NextFunction,
  ): Promise<void> {
    try {
      const client = await getRedisClient();

      // Fail open: no Redis client means no rate limiting.
      if (!client) {
        next();
        return;
      }

      const windowMs = windowSeconds * 1000;
      const now = Date.now();
      const windowStart = Math.floor(now / windowMs) * windowMs;
      const windowEnd = windowStart + windowMs;
      const identifier = req.ip ?? req.socket.remoteAddress ?? 'unknown';
      const key = `${keyPrefix}:${identifier}:${windowStart}`;

      const count = await client.incr(key);

      // Set the TTL only when the window key is created, so every later
      // increment keeps the original expiry.
      if (count === 1) {
        await client.expire(key, windowSeconds);
      }

      const remaining = Math.max(0, limit - count);
      res.setHeader('X-RateLimit-Limit', String(limit));
      res.setHeader('X-RateLimit-Remaining', String(remaining));
      res.setHeader('X-RateLimit-Reset', String(Math.ceil(windowEnd / 1000)));

      if (count > limit) {
        const retryAfter = Math.max(1, Math.ceil((windowEnd - now) / 1000));
        res.setHeader('Retry-After', String(retryAfter));
        next(
          new RateLimitError(
            `Rate limit exceeded (${limit} requests per ${windowSeconds}s). ` +
              `Try again in ${retryAfter}s.`,
          ),
        );
        return;
      }

      next();
    } catch (err) {
      // Redis errors must not reject traffic — fail open and keep serving.
      console.warn('[rateLimit] Redis error, allowing request:', (err as Error).message);
      next();
    }
  };
}
