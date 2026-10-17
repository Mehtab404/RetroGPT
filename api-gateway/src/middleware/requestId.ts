/**
 * Request-id middleware (Issue #24).
 *
 * - Reuses a well-formed inbound `x-request-id` (so callers, Nginx and
 *   frontends can correlate their own logs) or generates a UUID v4.
 * - Exposes it as `req.requestId` and echoes it on the response.
 * - Emits one `http_request` JSON log line when the response finishes,
 *   with method, path, status and duration — the same event name the AI
 *   service logs, both keyed by `request_id`.
 *
 * `/api/health` is logged at debug level so container healthchecks do
 * not flood the production log stream.
 */

import { randomUUID } from 'crypto';
import { Request, Response, NextFunction } from 'express';
import { logger } from '../utils/logger';

/** Accepted trace ids: bounded ASCII so log lines stay parseable. */
const VALID_REQUEST_ID = /^[A-Za-z0-9._-]{1,64}$/;

/**
 * Return `value` when it is a safe trace id, else a fresh UUID v4.
 *
 * @param value - Inbound `x-request-id` header (may be absent/duplicated).
 */
export function sanitizeRequestId(value: string | string[] | undefined): string {
  const candidate = Array.isArray(value) ? value[0] : value;
  if (candidate !== undefined && VALID_REQUEST_ID.test(candidate)) {
    return candidate;
  }
  return randomUUID();
}

/** Express middleware that tags the request and logs its completion. */
export function requestId(req: Request, res: Response, next: NextFunction): void {
  const id = sanitizeRequestId(req.headers['x-request-id']);
  req.requestId = id;
  res.setHeader('X-Request-Id', id);

  const start = process.hrtime.bigint();
  res.on('finish', () => {
    const durationMs = Number(process.hrtime.bigint() - start) / 1e6;
    const fields = {
      request_id: id,
      method: req.method,
      path: req.path,
      status: res.statusCode,
      duration_ms: Math.round(durationMs * 100) / 100,
    };

    if (req.path === '/api/health') {
      logger.debug('http_request', fields);
    } else {
      logger.info('http_request', fields);
    }
  });

  next();
}
