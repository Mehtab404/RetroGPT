/**
 * API key authentication middleware.
 *
 * Checks the `x-api-key` request header against the configured API key.
 * Forwards the request if valid; calls `next(UnauthorizedError)` otherwise.
 *
 * Usage:
 *   router.use(requireApiKey);
 */

import { Request, Response, NextFunction } from 'express';
import { config } from '../config';
import { UnauthorizedError } from './errorHandler';

/**
 * Express middleware that enforces API key authentication.
 *
 * The key must be supplied in the `x-api-key` request header.
 * Returns 401 with code `UNAUTHORIZED` when the header is missing or wrong.
 */
export function requireApiKey(req: Request, _res: Response, next: NextFunction): void {
  const providedKey = req.headers['x-api-key'];

  if (!providedKey || typeof providedKey !== 'string') {
    return next(new UnauthorizedError('API key is required'));
  }

  if (providedKey !== config.apiKey) {
    return next(new UnauthorizedError('Invalid API key'));
  }

  next();
}
