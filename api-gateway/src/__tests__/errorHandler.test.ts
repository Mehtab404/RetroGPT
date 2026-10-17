/**
 * Tests for src/middleware/errorHandler.ts
 */

import { Request, Response, NextFunction } from 'express';
import { ZodError, z } from 'zod';
import {
  errorHandler,
  AppError,
  ValidationError,
  UnauthorizedError,
  NotFoundError,
  RateLimitError,
} from '../middleware/errorHandler';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Create a minimal mock Response that records json() calls. */
function makeRes(): {
  res: Response;
  status: jest.Mock;
  json: jest.Mock;
} {
  const json = jest.fn().mockReturnThis();
  const status = jest.fn().mockReturnValue({ json });
  const res = { status, json } as unknown as Response;
  return { res, status, json };
}

const req = {} as Request;
const next: NextFunction = jest.fn();

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('errorHandler middleware', () => {
  it('returns 400 for a ZodError', () => {
    const { res, status, json } = makeRes();

    // Trigger a real ZodError via a failing parse
    const schema = z.object({ name: z.string() });
    const result = schema.safeParse({ name: 123 });
    const zodErr = (result as { error: ZodError }).error;

    errorHandler(zodErr, req, res, next);

    expect(status).toHaveBeenCalledWith(400);
    expect(json).toHaveBeenCalledWith(
      expect.objectContaining({
        error: expect.objectContaining({ code: 'VALIDATION_ERROR' }),
      }),
    );
  });

  it('returns 500 for a generic Error', () => {
    const { res, status, json } = makeRes();
    const err: AppError = new Error('Something went wrong');

    errorHandler(err, req, res, next);

    expect(status).toHaveBeenCalledWith(500);
    expect(json).toHaveBeenCalledWith(
      expect.objectContaining({
        error: expect.objectContaining({ code: 'INTERNAL_ERROR' }),
      }),
    );
  });

  it('uses statusCode and code from AppError when provided', () => {
    const { res, status, json } = makeRes();
    const err: AppError = new Error('Custom error');
    err.statusCode = 418;
    err.code = 'IM_A_TEAPOT';

    errorHandler(err, req, res, next);

    expect(status).toHaveBeenCalledWith(418);
    expect(json).toHaveBeenCalledWith(
      expect.objectContaining({
        error: expect.objectContaining({ code: 'IM_A_TEAPOT' }),
      }),
    );
  });
});

describe('Error subclasses', () => {
  it('ValidationError has statusCode 400 and code VALIDATION_ERROR', () => {
    const err = new ValidationError('bad input');
    expect(err.statusCode).toBe(400);
    expect(err.code).toBe('VALIDATION_ERROR');
    expect(err.message).toBe('bad input');
  });

  it('UnauthorizedError has statusCode 401 and code UNAUTHORIZED', () => {
    const err = new UnauthorizedError();
    expect(err.statusCode).toBe(401);
    expect(err.code).toBe('UNAUTHORIZED');
  });

  it('UnauthorizedError accepts custom message', () => {
    const err = new UnauthorizedError('Token expired');
    expect(err.message).toBe('Token expired');
  });

  it('NotFoundError has statusCode 404 and code NOT_FOUND', () => {
    const err = new NotFoundError();
    expect(err.statusCode).toBe(404);
    expect(err.code).toBe('NOT_FOUND');
  });

  it('NotFoundError accepts custom message', () => {
    const err = new NotFoundError('Session not found');
    expect(err.message).toBe('Session not found');
  });

  it('RateLimitError has statusCode 429 and code RATE_LIMIT_EXCEEDED', () => {
    const err = new RateLimitError();
    expect(err.statusCode).toBe(429);
    expect(err.code).toBe('RATE_LIMIT_EXCEEDED');
  });

  it('RateLimitError accepts custom message', () => {
    const err = new RateLimitError('Slow down');
    expect(err.message).toBe('Slow down');
  });
});
