/**
 * Unit tests for the Redis-backed rate limiting middleware.
 *
 * `getRedisClient` is mocked so tests control counters deterministically
 * without touching Redis.  Express req/res objects are minimal fakes.
 */

import { Request, Response, NextFunction } from 'express';
import { rateLimit } from '../middleware/rateLimit';
import { RateLimitError } from '../middleware/errorHandler';
import * as cache from '../services/cache';

jest.mock('../services/cache', () => ({
  getRedisClient: jest.fn(),
}));

const mockGetRedisClient = cache.getRedisClient as jest.Mock;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

interface MockRedis {
  incr: jest.Mock;
  expire: jest.Mock;
}

function makeClient(overrides: Partial<MockRedis> = {}): MockRedis {
  return {
    incr: overrides.incr ?? jest.fn().mockResolvedValue(1),
    expire: overrides.expire ?? jest.fn().mockResolvedValue(1),
  };
}

function makeReq(ip = '203.0.113.7'): Request {
  return { ip, socket: { remoteAddress: ip } } as unknown as Request;
}

interface MockRes {
  res: Response;
  setHeader: jest.Mock;
}

function makeRes(): MockRes {
  const setHeader = jest.fn();
  const res = { setHeader } as unknown as Response;
  return { res, setHeader };
}

function headerValue(setHeader: jest.Mock, name: string): string | undefined {
  const call = setHeader.mock.calls.find(([key]) => key === name);
  return call?.[1] as string | undefined;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('rateLimit middleware', () => {
  let next: NextFunction;

  beforeEach(() => {
    jest.clearAllMocks();
    next = jest.fn();
  });

  describe('under the limit', () => {
    it('allows the request and calls next() with no error', async () => {
      mockGetRedisClient.mockResolvedValue(makeClient({ incr: jest.fn().mockResolvedValue(1) }));
      const { res } = makeRes();

      await rateLimit({ limit: 5, windowSeconds: 60 })(makeReq(), res, next);

      expect(next).toHaveBeenCalledWith();
    });

    it('sets X-RateLimit-Limit and X-RateLimit-Remaining headers', async () => {
      mockGetRedisClient.mockResolvedValue(makeClient({ incr: jest.fn().mockResolvedValue(3) }));
      const { res, setHeader } = makeRes();

      await rateLimit({ limit: 30, windowSeconds: 60 })(makeReq(), res, next);

      expect(headerValue(setHeader, 'X-RateLimit-Limit')).toBe('30');
      expect(headerValue(setHeader, 'X-RateLimit-Remaining')).toBe('27');
      expect(headerValue(setHeader, 'X-RateLimit-Reset')).toBeDefined();
    });

    it('allows the request exactly at the limit', async () => {
      mockGetRedisClient.mockResolvedValue(makeClient({ incr: jest.fn().mockResolvedValue(5) }));
      const { res } = makeRes();

      await rateLimit({ limit: 5, windowSeconds: 60 })(makeReq(), res, next);

      expect(next).toHaveBeenCalledWith();
      expect(next).not.toHaveBeenCalledWith(expect.any(RateLimitError));
    });

    it('builds a window-scoped key containing the client IP', async () => {
      const incr = jest.fn().mockResolvedValue(1);
      mockGetRedisClient.mockResolvedValue(makeClient({ incr }));
      const { res } = makeRes();

      await rateLimit({ keyPrefix: 'rl-test' })(makeReq('198.51.100.9'), res, next);

      const key = incr.mock.calls[0][0] as string;
      expect(key).toMatch(/^rl-test:198\.51\.100\.9:\d+$/);
    });

    it('sets an expiry when the counter is created (count === 1)', async () => {
      const expire = jest.fn().mockResolvedValue(1);
      mockGetRedisClient.mockResolvedValue(
        makeClient({ incr: jest.fn().mockResolvedValue(1), expire }),
      );
      const { res } = makeRes();

      await rateLimit({ limit: 5, windowSeconds: 42 })(makeReq(), res, next);

      expect(expire).toHaveBeenCalledTimes(1);
      expect(expire).toHaveBeenCalledWith(expect.stringMatching(/^ratelimit:/), 42);
    });

    it('does NOT reset the expiry on later increments', async () => {
      const expire = jest.fn();
      mockGetRedisClient.mockResolvedValue(
        makeClient({ incr: jest.fn().mockResolvedValue(2), expire }),
      );
      const { res } = makeRes();

      await rateLimit({ limit: 5 })(makeReq(), res, next);

      expect(expire).not.toHaveBeenCalled();
    });
  });

  describe('over the limit', () => {
    it('passes a RateLimitError to next() when the counter exceeds the limit', async () => {
      mockGetRedisClient.mockResolvedValue(makeClient({ incr: jest.fn().mockResolvedValue(6) }));
      const { res } = makeRes();

      await rateLimit({ limit: 5, windowSeconds: 60 })(makeReq(), res, next);

      const err = (next as jest.Mock).mock.calls[0][0] as RateLimitError;
      expect(err).toBeInstanceOf(RateLimitError);
      expect(err.statusCode).toBe(429);
      expect(err.code).toBe('RATE_LIMIT_EXCEEDED');
    });

    it('sets X-RateLimit-Remaining to 0 and a Retry-After header', async () => {
      mockGetRedisClient.mockResolvedValue(makeClient({ incr: jest.fn().mockResolvedValue(31) }));
      const { res, setHeader } = makeRes();

      await rateLimit({ limit: 30, windowSeconds: 60 })(makeReq(), res, next);

      expect(headerValue(setHeader, 'X-RateLimit-Remaining')).toBe('0');
      expect(headerValue(setHeader, 'Retry-After')).toBeDefined();
      expect(Number(headerValue(setHeader, 'Retry-After'))).toBeGreaterThan(0);
    });
  });

  describe('fail-open behaviour', () => {
    it('allows the request when no Redis client is available', async () => {
      mockGetRedisClient.mockResolvedValue(null);
      const { res } = makeRes();

      await rateLimit({ limit: 1 })(makeReq(), res, next);

      expect(next).toHaveBeenCalledWith();
    });

    it('allows the request when Redis throws on incr', async () => {
      const consoleSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
      mockGetRedisClient.mockResolvedValue(
        makeClient({ incr: jest.fn().mockRejectedValue(new Error('connection lost')) }),
      );
      const { res } = makeRes();

      await rateLimit({ limit: 1 })(makeReq(), res, next);

      expect(next).toHaveBeenCalledWith();
      consoleSpy.mockRestore();
    });
  });
});
