/**
 * Tests for services/cache.ts
 *
 * The `redis` module is mocked at the top level so `createClient` never
 * makes a real network connection.  `setRedisClient` is used to inject a
 * pre-built mock client for tests that exercise get/set/quit behaviour.
 */

import { createHash } from 'crypto';
import {
  buildCacheKey,
  getCachedResponse,
  setCachedResponse,
  disconnectRedis,
  setRedisClient,
  CachedResponse,
} from '../services/cache';
import { RedisClientType } from 'redis';

// ---------------------------------------------------------------------------
// Mock the entire `redis` package so no real TCP connections are attempted.
// Tests that need a connected client inject one via setRedisClient().
// ---------------------------------------------------------------------------

jest.mock('redis', () => ({
  createClient: jest.fn(() => ({
    on: jest.fn(),
    connect: jest.fn().mockRejectedValue(new Error('No real Redis in tests')),
    get: jest.fn(),
    set: jest.fn(),
    quit: jest.fn(),
  })),
}));

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Minimal cached response fixture. */
function makeCachedResponse(overrides: Partial<CachedResponse> = {}): CachedResponse {
  return {
    answer: 'You score by destroying asteroids.',
    sources: [
      {
        id: 'asteroids_manual_p3_c2',
        title: 'Asteroids Manual',
        score: 0.91,
        snippet: 'Large asteroids: 20 points...',
      },
    ],
    latency_ms: 1820,
    ...overrides,
  };
}

/** Build a mock Redis client with jest.fn() for `get`, `set`, and `quit`. */
function makeMockClient(
  overrides: {
    get?: jest.Mock;
    set?: jest.Mock;
    quit?: jest.Mock;
  } = {},
): RedisClientType {
  return {
    get: overrides.get ?? jest.fn(),
    set: overrides.set ?? jest.fn().mockResolvedValue('OK'),
    quit: overrides.quit ?? jest.fn().mockResolvedValue('OK'),
    on: jest.fn(),
  } as unknown as RedisClientType;
}

// ---------------------------------------------------------------------------
// buildCacheKey
// ---------------------------------------------------------------------------

describe('buildCacheKey()', () => {
  it('returns a key prefixed with retrogpt:chat:', () => {
    const key = buildCacheKey('How do you score in Asteroids?');
    expect(key).toMatch(/^retrogpt:chat:[0-9a-f]{64}$/);
  });

  it('produces a SHA-256 hash of the normalised question', () => {
    const question = 'How do you score in Asteroids?';
    const normalised = question.trim().toLowerCase();
    const expectedHash = createHash('sha256').update(normalised, 'utf8').digest('hex');
    expect(buildCacheKey(question)).toBe(`retrogpt:chat:${expectedHash}`);
  });

  it('normalises case — upper and lower produce the same key', () => {
    expect(buildCacheKey('ASTEROIDS')).toBe(buildCacheKey('asteroids'));
  });

  it('normalises leading/trailing whitespace', () => {
    expect(buildCacheKey('  asteroids  ')).toBe(buildCacheKey('asteroids'));
  });

  it('different questions produce different keys', () => {
    expect(buildCacheKey('Asteroids')).not.toBe(buildCacheKey('Pong'));
  });
});

// ---------------------------------------------------------------------------
// getCachedResponse
// ---------------------------------------------------------------------------

describe('getCachedResponse()', () => {
  afterEach(() => {
    // Ensure the singleton is cleared between tests
    setRedisClient(null);
  });

  it('returns null when no Redis client is available', async () => {
    // Client is null; getRedisClient() will try createClient().connect() which
    // is mocked to reject — so it returns null.
    setRedisClient(null);
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    const result = await getCachedResponse('What is Pong?');
    consoleSpy.mockRestore();
    expect(result).toBeNull();
  });

  it('returns null on a cache miss (Redis returns null)', async () => {
    const mockGet = jest.fn().mockResolvedValue(null);
    setRedisClient(makeMockClient({ get: mockGet }));

    const result = await getCachedResponse('What is Asteroids?');
    expect(result).toBeNull();
  });

  it('returns parsed CachedResponse on a cache hit', async () => {
    const payload = makeCachedResponse();
    const mockGet = jest.fn().mockResolvedValue(JSON.stringify(payload));
    setRedisClient(makeMockClient({ get: mockGet }));

    const result = await getCachedResponse('How do you score?');
    expect(result).toEqual(payload);
  });

  it('calls Redis get with the correct key', async () => {
    const question = 'How do you score in Asteroids?';
    const expectedKey = buildCacheKey(question);
    const mockGet = jest.fn().mockResolvedValue(null);
    setRedisClient(makeMockClient({ get: mockGet }));

    await getCachedResponse(question);

    expect(mockGet).toHaveBeenCalledWith(expectedKey);
  });

  it('returns null (and logs) when Redis.get throws', async () => {
    const mockGet = jest.fn().mockRejectedValue(new Error('RESP3 parse error'));
    setRedisClient(makeMockClient({ get: mockGet }));

    const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    const result = await getCachedResponse('What is Pong?');
    consoleSpy.mockRestore();

    expect(result).toBeNull();
  });

  it('uses the normalised question for key lookup', async () => {
    const mockGet = jest.fn().mockResolvedValue(null);
    setRedisClient(makeMockClient({ get: mockGet }));

    // Leading spaces + upper-case should normalise to 'asteroids'
    await getCachedResponse('  ASTEROIDS  ');

    const expectedKey = buildCacheKey('asteroids');
    expect(mockGet).toHaveBeenCalledWith(expectedKey);
  });
});

// ---------------------------------------------------------------------------
// setCachedResponse
// ---------------------------------------------------------------------------

describe('setCachedResponse()', () => {
  afterEach(() => {
    setRedisClient(null);
  });

  it('is a no-op when no Redis client is available', async () => {
    // createClient().connect() is mocked to reject → getRedisClient returns null
    setRedisClient(null);
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    await expect(setCachedResponse('What is Pong?', makeCachedResponse())).resolves.toBeUndefined();
    consoleSpy.mockRestore();
  });

  it('calls Redis.set with the correct key, JSON value, and EX option', async () => {
    const mockSet = jest.fn().mockResolvedValue('OK');
    setRedisClient(makeMockClient({ set: mockSet }));

    const question = 'How do you score in Asteroids?';
    const payload = makeCachedResponse();
    const ttl = 3600;

    await setCachedResponse(question, payload, ttl);

    const expectedKey = buildCacheKey(question);
    expect(mockSet).toHaveBeenCalledWith(expectedKey, JSON.stringify(payload), { EX: ttl });
  });

  it('uses config.cacheTtlSeconds when no explicit TTL is provided', async () => {
    const mockSet = jest.fn().mockResolvedValue('OK');
    setRedisClient(makeMockClient({ set: mockSet }));

    await setCachedResponse('What is Galaga?', makeCachedResponse());

    // Third argument (index 2) to set() is the options object
    const [, , options] = mockSet.mock.calls[0] as [string, string, { EX: number }];
    expect(options.EX).toBeGreaterThan(0);
  });

  it('swallows Redis.set errors (logs but does not throw)', async () => {
    const mockSet = jest.fn().mockRejectedValue(new Error('OOM'));
    setRedisClient(makeMockClient({ set: mockSet }));

    const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    await expect(
      setCachedResponse('What is Breakout?', makeCachedResponse()),
    ).resolves.toBeUndefined();
    consoleSpy.mockRestore();
  });
});

// ---------------------------------------------------------------------------
// disconnectRedis
// ---------------------------------------------------------------------------

describe('disconnectRedis()', () => {
  afterEach(() => {
    setRedisClient(null);
  });

  it('is a no-op when no client is connected', async () => {
    setRedisClient(null);
    await expect(disconnectRedis()).resolves.toBeUndefined();
  });

  it('calls quit() on the active client', async () => {
    const mockQuit = jest.fn().mockResolvedValue('OK');
    setRedisClient(makeMockClient({ quit: mockQuit }));

    await disconnectRedis();

    expect(mockQuit).toHaveBeenCalledTimes(1);
  });

  it('sets the internal client to null after disconnect', async () => {
    const mockQuit = jest.fn().mockResolvedValue('OK');
    setRedisClient(makeMockClient({ quit: mockQuit }));

    await disconnectRedis();

    // After disconnect the singleton is null — a subsequent get call returns null
    // (createClient().connect() is mocked to reject → null)
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    const result = await getCachedResponse('test');
    consoleSpy.mockRestore();
    expect(result).toBeNull();
  });

  it('logs but does not throw when quit() fails', async () => {
    const mockQuit = jest.fn().mockRejectedValue(new Error('already closed'));
    setRedisClient(makeMockClient({ quit: mockQuit }));

    const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    await expect(disconnectRedis()).resolves.toBeUndefined();
    consoleSpy.mockRestore();
  });
});
