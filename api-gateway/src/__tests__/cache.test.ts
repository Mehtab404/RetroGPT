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
import { logger } from '../utils/logger';
import { type HistoryMessage } from '../services/history';

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

/** Minimal bounded conversation fixture (Issue #27). */
const HISTORY: HistoryMessage[] = [
  { role: 'user', content: 'How do you score in Asteroids?' },
  { role: 'assistant', content: 'Destroy asteroids for points [1].' },
];

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
// buildCacheKey with conversation history (Issue #27)
// ---------------------------------------------------------------------------

describe('buildCacheKey() with conversation history', () => {
  it('keeps the exact legacy key when history is empty', () => {
    const question = 'How do you score in Asteroids?';
    expect(buildCacheKey(question, [])).toBe(buildCacheKey(question));
  });

  it('appends an order-sensitive history fingerprint', () => {
    const key = buildCacheKey('What about the saucer?', HISTORY);
    expect(key).toMatch(/^retrogpt:chat:[0-9a-f]{64}:h:[0-9a-f]{64}$/);
  });

  it('differs from the no-history key for the same question', () => {
    const question = 'What about the saucer?';
    expect(buildCacheKey(question, HISTORY)).not.toBe(buildCacheKey(question));
  });

  it('different message contents produce different keys', () => {
    const other: HistoryMessage[] = [{ role: 'user', content: 'How do you score in Pong?' }];
    expect(buildCacheKey('What about the saucer?', [HISTORY[0]])).not.toBe(
      buildCacheKey('What about the saucer?', other),
    );
  });

  it('different roles produce different keys', () => {
    const userMessage: HistoryMessage[] = [{ role: 'user', content: 'same text' }];
    const assistantMessage: HistoryMessage[] = [{ role: 'assistant', content: 'same text' }];
    expect(buildCacheKey('q?', userMessage)).not.toBe(buildCacheKey('q?', assistantMessage));
  });

  it('different message order produces different keys', () => {
    const reversed = [...HISTORY].reverse();
    expect(buildCacheKey('What about the saucer?', HISTORY)).not.toBe(
      buildCacheKey('What about the saucer?', reversed),
    );
  });

  it('identical histories produce deterministic keys', () => {
    const clone = HISTORY.map((message) => ({ ...message }));
    expect(buildCacheKey('What about the saucer?', HISTORY)).toBe(
      buildCacheKey('What about the saucer?', clone),
    );
  });

  it('different questions under the same history produce different keys', () => {
    expect(buildCacheKey('What about the saucer?', HISTORY)).not.toBe(
      buildCacheKey('What about the UFO?', HISTORY),
    );
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
    const errorSpy = jest.spyOn(logger, 'error').mockImplementation(() => undefined);
    const result = await getCachedResponse('What is Pong?');
    errorSpy.mockRestore();
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

    const errorSpy = jest.spyOn(logger, 'error').mockImplementation(() => undefined);
    const result = await getCachedResponse('What is Pong?');
    expect(errorSpy).toHaveBeenCalledWith('cache_read_failed', { error: expect.any(Error) });
    errorSpy.mockRestore();

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

  it('looks up the history-aware key when history is provided', async () => {
    const mockGet = jest.fn().mockResolvedValue(null);
    setRedisClient(makeMockClient({ get: mockGet }));

    await getCachedResponse('What about the saucer?', HISTORY);

    expect(mockGet).toHaveBeenCalledWith(buildCacheKey('What about the saucer?', HISTORY));
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
    const errorSpy = jest.spyOn(logger, 'error').mockImplementation(() => undefined);
    await expect(setCachedResponse('What is Pong?', makeCachedResponse())).resolves.toBeUndefined();
    errorSpy.mockRestore();
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

    const errorSpy = jest.spyOn(logger, 'error').mockImplementation(() => undefined);
    await expect(
      setCachedResponse('What is Breakout?', makeCachedResponse()),
    ).resolves.toBeUndefined();
    expect(errorSpy).toHaveBeenCalledWith('cache_write_failed', { error: expect.any(Error) });
    errorSpy.mockRestore();
  });

  it('stores under the history-aware key when history is provided', async () => {
    const mockSet = jest.fn().mockResolvedValue('OK');
    setRedisClient(makeMockClient({ set: mockSet }));

    const payload = makeCachedResponse();
    await setCachedResponse('What about the saucer?', payload, 3600, HISTORY);

    expect(mockSet).toHaveBeenCalledWith(
      buildCacheKey('What about the saucer?', HISTORY),
      JSON.stringify(payload),
      { EX: 3600 },
    );
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
    const errorSpy = jest.spyOn(logger, 'error').mockImplementation(() => undefined);
    const result = await getCachedResponse('test');
    errorSpy.mockRestore();
    expect(result).toBeNull();
  });

  it('logs but does not throw when quit() fails', async () => {
    const mockQuit = jest.fn().mockRejectedValue(new Error('already closed'));
    setRedisClient(makeMockClient({ quit: mockQuit }));

    const errorSpy = jest.spyOn(logger, 'error').mockImplementation(() => undefined);
    await expect(disconnectRedis()).resolves.toBeUndefined();
    expect(errorSpy).toHaveBeenCalledWith('cache_disconnect_failed', { error: expect.any(Error) });
    errorSpy.mockRestore();
  });
});
