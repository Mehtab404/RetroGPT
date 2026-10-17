/**
 * Redis response cache for RetroGPT.
 *
 * Caches AI service responses keyed by a SHA-256 hash of the normalised
 * question text, with a configurable TTL (default 24 h).
 *
 * Design choices:
 * - Cache errors are logged but never re-thrown so a Redis outage does NOT
 *   break the chat endpoint.
 * - The client is a lazy singleton: `getRedisClient()` connects on first
 *   call.  Calling `disconnectRedis()` tears the connection down (used in
 *   graceful-shutdown and tests).
 * - Question normalisation: trim whitespace + fold to lower-case, so
 *   "Asteroids?" and "  asteroids?  " resolve to the same cache key.
 */

import { createClient, RedisClientType } from 'redis';
import { createHash } from 'crypto';
import { config } from '../config';
import { logger } from '../utils/logger';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** The JSON-serialisable payload stored in Redis. */
export interface CachedResponse {
  answer: string;
  sources: {
    id: string;
    title: string;
    score: number;
    snippet: string;
    page?: number;
  }[];
  latency_ms: number;
}

// ---------------------------------------------------------------------------
// Singleton client
// ---------------------------------------------------------------------------

let redisClient: RedisClientType | null = null;

/**
 * Return (and lazily create) the shared Redis client.
 *
 * The client connects once; subsequent calls return the already-connected
 * instance.  All connection errors are logged but not re-thrown.
 */
export async function getRedisClient(): Promise<RedisClientType | null> {
  if (redisClient !== null) {
    return redisClient;
  }

  const client = createClient({ url: config.redisUrl }) as RedisClientType;

  client.on('error', (err: Error) => {
    logger.error('cache_client_error', { error: err });
  });

  try {
    await client.connect();
    redisClient = client;
    return redisClient;
  } catch (err) {
    logger.error('cache_connection_failed', { error: err });
    return null;
  }
}

/**
 * Gracefully disconnect from Redis.
 * Safe to call even when no client exists (no-op in that case).
 */
export async function disconnectRedis(): Promise<void> {
  if (redisClient === null) return;

  try {
    await redisClient.quit();
  } catch (err) {
    logger.error('cache_disconnect_failed', { error: err });
  } finally {
    redisClient = null;
  }
}

/** Expose a setter so tests can inject a mock client. */
export function setRedisClient(client: RedisClientType | null): void {
  redisClient = client;
}

// ---------------------------------------------------------------------------
// Key derivation
// ---------------------------------------------------------------------------

/**
 * Derive a deterministic cache key from a question string.
 *
 * The question is normalised (trim + lower-case) before hashing so that
 * superficially different but semantically identical questions map to the
 * same bucket.
 *
 * @param question - Raw question text from the request body.
 * @returns A `retrogpt:chat:<sha256hex>` string.
 */
export function buildCacheKey(question: string): string {
  const normalised = question.trim().toLowerCase();
  const hash = createHash('sha256').update(normalised, 'utf8').digest('hex');
  return `retrogpt:chat:${hash}`;
}

// ---------------------------------------------------------------------------
// Public cache operations
// ---------------------------------------------------------------------------

/**
 * Attempt to retrieve a cached response for the given question.
 *
 * @returns The parsed {@link CachedResponse} on a hit, or `null` on a miss
 *          or any Redis error.
 */
export async function getCachedResponse(question: string): Promise<CachedResponse | null> {
  const client = await getRedisClient();
  if (client === null) return null;

  const key = buildCacheKey(question);

  try {
    const raw = await client.get(key);
    if (raw === null) return null;
    return JSON.parse(raw) as CachedResponse;
  } catch (err) {
    logger.error('cache_read_failed', { error: err });
    return null;
  }
}

/**
 * Store a response in the cache with the configured TTL.
 *
 * Errors are silently logged; cache failures never propagate to callers.
 *
 * @param question  - The original question (will be normalised internally).
 * @param response  - The payload to cache.
 * @param ttlSeconds - Time-to-live in seconds (defaults to `config.cacheTtlSeconds`).
 */
export async function setCachedResponse(
  question: string,
  response: CachedResponse,
  ttlSeconds: number = config.cacheTtlSeconds,
): Promise<void> {
  const client = await getRedisClient();
  if (client === null) return;

  const key = buildCacheKey(question);

  try {
    await client.set(key, JSON.stringify(response), { EX: ttlSeconds });
  } catch (err) {
    logger.error('cache_write_failed', { error: err });
  }
}
