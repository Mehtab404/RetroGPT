/**
 * Tests for POST /api/chat/stream (Issue #25).
 *
 * The AI service stream client, Redis cache and Session model are mocked so
 * no real network/database calls are made.
 */

import request from 'supertest';
import app from '../app';
import * as aiStream from '../services/aiStream';
import * as cache from '../services/cache';
import { Session } from '../models/session';

// Mock Session model so no real MongoDB calls are made
jest.mock('../models/session', () => ({
  Session: {
    findOne: jest.fn(),
    findOneAndUpdate: jest.fn().mockResolvedValue(null),
  },
}));

// Mock cache service so no real Redis calls are made
jest.mock('../services/cache', () => ({
  getCachedResponse: jest.fn().mockResolvedValue(null),
  setCachedResponse: jest.fn().mockResolvedValue(undefined),
  // Rate limiter fails open in tests (no Redis)
  getRedisClient: jest.fn().mockResolvedValue(null),
}));

const API_KEY = process.env.API_KEY || 'change-me';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Build a fake AiStream that yields the given [event, data] pairs. */
function makeStream(events: Array<[string, Record<string, unknown>]>): aiStream.AiStream {
  async function* generator() {
    for (const [event, data] of events) yield { event, data };
  }
  return { events: generator() };
}

/** Build a fake AiStream whose generator throws after yielding events. */
function makeFailingStream(
  events: Array<[string, Record<string, unknown>]>,
  error: Error,
): aiStream.AiStream {
  async function* generator() {
    for (const [event, data] of events) yield { event, data };
    throw error;
  }
  return { events: generator() };
}

/** Parse an SSE response body into [event, payload] pairs. */
function parseSseBody(text: string): Array<[string, Record<string, unknown>]> {
  const events: Array<[string, Record<string, unknown>]> = [];
  for (const frame of text.split('\n\n')) {
    if (!frame.trim()) continue;
    let event = 'message';
    let data = '';
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data = line.slice('data:'.length).trim();
    }
    if (data) events.push([event, JSON.parse(data) as Record<string, unknown>]);
  }
  return events;
}

const SOURCES = [
  {
    id: 'asteroids_manual_p3_c2',
    title: 'Asteroids Manual',
    score: 0.91,
    snippet: 'Large asteroids: 20 points...',
  },
];

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('POST /api/chat/stream', () => {
  let openSpy: jest.SpyInstance;

  beforeEach(() => {
    jest.clearAllMocks();
    (cache.getCachedResponse as jest.Mock).mockResolvedValue(null);
    (cache.setCachedResponse as jest.Mock).mockResolvedValue(undefined);
    (Session.findOneAndUpdate as jest.Mock).mockResolvedValue(null);
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  // -------------------------------------------------------------------------
  // Happy path
  // -------------------------------------------------------------------------

  it('relays meta/token/done events and echoes the sessionId', async () => {
    const sessionId = '550e8400-e29b-41d4-a716-446655440000';
    openSpy = jest.spyOn(aiStream, 'openAiStream').mockResolvedValue(
      makeStream([
        ['meta', { sources: SOURCES }],
        ['token', { text: 'Large ' }],
        ['token', { text: 'asteroids are worth 20 points [1].' }],
        ['done', { latency_ms: 1234 }],
      ]),
    );

    const res = await request(app)
      .post('/api/chat/stream')
      .set('x-api-key', API_KEY)
      .send({ question: 'How do you score in Asteroids?', sessionId });

    expect(res.status).toBe(200);
    expect(res.headers['content-type']).toContain('text/event-stream');

    const events = parseSseBody(res.text);
    expect(events.map(([name]) => name)).toEqual(['meta', 'token', 'token', 'done']);
    expect(events[0][1].sources).toEqual(SOURCES);
    expect(events[3][1]).toEqual({ sessionId, latency_ms: 1234, cached: false });
  });

  it('generates a sessionId when none is supplied', async () => {
    openSpy = jest.spyOn(aiStream, 'openAiStream').mockResolvedValue(
      makeStream([
        ['meta', { sources: [] }],
        ['token', { text: 'Answer' }],
        ['done', { latency_ms: 5 }],
      ]),
    );

    const res = await request(app)
      .post('/api/chat/stream')
      .set('x-api-key', API_KEY)
      .send({ question: 'What is Pong?' });

    const done = parseSseBody(res.text).find(([name]) => name === 'done');
    expect(done?.[1].sessionId).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i,
    );
  });

  it('caches and persists the completed answer', async () => {
    openSpy = jest.spyOn(aiStream, 'openAiStream').mockResolvedValue(
      makeStream([
        ['meta', { sources: SOURCES }],
        ['token', { text: 'Large ' }],
        ['token', { text: 'asteroids [1].' }],
        ['done', { latency_ms: 900 }],
      ]),
    );

    await request(app)
      .post('/api/chat/stream')
      .set('x-api-key', API_KEY)
      .send({ question: 'How do you score in Asteroids?' });

    expect(cache.setCachedResponse).toHaveBeenCalledWith('How do you score in Asteroids?', {
      answer: 'Large asteroids [1].',
      sources: SOURCES,
      latency_ms: 900,
    });
    expect(Session.findOneAndUpdate).toHaveBeenCalledTimes(1);
  });

  it('passes the request id to the upstream stream', async () => {
    openSpy = jest.spyOn(aiStream, 'openAiStream').mockResolvedValue(
      makeStream([
        ['meta', { sources: [] }],
        ['done', { latency_ms: 1 }],
      ]),
    );

    await request(app)
      .post('/api/chat/stream')
      .set('x-api-key', API_KEY)
      .set('x-request-id', 'trace-stream-42')
      .send({ question: 'What is Pong?' });

    expect(openSpy).toHaveBeenCalledWith({ question: 'What is Pong?' }, 'trace-stream-42');
  });

  // -------------------------------------------------------------------------
  // Cache hit
  // -------------------------------------------------------------------------

  it('replays cached answers as SSE without calling the AI service', async () => {
    // Spy so "not called" can be asserted; the route must short-circuit.
    jest.spyOn(aiStream, 'openAiStream');

    (cache.getCachedResponse as jest.Mock).mockResolvedValue({
      answer: 'Cached answer [1].',
      sources: SOURCES,
      latency_ms: 42,
    });

    const res = await request(app)
      .post('/api/chat/stream')
      .set('x-api-key', API_KEY)
      .send({ question: 'How do you score in Asteroids?' });

    expect(res.status).toBe(200);
    const events = parseSseBody(res.text);
    expect(events.map(([name]) => name)).toEqual(['meta', 'token', 'done']);
    expect(events[1][1]).toEqual({ text: 'Cached answer [1].' });
    expect(events[2][1]).toMatchObject({ latency_ms: 42, cached: true });

    expect(aiStream.openAiStream).not.toHaveBeenCalled();
    expect(Session.findOneAndUpdate).toHaveBeenCalledTimes(1);
  });

  // -------------------------------------------------------------------------
  // Upstream opening failures (before any SSE byte)
  // -------------------------------------------------------------------------

  it('maps an upstream 502 to a JSON 502 envelope', async () => {
    openSpy = jest
      .spyOn(aiStream, 'openAiStream')
      .mockRejectedValue(new aiStream.AiStreamHttpError(502, 'bad gateway', { detail: 'x' }));

    const res = await request(app)
      .post('/api/chat/stream')
      .set('x-api-key', API_KEY)
      .send({ question: 'How do you score in Asteroids?' });

    expect(res.status).toBe(502);
    expect(res.headers['content-type']).toContain('application/json');
    expect(res.body).toEqual({
      error: {
        code: 'AI_SERVICE_ERROR',
        message: 'AI service is unavailable. Please try again later.',
      },
    });
  });

  it('maps an upstream 422 to a 400 validation envelope', async () => {
    openSpy = jest
      .spyOn(aiStream, 'openAiStream')
      .mockRejectedValue(
        new aiStream.AiStreamHttpError(422, 'unprocessable', { detail: 'question too long' }),
      );

    const res = await request(app)
      .post('/api/chat/stream')
      .set('x-api-key', API_KEY)
      .send({ question: 'How do you score in Asteroids?' });

    expect(res.status).toBe(400);
    expect(res.body.error.code).toBe('VALIDATION_ERROR');
    expect(res.body.error.details).toEqual({ detail: 'question too long' });
  });

  it('maps a network failure to a JSON 502 envelope', async () => {
    openSpy = jest
      .spyOn(aiStream, 'openAiStream')
      .mockRejectedValue(new aiStream.AiStreamNetworkError('unreachable'));

    const res = await request(app)
      .post('/api/chat/stream')
      .set('x-api-key', API_KEY)
      .send({ question: 'How do you score in Asteroids?' });

    expect(res.status).toBe(502);
    expect(res.body.error.code).toBe('AI_SERVICE_UNREACHABLE');
  });

  it('maps a header timeout to a JSON 504 envelope', async () => {
    openSpy = jest
      .spyOn(aiStream, 'openAiStream')
      .mockRejectedValue(new aiStream.AiStreamNetworkError('timed out', true));

    const res = await request(app)
      .post('/api/chat/stream')
      .set('x-api-key', API_KEY)
      .send({ question: 'How do you score in Asteroids?' });

    expect(res.status).toBe(504);
    expect(res.body.error.code).toBe('AI_SERVICE_TIMEOUT');
  });

  // -------------------------------------------------------------------------
  // Mid-stream failures (SSE error event)
  // -------------------------------------------------------------------------

  it('relays a mid-stream error event without caching or persisting', async () => {
    openSpy = jest.spyOn(aiStream, 'openAiStream').mockResolvedValue(
      makeStream([
        ['meta', { sources: SOURCES }],
        ['token', { text: 'Partial ' }],
        ['error', { code: 'LLM_ERROR', message: 'provider exploded' }],
      ]),
    );

    const res = await request(app)
      .post('/api/chat/stream')
      .set('x-api-key', API_KEY)
      .send({ question: 'How do you score in Asteroids?' });

    expect(res.status).toBe(200);
    const events = parseSseBody(res.text);
    expect(events.map(([name]) => name)).toEqual(['meta', 'token', 'error']);
    expect(events[2][1]).toEqual({ code: 'LLM_ERROR', message: 'provider exploded' });

    expect(cache.setCachedResponse).not.toHaveBeenCalled();
    expect(Session.findOneAndUpdate).not.toHaveBeenCalled();
  });

  it('emits an error event when the upstream stream breaks', async () => {
    openSpy = jest
      .spyOn(aiStream, 'openAiStream')
      .mockResolvedValue(
        makeFailingStream([['token', { text: 'Partial ' }]], new Error('socket hang up')),
      );

    const res = await request(app)
      .post('/api/chat/stream')
      .set('x-api-key', API_KEY)
      .send({ question: 'How do you score in Asteroids?' });

    const events = parseSseBody(res.text);
    expect(events.map(([name]) => name)).toEqual(['token', 'error']);
    expect(events[1][1]).toMatchObject({ code: 'AI_SERVICE_ERROR' });

    expect(cache.setCachedResponse).not.toHaveBeenCalled();
  });

  // -------------------------------------------------------------------------
  // Validation
  // -------------------------------------------------------------------------

  it('rejects invalid bodies with a 400 validation envelope', async () => {
    jest.spyOn(aiStream, 'openAiStream');

    const res = await request(app)
      .post('/api/chat/stream')
      .set('x-api-key', API_KEY)
      .send({ question: 'x' });

    expect(res.status).toBe(400);
    expect(res.body.error.code).toBe('VALIDATION_ERROR');
    expect(aiStream.openAiStream).not.toHaveBeenCalled();
  });
});
