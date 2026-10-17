/**
 * Tests for POST /api/chat
 *
 * The AI service HTTP client and Redis cache are mocked so no real network
 * calls are made during testing.
 */

import request from 'supertest';
import axios, { AxiosError } from 'axios';
import app from '../app';
import * as aiClient from '../services/aiClient';
import * as cache from '../services/cache';
import { logger } from '../utils/logger';
import { Session } from '../models/session';

// Mock Session model so no real MongoDB calls are made in chat tests
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

/** Build a minimal valid AiQueryResponse. */
function makeAiResponse(
  overrides: Partial<aiClient.AiQueryResponse> = {},
): aiClient.AiQueryResponse {
  return {
    answer: 'You score points by destroying asteroids. [1]',
    sources: [
      {
        id: 'asteroids_manual_p3_c2',
        title: 'Asteroids Manual',
        score: 0.91,
        snippet: 'Large asteroids: 20 points...',
      },
    ],
    latency_ms: 1820,
    cached: false,
    ...overrides,
  };
}

/** Build an AxiosError with an HTTP response attached. */
function makeAxiosError(status: number, data: unknown = {}, code?: string): AxiosError {
  const err = new axios.AxiosError('Request failed', code ?? String(status));
  err.response = {
    status,
    data,
    statusText: String(status),
    headers: {},
    config: {} as never,
  };
  return err;
}

/** Build an AxiosError without a response (network failure). */
function makeNetworkError(code: string): AxiosError {
  const err = new axios.AxiosError('Network error', code);
  return err;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('POST /api/chat', () => {
  let querySpy: jest.SpyInstance;
  let getCacheSpy: jest.SpyInstance;
  let setCacheSpy: jest.SpyInstance;

  beforeEach(() => {
    jest.clearAllMocks();

    querySpy = jest.spyOn(aiClient, 'queryAiService').mockResolvedValue(makeAiResponse());

    getCacheSpy = jest.spyOn(cache, 'getCachedResponse').mockResolvedValue(null);
    setCacheSpy = jest.spyOn(cache, 'setCachedResponse').mockResolvedValue(undefined);
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  // -------------------------------------------------------------------------
  // Happy path
  // -------------------------------------------------------------------------

  describe('success cases', () => {
    it('returns 200 with answer and sources for a valid question', async () => {
      const res = await request(app)
        .post('/api/chat')
        .set('x-api-key', API_KEY)
        .send({ question: 'How do you score points in Asteroids?' });

      expect(res.status).toBe(200);
      expect(res.body).toMatchObject({
        answer: expect.any(String),
        sources: expect.any(Array),
        latency_ms: expect.any(Number),
        cached: false,
      });
    });

    it('returns a sessionId in the response', async () => {
      const res = await request(app)
        .post('/api/chat')
        .set('x-api-key', API_KEY)
        .send({ question: 'What is Pong?' });

      expect(res.status).toBe(200);
      expect(res.body).toHaveProperty('sessionId');
      expect(typeof res.body.sessionId).toBe('string');
      expect(res.body.sessionId.length).toBeGreaterThan(0);
    });

    it('echoes back the provided sessionId', async () => {
      const sessionId = '550e8400-e29b-41d4-a716-446655440000';
      const res = await request(app)
        .post('/api/chat')
        .set('x-api-key', API_KEY)
        .send({ question: 'What is Pong?', sessionId });

      expect(res.status).toBe(200);
      expect(res.body.sessionId).toBe(sessionId);
    });

    it('auto-generates a sessionId when not provided', async () => {
      const res = await request(app)
        .post('/api/chat')
        .set('x-api-key', API_KEY)
        .send({ question: 'How to play Breakout?' });

      expect(res.status).toBe(200);
      // UUID v4 pattern
      expect(res.body.sessionId).toMatch(
        /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i,
      );
    });

    it('forwards the question to the AI service', async () => {
      const question = 'How many lives do you start with in Missile Command?';
      await request(app).post('/api/chat').set('x-api-key', API_KEY).send({ question });

      expect(querySpy).toHaveBeenCalledTimes(1);
      expect(querySpy).toHaveBeenCalledWith(
        expect.objectContaining({ question }),
        expect.any(String),
      );
    });

    it('accepts a question at the minimum length boundary (3 chars)', async () => {
      const res = await request(app)
        .post('/api/chat')
        .set('x-api-key', API_KEY)
        .send({ question: 'Hi?' });

      expect(res.status).toBe(200);
    });

    it('accepts a question at the maximum length boundary (500 chars)', async () => {
      const question = 'A'.repeat(500);
      const res = await request(app).post('/api/chat').set('x-api-key', API_KEY).send({ question });

      expect(res.status).toBe(200);
    });

    it('returns sources array from AI service response', async () => {
      const res = await request(app)
        .post('/api/chat')
        .set('x-api-key', API_KEY)
        .send({ question: 'How do asteroids split?' });

      expect(res.body.sources).toHaveLength(1);
      expect(res.body.sources[0]).toMatchObject({
        id: 'asteroids_manual_p3_c2',
        title: 'Asteroids Manual',
        score: expect.any(Number),
        snippet: expect.any(String),
      });
    });
  });

  // -------------------------------------------------------------------------
  // Request tracing (Issue #24)
  // -------------------------------------------------------------------------

  describe('request tracing', () => {
    it('forwards x-request-id to the AI service and echoes it', async () => {
      const res = await request(app)
        .post('/api/chat')
        .set('x-api-key', API_KEY)
        .set('x-request-id', 'trace-chat-1')
        .send({ question: 'How do you score in Asteroids?' });

      expect(res.headers['x-request-id']).toBe('trace-chat-1');
      expect(querySpy).toHaveBeenCalledWith(
        expect.objectContaining({ question: expect.any(String) }),
        'trace-chat-1',
      );
    });

    it('generates a request id when the client does not send one', async () => {
      const res = await request(app)
        .post('/api/chat')
        .set('x-api-key', API_KEY)
        .send({ question: 'What is Pong?' });

      expect(res.headers['x-request-id']).toMatch(
        /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i,
      );
      expect(querySpy).toHaveBeenCalledWith(expect.anything(), res.headers['x-request-id']);
    });

    it('logs chat_completed with the request id on a cache miss', async () => {
      const infoSpy = jest.spyOn(logger, 'info').mockImplementation(() => undefined);

      await request(app)
        .post('/api/chat')
        .set('x-api-key', API_KEY)
        .set('x-request-id', 'trace-chat-2')
        .send({ question: 'How do you score in Asteroids?' });

      expect(infoSpy).toHaveBeenCalledWith(
        'chat_completed',
        expect.objectContaining({
          request_id: 'trace-chat-2',
          cached: false,
          upstream_ms: expect.any(Number),
          sources: 1,
        }),
      );
    });

    it('logs chat_completed with cached: true on a cache hit', async () => {
      getCacheSpy.mockResolvedValue({
        answer: 'Cached answer.',
        sources: [],
        latency_ms: 42,
      });
      const infoSpy = jest.spyOn(logger, 'info').mockImplementation(() => undefined);

      await request(app)
        .post('/api/chat')
        .set('x-api-key', API_KEY)
        .set('x-request-id', 'trace-chat-3')
        .send({ question: 'How do you score in Asteroids?' });

      expect(infoSpy).toHaveBeenCalledWith(
        'chat_completed',
        expect.objectContaining({
          request_id: 'trace-chat-3',
          cached: true,
          latency_ms: 42,
        }),
      );
    });
  });

  // -------------------------------------------------------------------------
  // Caching behaviour
  // -------------------------------------------------------------------------

  describe('caching behaviour', () => {
    it('returns cached: false on a cache miss', async () => {
      getCacheSpy.mockResolvedValue(null);

      const res = await request(app)
        .post('/api/chat')
        .set('x-api-key', API_KEY)
        .send({ question: 'How do you score in Asteroids?' });

      expect(res.status).toBe(200);
      expect(res.body.cached).toBe(false);
    });

    it('calls getCachedResponse before querying the AI service', async () => {
      getCacheSpy.mockResolvedValue(null);
      const question = 'How do you score in Asteroids?';

      await request(app).post('/api/chat').set('x-api-key', API_KEY).send({ question });

      expect(getCacheSpy).toHaveBeenCalledWith(question);
      // AI service should still be called on a miss
      expect(querySpy).toHaveBeenCalledTimes(1);
    });

    it('calls setCachedResponse with the AI response on a cache miss', async () => {
      getCacheSpy.mockResolvedValue(null);
      const aiResp = makeAiResponse();
      querySpy.mockResolvedValue(aiResp);

      const question = 'How do you score in Asteroids?';
      await request(app).post('/api/chat').set('x-api-key', API_KEY).send({ question });

      expect(setCacheSpy).toHaveBeenCalledWith(
        question,
        expect.objectContaining({
          answer: aiResp.answer,
          sources: aiResp.sources,
          latency_ms: aiResp.latency_ms,
        }),
      );
    });

    it('returns cached: true on a cache hit', async () => {
      const cachedPayload = {
        answer: 'Cached answer about Asteroids.',
        sources: [
          {
            id: 'asteroids_manual_p3_c2',
            title: 'Asteroids Manual',
            score: 0.91,
            snippet: 'Large asteroids: 20 points...',
          },
        ],
        latency_ms: 50,
      };
      getCacheSpy.mockResolvedValue(cachedPayload);

      const res = await request(app)
        .post('/api/chat')
        .set('x-api-key', API_KEY)
        .send({ question: 'How do you score in Asteroids?' });

      expect(res.status).toBe(200);
      expect(res.body.cached).toBe(true);
      expect(res.body.answer).toBe(cachedPayload.answer);
      expect(res.body.latency_ms).toBe(cachedPayload.latency_ms);
    });

    it('does NOT call AI service when cache hit', async () => {
      const cachedPayload = {
        answer: 'Cached answer.',
        sources: [],
        latency_ms: 10,
      };
      getCacheSpy.mockResolvedValue(cachedPayload);

      await request(app)
        .post('/api/chat')
        .set('x-api-key', API_KEY)
        .send({ question: 'How do you score in Asteroids?' });

      expect(querySpy).not.toHaveBeenCalled();
    });

    it('returns the correct sessionId on a cache hit', async () => {
      const cachedPayload = {
        answer: 'Cached answer.',
        sources: [],
        latency_ms: 10,
      };
      getCacheSpy.mockResolvedValue(cachedPayload);

      const sessionId = '550e8400-e29b-41d4-a716-446655440000';
      const res = await request(app)
        .post('/api/chat')
        .set('x-api-key', API_KEY)
        .send({ question: 'How do you score in Asteroids?', sessionId });

      expect(res.status).toBe(200);
      expect(res.body.sessionId).toBe(sessionId);
    });

    it('does not call getCachedResponse when validation fails', async () => {
      await request(app).post('/api/chat').set('x-api-key', API_KEY).send({ question: 'AB' });

      expect(getCacheSpy).not.toHaveBeenCalled();
    });
  });

  // -------------------------------------------------------------------------
  // Conversation history (Issue #27)
  // -------------------------------------------------------------------------

  describe('conversation history', () => {
    const SESSION_ID = '550e8400-e29b-41d4-a716-446655440000';

    /** Build a stored-session message fixture. */
    function storedMessage(role: 'user' | 'assistant', content: string) {
      return { role, content, createdAt: new Date() };
    }

    beforeEach(() => {
      // Default: the session does not exist → no history.
      (Session.findOne as unknown as jest.Mock).mockResolvedValue(null);
    });

    it('loads the stored session history server-side and forwards it', async () => {
      (Session.findOne as unknown as jest.Mock).mockResolvedValue({
        messages: [
          storedMessage('user', 'How do you score in Asteroids?'),
          storedMessage('assistant', 'Destroy asteroids for points [1].'),
        ],
      });

      await request(app)
        .post('/api/chat')
        .set('x-api-key', API_KEY)
        .send({ question: 'What about the saucer?', sessionId: SESSION_ID });

      expect(Session.findOne).toHaveBeenCalledWith({ sessionId: SESSION_ID });
      expect(querySpy).toHaveBeenCalledWith(
        {
          question: 'What about the saucer?',
          history: [
            { role: 'user', content: 'How do you score in Asteroids?' },
            { role: 'assistant', content: 'Destroy asteroids for points [1].' },
          ],
        },
        expect.any(String),
      );
    });

    it('forwards only the last six messages, oldest first', async () => {
      const messages = Array.from({ length: 8 }, (_, i) =>
        storedMessage(i % 2 === 0 ? 'user' : 'assistant', `message ${i + 1}`),
      );
      (Session.findOne as unknown as jest.Mock).mockResolvedValue({ messages });

      await request(app)
        .post('/api/chat')
        .set('x-api-key', API_KEY)
        .send({ question: 'What about the saucer?', sessionId: SESSION_ID });

      const [aiRequest] = querySpy.mock.calls[0];
      expect(aiRequest.history).toEqual([
        { role: 'user', content: 'message 3' },
        { role: 'assistant', content: 'message 4' },
        { role: 'user', content: 'message 5' },
        { role: 'assistant', content: 'message 6' },
        { role: 'user', content: 'message 7' },
        { role: 'assistant', content: 'message 8' },
      ]);
    });

    it('applies the character cap, preferring the newest messages', async () => {
      const halfCap = 'a'.repeat(2000);
      (Session.findOne as unknown as jest.Mock).mockResolvedValue({
        messages: [
          storedMessage('user', halfCap),
          storedMessage('assistant', halfCap),
          storedMessage('user', 'newest question'),
        ],
      });

      await request(app)
        .post('/api/chat')
        .set('x-api-key', API_KEY)
        .send({ question: 'What about the saucer?', sessionId: SESSION_ID });

      const [aiRequest] = querySpy.mock.calls[0];
      expect(aiRequest.history).toEqual([
        { role: 'assistant', content: halfCap },
        { role: 'user', content: 'newest question' },
      ]);
    });

    it('proceeds without history when the session does not exist', async () => {
      await request(app)
        .post('/api/chat')
        .set('x-api-key', API_KEY)
        .send({ question: 'What about the saucer?', sessionId: SESSION_ID });

      expect(querySpy).toHaveBeenCalledWith(
        { question: 'What about the saucer?' },
        expect.any(String),
      );
    });

    it('proceeds without history when the session has no messages', async () => {
      (Session.findOne as unknown as jest.Mock).mockResolvedValue({ messages: [] });

      await request(app)
        .post('/api/chat')
        .set('x-api-key', API_KEY)
        .send({ question: 'What about the saucer?', sessionId: SESSION_ID });

      expect(querySpy).toHaveBeenCalledWith(
        { question: 'What about the saucer?' },
        expect.any(String),
      );
    });

    it('fails open (200, no history) when the session lookup throws', async () => {
      (Session.findOne as unknown as jest.Mock).mockRejectedValue(new Error('mongo down'));
      const warnSpy = jest.spyOn(logger, 'warn').mockImplementation(() => undefined);

      const res = await request(app)
        .post('/api/chat')
        .set('x-api-key', API_KEY)
        .set('x-request-id', 'trace-history-1')
        .send({ question: 'What about the saucer?', sessionId: SESSION_ID });

      expect(res.status).toBe(200);
      expect(querySpy).toHaveBeenCalledWith(
        { question: 'What about the saucer?' },
        expect.any(String),
      );
      expect(warnSpy).toHaveBeenCalledWith(
        'session_history_load_failed',
        expect.objectContaining({
          request_id: 'trace-history-1',
          session_id: SESSION_ID,
          error: expect.any(Error),
        }),
      );
    });

    it('does not query sessions when no sessionId is provided', async () => {
      await request(app)
        .post('/api/chat')
        .set('x-api-key', API_KEY)
        .send({ question: 'What about the saucer?' });

      expect(Session.findOne).not.toHaveBeenCalled();
    });

    it('ignores browser-supplied history in the request body', async () => {
      await request(app)
        .post('/api/chat')
        .set('x-api-key', API_KEY)
        .send({
          question: 'What about the saucer?',
          sessionId: SESSION_ID,
          history: [{ role: 'user', content: 'BROWSER-SUPPLIED HISTORY' }],
        });

      // The stored session does not exist → no history is forwarded even
      // though the browser sent some.
      expect(querySpy).toHaveBeenCalledWith(
        { question: 'What about the saucer?' },
        expect.any(String),
      );
    });

    it('uses a history-aware cache key when history is present', async () => {
      (Session.findOne as unknown as jest.Mock).mockResolvedValue({
        messages: [storedMessage('user', 'How do you score in Asteroids?')],
      });
      getCacheSpy.mockResolvedValue(null);

      const question = 'What about the saucer?';
      await request(app)
        .post('/api/chat')
        .set('x-api-key', API_KEY)
        .send({ question, sessionId: SESSION_ID });

      const expectedHistory = [{ role: 'user', content: 'How do you score in Asteroids?' }];
      expect(getCacheSpy).toHaveBeenCalledWith(question, expectedHistory);
      expect(setCacheSpy).toHaveBeenCalledWith(
        question,
        expect.objectContaining({ answer: expect.any(String) }),
        undefined,
        expectedHistory,
      );
    });

    it('keeps the legacy cache call shape when no history is present', async () => {
      const question = 'How do you score in Asteroids?';
      await request(app).post('/api/chat').set('x-api-key', API_KEY).send({ question });

      expect(getCacheSpy).toHaveBeenCalledWith(question);
      expect(setCacheSpy).toHaveBeenCalledWith(
        question,
        expect.objectContaining({ answer: expect.any(String) }),
      );
    });

    it('replays a history-aware cache hit without calling the AI service', async () => {
      (Session.findOne as unknown as jest.Mock).mockResolvedValue({
        messages: [storedMessage('user', 'How do you score in Asteroids?')],
      });
      getCacheSpy.mockResolvedValue({
        answer: 'Cached follow-up answer.',
        sources: [],
        latency_ms: 25,
      });

      const question = 'What about the saucer?';
      const res = await request(app)
        .post('/api/chat')
        .set('x-api-key', API_KEY)
        .send({ question, sessionId: SESSION_ID });

      expect(res.status).toBe(200);
      expect(res.body.cached).toBe(true);
      expect(querySpy).not.toHaveBeenCalled();
      expect(getCacheSpy).toHaveBeenCalledWith(question, [
        { role: 'user', content: 'How do you score in Asteroids?' },
      ]);
    });
  });

  // -------------------------------------------------------------------------
  // Validation errors
  // -------------------------------------------------------------------------

  describe('validation errors', () => {
    it('returns 400 when question is missing', async () => {
      const res = await request(app).post('/api/chat').set('x-api-key', API_KEY).send({});

      expect(res.status).toBe(400);
      expect(res.body.error.code).toBe('VALIDATION_ERROR');
    });

    it('returns 400 when question is too short (< 3 chars)', async () => {
      const res = await request(app)
        .post('/api/chat')
        .set('x-api-key', API_KEY)
        .send({ question: 'Hi' });

      expect(res.status).toBe(400);
      expect(res.body.error.code).toBe('VALIDATION_ERROR');
      expect(res.body.error.details).toEqual(
        expect.arrayContaining([
          expect.objectContaining({ message: expect.stringContaining('3') }),
        ]),
      );
    });

    it('returns 400 when question exceeds 500 characters', async () => {
      const res = await request(app)
        .post('/api/chat')
        .set('x-api-key', API_KEY)
        .send({ question: 'A'.repeat(501) });

      expect(res.status).toBe(400);
      expect(res.body.error.code).toBe('VALIDATION_ERROR');
    });

    it('returns 400 when sessionId is not a valid UUID', async () => {
      const res = await request(app).post('/api/chat').set('x-api-key', API_KEY).send({
        question: 'What is Centipede?',
        sessionId: 'not-a-uuid',
      });

      expect(res.status).toBe(400);
      expect(res.body.error.code).toBe('VALIDATION_ERROR');
    });

    it('returns 400 when question is not a string', async () => {
      const res = await request(app)
        .post('/api/chat')
        .set('x-api-key', API_KEY)
        .send({ question: 12345 });

      expect(res.status).toBe(400);
      expect(res.body.error.code).toBe('VALIDATION_ERROR');
    });

    it('does not call AI service when validation fails', async () => {
      await request(app).post('/api/chat').set('x-api-key', API_KEY).send({ question: 'AB' });

      expect(querySpy).not.toHaveBeenCalled();
    });
  });

  // -------------------------------------------------------------------------
  // AI service error handling
  // -------------------------------------------------------------------------

  describe('AI service errors', () => {
    it('returns 502 when AI service returns a 500 error', async () => {
      querySpy.mockRejectedValue(makeAxiosError(500, { detail: 'Internal error' }));

      const res = await request(app)
        .post('/api/chat')
        .set('x-api-key', API_KEY)
        .send({ question: 'What is Space Invaders?' });

      expect(res.status).toBe(502);
      expect(res.body.error.code).toBe('AI_SERVICE_ERROR');
    });

    it('returns 400 when AI service returns 422 (upstream validation error)', async () => {
      querySpy.mockRejectedValue(makeAxiosError(422, { detail: 'question too long' }));

      const res = await request(app)
        .post('/api/chat')
        .set('x-api-key', API_KEY)
        .send({ question: 'What is Galaga?' });

      expect(res.status).toBe(400);
      expect(res.body.error.code).toBe('VALIDATION_ERROR');
    });

    it('returns 504 when AI service times out', async () => {
      querySpy.mockRejectedValue(makeNetworkError('ECONNABORTED'));

      const res = await request(app)
        .post('/api/chat')
        .set('x-api-key', API_KEY)
        .send({ question: 'What is Frogger?' });

      expect(res.status).toBe(504);
      expect(res.body.error.code).toBe('AI_SERVICE_TIMEOUT');
    });

    it('returns 504 for ETIMEDOUT errors', async () => {
      querySpy.mockRejectedValue(makeNetworkError('ETIMEDOUT'));

      const res = await request(app)
        .post('/api/chat')
        .set('x-api-key', API_KEY)
        .send({ question: 'What is Donkey Kong?' });

      expect(res.status).toBe(504);
      expect(res.body.error.code).toBe('AI_SERVICE_TIMEOUT');
    });

    it('returns 502 when AI service is unreachable (ECONNREFUSED)', async () => {
      querySpy.mockRejectedValue(makeNetworkError('ECONNREFUSED'));

      const res = await request(app)
        .post('/api/chat')
        .set('x-api-key', API_KEY)
        .send({ question: 'What is Pac-Man?' });

      expect(res.status).toBe(502);
      expect(res.body.error.code).toBe('AI_SERVICE_UNREACHABLE');
    });
  });
});
