/**
 * Tests for POST /api/chat
 *
 * The AI service HTTP client is mocked via jest.mock so no real network calls
 * are made during testing.
 */

import request from 'supertest';
import axios, { AxiosError } from 'axios';
import app from '../app';
import * as aiClient from '../services/aiClient';

// Mock Session model so no real MongoDB calls are made in chat tests
jest.mock('../models/session', () => ({
  Session: {
    findOne: jest.fn(),
    findOneAndUpdate: jest.fn().mockResolvedValue(null),
  },
}));

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Build a minimal valid AiQueryResponse. */
function makeAiResponse(overrides: Partial<aiClient.AiQueryResponse> = {}): aiClient.AiQueryResponse {
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
function makeAxiosError(
  status: number,
  data: unknown = {},
  code?: string,
): AxiosError {
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

  beforeEach(() => {
    querySpy = jest
      .spyOn(aiClient, 'queryAiService')
      .mockResolvedValue(makeAiResponse());
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
        .send({ question: 'What is Pong?', sessionId });

      expect(res.status).toBe(200);
      expect(res.body.sessionId).toBe(sessionId);
    });

    it('auto-generates a sessionId when not provided', async () => {
      const res = await request(app)
        .post('/api/chat')
        .send({ question: 'How to play Breakout?' });

      expect(res.status).toBe(200);
      // UUID v4 pattern
      expect(res.body.sessionId).toMatch(
        /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i,
      );
    });

    it('forwards the question to the AI service', async () => {
      const question = 'How many lives do you start with in Missile Command?';
      await request(app).post('/api/chat').send({ question });

      expect(querySpy).toHaveBeenCalledTimes(1);
      expect(querySpy).toHaveBeenCalledWith(
        expect.objectContaining({ question }),
      );
    });

    it('accepts a question at the minimum length boundary (3 chars)', async () => {
      const res = await request(app)
        .post('/api/chat')
        .send({ question: 'Hi?' });

      expect(res.status).toBe(200);
    });

    it('accepts a question at the maximum length boundary (500 chars)', async () => {
      const question = 'A'.repeat(500);
      const res = await request(app)
        .post('/api/chat')
        .send({ question });

      expect(res.status).toBe(200);
    });

    it('returns sources array from AI service response', async () => {
      const res = await request(app)
        .post('/api/chat')
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
  // Validation errors
  // -------------------------------------------------------------------------

  describe('validation errors', () => {
    it('returns 400 when question is missing', async () => {
      const res = await request(app).post('/api/chat').send({});

      expect(res.status).toBe(400);
      expect(res.body.error.code).toBe('VALIDATION_ERROR');
    });

    it('returns 400 when question is too short (< 3 chars)', async () => {
      const res = await request(app)
        .post('/api/chat')
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
        .send({ question: 'A'.repeat(501) });

      expect(res.status).toBe(400);
      expect(res.body.error.code).toBe('VALIDATION_ERROR');
    });

    it('returns 400 when sessionId is not a valid UUID', async () => {
      const res = await request(app).post('/api/chat').send({
        question: 'What is Centipede?',
        sessionId: 'not-a-uuid',
      });

      expect(res.status).toBe(400);
      expect(res.body.error.code).toBe('VALIDATION_ERROR');
    });

    it('returns 400 when question is not a string', async () => {
      const res = await request(app)
        .post('/api/chat')
        .send({ question: 12345 });

      expect(res.status).toBe(400);
      expect(res.body.error.code).toBe('VALIDATION_ERROR');
    });

    it('does not call AI service when validation fails', async () => {
      await request(app).post('/api/chat').send({ question: 'AB' });

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
        .send({ question: 'What is Space Invaders?' });

      expect(res.status).toBe(502);
      expect(res.body.error.code).toBe('AI_SERVICE_ERROR');
    });

    it('returns 400 when AI service returns 422 (upstream validation error)', async () => {
      querySpy.mockRejectedValue(
        makeAxiosError(422, { detail: 'question too long' }),
      );

      const res = await request(app)
        .post('/api/chat')
        .send({ question: 'What is Galaga?' });

      expect(res.status).toBe(400);
      expect(res.body.error.code).toBe('VALIDATION_ERROR');
    });

    it('returns 504 when AI service times out', async () => {
      querySpy.mockRejectedValue(makeNetworkError('ECONNABORTED'));

      const res = await request(app)
        .post('/api/chat')
        .send({ question: 'What is Frogger?' });

      expect(res.status).toBe(504);
      expect(res.body.error.code).toBe('AI_SERVICE_TIMEOUT');
    });

    it('returns 504 for ETIMEDOUT errors', async () => {
      querySpy.mockRejectedValue(makeNetworkError('ETIMEDOUT'));

      const res = await request(app)
        .post('/api/chat')
        .send({ question: 'What is Donkey Kong?' });

      expect(res.status).toBe(504);
      expect(res.body.error.code).toBe('AI_SERVICE_TIMEOUT');
    });

    it('returns 502 when AI service is unreachable (ECONNREFUSED)', async () => {
      querySpy.mockRejectedValue(makeNetworkError('ECONNREFUSED'));

      const res = await request(app)
        .post('/api/chat')
        .send({ question: 'What is Pac-Man?' });

      expect(res.status).toBe(502);
      expect(res.body.error.code).toBe('AI_SERVICE_UNREACHABLE');
    });
  });
});
