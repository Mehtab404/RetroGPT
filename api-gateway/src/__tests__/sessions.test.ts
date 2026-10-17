/**
 * Tests for GET /api/sessions/:id
 *
 * The Mongoose Session model is mocked so no real database calls occur.
 */

import request from 'supertest';
import app from '../app';

// ---------------------------------------------------------------------------
// Mock the Session model
// ---------------------------------------------------------------------------

// jest.mock is hoisted, so variables defined outside can't be referenced inside.
// Instead, define the mocks inside the factory and retrieve them after import.
jest.mock('../models/session', () => ({
  Session: {
    findOne: jest.fn(),
    findOneAndUpdate: jest.fn(),
  },
}));

// Rate limiter + cache: no real Redis calls in tests.
jest.mock('../services/cache', () => ({
  getRedisClient: jest.fn().mockResolvedValue(null),
}));

const API_KEY = process.env.API_KEY || 'change-me';

// Retrieve mock references after the module has been set up
import { Session } from '../models/session';
const mockFindOne = Session.findOne as jest.Mock;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const VALID_UUID = '550e8400-e29b-41d4-a716-446655440000';
const ANOTHER_UUID = '6ba7b810-9dad-11d1-80b4-00c04fd430c8';

function makeSession(sessionId: string = VALID_UUID) {
  return {
    sessionId,
    messages: [
      {
        role: 'user',
        content: 'How do you score in Asteroids?',
        createdAt: new Date('2026-10-17T10:00:00.000Z'),
      },
      {
        role: 'assistant',
        content: 'You score by destroying asteroids.',
        sources: [
          {
            id: 'asteroids_manual_p3_c2',
            title: 'Asteroids Manual',
            score: 0.91,
            snippet: 'Large asteroids: 20 points...',
          },
        ],
        createdAt: new Date('2026-10-17T10:00:02.000Z'),
      },
    ],
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('GET /api/sessions/:id', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  // -------------------------------------------------------------------------
  // Happy path
  // -------------------------------------------------------------------------

  describe('success cases', () => {
    it('returns 200 with session messages when session exists', async () => {
      mockFindOne.mockReturnValue({ lean: () => Promise.resolve(makeSession()) });

      const res = await request(app).get(`/api/sessions/${VALID_UUID}`).set('x-api-key', API_KEY);

      expect(res.status).toBe(200);
      expect(res.body.sessionId).toBe(VALID_UUID);
      expect(res.body.messages).toHaveLength(2);
    });

    it('shapes user messages correctly (no sources field)', async () => {
      mockFindOne.mockReturnValue({ lean: () => Promise.resolve(makeSession()) });

      const res = await request(app).get(`/api/sessions/${VALID_UUID}`).set('x-api-key', API_KEY);

      expect(res.status).toBe(200);
      const userMsg = res.body.messages[0];
      expect(userMsg.role).toBe('user');
      expect(userMsg.content).toBe('How do you score in Asteroids?');
      expect(userMsg).not.toHaveProperty('sources');
    });

    it('shapes assistant messages correctly (with sources)', async () => {
      mockFindOne.mockReturnValue({ lean: () => Promise.resolve(makeSession()) });

      const res = await request(app).get(`/api/sessions/${VALID_UUID}`).set('x-api-key', API_KEY);

      expect(res.status).toBe(200);
      const assistantMsg = res.body.messages[1];
      expect(assistantMsg.role).toBe('assistant');
      expect(assistantMsg.sources).toHaveLength(1);
      expect(assistantMsg.sources[0]).toMatchObject({
        id: 'asteroids_manual_p3_c2',
        title: 'Asteroids Manual',
        score: 0.91,
      });
    });

    it('returns ISO8601 createdAt strings', async () => {
      mockFindOne.mockReturnValue({ lean: () => Promise.resolve(makeSession()) });

      const res = await request(app).get(`/api/sessions/${VALID_UUID}`).set('x-api-key', API_KEY);

      expect(res.status).toBe(200);
      expect(res.body.messages[0].createdAt).toBe('2026-10-17T10:00:00.000Z');
      expect(res.body.messages[1].createdAt).toBe('2026-10-17T10:00:02.000Z');
    });

    it('returns empty messages array for a session with no messages', async () => {
      mockFindOne.mockReturnValue({
        lean: () => Promise.resolve({ sessionId: VALID_UUID, messages: [] }),
      });

      const res = await request(app).get(`/api/sessions/${VALID_UUID}`).set('x-api-key', API_KEY);

      expect(res.status).toBe(200);
      expect(res.body.messages).toEqual([]);
    });

    it('queries MongoDB with the correct sessionId', async () => {
      mockFindOne.mockReturnValue({ lean: () => Promise.resolve(makeSession(ANOTHER_UUID)) });

      await request(app).get(`/api/sessions/${ANOTHER_UUID}`).set('x-api-key', API_KEY);

      expect(mockFindOne).toHaveBeenCalledWith({ sessionId: ANOTHER_UUID });
    });

    it('handles message with a non-Date createdAt (string fallback)', async () => {
      mockFindOne.mockReturnValue({
        lean: () =>
          Promise.resolve({
            sessionId: VALID_UUID,
            messages: [
              {
                role: 'user' as const,
                content: 'Hello',
                createdAt: '2026-10-17T10:00:00.000Z' as unknown as Date,
              },
            ],
          }),
      });

      const res = await request(app).get(`/api/sessions/${VALID_UUID}`).set('x-api-key', API_KEY);

      expect(res.status).toBe(200);
      expect(typeof res.body.messages[0].createdAt).toBe('string');
    });
  });

  // -------------------------------------------------------------------------
  // Not found
  // -------------------------------------------------------------------------

  describe('not found', () => {
    it('returns 404 when session does not exist', async () => {
      mockFindOne.mockReturnValue({ lean: () => Promise.resolve(null) });

      const res = await request(app).get(`/api/sessions/${VALID_UUID}`).set('x-api-key', API_KEY);

      expect(res.status).toBe(404);
      expect(res.body.error.code).toBe('NOT_FOUND');
    });

    it('includes the sessionId in the 404 message', async () => {
      mockFindOne.mockReturnValue({ lean: () => Promise.resolve(null) });

      const res = await request(app).get(`/api/sessions/${VALID_UUID}`).set('x-api-key', API_KEY);

      expect(res.body.error.message).toContain(VALID_UUID);
    });
  });

  // -------------------------------------------------------------------------
  // Validation errors
  // -------------------------------------------------------------------------

  describe('validation errors', () => {
    it('returns 400 for a non-UUID id', async () => {
      const res = await request(app).get('/api/sessions/not-a-uuid').set('x-api-key', API_KEY);

      expect(res.status).toBe(400);
      expect(res.body.error.code).toBe('VALIDATION_ERROR');
    });

    it('returns 400 for a numeric id', async () => {
      const res = await request(app).get('/api/sessions/12345').set('x-api-key', API_KEY);

      expect(res.status).toBe(400);
      expect(res.body.error.code).toBe('VALIDATION_ERROR');
    });

    it('does not query MongoDB when id is invalid', async () => {
      await request(app).get('/api/sessions/bad-id').set('x-api-key', API_KEY);

      expect(mockFindOne).not.toHaveBeenCalled();
    });
  });

  // -------------------------------------------------------------------------
  // Database errors
  // -------------------------------------------------------------------------

  describe('database errors', () => {
    it('returns 500 when MongoDB throws an error', async () => {
      mockFindOne.mockReturnValue({
        lean: () => Promise.reject(new Error('DB connection lost')),
      });

      const res = await request(app).get(`/api/sessions/${VALID_UUID}`).set('x-api-key', API_KEY);

      expect(res.status).toBe(500);
    });
  });
});
