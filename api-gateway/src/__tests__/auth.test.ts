/**
 * Tests for API key authentication (x-api-key header) on the gateway.
 *
 * Hits the real Express app; the Redis cache module is mocked so the
 * rate-limiter fails open and no real network/database calls occur.
 */

import request from 'supertest';
import app from '../app';

// Fail-open rate limiter + cache: no Redis in unit tests.
jest.mock('../services/cache', () => ({
  getCachedResponse: jest.fn().mockResolvedValue(null),
  setCachedResponse: jest.fn().mockResolvedValue(undefined),
  getRedisClient: jest.fn().mockResolvedValue(null),
}));

const API_KEY = process.env.API_KEY || 'change-me';

describe('API key authentication', () => {
  // -------------------------------------------------------------------------
  // Protected routes
  // -------------------------------------------------------------------------

  describe('protected /api routes', () => {
    it('returns 401 when the x-api-key header is missing', async () => {
      const res = await request(app)
        .post('/api/chat')
        .send({ question: 'How do you score in Asteroids?' });

      expect(res.status).toBe(401);
      expect(res.body.error.code).toBe('UNAUTHORIZED');
    });

    it('returns 401 when the x-api-key header is wrong', async () => {
      const res = await request(app)
        .post('/api/chat')
        .set('x-api-key', 'definitely-not-the-key')
        .send({ question: 'How do you score in Asteroids?' });

      expect(res.status).toBe(401);
      expect(res.body.error.code).toBe('UNAUTHORIZED');
    });

    it('returns 401 for session routes without a key', async () => {
      const res = await request(app).get('/api/sessions/550e8400-e29b-41d4-a716-446655440000');

      expect(res.status).toBe(401);
      expect(res.body.error.code).toBe('UNAUTHORIZED');
    });

    it('lets a request with the correct key reach the route handler', async () => {
      // An intentionally invalid body proves we passed auth and reached the
      // chat route's own validation (which returns 400, not 401).
      const res = await request(app)
        .post('/api/chat')
        .set('x-api-key', API_KEY)
        .send({ question: 'AB' });

      expect(res.status).toBe(400);
      expect(res.body.error.code).toBe('VALIDATION_ERROR');
    });

    it('does not accept the key from a query parameter', async () => {
      const res = await request(app)
        .post(`/api/chat?x-api-key=${encodeURIComponent(API_KEY)}`)
        .send({ question: 'AB' });

      expect(res.status).toBe(401);
    });

    it('invalidates an empty-string key', async () => {
      const res = await request(app)
        .post('/api/chat')
        .set('x-api-key', '')
        .send({ question: 'AB' });

      expect(res.status).toBe(401);
    });
  });

  // -------------------------------------------------------------------------
  // Public routes
  // -------------------------------------------------------------------------

  describe('public routes', () => {
    it('GET /api/health requires no API key', async () => {
      const res = await request(app).get('/api/health');

      expect(res.status).toBe(200);
      expect(res.body.status).toBe('ok');
    });

    it('GET / requires no API key', async () => {
      const res = await request(app).get('/');

      expect(res.status).toBe(200);
    });
  });
});
