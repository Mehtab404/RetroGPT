import request from 'supertest';
import app from '../app';

describe('Health Check Endpoints', () => {
  describe('GET /api/health', () => {
    it('should return 200 with status ok', async () => {
      const response = await request(app).get('/api/health');

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('status', 'ok');
      expect(response.body).toHaveProperty('service', 'api-gateway');
      expect(response.body).toHaveProperty('timestamp');
    });
  });

  describe('GET /', () => {
    it('should return service information', async () => {
      const response = await request(app).get('/');

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('service', 'RetroGPT API Gateway');
      expect(response.body).toHaveProperty('version', '0.1.0');
      expect(response.body).toHaveProperty('health', '/api/health');
    });
  });
});
