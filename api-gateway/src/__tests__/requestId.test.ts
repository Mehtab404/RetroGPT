/**
 * Tests for the request-id middleware and per-request logging (Issue #24).
 */

import request from 'supertest';
import app from '../app';
import { logger } from '../utils/logger';
import { sanitizeRequestId } from '../middleware/requestId';

const UUID_V4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

/** Let the server-side `finish` listener run before asserting on logs. */
function flushLogs(): Promise<void> {
  return new Promise((resolve) => setImmediate(resolve));
}

describe('sanitizeRequestId', () => {
  it('keeps a safe inbound id', () => {
    expect(sanitizeRequestId('trace-abc.123_XYZ')).toBe('trace-abc.123_XYZ');
  });

  it('keeps the maximum-length id', () => {
    const maxId = 'a'.repeat(64);
    expect(sanitizeRequestId(maxId)).toBe(maxId);
  });

  it('uses the first value when the header is repeated', () => {
    expect(sanitizeRequestId(['trace-1', 'trace-2'])).toBe('trace-1');
  });

  it('generates a UUID when the header is missing or empty', () => {
    expect(sanitizeRequestId(undefined)).toMatch(UUID_V4);
    expect(sanitizeRequestId('')).toMatch(UUID_V4);
  });

  it('replaces unsafe ids (spaces, control chars, overlong values)', () => {
    expect(sanitizeRequestId('bad id!')).toMatch(UUID_V4);
    expect(sanitizeRequestId('a'.repeat(65))).toMatch(UUID_V4);
    expect(sanitizeRequestId('line\nbreak')).toMatch(UUID_V4);
  });
});

describe('requestId middleware', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('echoes an inbound id on the response', async () => {
    const res = await request(app).get('/').set('x-request-id', 'trace-echo-1');

    expect(res.status).toBe(200);
    expect(res.headers['x-request-id']).toBe('trace-echo-1');
  });

  it('generates and echoes a UUID when the header is absent', async () => {
    const res = await request(app).get('/');

    expect(res.headers['x-request-id']).toMatch(UUID_V4);
  });

  it('replaces an invalid inbound id', async () => {
    const res = await request(app).get('/').set('x-request-id', 'not valid!');

    expect(res.headers['x-request-id']).not.toBe('not valid!');
    expect(res.headers['x-request-id']).toMatch(UUID_V4);
  });

  it('logs one http_request event with status and duration', async () => {
    const infoSpy = jest.spyOn(logger, 'info').mockImplementation(() => undefined);

    await request(app).get('/').set('x-request-id', 'trace-log-9');
    await flushLogs();

    expect(infoSpy).toHaveBeenCalledWith('http_request', {
      request_id: 'trace-log-9',
      method: 'GET',
      path: '/',
      status: 200,
      duration_ms: expect.any(Number),
    });
  });

  it('logs health checks at debug level instead of info', async () => {
    const infoSpy = jest.spyOn(logger, 'info').mockImplementation(() => undefined);
    const debugSpy = jest.spyOn(logger, 'debug').mockImplementation(() => undefined);

    await request(app).get('/api/health');
    await flushLogs();

    expect(debugSpy).toHaveBeenCalledWith(
      'http_request',
      expect.objectContaining({ path: '/api/health', status: 200 }),
    );
    expect(infoSpy).not.toHaveBeenCalled();
  });
});
