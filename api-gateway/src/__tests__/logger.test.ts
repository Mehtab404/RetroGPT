/**
 * Tests for the structured JSON logger (Issue #24).
 *
 * stdout/stderr writes are mocked so no log lines leak into the test run.
 */

import { logger, setLogLevel } from '../utils/logger';

describe('logger', () => {
  let stdoutSpy: jest.SpyInstance;
  let stderrSpy: jest.SpyInstance;

  beforeEach(() => {
    setLogLevel('debug');
    stdoutSpy = jest.spyOn(process.stdout, 'write').mockImplementation(() => true);
    stderrSpy = jest.spyOn(process.stderr, 'write').mockImplementation(() => true);
  });

  afterEach(() => {
    setLogLevel('info');
    jest.restoreAllMocks();
  });

  it('writes one JSON line to stdout with the shared base fields', () => {
    logger.info('http_request', { request_id: 'trace-1', status: 200 });

    expect(stdoutSpy).toHaveBeenCalledTimes(1);
    const line = stdoutSpy.mock.calls[0][0] as string;
    expect(line.endsWith('\n')).toBe(true);

    const payload = JSON.parse(line);
    expect(payload).toMatchObject({
      level: 'info',
      service: 'api-gateway',
      event: 'http_request',
      request_id: 'trace-1',
      status: 200,
    });
    expect(payload.timestamp).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/);
  });

  it('writes warn/error records to stderr with normalised level names', () => {
    logger.warn('ratelimit_redis_error', { request_id: 'r1' });
    logger.error('request_failed', { status: 500 });

    expect(stdoutSpy).not.toHaveBeenCalled();
    expect(stderrSpy).toHaveBeenCalledTimes(2);
    expect(JSON.parse(stderrSpy.mock.calls[0][0] as string).level).toBe('warning');
    expect(JSON.parse(stderrSpy.mock.calls[1][0] as string).level).toBe('error');
  });

  it('expands Error instances in the error field', () => {
    logger.error('uncaught_exception', { error: new Error('boom') });

    const payload = JSON.parse(stderrSpy.mock.calls[0][0] as string);
    expect(payload.error).toMatchObject({ name: 'Error', message: 'boom' });
    expect(typeof payload.error.stack).toBe('string');
  });

  it('suppresses records below the configured level', () => {
    setLogLevel('warn');

    logger.debug('cache_hit');
    logger.info('cache_hit');
    expect(stdoutSpy).not.toHaveBeenCalled();

    logger.warn('cache_error');
    expect(stderrSpy).toHaveBeenCalledTimes(1);
  });

  it('serialises nested context objects', () => {
    logger.info('query', { stage_ms: { embed: 1.5, generate: 10 } });

    const payload = JSON.parse(stdoutSpy.mock.calls[0][0] as string);
    expect(payload.stage_ms).toEqual({ embed: 1.5, generate: 10 });
  });
});
