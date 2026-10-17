/**
 * Unit tests for the AI service HTTP client (aiClient.ts).
 *
 * Uses `setAxiosInstance` to inject a mock Axios instance so no real HTTP
 * calls are made.  Retry / back-off logic is exercised by making the mock
 * reject on the first N attempts and resolve on the final attempt.
 *
 * Jest's fake timers are NOT needed here because the test replaces the
 * `sleep` indirection by mocking the axios instance itself — the delays are
 * bypassed by the mock's immediate rejection/resolution.
 */

import axios, { AxiosError, AxiosInstance } from 'axios';
import {
  queryAiService,
  setAxiosInstance,
  resetAxiosInstance,
  AiQueryResponse,
} from '../services/aiClient';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Minimal valid AI response fixture. */
function makeAiResponse(overrides: Partial<AiQueryResponse> = {}): AiQueryResponse {
  return {
    answer: 'Large asteroids are worth 20 points.',
    sources: [],
    latency_ms: 500,
    cached: false,
    ...overrides,
  };
}

/** Build an AxiosError with an HTTP response. */
function makeHttpError(status: number, data: unknown = {}): AxiosError {
  const err = new axios.AxiosError(`Request failed with status ${status}`, String(status));
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
  return new axios.AxiosError('Network error', code);
}

/**
 * Create a mock Axios instance whose `post` method returns the provided
 * sequence of outcomes in order.  Each element is either a resolved value or
 * a rejected error.
 */
function makeMockInstance(
  outcomes: Array<{ resolve: AiQueryResponse } | { reject: AxiosError }>,
): AxiosInstance {
  let callCount = 0;

  const mockPost = jest.fn().mockImplementation(() => {
    const outcome = outcomes[callCount];
    callCount++;
    if ('resolve' in outcome) {
      return Promise.resolve({ data: outcome.resolve });
    }
    return Promise.reject(outcome.reject);
  });

  return { post: mockPost } as unknown as AxiosInstance;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('aiClient.queryAiService', () => {
  afterEach(() => {
    resetAxiosInstance();
    jest.restoreAllMocks();
  });

  // -------------------------------------------------------------------------
  // Happy path
  // -------------------------------------------------------------------------

  it('returns AI response data on a successful first attempt', async () => {
    const expected = makeAiResponse({ answer: 'Asteroids score 20 points.' });
    setAxiosInstance(makeMockInstance([{ resolve: expected }]));

    const result = await queryAiService({ question: 'Score in Asteroids?' });

    expect(result).toEqual(expected);
  });

  it('sends the question to POST /query', async () => {
    const mockInstance = makeMockInstance([{ resolve: makeAiResponse() }]);
    setAxiosInstance(mockInstance);

    const question = 'How many shields does Centipede have?';
    await queryAiService({ question });

    expect(mockInstance.post).toHaveBeenCalledWith('/query', expect.objectContaining({ question }));
  });

  it('passes optional game_slug field to the AI service', async () => {
    const mockInstance = makeMockInstance([{ resolve: makeAiResponse() }]);
    setAxiosInstance(mockInstance);

    await queryAiService({ question: 'How to fire?', game_slug: 'asteroids' });

    expect(mockInstance.post).toHaveBeenCalledWith(
      '/query',
      expect.objectContaining({ game_slug: 'asteroids' }),
    );
  });

  // -------------------------------------------------------------------------
  // Request-id propagation (Issue #24)
  // -------------------------------------------------------------------------

  it('forwards the request id as x-request-id when provided', async () => {
    const mockInstance = makeMockInstance([{ resolve: makeAiResponse() }]);
    setAxiosInstance(mockInstance);

    await queryAiService({ question: 'How to fire?' }, 'trace-ai-1');

    expect(mockInstance.post).toHaveBeenCalledWith(
      '/query',
      expect.objectContaining({ question: 'How to fire?' }),
      { headers: { 'x-request-id': 'trace-ai-1' } },
    );
  });

  it('omits the request-id header when no trace id is given', async () => {
    const mockInstance = makeMockInstance([{ resolve: makeAiResponse() }]);
    setAxiosInstance(mockInstance);

    await queryAiService({ question: 'How to fire?' });

    expect(mockInstance.post).toHaveBeenCalledWith(
      '/query',
      expect.objectContaining({ question: 'How to fire?' }),
    );
  });

  it('reuses the same request id across retries', async () => {
    const mockInstance = makeMockInstance([
      { reject: makeHttpError(500) },
      { resolve: makeAiResponse() },
    ]);
    setAxiosInstance(mockInstance);

    jest.spyOn(global, 'setTimeout').mockImplementation((fn) => {
      (fn as () => void)();
      return 0 as unknown as ReturnType<typeof setTimeout>;
    });

    await queryAiService({ question: 'retry me' }, 'trace-retry-1');

    const expectedConfig = { headers: { 'x-request-id': 'trace-retry-1' } };
    expect(mockInstance.post).toHaveBeenNthCalledWith(
      1,
      '/query',
      expect.anything(),
      expectedConfig,
    );
    expect(mockInstance.post).toHaveBeenNthCalledWith(
      2,
      '/query',
      expect.anything(),
      expectedConfig,
    );
  });

  // -------------------------------------------------------------------------
  // Retry logic
  // -------------------------------------------------------------------------

  it('retries once on a 500 error and returns the successful response', async () => {
    const expected = makeAiResponse();
    const mockInstance = makeMockInstance([{ reject: makeHttpError(500) }, { resolve: expected }]);
    setAxiosInstance(mockInstance);

    // Bypass the real sleep to keep tests fast
    jest.spyOn(global, 'setTimeout').mockImplementation((fn) => {
      (fn as () => void)();
      return 0 as unknown as ReturnType<typeof setTimeout>;
    });

    const result = await queryAiService({ question: 'Test retry' });

    expect(result).toEqual(expected);
    expect(mockInstance.post).toHaveBeenCalledTimes(2);
  });

  it('retries up to 3 times and returns after two failures + one success', async () => {
    const expected = makeAiResponse({ answer: 'Third try charm.' });
    const mockInstance = makeMockInstance([
      { reject: makeHttpError(503) },
      { reject: makeHttpError(503) },
      { resolve: expected },
    ]);
    setAxiosInstance(mockInstance);

    jest.spyOn(global, 'setTimeout').mockImplementation((fn) => {
      (fn as () => void)();
      return 0 as unknown as ReturnType<typeof setTimeout>;
    });

    const result = await queryAiService({ question: 'Three retries' });

    expect(result).toEqual(expected);
    expect(mockInstance.post).toHaveBeenCalledTimes(3);
  });

  it('throws after all 3 attempts fail with 5xx errors', async () => {
    const serverError = makeHttpError(502);
    const mockInstance = makeMockInstance([
      { reject: serverError },
      { reject: serverError },
      { reject: serverError },
    ]);
    setAxiosInstance(mockInstance);

    jest.spyOn(global, 'setTimeout').mockImplementation((fn) => {
      (fn as () => void)();
      return 0 as unknown as ReturnType<typeof setTimeout>;
    });

    await expect(queryAiService({ question: 'All fail' })).rejects.toThrow();
    expect(mockInstance.post).toHaveBeenCalledTimes(3);
  });

  it('does not retry on a 400 client error (not retryable)', async () => {
    const clientError = makeHttpError(400, { detail: 'Bad request' });
    const mockInstance = makeMockInstance([{ reject: clientError }]);
    setAxiosInstance(mockInstance);

    await expect(queryAiService({ question: 'Client error test' })).rejects.toThrow();

    expect(mockInstance.post).toHaveBeenCalledTimes(1);
  });

  it('does not retry on a 422 validation error (not retryable)', async () => {
    const validationError = makeHttpError(422, { detail: 'Unprocessable' });
    const mockInstance = makeMockInstance([{ reject: validationError }]);
    setAxiosInstance(mockInstance);

    await expect(queryAiService({ question: 'Validation error test' })).rejects.toThrow();

    expect(mockInstance.post).toHaveBeenCalledTimes(1);
  });

  it('retries on a network error (no response)', async () => {
    const networkErr = makeNetworkError('ECONNRESET');
    const expected = makeAiResponse();
    const mockInstance = makeMockInstance([{ reject: networkErr }, { resolve: expected }]);
    setAxiosInstance(mockInstance);

    jest.spyOn(global, 'setTimeout').mockImplementation((fn) => {
      (fn as () => void)();
      return 0 as unknown as ReturnType<typeof setTimeout>;
    });

    const result = await queryAiService({ question: 'Network retry test' });

    expect(result).toEqual(expected);
    expect(mockInstance.post).toHaveBeenCalledTimes(2);
  });
});
