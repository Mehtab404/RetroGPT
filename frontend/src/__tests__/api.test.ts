import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiError, getSessionHistory, sendQuestion, streamQuestion } from '../api';

// ---------------------------------------------------------------------------
// fetch mock helpers
// ---------------------------------------------------------------------------

const fetchMock = vi.fn();

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as Response;
}

/** Encode an ASCII string as UTF-8 bytes (no TextEncoder dependency). */
function bytes(text: string): Uint8Array {
  return Uint8Array.from([...text].map((char) => char.charCodeAt(0)));
}

/** Response whose body streams the given raw chunks via a fake reader. */
function sseResponse(chunks: string[], status = 200): Response {
  const parts = chunks.map(bytes);
  let index = 0;
  return {
    ok: status >= 200 && status < 300,
    status,
    body: {
      getReader: () => ({
        read: async () =>
          index < parts.length
            ? { done: false, value: parts[index++] }
            : { done: true, value: undefined },
        cancel: async () => undefined,
      }),
    },
  } as unknown as Response;
}

beforeEach(() => {
  vi.stubGlobal('fetch', fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
  fetchMock.mockReset();
});

// ---------------------------------------------------------------------------
// sendQuestion
// ---------------------------------------------------------------------------

describe('sendQuestion()', () => {
  it('POSTs the question to /api/chat with the api key header', async () => {
    fetchMock.mockResolvedValue(
      jsonResponse({
        sessionId: 's1',
        answer: 'Answer',
        sources: [],
        latency_ms: 5,
        cached: false,
      }),
    );

    await sendQuestion('How do you score in Asteroids?');

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('/api/chat');
    expect(init.method).toBe('POST');
    expect((init.headers as Record<string, string>)['x-api-key']).toBeTruthy();
    expect(JSON.parse(init.body as string)).toEqual({
      question: 'How do you score in Asteroids?',
    });
  });

  it('includes the sessionId in the body when provided', async () => {
    fetchMock.mockResolvedValue(
      jsonResponse({ sessionId: 's1', answer: '', sources: [], latency_ms: 0, cached: false }),
    );

    await sendQuestion('Follow-up question', 's1');

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(JSON.parse(init.body as string)).toEqual({
      question: 'Follow-up question',
      sessionId: 's1',
    });
  });

  it('returns the parsed chat response', async () => {
    const payload = {
      sessionId: 's1',
      answer: 'Large asteroids score 20 points [1].',
      sources: [{ id: 'c1', title: 'Asteroids Manual', score: 0.91, snippet: '...' }],
      latency_ms: 42,
      cached: true,
    };
    fetchMock.mockResolvedValue(jsonResponse(payload));

    await expect(sendQuestion('question')).resolves.toEqual(payload);
  });

  it('maps the gateway error envelope to an ApiError', async () => {
    fetchMock.mockResolvedValue(
      jsonResponse(
        { error: { code: 'AI_SERVICE_ERROR', message: 'AI service is unavailable.' } },
        502,
      ),
    );

    const error = await sendQuestion('question').catch((e: unknown) => e);

    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).code).toBe('AI_SERVICE_ERROR');
    expect((error as ApiError).status).toBe(502);
    expect((error as ApiError).message).toBe('AI service is unavailable.');
  });

  it('maps network failures to a NETWORK_ERROR ApiError', async () => {
    fetchMock.mockRejectedValue(new TypeError('Failed to fetch'));

    const error = await sendQuestion('question').catch((e: unknown) => e);

    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).code).toBe('NETWORK_ERROR');
  });
});

// ---------------------------------------------------------------------------
// getSessionHistory
// ---------------------------------------------------------------------------

describe('getSessionHistory()', () => {
  const sessionId = '550e8400-e29b-41d4-a716-446655440000';

  it('GETs /api/sessions/:id with the api key header', async () => {
    fetchMock.mockResolvedValue(jsonResponse({ sessionId, messages: [] }));

    await getSessionHistory(sessionId);

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe(`/api/sessions/${sessionId}`);
    expect((init.headers as Record<string, string>)['x-api-key']).toBeTruthy();
  });

  it('returns the parsed history', async () => {
    const payload = {
      sessionId,
      messages: [{ role: 'user', content: 'Hi', createdAt: '2026-10-17T10:00:00.000Z' }],
    };
    fetchMock.mockResolvedValue(jsonResponse(payload));

    await expect(getSessionHistory(sessionId)).resolves.toEqual(payload);
  });

  it('maps 404 responses to an ApiError with status 404', async () => {
    fetchMock.mockResolvedValue(
      jsonResponse({ error: { code: 'NOT_FOUND', message: 'Session not found' } }, 404),
    );

    const error = await getSessionHistory(sessionId).catch((e: unknown) => e);

    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).status).toBe(404);
  });
});

// ---------------------------------------------------------------------------
// streamQuestion
// ---------------------------------------------------------------------------

describe('streamQuestion()', () => {
  it('POSTs to /api/chat/stream with SSE headers and the sessionId', async () => {
    fetchMock.mockResolvedValue(sseResponse([]));

    await streamQuestion('How do you score in Asteroids?', { sessionId: 's1' });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('/api/chat/stream');
    expect(init.method).toBe('POST');
    const headers = init.headers as Record<string, string>;
    expect(headers.Accept).toBe('text/event-stream');
    expect(headers['x-api-key']).toBeTruthy();
    expect(JSON.parse(init.body as string)).toEqual({
      question: 'How do you score in Asteroids?',
      sessionId: 's1',
    });
  });

  it('dispatches meta/token/done callbacks in order across chunk boundaries', async () => {
    fetchMock.mockResolvedValue(
      sseResponse([
        'event: meta\ndata: {"sources":[{"id":"c1","title":"Asteroids Manual","score":0.9,"snippet":"s"}]}\n\nev',
        'ent: token\ndata: {"text":"Large "}\n\nevent: token\ndata: {"text":"asteroids [1]."}\n\nevent: done\ndata: {"sessionId":"s1","latency_ms":42,"cached":false}\n\n',
      ]),
    );

    const events: string[] = [];
    const sources: unknown[] = [];
    let done: unknown = null;

    await streamQuestion('How do you score in Asteroids?', {
      onMeta: (s) => {
        events.push('meta');
        sources.push(...s);
      },
      onToken: (text) => events.push(`token:${text}`),
      onDone: (d) => {
        events.push('done');
        done = d;
      },
    });

    expect(events).toEqual(['meta', 'token:Large ', 'token:asteroids [1].', 'done']);
    expect(sources).toHaveLength(1);
    expect(done).toEqual({ sessionId: 's1', latency_ms: 42, cached: false });
  });

  it('parses CRLF frames', async () => {
    fetchMock.mockResolvedValue(
      sseResponse([
        'event: token\r\ndata: {"text":"Hi"}\r\n\r\nevent: done\r\ndata: {"sessionId":"s1","latency_ms":1,"cached":true}\r\n\r\n',
      ]),
    );

    const tokens: string[] = [];
    let done: unknown = null;
    await streamQuestion('question', {
      onToken: (text) => tokens.push(text),
      onDone: (d) => {
        done = d;
      },
    });

    expect(tokens).toEqual(['Hi']);
    expect(done).toEqual({ sessionId: 's1', latency_ms: 1, cached: true });
  });

  it('throws an ApiError when a mid-stream error event arrives', async () => {
    fetchMock.mockResolvedValue(
      sseResponse([
        'event: token\ndata: {"text":"Partial "}\n\nevent: error\ndata: {"code":"LLM_ERROR","message":"provider exploded"}\n\n',
      ]),
    );

    const tokens: string[] = [];
    const error = await streamQuestion('question', {
      onToken: (text) => tokens.push(text),
    }).catch((e: unknown) => e);

    expect(tokens).toEqual(['Partial ']);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).code).toBe('LLM_ERROR');
    expect((error as ApiError).message).toBe('provider exploded');
  });

  it('maps pre-stream JSON errors to an ApiError', async () => {
    fetchMock.mockResolvedValue(
      jsonResponse(
        { error: { code: 'AI_SERVICE_ERROR', message: 'AI service is unavailable.' } },
        502,
      ),
    );

    const error = await streamQuestion('question').catch((e: unknown) => e);

    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).code).toBe('AI_SERVICE_ERROR');
    expect((error as ApiError).status).toBe(502);
  });

  it('maps network failures to a NETWORK_ERROR ApiError', async () => {
    fetchMock.mockRejectedValue(new TypeError('Failed to fetch'));

    const error = await streamQuestion('question').catch((e: unknown) => e);

    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).code).toBe('NETWORK_ERROR');
  });
});
