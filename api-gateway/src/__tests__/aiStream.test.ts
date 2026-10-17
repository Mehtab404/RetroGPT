/**
 * Tests for the streaming AI service client (`openAiStream`, Issue #25).
 *
 * The global `fetch` is mocked so no real network calls are made.
 */

import { AiStreamNetworkError, openAiStream, type AiStreamEvent } from '../services/aiStream';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const fetchMock = jest.fn();

function sseResponse(chunks: string[], status = 200): Response {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
  return new Response(body, { status });
}

function errorResponse(status: number, payload: unknown = {}): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

async function collect(stream: AsyncGenerator<AiStreamEvent>): Promise<AiStreamEvent[]> {
  const events: AiStreamEvent[] = [];
  for await (const event of stream) events.push(event);
  return events;
}

beforeEach(() => {
  fetchMock.mockReset();
  global.fetch = fetchMock as unknown as typeof fetch;
});

afterEach(() => {
  jest.restoreAllMocks();
});

// ---------------------------------------------------------------------------
// Happy path
// ---------------------------------------------------------------------------

describe('openAiStream', () => {
  it('POSTs to /query/stream with SSE headers and the request id', async () => {
    fetchMock.mockResolvedValue(sseResponse(['event: done\ndata: {"latency_ms":1}\n\n']));

    const stream = await openAiStream({ question: 'How do you score?' }, 'trace-1');
    await collect(stream.events);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toContain('/query/stream');
    expect(init.method).toBe('POST');
    const headers = init.headers as Record<string, string>;
    expect(headers['Accept']).toBe('text/event-stream');
    expect(headers['x-request-id']).toBe('trace-1');
    expect(JSON.parse(init.body as string)).toEqual({ question: 'How do you score?' });
  });

  it('parses events across chunk boundaries', async () => {
    fetchMock.mockResolvedValue(
      sseResponse([
        'event: meta\ndata: {"sources":[]}\n\nevent: tok',
        'en\ndata: {"text":"Hi"}\n\nevent: done\ndata: {"latency_ms":2}\n\n',
      ]),
    );

    const stream = await openAiStream({ question: 'q?' });
    const events = await collect(stream.events);

    expect(events).toEqual([
      { event: 'meta', data: { sources: [] } },
      { event: 'token', data: { text: 'Hi' } },
      { event: 'done', data: { latency_ms: 2 } },
    ]);
  });

  it('skips frames whose data is not a JSON object', async () => {
    fetchMock.mockResolvedValue(
      sseResponse(['event: token\ndata: not-json\n\nevent: done\ndata: {"latency_ms":3}\n\n']),
    );

    const stream = await openAiStream({ question: 'q?' });
    const events = await collect(stream.events);

    expect(events).toEqual([{ event: 'done', data: { latency_ms: 3 } }]);
  });

  // -------------------------------------------------------------------------
  // Opening failures
  // -------------------------------------------------------------------------

  it('retries 5xx responses before giving up', async () => {
    fetchMock
      .mockResolvedValueOnce(errorResponse(503, { detail: 'busy' }))
      .mockResolvedValueOnce(sseResponse(['event: done\ndata: {"latency_ms":1}\n\n']));

    const stream = await openAiStream({ question: 'q?' });
    const events = await collect(stream.events);

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(events[0].event).toBe('done');
  });

  it('throws AiStreamHttpError immediately for non-retryable statuses', async () => {
    fetchMock.mockResolvedValueOnce(errorResponse(422, { detail: 'bad request' }));

    await expect(openAiStream({ question: 'q?' })).rejects.toMatchObject({
      name: 'AiStreamHttpError',
      status: 422,
      data: { detail: 'bad request' },
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('throws AiStreamNetworkError when fetch fails', async () => {
    fetchMock.mockRejectedValue(new TypeError('fetch failed'));

    const error = await openAiStream({ question: 'q?' }).catch((err: unknown) => err);

    expect(error).toBeInstanceOf(AiStreamNetworkError);
    expect((error as AiStreamNetworkError).timedOut).toBe(false);
  });

  it('marks the error as timedOut when the header wait aborts', async () => {
    jest.useFakeTimers();
    try {
      fetchMock.mockImplementation(
        (_url: string, init: RequestInit) =>
          new Promise((_resolve, reject) => {
            init.signal?.addEventListener('abort', () =>
              reject(new DOMException('Aborted', 'AbortError')),
            );
          }),
      );

      const promise = openAiStream({ question: 'q?' });
      const assertion = expect(promise).rejects.toMatchObject({
        name: 'AiStreamNetworkError',
        timedOut: true,
      });
      await jest.advanceTimersByTimeAsync(120_000);
      await assertion;
    } finally {
      jest.useRealTimers();
    }
  });
});
