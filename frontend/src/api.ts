/**
 * Typed API client for the RetroGPT gateway.
 *
 * All requests are made relative to the current origin (`/api/...`) so the
 * same build works behind Nginx in compose and behind the Vite dev proxy.
 *
 * The gateway requires the `x-api-key` header.  In the browser build the key
 * comes from `VITE_API_KEY` (public in the bundle by design — this is a
 * demo-grade auth layer, not a secret storage mechanism).
 */

const API_KEY: string = (import.meta.env.VITE_API_KEY as string | undefined) ?? 'change-me';

// ---------------------------------------------------------------------------
// Types (mirror the gateway response contracts)
// ---------------------------------------------------------------------------

export interface Source {
  id: string;
  title: string;
  score: number;
  snippet: string;
  page?: number;
}

export interface ChatResponse {
  sessionId: string;
  answer: string;
  sources: Source[];
  latency_ms: number;
  cached: boolean;
}

export interface SessionMessage {
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  createdAt: string;
}

export interface SessionHistory {
  sessionId: string;
  messages: SessionMessage[];
}

/** Standard gateway error envelope. */
export class ApiError extends Error {
  public readonly code: string;
  public readonly status: number;

  constructor(message: string, code = 'UNKNOWN_ERROR', status = 0) {
    super(message);
    this.name = 'ApiError';
    this.code = code;
    this.status = status;
  }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

async function parseError(response: Response): Promise<ApiError> {
  try {
    const body = (await response.json()) as {
      error?: { code?: string; message?: string };
    };
    return new ApiError(
      body.error?.message ?? `Request failed with status ${response.status}`,
      body.error?.code ?? 'HTTP_ERROR',
      response.status,
    );
  } catch {
    return new ApiError(
      `Request failed with status ${response.status}`,
      'HTTP_ERROR',
      response.status,
    );
  }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Send a question to `POST /api/chat`.
 *
 * @param question  The user's question (3–500 characters).
 * @param sessionId Optional existing session to append to.
 * @throws ApiError on non-2xx responses or network failures.
 */
export async function sendQuestion(question: string, sessionId?: string): Promise<ChatResponse> {
  let response: Response;
  try {
    response = await fetch('/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': API_KEY,
      },
      body: JSON.stringify(sessionId ? { question, sessionId } : { question }),
    });
  } catch {
    throw new ApiError('Unable to reach the server. Is the API gateway running?', 'NETWORK_ERROR');
  }

  if (!response.ok) {
    throw await parseError(response);
  }

  return (await response.json()) as ChatResponse;
}

/**
 * Fetch the full message history for a session.
 *
 * @param sessionId Session UUID.
 * @throws ApiError on non-2xx responses or network failures.
 */
export async function getSessionHistory(sessionId: string): Promise<SessionHistory> {
  let response: Response;
  try {
    response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`, {
      headers: { 'x-api-key': API_KEY },
    });
  } catch {
    throw new ApiError('Unable to reach the server. Is the API gateway running?', 'NETWORK_ERROR');
  }

  if (!response.ok) {
    throw await parseError(response);
  }

  return (await response.json()) as SessionHistory;
}

// ---------------------------------------------------------------------------
// Streaming API (Issue #25)
// ---------------------------------------------------------------------------

/** Payload of the terminal `done` SSE event. */
export interface StreamDonePayload {
  sessionId: string;
  latency_ms: number;
  cached: boolean;
}

/** Callbacks invoked while an answer streams in. All are optional. */
export interface StreamCallbacks {
  /** Retrieval finished — the sources the answer will be grounded in. */
  onMeta?: (sources: Source[]) => void;
  /** A generated text chunk arrived. */
  onToken?: (text: string) => void;
  /** The stream finished successfully. */
  onDone?: (done: StreamDonePayload) => void;
}

/** Options for {@link streamQuestion}. */
export interface StreamQuestionOptions extends StreamCallbacks {
  /** Optional existing session to append to. */
  sessionId?: string;
}

interface SseFrame {
  event: string;
  data: string;
}

/** Normalise a chunk and split out complete `event:`/`data:` frames. */
function createFrameParser(): (chunk: string) => SseFrame[] {
  let buffer = '';

  return (chunk: string): SseFrame[] => {
    // Normalise CRLF (even when split across chunks) so frame boundaries
    // always match the "\n\n" search below.
    buffer = (buffer + chunk).replace(/\r\n/g, '\n');
    const frames: SseFrame[] = [];

    let boundary = buffer.indexOf('\n\n');
    while (boundary !== -1) {
      const rawFrame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);

      let event = 'message';
      const dataLines: string[] = [];
      for (const line of rawFrame.split('\n')) {
        if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
        else if (line.startsWith('data:'))
          dataLines.push(line.slice('data:'.length).replace(/^ /, ''));
      }
      // Frames without data are keep-alives / comments — skip them.
      if (dataLines.length > 0) frames.push({ event, data: dataLines.join('\n') });

      boundary = buffer.indexOf('\n\n');
    }

    return frames;
  };
}

/** Route one parsed SSE frame to the matching callback. */
function dispatchFrame(frame: SseFrame, callbacks: StreamCallbacks): void {
  let payload: Record<string, unknown>;
  try {
    payload = JSON.parse(frame.data) as Record<string, unknown>;
  } catch {
    return; // Ignore malformed frames — the stream stays usable.
  }

  switch (frame.event) {
    case 'meta':
      callbacks.onMeta?.(Array.isArray(payload.sources) ? (payload.sources as Source[]) : []);
      break;
    case 'token':
      if (typeof payload.text === 'string' && payload.text.length > 0) {
        callbacks.onToken?.(payload.text);
      }
      break;
    case 'done':
      callbacks.onDone?.({
        sessionId: typeof payload.sessionId === 'string' ? payload.sessionId : '',
        latency_ms: typeof payload.latency_ms === 'number' ? payload.latency_ms : 0,
        cached: payload.cached === true,
      });
      break;
    case 'error': {
      const code = typeof payload.code === 'string' ? payload.code : 'STREAM_ERROR';
      const message = typeof payload.message === 'string' ? payload.message : 'The stream failed.';
      throw new ApiError(message, code);
    }
    default:
      break; // Unknown event types are ignored for forward compatibility.
  }
}

/**
 * Stream an answer from `POST /api/chat/stream` (Server-Sent Events).
 *
 * Failures before the first event (validation, AI service down, timeout)
 * reject with the same `ApiError` codes as {@link sendQuestion}. A failure
 * *mid-stream* arrives as an SSE `error` event and also rejects with an
 * `ApiError` — any tokens already delivered through `onToken` remain valid.
 *
 * @throws ApiError on JSON error responses, network failures, or a
 *   mid-stream `error` event.
 */
export async function streamQuestion(
  question: string,
  options: StreamQuestionOptions = {},
): Promise<void> {
  let response: Response;
  try {
    response = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
        'x-api-key': API_KEY,
      },
      body: JSON.stringify(
        options.sessionId ? { question, sessionId: options.sessionId } : { question },
      ),
    });
  } catch {
    throw new ApiError('Unable to reach the server. Is the API gateway running?', 'NETWORK_ERROR');
  }

  if (!response.ok) {
    throw await parseError(response);
  }

  if (!response.body) {
    throw new ApiError('Streaming is not supported in this browser.', 'STREAM_UNSUPPORTED');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const parseFrames = createFrameParser();

  try {
    for (;;) {
      let result: ReadableStreamReadResult<Uint8Array>;
      try {
        result = await reader.read();
      } catch {
        throw new ApiError(
          'The connection was interrupted while streaming the answer.',
          'STREAM_INTERRUPTED',
        );
      }
      if (result.done) break;

      for (const frame of parseFrames(decoder.decode(result.value, { stream: true }))) {
        dispatchFrame(frame, options);
      }
    }
  } finally {
    // No-op once the stream has ended; aborts the body when an error was
    // thrown from `dispatchFrame` mid-read.
    reader.cancel().catch(() => undefined);
  }
}
