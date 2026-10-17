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
