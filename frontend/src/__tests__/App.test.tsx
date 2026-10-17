import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from '../App';
import { ApiError } from '../api';
import * as api from '../api';

vi.mock('../api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api')>();
  return {
    ...actual,
    sendQuestion: vi.fn(),
    streamQuestion: vi.fn(),
    getSessionHistory: vi.fn(),
  };
});

const mockStreamQuestion = vi.mocked(api.streamQuestion);
const mockGetSessionHistory = vi.mocked(api.getSessionHistory);

const SESSION_KEY = 'retrogpt.sessionId';

function chatResponse(overrides: Partial<api.ChatResponse> = {}): api.ChatResponse {
  return {
    sessionId: '550e8400-e29b-41d4-a716-446655440000',
    answer: 'Large asteroids score 20 points [1].',
    sources: [
      {
        id: 'chunk-1',
        title: 'Asteroids Manual',
        score: 0.91,
        snippet: 'Large asteroids: 20 points...',
      },
    ],
    latency_ms: 120,
    cached: false,
    ...overrides,
  };
}

/** Make the next streamQuestion call deliver a full answer immediately. */
function deliverAnswer(overrides: Partial<api.ChatResponse> = {}): void {
  const response = chatResponse(overrides);
  mockStreamQuestion.mockImplementation(async (_question, options) => {
    options?.onMeta?.(response.sources);
    options?.onToken?.(response.answer);
    options?.onDone?.({
      sessionId: response.sessionId,
      latency_ms: response.latency_ms,
      cached: response.cached,
    });
  });
}

beforeEach(() => {
  localStorage.clear();
  mockStreamQuestion.mockReset();
  mockGetSessionHistory.mockReset();
  mockGetSessionHistory.mockResolvedValue({ sessionId: 'no-session', messages: [] });
});

// ---------------------------------------------------------------------------
// Composer
// ---------------------------------------------------------------------------

describe('App — composer', () => {
  it('renders a labelled question input and submit button', () => {
    render(<App />);

    expect(screen.getByLabelText('Your question')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Ask' })).toBeInTheDocument();
  });

  it('disables the submit button until the question is at least 3 characters', async () => {
    const user = userEvent.setup();
    render(<App />);

    const button = screen.getByRole('button', { name: 'Ask' });
    expect(button).toBeDisabled();

    await user.type(screen.getByLabelText('Your question'), 'Hi');
    expect(button).toBeDisabled();

    await user.type(screen.getByLabelText('Your question'), '!');
    expect(button).toBeEnabled();
  });
});

// ---------------------------------------------------------------------------
// Conversation flow
// ---------------------------------------------------------------------------

describe('App — conversation flow', () => {
  it('streams an answer and renders it with sources', async () => {
    const user = userEvent.setup();
    deliverAnswer();
    render(<App />);

    await user.type(
      screen.getByLabelText('Your question'),
      'How do you score points in Asteroids?',
    );
    await user.click(screen.getByRole('button', { name: 'Ask' }));

    expect(mockStreamQuestion).toHaveBeenCalledWith(
      'How do you score points in Asteroids?',
      expect.objectContaining({ sessionId: undefined }),
    );

    // User message and assistant answer both appear.
    expect(await screen.findByText('How do you score points in Asteroids?')).toBeInTheDocument();
    expect(screen.getByText(/Large asteroids score 20 points/)).toBeInTheDocument();

    // Source list with title and snippet.
    expect(screen.getByRole('heading', { name: 'Sources' })).toBeInTheDocument();
    expect(screen.getByText('Asteroids Manual')).toBeInTheDocument();
    expect(screen.getByText('Large asteroids: 20 points...')).toBeInTheDocument();
  });

  it('renders tokens incrementally and hides the typing indicator after the first token', async () => {
    const user = userEvent.setup();
    let pushToken!: (text: string) => void;
    let finish!: () => void;

    mockStreamQuestion.mockImplementation(
      (_question, options) =>
        new Promise<void>((resolve) => {
          pushToken = (text) => options?.onToken?.(text);
          finish = () => {
            options?.onDone?.({
              sessionId: 'session-stream',
              latency_ms: 42,
              cached: false,
            });
            resolve();
          };
        }),
    );

    render(<App />);

    await user.type(
      screen.getByLabelText('Your question'),
      'How do you score points in Asteroids?',
    );
    await user.click(screen.getByRole('button', { name: 'Ask' }));

    // The stream is open but no token arrived yet.
    expect(screen.getByText('Thinking…')).toBeInTheDocument();
    expect(screen.getByLabelText('Your question')).toBeDisabled();

    // First token: indicator disappears, the partial answer shows.
    act(() => pushToken('Large asteroids '));
    expect(screen.queryByText('Thinking…')).not.toBeInTheDocument();
    expect(screen.getByText(/Large asteroids/)).toBeInTheDocument();

    // Remaining token plus done finalise the answer.
    act(() => {
      pushToken('score 20 points [1].');
      finish();
    });

    await waitFor(() => expect(screen.getByLabelText('Your question')).toBeEnabled());
    expect(screen.getByText(/score 20 points/)).toBeInTheDocument();
    expect(localStorage.getItem(SESSION_KEY)).toBe('session-stream');
  });

  it('submits on Enter and clears the input', async () => {
    const user = userEvent.setup();
    deliverAnswer();
    render(<App />);

    const input = screen.getByLabelText('Your question');
    await user.type(input, 'How do you score points in Asteroids?{Enter}');

    await waitFor(() => expect(mockStreamQuestion).toHaveBeenCalledTimes(1));
    expect(input).toHaveValue('');
  });

  it('persists the returned sessionId in localStorage', async () => {
    const user = userEvent.setup();
    deliverAnswer({ sessionId: 'session-abc' });
    render(<App />);

    await user.type(screen.getByLabelText('Your question'), 'What is Pong?');
    await user.click(screen.getByRole('button', { name: 'Ask' }));

    await waitFor(() => expect(localStorage.getItem(SESSION_KEY)).toBe('session-abc'));
  });

  it('sends the stored sessionId on follow-up questions', async () => {
    const user = userEvent.setup();
    localStorage.setItem(SESSION_KEY, 'session-abc');
    mockGetSessionHistory.mockResolvedValue({ sessionId: 'session-abc', messages: [] });
    deliverAnswer({ sessionId: 'session-abc' });
    render(<App />);

    await user.type(screen.getByLabelText('Your question'), 'And how do saucers score?');
    await user.click(screen.getByRole('button', { name: 'Ask' }));

    expect(mockStreamQuestion).toHaveBeenCalledWith(
      'And how do saucers score?',
      expect.objectContaining({ sessionId: 'session-abc' }),
    );
  });

  it('restores conversation history from the API on load', async () => {
    localStorage.setItem(SESSION_KEY, 'session-abc');
    mockGetSessionHistory.mockResolvedValue({
      sessionId: 'session-abc',
      messages: [
        {
          role: 'user',
          content: 'How do you score in Asteroids?',
          createdAt: '2026-10-17T10:00:00Z',
        },
        {
          role: 'assistant',
          content: 'Destroy asteroids [1].',
          sources: [{ id: 'c1', title: 'Asteroids Manual', score: 0.9, snippet: 's' }],
          createdAt: '2026-10-17T10:00:01Z',
        },
      ],
    });

    render(<App />);

    expect(await screen.findByText('How do you score in Asteroids?')).toBeInTheDocument();
    expect(screen.getByText('Destroy asteroids')).toBeInTheDocument();
    expect(screen.getByText('Asteroids Manual')).toBeInTheDocument();
  });

  it('drops a stale session id when history returns 404', async () => {
    localStorage.setItem(SESSION_KEY, 'stale-session');
    mockGetSessionHistory.mockRejectedValue(new ApiError('Session not found', 'NOT_FOUND', 404));

    render(<App />);

    await waitFor(() => expect(localStorage.getItem(SESSION_KEY)).toBeNull());
  });
});

// ---------------------------------------------------------------------------
// Citations
// ---------------------------------------------------------------------------

describe('App — source citations', () => {
  it('scrolls to and highlights the source when a citation is clicked', async () => {
    const user = userEvent.setup();
    deliverAnswer();
    const scrollSpy = vi.spyOn(Element.prototype, 'scrollIntoView').mockImplementation(() => {});
    render(<App />);

    await user.type(
      screen.getByLabelText('Your question'),
      'How do you score points in Asteroids?',
    );
    await user.click(screen.getByRole('button', { name: 'Ask' }));

    const citation = await screen.findByRole('button', { name: 'Show source 1' });
    await user.click(citation);

    expect(scrollSpy).toHaveBeenCalled();
    expect(screen.getByText('Asteroids Manual').closest('li')).toHaveClass('source-highlighted');
  });

  it('renders citation markers inside the answer as buttons', async () => {
    const user = userEvent.setup();
    deliverAnswer();
    render(<App />);

    await user.type(
      screen.getByLabelText('Your question'),
      'How do you score points in Asteroids?',
    );
    await user.click(screen.getByRole('button', { name: 'Ask' }));

    expect(await screen.findByRole('button', { name: 'Show source 1' })).toHaveTextContent('[1]');
  });
});

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

describe('App — errors', () => {
  it('shows an alert banner and removes the empty placeholder when the stream fails to open', async () => {
    const user = userEvent.setup();
    mockStreamQuestion.mockRejectedValue(
      new ApiError('AI service is unavailable.', 'AI_SERVICE_ERROR', 502),
    );
    render(<App />);

    await user.type(
      screen.getByLabelText('Your question'),
      'How do you score points in Asteroids?',
    );
    await user.click(screen.getByRole('button', { name: 'Ask' }));

    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent('AI service is unavailable.');

    // Only the user turn remains — the empty assistant bubble was dropped.
    expect(screen.getAllByRole('article')).toHaveLength(1);
  });

  it('keeps partially streamed text when the stream breaks mid-answer', async () => {
    const user = userEvent.setup();
    mockStreamQuestion.mockImplementation(async (_question, options) => {
      options?.onToken?.('Large asteroids ');
      throw new ApiError('The connection was interrupted.', 'STREAM_INTERRUPTED');
    });
    render(<App />);

    await user.type(
      screen.getByLabelText('Your question'),
      'How do you score points in Asteroids?',
    );
    await user.click(screen.getByRole('button', { name: 'Ask' }));

    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent('The connection was interrupted.');
    expect(screen.getByText(/Large asteroids/)).toBeInTheDocument();
  });

  it('re-enables the composer after an error', async () => {
    const user = userEvent.setup();
    mockStreamQuestion.mockRejectedValue(new ApiError('Boom', 'AI_SERVICE_ERROR', 502));
    render(<App />);

    await user.type(
      screen.getByLabelText('Your question'),
      'How do you score points in Asteroids?',
    );
    await user.click(screen.getByRole('button', { name: 'Ask' }));

    await screen.findByRole('alert');
    expect(screen.getByLabelText('Your question')).toBeEnabled();
  });
});

// ---------------------------------------------------------------------------
// New chat
// ---------------------------------------------------------------------------

describe('App — new chat', () => {
  it('clears messages and the stored session', async () => {
    const user = userEvent.setup();
    localStorage.setItem(SESSION_KEY, 'session-abc');
    mockGetSessionHistory.mockResolvedValue({
      sessionId: 'session-abc',
      messages: [{ role: 'user', content: 'Old question', createdAt: '2026-10-17T10:00:00Z' }],
    });
    render(<App />);

    expect(await screen.findByText('Old question')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'New chat' }));

    expect(screen.queryByText('Old question')).not.toBeInTheDocument();
    expect(localStorage.getItem(SESSION_KEY)).toBeNull();
  });
});
