import { useEffect, useRef, useState, type FormEvent } from 'react';
import type { ChatMessage } from '../App';
import { Message } from './Message';

export interface ChatWindowProps {
  messages: ChatMessage[];
  loading: boolean;
  error: string | null;
  onSend: (question: string) => void;
  onCitationClick: (messageId: string, sourceIndex: number) => void;
  highlightedSource: string | null;
}

/**
 * Main chat surface: scrolling transcript, composer form, and status states.
 *
 * Accessibility:
 * - Transcript is `aria-live="polite"` and `aria-busy` while loading.
 * - The composer input has a visible label and submits on Enter (native form).
 * - Errors surface in a `role="alert"` banner.
 */
export function ChatWindow({
  messages,
  loading,
  error,
  onSend,
  onCitationClick,
  highlightedSource,
}: ChatWindowProps) {
  const [question, setQuestion] = useState('');
  const transcriptEndRef = useRef<HTMLDivElement>(null);

  // Keep the newest content in view as the conversation grows.
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages, loading, error]);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = question.trim();
    if (trimmed.length < 3 || loading) return;
    onSend(trimmed);
    setQuestion('');
  }

  const showEmptyState = messages.length === 0 && !loading && !error;

  return (
    <main className="chat-window">
      <div
        className="transcript"
        role="log"
        aria-live="polite"
        aria-busy={loading}
        aria-label="Conversation"
      >
        {showEmptyState && (
          <div className="empty-state">
            <h2>Welcome!</h2>
            <p>
              Try asking something like <em>“How do you score points in Asteroids?”</em> or{' '}
              <em>“How does the Centipede move?”</em>
            </p>
          </div>
        )}

        {messages.map((message) => (
          <Message
            key={message.id}
            message={message}
            onCitationClick={onCitationClick}
            highlightedSource={highlightedSource}
          />
        ))}

        {loading && (
          <div className="typing-indicator" role="status">
            <span className="typing-dots" aria-hidden="true">
              ● ● ●
            </span>
            <span>Thinking…</span>
          </div>
        )}

        {error && (
          <div className="error-banner" role="alert">
            <strong>Something went wrong:</strong> {error}
          </div>
        )}

        <div ref={transcriptEndRef} />
      </div>

      <form className="composer" onSubmit={handleSubmit} aria-label="Ask a question">
        <label htmlFor="question-input" className="composer-label">
          Your question
        </label>
        <div className="composer-row">
          <input
            id="question-input"
            name="question"
            type="text"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="e.g. How do you score points in Asteroids?"
            minLength={3}
            maxLength={500}
            autoComplete="off"
            disabled={loading}
          />
          <button type="submit" disabled={loading || question.trim().length < 3}>
            {loading ? 'Sending…' : 'Ask'}
          </button>
        </div>
      </form>
    </main>
  );
}
