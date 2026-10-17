import { useCallback, useEffect, useState } from 'react';
import { ChatWindow } from './components/ChatWindow';
import { ApiError, getSessionHistory, streamQuestion, type Source } from './api';
import { sourceAnchorId } from './citations';

const SESSION_STORAGE_KEY = 'retrogpt.sessionId';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources: Source[];
}

let messageCounter = 0;
function makeMessageId(): string {
  messageCounter += 1;
  return `msg-${Date.now()}-${messageCounter}`;
}

function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(() =>
    localStorage.getItem(SESSION_STORAGE_KEY),
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [highlightedSource, setHighlightedSource] = useState<string | null>(null);

  // Restore the previous conversation on first render.
  useEffect(() => {
    const stored = localStorage.getItem(SESSION_STORAGE_KEY);
    if (!stored) return;

    let cancelled = false;

    getSessionHistory(stored)
      .then((history) => {
        if (cancelled) return;
        setMessages(
          history.messages.map((msg) => ({
            id: makeMessageId(),
            role: msg.role,
            content: msg.content,
            sources: msg.sources ?? [],
          })),
        );
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        // A stale session id (e.g. after a database reset) must not block a
        // fresh conversation — drop it and start clean.
        if (err instanceof ApiError && err.status === 404) {
          localStorage.removeItem(SESSION_STORAGE_KEY);
          setSessionId(null);
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const handleSend = useCallback(
    async (question: string) => {
      const trimmed = question.trim();
      if (!trimmed || loading) return;

      // Add the user turn and an empty assistant placeholder that streamed
      // tokens fill in. The placeholder is removed again if the request
      // fails before any content arrived.
      const assistantId = makeMessageId();
      setMessages((prev) => [
        ...prev,
        { id: makeMessageId(), role: 'user', content: trimmed, sources: [] },
        { id: assistantId, role: 'assistant', content: '', sources: [] },
      ]);
      setLoading(true);
      setError(null);

      let receivedContent = false;
      const updateAssistant = (update: (message: ChatMessage) => ChatMessage) => {
        setMessages((prev) =>
          prev.map((message) => (message.id === assistantId ? update(message) : message)),
        );
      };

      try {
        await streamQuestion(trimmed, {
          sessionId: sessionId ?? undefined,
          onMeta: (sources) => {
            receivedContent = true;
            updateAssistant((message) => ({ ...message, sources }));
          },
          onToken: (text) => {
            receivedContent = true;
            updateAssistant((message) => ({ ...message, content: message.content + text }));
          },
          onDone: (done) => {
            if (done.sessionId) {
              setSessionId(done.sessionId);
              localStorage.setItem(SESSION_STORAGE_KEY, done.sessionId);
            }
          },
        });
      } catch (err) {
        const message =
          err instanceof ApiError
            ? err.message
            : 'Something went wrong while contacting the server. Please try again.';
        setError(message);
        if (!receivedContent) {
          // Nothing streamed — drop the empty assistant placeholder.
          setMessages((prev) => prev.filter((m) => m.id !== assistantId));
        }
      } finally {
        setLoading(false);
      }
    },
    [loading, sessionId],
  );

  const handleNewChat = useCallback(() => {
    localStorage.removeItem(SESSION_STORAGE_KEY);
    setSessionId(null);
    setMessages([]);
    setError(null);
  }, []);

  const handleCitationClick = useCallback((messageId: string, sourceIndex: number) => {
    const anchor = sourceAnchorId(messageId, sourceIndex);
    const element = document.getElementById(anchor);
    element?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    setHighlightedSource(anchor);
    window.setTimeout(() => {
      setHighlightedSource((prev) => (prev === anchor ? null : prev));
    }, 2000);
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-title">
          <h1>
            <span aria-hidden="true">🕹️</span> RetroGPT
          </h1>
          <p className="app-subtitle">
            Ask about classic arcade games — answers cited from game manuals.
          </p>
        </div>
        <button type="button" className="new-chat-button" onClick={handleNewChat}>
          New chat
        </button>
      </header>

      <ChatWindow
        messages={messages}
        loading={loading}
        error={error}
        onSend={handleSend}
        onCitationClick={handleCitationClick}
        highlightedSource={highlightedSource}
      />

      <footer className="app-footer">
        <p>
          Answers are grounded in retrieved game manuals. Verify important details against the cited
          sources.
        </p>
      </footer>
    </div>
  );
}

export default App;
