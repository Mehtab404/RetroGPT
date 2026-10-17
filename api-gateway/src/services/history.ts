/**
 * Bounded conversation-history helpers (Issue #27).
 *
 * The API gateway stores every exchange on the session document
 * (`api-gateway/src/models/session.ts`); before calling the AI service it
 * reloads a small window of recent messages *server-side* so follow-up
 * questions ("What about the saucer?") can be interpreted against the prior
 * turns.  Browser-supplied history is never trusted.
 *
 * Design choices:
 * - Bounding is deterministic and applied newest-first: the most recent
 *   messages are kept until either bound is reached.  The returned window is
 *   in chronological (oldest-first) order, matching the stored order.
 * - {@link serializeHistory} produces an unambiguous, order-sensitive
 *   encoding used only to fingerprint history-aware cache keys.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** One prior conversation turn forwarded to the AI service. */
export interface HistoryMessage {
  role: 'user' | 'assistant';
  content: string;
}

// ---------------------------------------------------------------------------
// Bounds
// ---------------------------------------------------------------------------

/** Maximum number of prior messages attached to a query. */
export const HISTORY_MAX_MESSAGES = 6;

/** Maximum total characters across the attached prior messages. */
export const HISTORY_MAX_CHARS = 4000;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Return the bounded window of the most recent messages.
 *
 * Walks the messages newest-first, keeping at most
 * {@link HISTORY_MAX_MESSAGES} messages whose combined content length stays
 * within {@link HISTORY_MAX_CHARS}, then reverses the window back into
 * chronological order.  Returns an empty array when nothing fits.
 */
export function boundHistory(messages: HistoryMessage[]): HistoryMessage[] {
  const bounded: HistoryMessage[] = [];
  let totalChars = 0;

  for (let i = messages.length - 1; i >= 0 && bounded.length < HISTORY_MAX_MESSAGES; i -= 1) {
    const message = messages[i];
    if (totalChars + message.content.length > HISTORY_MAX_CHARS) break;
    bounded.push(message);
    totalChars += message.content.length;
  }

  return bounded.reverse();
}

/**
 * Serialise history into a deterministic, order-sensitive string.
 *
 * The array-of-pairs JSON encoding is unambiguous: JSON escaping keeps roles
 * and contents from bleeding into one another, and reordering the same
 * messages produces a different string.
 */
export function serializeHistory(history: HistoryMessage[]): string {
  return JSON.stringify(history.map((message) => [message.role, message.content]));
}
