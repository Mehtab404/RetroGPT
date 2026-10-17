/**
 * Tests for services/history.ts (Issue #27).
 *
 * Pure bounding and serialisation helpers — no network or database access.
 */

import {
  boundHistory,
  serializeHistory,
  HISTORY_MAX_CHARS,
  HISTORY_MAX_MESSAGES,
  type HistoryMessage,
} from '../services/history';

/** Build a user message with the given content. */
function user(content: string): HistoryMessage {
  return { role: 'user', content };
}

/** Build an assistant message with the given content. */
function assistant(content: string): HistoryMessage {
  return { role: 'assistant', content };
}

// ---------------------------------------------------------------------------
// boundHistory
// ---------------------------------------------------------------------------

describe('boundHistory()', () => {
  it('returns an empty array for empty input', () => {
    expect(boundHistory([])).toEqual([]);
  });

  it('returns all messages when within both bounds', () => {
    const messages = [user('first'), assistant('second')];
    expect(boundHistory(messages)).toEqual(messages);
  });

  it('keeps chronological (oldest-first) order', () => {
    const messages = [user('one'), assistant('two'), user('three')];
    expect(boundHistory(messages).map((message) => message.content)).toEqual([
      'one',
      'two',
      'three',
    ]);
  });

  it('keeps at most the last six messages', () => {
    const messages = Array.from({ length: 8 }, (_, i) => user(`message ${i + 1}`));
    const bounded = boundHistory(messages);

    expect(bounded).toHaveLength(HISTORY_MAX_MESSAGES);
    expect(bounded.map((message) => message.content)).toEqual([
      'message 3',
      'message 4',
      'message 5',
      'message 6',
      'message 7',
      'message 8',
    ]);
  });

  it('drops the oldest messages once the character cap is reached', () => {
    const half = 'a'.repeat(HISTORY_MAX_CHARS / 2);
    const messages = [user(half), assistant(half), user('newest')];
    const bounded = boundHistory(messages);

    // 'newest' + one half-cap message fit; the oldest would overflow.
    expect(bounded.map((message) => message.content)).toEqual([half, 'newest']);
  });

  it('keeps a message whose content exactly reaches the cap', () => {
    const messages = [user('x'.repeat(HISTORY_MAX_CHARS))];
    expect(boundHistory(messages)).toHaveLength(1);
  });

  it('returns an empty array when even the newest message exceeds the cap', () => {
    const messages = [user('y'.repeat(HISTORY_MAX_CHARS + 1))];
    expect(boundHistory(messages)).toEqual([]);
  });

  it('does not count role strings toward the character cap', () => {
    const messages = Array.from({ length: HISTORY_MAX_MESSAGES }, () => user('tiny'));
    expect(boundHistory(messages)).toHaveLength(HISTORY_MAX_MESSAGES);
  });
});

// ---------------------------------------------------------------------------
// serializeHistory
// ---------------------------------------------------------------------------

describe('serializeHistory()', () => {
  it('is deterministic for identical histories', () => {
    const history = [user('question'), assistant('answer')];
    const clone = history.map((message) => ({ ...message }));
    expect(serializeHistory(history)).toBe(serializeHistory(clone));
  });

  it('distinguishes different roles', () => {
    expect(serializeHistory([user('same')])).not.toBe(serializeHistory([assistant('same')]));
  });

  it('distinguishes different contents', () => {
    expect(serializeHistory([user('a')])).not.toBe(serializeHistory([user('b')]));
  });

  it('distinguishes message order', () => {
    const a = [user('one'), assistant('two')];
    const b = [assistant('two'), user('one')];
    expect(serializeHistory(a)).not.toBe(serializeHistory(b));
  });

  it('cannot be ambiguous about content boundaries', () => {
    // Plain concatenation would collide here; the JSON pair encoding does not.
    const a = [user('ab'), assistant('c')];
    const b = [user('a'), assistant('bc')];
    expect(serializeHistory(a)).not.toBe(serializeHistory(b));
  });
});
