import { describe, expect, it } from 'vitest';
import { parseCitations, sourceAnchorId } from '../citations';

describe('parseCitations()', () => {
  it('splits text around a single citation', () => {
    const segments = parseCitations('Large asteroids score 20 points [1].', 1);

    expect(segments).toEqual([
      { type: 'text', content: 'Large asteroids score 20 points ' },
      { type: 'citation', index: 1, content: '[1]' },
      { type: 'text', content: '.' },
    ]);
  });

  it('finds multiple citations in order', () => {
    const segments = parseCitations('First [1] second [2] done.', 2);

    expect(segments.filter((s) => s.type === 'citation')).toHaveLength(2);
    expect(segments[1]).toMatchObject({ type: 'citation', index: 1 });
    expect(segments[3]).toMatchObject({ type: 'citation', index: 2 });
  });

  it('returns a single text segment when there are no citations', () => {
    const segments = parseCitations('No citations here.', 0);

    expect(segments).toEqual([{ type: 'text', content: 'No citations here.' }]);
  });

  it('leaves out-of-range citations as plain text', () => {
    const segments = parseCitations('See [5] for details.', 2);

    expect(segments).toEqual([{ type: 'text', content: 'See [5] for details.' }]);
  });

  it('handles a citation at the start of the answer', () => {
    const segments = parseCitations('[1] is the answer.', 1);

    expect(segments[0]).toEqual({ type: 'citation', index: 1, content: '[1]' });
    expect(segments[1]).toEqual({ type: 'text', content: ' is the answer.' });
  });

  it('treats bare brackets and zero index as plain text', () => {
    const segments = parseCitations('Use [abc] or [0] notation.', 3);

    expect(segments).toEqual([{ type: 'text', content: 'Use [abc] or [0] notation.' }]);
  });

  it('treats citations as valid when maxSources is omitted', () => {
    const segments = parseCitations('A [7] B');

    expect(segments[1]).toMatchObject({ type: 'citation', index: 7 });
  });
});

describe('sourceAnchorId()', () => {
  it('builds a stable anchor id from message id and index', () => {
    expect(sourceAnchorId('msg-1', 2)).toBe('source-msg-1-2');
  });
});
