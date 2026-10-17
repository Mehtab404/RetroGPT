/**
 * Citation parsing for assistant answers.
 *
 * The LLM is instructed to cite sources inline as `[N]` (1-based passage
 * numbers).  These helpers split an answer into text and citation segments
 * so the UI can render clickable citation references.
 */

export interface TextSegment {
  type: 'text';
  content: string;
}

export interface CitationSegment {
  type: 'citation';
  /** 1-based source number as written in the answer. */
  index: number;
  /** The raw matched text, e.g. `[2]`. */
  content: string;
}

export type AnswerSegment = TextSegment | CitationSegment;

/**
 * Split an answer string into text and `[N]` citation segments.
 *
 * Only citation numbers that fall within `[1, maxSources]` are turned into
 * citation segments; anything else stays plain text so stray brackets in the
 * prose are not misinterpreted.
 *
 * @param answer      Raw assistant answer text.
 * @param maxSources  Number of sources available (defaults to `Infinity`).
 */
export function parseCitations(answer: string, maxSources = Infinity): AnswerSegment[] {
  const segments: AnswerSegment[] = [];
  const pattern = /\[(\d+)\]/g;

  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = pattern.exec(answer)) !== null) {
    const index = Number.parseInt(match[1], 10);
    const valid = index >= 1 && index <= maxSources;

    if (!valid) {
      continue;
    }

    if (match.index > lastIndex) {
      segments.push({ type: 'text', content: answer.slice(lastIndex, match.index) });
    }
    segments.push({ type: 'citation', index, content: match[0] });
    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < answer.length) {
    segments.push({ type: 'text', content: answer.slice(lastIndex) });
  }

  return segments;
}

/** Build the DOM id used to anchor a source entry inside a message. */
export function sourceAnchorId(messageId: string, sourceIndex: number): string {
  return `source-${messageId}-${sourceIndex}`;
}
