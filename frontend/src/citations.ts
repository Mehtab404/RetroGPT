/**
 * Citation parsing for assistant answers.
 *
 * The LLM is instructed to cite sources inline as `[N]` (1-based passage
 * numbers), but it occasionally emits full-width variants such as `【N】`
 * or `［N］` (even mixed pairs like `【2]`).  These helpers accept every
 * bracket family, split an answer into text and citation segments, and
 * normalize the rendered marker text back to ASCII.
 */

export interface TextSegment {
  type: 'text';
  content: string;
}

export interface CitationSegment {
  type: 'citation';
  /** 1-based source number as written in the answer. */
  index: number;
  /** Normalized marker text, always ASCII, e.g. `[2]`. */
  content: string;
}

export type AnswerSegment = TextSegment | CitationSegment;

/**
 * Split an answer string into text and `[N]` citation segments.
 *
 * Accepts ASCII brackets and the full-width variants `【N】` / `［N］`
 * (mixed pairs included).  Only citation numbers that fall within
 * `[1, maxSources]` are turned into citation segments; anything else stays
 * plain text so stray brackets in the prose are not misinterpreted.
 *
 * @param answer      Raw assistant answer text.
 * @param maxSources  Number of sources available (defaults to `Infinity`).
 */
export function parseCitations(answer: string, maxSources = Infinity): AnswerSegment[] {
  const segments: AnswerSegment[] = [];
  // Opening/closing bracket families may be mixed (e.g. `【2]`).
  const pattern = /[【［[](\d+)[\]】］]/g;

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
    segments.push({ type: 'citation', index, content: `[${index}]` });
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
