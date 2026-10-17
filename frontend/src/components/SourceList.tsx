import type { Source } from '../api';
import { sourceAnchorId } from '../citations';

export interface SourceListProps {
  messageId: string;
  sources: Source[];
  highlightedSource: string | null;
}

/**
 * Numbered list of retrieved passages below an assistant answer.
 *
 * Each entry carries a stable DOM id (`source-<messageId>-<n>`) that inline
 * citation buttons scroll to and briefly highlight.
 */
export function SourceList({ messageId, sources, highlightedSource }: SourceListProps) {
  if (sources.length === 0) return null;

  return (
    <section className="sources" aria-label="Sources">
      <h3>Sources</h3>
      <ol>
        {sources.map((source, i) => {
          const number = i + 1;
          const anchor = sourceAnchorId(messageId, number);
          const isHighlighted = highlightedSource === anchor;

          return (
            <li
              key={source.id || anchor}
              id={anchor}
              className={isHighlighted ? 'source source-highlighted' : 'source'}
            >
              <div className="source-header">
                <span className="source-number">[{number}]</span>
                <span className="source-title">{source.title || source.id}</span>
                {typeof source.score === 'number' && (
                  <span className="source-score">{Math.round(source.score * 100)}% match</span>
                )}
              </div>
              {source.snippet && <p className="source-snippet">{source.snippet}</p>}
            </li>
          );
        })}
      </ol>
    </section>
  );
}
