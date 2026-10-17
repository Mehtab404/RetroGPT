import type { ChatMessage } from '../App';
import { parseCitations } from '../citations';
import { SourceList } from './SourceList';

export interface MessageProps {
  message: ChatMessage;
  onCitationClick: (messageId: string, sourceIndex: number) => void;
  highlightedSource: string | null;
}

/**
 * Renders a single conversation turn.
 *
 * Assistant answers have their inline `[N]` citations replaced by clickable
 * buttons that scroll to and highlight the matching entry in the source list.
 */
export function Message({ message, onCitationClick, highlightedSource }: MessageProps) {
  const isAssistant = message.role === 'assistant';
  const segments = isAssistant
    ? parseCitations(message.content, message.sources.length)
    : [{ type: 'text' as const, content: message.content }];

  return (
    <article className={`message message-${message.role}`}>
      <div className="message-role" aria-hidden="true">
        {isAssistant ? '🤖 RetroGPT' : '🧑 You'}
      </div>
      <div className="message-content">
        {segments.map((segment, i) =>
          segment.type === 'citation' ? (
            <button
              key={`${message.id}-cite-${i}`}
              type="button"
              className="citation-link"
              onClick={() => onCitationClick(message.id, segment.index)}
              aria-label={`Show source ${segment.index}`}
            >
              {segment.content}
            </button>
          ) : (
            <span key={`${message.id}-text-${i}`}>{segment.content}</span>
          ),
        )}
      </div>

      {isAssistant && (
        <SourceList
          messageId={message.id}
          sources={message.sources}
          highlightedSource={highlightedSource}
        />
      )}
    </article>
  );
}
