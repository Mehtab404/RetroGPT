/**
 * Minimal Server-Sent Events helpers (Issue #25).
 *
 * RetroGPT streams answers as SSE: `event:` + single-line JSON `data:`
 * frames separated by a blank line.  These helpers format frames for
 * responses and parse incoming frames incrementally, so chunks that split
 * a frame across network reads are handled correctly.
 */

export interface SseFrame {
  event: string;
  data: string;
}

/** Format one SSE frame from an event name and JSON-serialisable payload. */
export function formatSseEvent(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

/**
 * Create an incremental SSE parser.
 *
 * Feed decoded text chunks in; each call returns the complete frames
 * contained so far.  Partial frames are buffered until the terminating
 * blank line arrives.
 */
export function createSseParser(): (chunk: string) => SseFrame[] {
  let buffer = '';

  return (chunk: string): SseFrame[] => {
    // Normalise CRLF (and split CRLF across chunks) so frame boundaries
    // always match the "\n\n" search below.
    buffer = (buffer + chunk).replace(/\r\n/g, '\n');
    const frames: SseFrame[] = [];

    let boundary = buffer.indexOf('\n\n');
    while (boundary !== -1) {
      const rawFrame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const frame = parseSseFrame(rawFrame);
      if (frame !== null) frames.push(frame);
      boundary = buffer.indexOf('\n\n');
    }

    return frames;
  };
}

function parseSseFrame(rawFrame: string): SseFrame | null {
  let event = 'message';
  const dataLines: string[] = [];

  for (const rawLine of rawFrame.split('\n')) {
    const line = rawLine.endsWith('\r') ? rawLine.slice(0, -1) : rawLine;
    if (line.startsWith('event:')) {
      event = line.slice('event:'.length).trim();
    } else if (line.startsWith('data:')) {
      // A single leading space after "data:" is part of the SSE framing.
      dataLines.push(line.slice('data:'.length).replace(/^ /, ''));
    }
  }

  if (dataLines.length === 0) return null;
  return { event, data: dataLines.join('\n') };
}
