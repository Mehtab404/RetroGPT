import { createSseParser, formatSseEvent } from '../utils/sse';

describe('formatSseEvent', () => {
  it('formats an event with JSON data and a terminating blank line', () => {
    expect(formatSseEvent('token', { text: 'hi' })).toBe('event: token\ndata: {"text":"hi"}\n\n');
  });
});

describe('createSseParser', () => {
  it('parses complete frames', () => {
    const parse = createSseParser();
    const frames = parse(
      'event: meta\ndata: {"sources":[]}\n\nevent: done\ndata: {"latency_ms":5}\n\n',
    );

    expect(frames).toEqual([
      { event: 'meta', data: '{"sources":[]}' },
      { event: 'done', data: '{"latency_ms":5}' },
    ]);
  });

  it('buffers frames split across chunk boundaries', () => {
    const parse = createSseParser();

    expect(parse('event: tok')).toEqual([]);
    expect(parse('en\ndata: {"te')).toEqual([]);
    expect(parse('xt":"Hi"}\n\n')).toEqual([{ event: 'token', data: '{"text":"Hi"}' }]);
  });

  it('handles CRLF line endings', () => {
    const parse = createSseParser();
    const frames = parse('event: token\r\ndata: {"text":"Hi"}\r\n\r\n');

    expect(frames).toEqual([{ event: 'token', data: '{"text":"Hi"}' }]);
  });

  it('ignores keep-alive frames without data', () => {
    const parse = createSseParser();

    expect(parse(': keep-alive\n\n')).toEqual([]);
    expect(parse('event: ping\n\n')).toEqual([]);
  });

  it('joins multi-line data with newlines', () => {
    const parse = createSseParser();
    const frames = parse('event: token\ndata: line one\ndata: line two\n\n');

    expect(frames).toEqual([{ event: 'token', data: 'line one\nline two' }]);
  });

  it('defaults the event name to "message" when no event: line is present', () => {
    const parse = createSseParser();
    const frames = parse('data: {"text":"Hi"}\n\n');

    expect(frames).toEqual([{ event: 'message', data: '{"text":"Hi"}' }]);
  });
});
