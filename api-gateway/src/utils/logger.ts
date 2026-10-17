/**
 * Minimal zero-dependency structured logger (Issue #24).
 *
 * Emits one newline-delimited JSON object per log call, mirroring the
 * Python AI service's format (`app/logging_config.py`) so both services'
 * logs can be shipped and searched uniformly:
 *
 *   { "timestamp": "2026-10-17T12:00:00.123Z", "level": "info",
 *     "service": "api-gateway", "event": "http_request",
 *     "request_id": "…", …context fields }
 *
 * Design choices:
 * - No pino/winston dependency: we need four levels, JSON lines and
 *   request context — nothing a transport/tooling stack buys us yet.
 * - info/debug go to stdout, warn/error to stderr (12-factor log
 *   streams); Docker captures both.
 * - `error` fields holding an Error instance are expanded to
 *   `{ name, message, stack }` so JSON.stringify does not emit `{}`.
 *
 * Level filtering reads `LOG_LEVEL` once at import (default `info`);
 * `setLogLevel` exists for tests.
 */

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export type LogFields = Record<string, unknown>;

const LEVEL_VALUES: Record<LogLevel, number> = {
  debug: 10,
  info: 20,
  warn: 30,
  error: 40,
};

/** Emitted level names — `warn` becomes `warning` to match Python. */
const LEVEL_NAMES: Record<LogLevel, string> = {
  debug: 'debug',
  info: 'info',
  warn: 'warning',
  error: 'error',
};

const SERVICE_NAME = 'api-gateway';

function normaliseLevel(value: unknown): LogLevel {
  const candidate = typeof value === 'string' ? value.toLowerCase() : '';
  return candidate in LEVEL_VALUES ? (candidate as LogLevel) : 'info';
}

let minLevel: LogLevel = normaliseLevel(process.env.LOG_LEVEL);

/**
 * Override the minimum log level.
 * Primarily for tests; production uses the `LOG_LEVEL` env var.
 */
export function setLogLevel(level: LogLevel): void {
  minLevel = level;
}

/** Expand Error instances in the `error` field for JSON serialisation. */
function normaliseFields(fields: LogFields): LogFields {
  const error = fields.error;
  if (error instanceof Error) {
    return {
      ...fields,
      error: { name: error.name, message: error.message, stack: error.stack },
    };
  }
  return fields;
}

function write(level: LogLevel, event: string, fields: LogFields = {}): void {
  if (LEVEL_VALUES[level] < LEVEL_VALUES[minLevel]) return;

  const line = JSON.stringify({
    timestamp: new Date().toISOString(),
    level: LEVEL_NAMES[level],
    service: SERVICE_NAME,
    event,
    ...normaliseFields(fields),
  });

  if (level === 'warn' || level === 'error') {
    process.stderr.write(line + '\n');
  } else {
    process.stdout.write(line + '\n');
  }
}

/** Structured logger — call `logger.info('event_name', { …fields })`. */
export const logger = {
  debug: (event: string, fields?: LogFields): void => write('debug', event, fields),
  info: (event: string, fields?: LogFields): void => write('info', event, fields),
  warn: (event: string, fields?: LogFields): void => write('warn', event, fields),
  error: (event: string, fields?: LogFields): void => write('error', event, fields),
};
