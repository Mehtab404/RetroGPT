/**
 * Jest setup (Issue #24).
 *
 * Silences info/warn JSON log lines so test output stays readable —
 * requests emit an `http_request` line each. Tests that assert on
 * logging raise the level or spy on the logger methods directly.
 */

import { setLogLevel } from './utils/logger';

setLogLevel('error');
