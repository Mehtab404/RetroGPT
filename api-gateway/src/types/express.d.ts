/**
 * Express type augmentation: request-scoped trace id.
 *
 * `src/middleware/requestId.ts` assigns it before any route runs; every
 * handler can therefore log with `request_id: req.requestId` (Issue #24).
 */

declare global {
  // eslint-disable-next-line @typescript-eslint/no-namespace
  namespace Express {
    interface Request {
      requestId: string;
    }
  }
}

export {};
