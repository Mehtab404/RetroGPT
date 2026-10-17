"""ASGI middleware for request-id propagation and structured request logs.

Issue #24: the API gateway sends an ``x-request-id`` header with every
call to the AI service.  The middleware

1. reuses a well-formed inbound id (so the gateway's and Python logs can
   be joined on ``request_id``) or generates a UUID4,
2. exposes it to the JSON formatter through ``request_id_var`` so every
   log line emitted while the request is handled is correlated,
3. echoes it back on the response, and
4. logs one ``http_request`` event with method, path, status and duration.

The middleware is pure ASGI (not ``BaseHTTPMiddleware``) so it stays safe
for streaming responses — see Issue #25.
"""

from __future__ import annotations

import logging
import re
import time
import uuid

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.logging_config import request_id_var

logger = logging.getLogger("app.request")

REQUEST_ID_HEADER = "x-request-id"

#: Trace ids are opaque but bounded: ASCII letters, digits and ``._-``.
#: Anything else (control characters, separators, overlong strings) is
#: replaced to keep log lines clean and machine-parseable.
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

#: Paths whose request logs are emitted at DEBUG (container healthchecks).
_QUIET_PATHS = frozenset({"/health"})


def sanitize_request_id(value: str | None) -> str:
    """Return *value* when it is a safe trace id, else a fresh UUID4.

    Args:
        value: Inbound ``x-request-id`` header value (may be ``None``).

    Returns:
        A validated inbound id, or ``str(uuid.uuid4())``.
    """
    if value is not None and _VALID_REQUEST_ID.match(value):
        return value
    return str(uuid.uuid4())


class RequestIdMiddleware:
    """Pure-ASGI middleware adding request ids and per-request log lines."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one ASGI message; non-HTTP scopes pass through untouched."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = sanitize_request_id(Headers(scope=scope).get(REQUEST_ID_HEADER))
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        status_code = 500

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            duration_ms = (time.perf_counter() - start) * 1000.0
            log = logger.debug if scope["path"] in _QUIET_PATHS else logger.info
            log(
                "http_request",
                extra={
                    "request_id": request_id,
                    "method": scope["method"],
                    "path": scope["path"],
                    "status": status_code,
                    "duration_ms": round(duration_ms, 2),
                },
            )
            request_id_var.reset(token)
