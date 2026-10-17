"""Structured JSON logging and request-scoped context (Issue #24).

Both RetroGPT services emit one JSON object per log line so log shippers
(Docker, CloudWatch) can parse them without regex.  Every line carries:

======================  ====================================================
``timestamp``           ISO 8601 UTC with millisecond precision
``level``               ``debug`` / ``info`` / ``warning`` / ``error`` …
``service``             ``"ai-service"``
``logger``              Python logger name (e.g. ``app.main``)
``event``               human-readable event name (the log message)
``request_id``          in-flight request id; set by ``RequestIdMiddleware``
                        (fallback: :data:`request_id_var`)
======================  ====================================================

Keyword arguments passed via ``logging``'s ``extra=`` are merged into the
JSON object at the top level, e.g.::

    logger.info("query", extra={"stage_ms": {...}, "total_ms": 912.4})

``configure_logging()`` attaches a single JSON handler to the root logger
and routes uvicorn's loggers through it.  Uvicorn's plain-text access log
is silenced on purpose: :class:`~app.middleware.RequestIdMiddleware` emits
the structured equivalent (same event name + request id as the gateway).
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from app.config import settings

SERVICE_NAME = "ai-service"

#: Request id of the in-flight HTTP request (set by RequestIdMiddleware).
#: Read by :class:`JsonFormatter` so *every* log line emitted while a
#: request is handled — including retrieval warnings — is correlated.
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

#: Attributes that always exist on a ``LogRecord`` — anything else was
#: passed through ``extra=`` and should be merged into the JSON payload.
_STANDARD_ATTRS: frozenset[str] = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"message", "asctime", "color_message"}


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """Render *record* as a JSON string (one line, UTF-8)."""
        payload: dict[str, Any] = {
            "timestamp": _utc_timestamp(),
            "level": record.levelname.lower(),
            "service": SERVICE_NAME,
            "logger": record.name,
            "event": record.getMessage(),
        }

        # Merge ``extra=`` fields (ctx fields win over the defaults above).
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value

        # Correlate with the in-flight request when the caller did not
        # already pass an explicit ``request_id`` extra.
        if "request_id" not in payload:
            request_id = request_id_var.get()
            if request_id:
                payload["request_id"] = request_id

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, ensure_ascii=False)


def _utc_timestamp() -> str:
    """Return the current UTC time as ``2026-10-17T12:00:00.123Z``."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def configure_logging(level: str | int | None = None) -> None:
    """Install the JSON handler on the root and uvicorn loggers.

    Args:
        level: Minimum level (name or :mod:`logging` constant).  Defaults
            to ``settings.log_level`` (env ``LOG_LEVEL``, default ``INFO``).

    Safe to call multiple times — handlers are replaced, not appended.
    Uvicorn's access logger is set to ``WARNING`` because
    :class:`~app.middleware.RequestIdMiddleware` already emits one
    structured ``http_request`` line per request.
    """
    resolved_level: str | int = level if level is not None else settings.log_level
    if isinstance(resolved_level, str):
        resolved_level = resolved_level.upper()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(resolved_level)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers[:] = [handler]
        uvicorn_logger.propagate = False
        uvicorn_logger.setLevel(resolved_level)

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
