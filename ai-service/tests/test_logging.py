"""Tests for structured JSON logging (Issue #24).

Covers :class:`~app.logging_config.JsonFormatter` field mapping,
request-id context propagation, ``extra=`` merging, exception rendering,
and :func:`~app.logging_config.configure_logging` handler installation.
"""

from __future__ import annotations

import json
import logging
import sys

import pytest

from app.logging_config import (
    SERVICE_NAME,
    JsonFormatter,
    configure_logging,
    request_id_var,
)


def _make_record(**overrides: object) -> logging.LogRecord:
    """Build a LogRecord with sensible defaults for formatter tests."""
    values: dict[str, object] = {
        "name": "app.main",
        "levelname": "INFO",
        "levelno": logging.INFO,
        "msg": "query",
        "args": (),
    }
    values.update(overrides)
    return logging.makeLogRecord(values)


def _format_json(record: logging.LogRecord) -> dict:
    """Format *record* and parse the JSON payload."""
    return json.loads(JsonFormatter().format(record))


class TestJsonFormatter:
    """JsonFormatter renders one parseable JSON object per record."""

    def test_base_fields(self) -> None:
        payload = _format_json(_make_record())

        assert payload["level"] == "info"
        assert payload["service"] == SERVICE_NAME
        assert payload["logger"] == "app.main"
        assert payload["event"] == "query"
        # ISO 8601 UTC with millisecond precision, e.g. 2026-10-17T12:00:00.123Z
        assert payload["timestamp"].endswith("Z")
        assert "T" in payload["timestamp"]

    def test_message_interpolates_args(self) -> None:
        payload = _format_json(_make_record(msg="query %s", args=("abc",)))
        assert payload["event"] == "query abc"

    def test_level_is_lowercase(self) -> None:
        payload = _format_json(_make_record(levelname="WARNING", levelno=logging.WARNING))
        assert payload["level"] == "warning"

    def test_extra_fields_merged(self) -> None:
        payload = _format_json(
            _make_record(
                stage_ms={"embed": 1.2, "generate": 900.0},
                result_count=3,
            )
        )

        assert payload["stage_ms"] == {"embed": 1.2, "generate": 900.0}
        assert payload["result_count"] == 3

    def test_request_id_from_context(self) -> None:
        token = request_id_var.set("trace-context-1")
        try:
            payload = _format_json(_make_record())
        finally:
            request_id_var.reset(token)

        assert payload["request_id"] == "trace-context-1"

    def test_explicit_extra_request_id_wins_over_context(self) -> None:
        token = request_id_var.set("from-context")
        try:
            payload = _format_json(_make_record(request_id="from-extra"))
        finally:
            request_id_var.reset(token)

        assert payload["request_id"] == "from-extra"

    def test_no_request_id_outside_request(self) -> None:
        payload = _format_json(_make_record())
        assert "request_id" not in payload

    def test_exception_is_rendered(self) -> None:
        try:
            raise ValueError("boom")
        except ValueError:
            record = _make_record(levelname="ERROR", levelno=logging.ERROR)
            record.exc_info = sys.exc_info()

        payload = _format_json(record)
        assert "ValueError: boom" in payload["exception"]


class TestConfigureLogging:
    """configure_logging installs the JSON handler and quiets uvicorn access."""

    def test_installs_json_handler_on_root_and_uvicorn(self) -> None:
        try:
            configure_logging("DEBUG")

            root = logging.getLogger()
            assert root.level == logging.DEBUG
            assert len(root.handlers) == 1
            assert isinstance(root.handlers[0].formatter, JsonFormatter)

            for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
                uvicorn_logger = logging.getLogger(name)
                assert uvicorn_logger.handlers == [root.handlers[0]]
                assert uvicorn_logger.propagate is False

            # Structured request logs come from RequestIdMiddleware; uvicorn's
            # plain-text access lines are suppressed.
            assert logging.getLogger("uvicorn.access").level == logging.WARNING
        finally:
            # Back to the app defaults configured at import time.
            configure_logging()

    def test_is_idempotent(self) -> None:
        try:
            configure_logging("INFO")
            configure_logging("INFO")
            assert len(logging.getLogger().handlers) == 1
        finally:
            configure_logging()

    def test_respects_settings_log_level_by_default(self) -> None:
        try:
            configure_logging()
            assert logging.getLogger().level == logging.getLevelName("INFO")
        finally:
            configure_logging()


class TestRequestIdVar:
    """request_id_var is an isolated context variable."""

    def test_default_is_none(self) -> None:
        assert request_id_var.get() is None

    def test_set_and_reset(self) -> None:
        token = request_id_var.set("abc")
        try:
            assert request_id_var.get() == "abc"
        finally:
            request_id_var.reset(token)
        assert request_id_var.get() is None


@pytest.mark.parametrize(
    ("level_name", "expected"),
    [("INFO", logging.INFO), ("DEBUG", logging.DEBUG), ("WARNING", logging.WARNING)],
)
def test_configure_logging_accepts_level_names(level_name: str, expected: int) -> None:
    try:
        configure_logging(level_name)
        assert logging.getLogger().level == expected
    finally:
        configure_logging()
