"""Tests for the request-id ASGI middleware (Issue #24).

Covers id sanitisation, response header echo/generation, and the
structured ``http_request`` log line (status, duration, request id).
"""

from __future__ import annotations

import logging
import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.middleware import REQUEST_ID_HEADER, sanitize_request_id

client = TestClient(app)


class TestSanitizeRequestId:
    """sanitize_request_id accepts safe ids and replaces anything else."""

    def test_accepts_safe_id(self) -> None:
        assert sanitize_request_id("trace-abc.123_XYZ") == "trace-abc.123_XYZ"

    def test_accepts_max_length_id(self) -> None:
        max_id = "a" * 64
        assert sanitize_request_id(max_id) == max_id

    def test_replaces_none_with_uuid(self) -> None:
        value = sanitize_request_id(None)
        uuid.UUID(value)  # raises ValueError if not a UUID

    def test_replaces_spaces_and_special_chars(self) -> None:
        value = sanitize_request_id("bad id!;DROP")
        assert value != "bad id!;DROP"
        uuid.UUID(value)

    def test_replaces_overlong_id(self) -> None:
        value = sanitize_request_id("a" * 65)
        assert value != "a" * 65
        uuid.UUID(value)

    def test_replaces_empty_string(self) -> None:
        uuid.UUID(sanitize_request_id(""))


class TestRequestIdMiddleware:
    """The middleware echoes or generates x-request-id on every response."""

    def test_echoes_inbound_request_id(self) -> None:
        response = client.get("/", headers={REQUEST_ID_HEADER: "trace-abc"})
        assert response.headers[REQUEST_ID_HEADER] == "trace-abc"

    def test_generates_uuid_when_header_missing(self) -> None:
        response = client.get("/")
        uuid.UUID(response.headers[REQUEST_ID_HEADER])

    def test_replaces_invalid_inbound_id(self) -> None:
        response = client.get("/", headers={REQUEST_ID_HEADER: "not a valid id!"})
        assert response.headers[REQUEST_ID_HEADER] != "not a valid id!"
        uuid.UUID(response.headers[REQUEST_ID_HEADER])

    def test_health_response_carries_request_id(self) -> None:
        response = client.get("/health")
        uuid.UUID(response.headers[REQUEST_ID_HEADER])


class TestRequestLogging:
    """RequestIdMiddleware emits one structured http_request event."""

    def test_logs_http_request_event_with_context(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="app.request"):
            client.get("/", headers={REQUEST_ID_HEADER: "trace-log-1"})

        record = next(
            r
            for r in caplog.records
            if r.name == "app.request" and r.getMessage() == "http_request"
        )
        assert record.request_id == "trace-log-1"
        assert record.method == "GET"
        assert record.path == "/"
        assert record.status == 200
        assert record.duration_ms >= 0.0

    def test_health_logged_at_debug_level(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.DEBUG, logger="app.request"):
            client.get("/health", headers={REQUEST_ID_HEADER: "trace-health-1"})

        record = next(
            r
            for r in caplog.records
            if r.name == "app.request" and r.getMessage() == "http_request"
        )
        assert record.levelno == logging.DEBUG
