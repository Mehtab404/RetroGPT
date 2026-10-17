"""Tests for health check endpoint."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check_success() -> None:
    """Test health check returns OK when Qdrant is healthy."""
    with patch("app.main.qdrant_client") as mock_client:
        mock_client.get_collections.return_value = MagicMock()
        
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "ai-service"
        assert data["dependencies"]["qdrant"] == "ok"


def test_health_check_qdrant_error() -> None:
    """Test health check returns degraded when Qdrant is unavailable."""
    with patch("app.main.qdrant_client") as mock_client:
        mock_client.get_collections.side_effect = Exception("Connection failed")
        
        response = client.get("/health")
        
        assert response.status_code == 503
        data = response.json()
        assert "error" in data["detail"]["dependencies"]["qdrant"]


def test_health_check_no_client() -> None:
    """Test health check when Qdrant client is not initialized."""
    with patch("app.main.qdrant_client", None):
        response = client.get("/health")
        
        assert response.status_code == 503
        data = response.json()
        assert data["detail"]["dependencies"]["qdrant"] == "not_initialized"


def test_root_endpoint() -> None:
    """Test root endpoint returns service info."""
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "RetroGPT AI Service"
    assert data["version"] == "0.1.0"
    assert data["docs"] == "/docs"
