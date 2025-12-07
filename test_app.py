#!/usr/bin/env python3
"""Basic unit tests for CodeSmith Ollama Helper.

Run with: pytest test_app.py -v
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import app components
from app import (
    app,
    validate_snapshot_file,
    cleanup_old_snapshots,
    UPLOAD_DIR,
)
from config_ollama_helper import (
    get_ollama_base,
    get_default_model,
    get_max_upload_size,
    require_ollama_on_startup,
    get_ollama_startup_retries,
    get_ollama_startup_retry_delay,
)
from ollama_client import check_ollama_health, wait_for_ollama
from codesmith_patch_utils import (
    normalize_patch_text,
    build_code_system_prompt,
)


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestConfigHelpers:
    """Test configuration helper functions."""

    def test_get_ollama_base_default(self):
        """Test default Ollama base URL."""
        base = get_ollama_base()
        assert base.startswith("http")
        assert not base.endswith("/")

    def test_get_default_model(self):
        """Test default model retrieval."""
        model = get_default_model()
        assert isinstance(model, str)
        assert len(model) > 0

    def test_get_max_upload_size(self):
        """Test max upload size is positive."""
        size = get_max_upload_size()
        assert size > 0
        assert isinstance(size, int)

    def test_require_ollama_on_startup(self):
        """Test require Ollama on startup config."""
        result = require_ollama_on_startup()
        assert isinstance(result, bool)

    def test_get_ollama_startup_retries(self):
        """Test startup retries config."""
        retries = get_ollama_startup_retries()
        assert isinstance(retries, int)
        assert retries >= 0

    def test_get_ollama_startup_retry_delay(self):
        """Test startup retry delay config."""
        delay = get_ollama_startup_retry_delay()
        assert isinstance(delay, float)
        assert delay >= 0


class TestPatchUtils:
    """Test patch normalization utilities."""

    def test_normalize_simple_patch(self):
        """Test normalizing a simple patch object."""
        raw = json.dumps({
            "edits": [
                {
                    "operation": "modify",
                    "path": "test.py",
                    "search": "old",
                    "replace": "new"
                }
            ]
        })

        result = normalize_patch_text(raw)
        assert result["schema"] == "codesmith.patch.v1"
        assert result["transactional"] is True
        assert len(result["edits"]) == 1

    def test_normalize_with_markdown_fences(self):
        """Test stripping markdown code fences."""
        raw = "```json\n{\"edits\": []}\n```"
        result = normalize_patch_text(raw)
        assert result["schema"] == "codesmith.patch.v1"

    def test_normalize_invalid_json(self):
        """Test error handling for invalid JSON."""
        with pytest.raises(ValueError, match="Could not parse"):
            normalize_patch_text("not valid json")

    def test_build_code_system_prompt(self):
        """Test system prompt generation."""
        prompt = build_code_system_prompt()
        assert "code assistant" in prompt.lower()
        assert "codesmith" in prompt.lower()

    def test_build_code_system_prompt_with_custom(self):
        """Test system prompt with custom prefix."""
        custom = "You are a helpful assistant."
        prompt = build_code_system_prompt(custom)
        assert custom in prompt
        assert "codesmith" in prompt.lower()


class TestSnapshotValidation:
    """Test snapshot file validation."""

    def test_validate_valid_snapshot(self):
        """Test validation of a valid snapshot."""
        snapshot = {
            "schema": "codesmith.snapshot.v1",
            "root_path": "/test",
            "files": [
                {"path": "test.py", "source": "print('hello')"}
            ]
        }
        content = json.dumps(snapshot).encode("utf-8")
        error = validate_snapshot_file("test.json", content)
        assert error is None

    def test_validate_wrong_extension(self):
        """Test rejection of non-JSON files."""
        error = validate_snapshot_file("test.txt", b"{}")
        assert error is not None
        assert "file type" in error.lower()

    def test_validate_invalid_json(self):
        """Test rejection of invalid JSON."""
        error = validate_snapshot_file("test.json", b"not json")
        assert error is not None
        assert "invalid json" in error.lower()

    def test_validate_missing_files_field(self):
        """Test rejection of snapshots without 'files' field."""
        content = json.dumps({"schema": "test"}).encode("utf-8")
        error = validate_snapshot_file("test.json", content)
        assert error is not None
        assert "files" in error.lower()


class TestAPIEndpoints:
    """Test Flask API endpoints."""

    def test_index_route(self, client):
        """Test root page returns HTML."""
        response = client.get("/")
        assert response.status_code == 200
        assert b"CodeSmith Ollama Helper" in response.data

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "status" in data
        assert "ollama_host" in data
        assert "default_model" in data

    @patch("app.post_ollama")
    def test_models_endpoint(self, mock_post, client):
        """Test models listing endpoint."""
        mock_post.return_value = {"models": ["model1", "model2"]}
        response = client.get("/api/models")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "models" in data

    def test_chat_missing_input(self, client):
        """Test chat endpoint with missing input."""
        response = client.post(
            "/api/chat",
            data=json.dumps({}),
            content_type="application/json"
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    @patch("app.post_ollama")
    def test_chat_with_prompt(self, mock_post, client):
        """Test successful chat with prompt."""
        mock_post.return_value = {
            "message": {"content": "Hello!"}
        }
        response = client.post(
            "/api/chat",
            data=json.dumps({"prompt": "Hi"}),
            content_type="application/json"
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "reply" in data

    def test_chat_invalid_messages(self, client):
        """Test chat with invalid message format."""
        response = client.post(
            "/api/chat",
            data=json.dumps({"messages": "not a list"}),
            content_type="application/json"
        )
        assert response.status_code == 400

    def test_patch_normalize_missing_raw(self, client):
        """Test patch normalization without raw input."""
        response = client.post(
            "/api/patch/normalize",
            data=json.dumps({}),
            content_type="application/json"
        )
        assert response.status_code == 400

    def test_patch_normalize_success(self, client):
        """Test successful patch normalization."""
        response = client.post(
            "/api/patch/normalize",
            data=json.dumps({
                "raw": json.dumps({"edits": []})
            }),
            content_type="application/json"
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "patch" in data
        assert data["patch"]["schema"] == "codesmith.patch.v1"


class TestFileUpload:
    """Test file upload functionality."""

    def test_upload_missing_file(self, client):
        """Test upload without file."""
        response = client.post("/api/upload-snapshot")
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_upload_invalid_json(self, client):
        """Test upload of invalid JSON file."""
        from io import BytesIO
        data = {
            "file": (BytesIO(b"not json"), "test.json")
        }
        response = client.post(
            "/api/upload-snapshot",
            data=data,
            content_type="multipart/form-data"
        )
        assert response.status_code == 400

    def test_upload_wrong_extension(self, client):
        """Test upload of non-JSON file."""
        from io import BytesIO
        data = {
            "file": (BytesIO(b"content"), "test.txt")
        }
        response = client.post(
            "/api/upload-snapshot",
            data=data,
            content_type="multipart/form-data"
        )
        assert response.status_code == 400


class TestCodeAssist:
    """Test code assistance endpoint."""

    def test_code_assist_no_input(self, client):
        """Test code assist without prompt or code."""
        response = client.post(
            "/api/code-assist",
            data=json.dumps({}),
            content_type="application/json"
        )
        assert response.status_code == 400

    @patch("app.post_ollama")
    def test_code_assist_with_code(self, mock_post, client):
        """Test code assist with code snippet."""
        mock_post.return_value = {
            "message": {"content": "This code looks good!"}
        }
        response = client.post(
            "/api/code-assist",
            data=json.dumps({
                "prompt": "Review this code",
                "code": "def hello(): pass"
            }),
            content_type="application/json"
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "reply" in data


class TestOllamaStartup:
    """Test Ollama startup verification."""

    def test_check_ollama_health_returns_bool(self):
        """Test health check returns boolean."""
        result = check_ollama_health()
        assert isinstance(result, bool)

    @patch("ollama_client.requests.get")
    def test_check_ollama_health_success(self, mock_get):
        """Test successful health check."""
        mock_response = Mock()
        mock_response.ok = True
        mock_get.return_value = mock_response

        result = check_ollama_health()
        assert result is True

    @patch("ollama_client.requests.get")
    def test_check_ollama_health_failure(self, mock_get):
        """Test failed health check."""
        mock_get.side_effect = Exception("Connection refused")

        result = check_ollama_health()
        assert result is False

    @patch("ollama_client.requests.get")
    @patch("ollama_client.time.sleep")
    def test_wait_for_ollama_success_first_try(self, mock_sleep, mock_get):
        """Test wait_for_ollama succeeds on first attempt."""
        mock_response = Mock()
        mock_response.ok = True
        mock_get.return_value = mock_response

        result = wait_for_ollama(max_retries=3, initial_delay=1.0)
        assert result is True
        mock_sleep.assert_not_called()

    @patch("ollama_client.requests.get")
    @patch("ollama_client.time.sleep")
    def test_wait_for_ollama_retry_then_success(self, mock_sleep, mock_get):
        """Test wait_for_ollama retries then succeeds."""
        mock_response_fail = Mock()
        mock_response_fail.ok = False
        mock_response_success = Mock()
        mock_response_success.ok = True

        # Fail first two times, succeed on third
        mock_get.side_effect = [
            Exception("Connection refused"),
            Exception("Connection refused"),
            mock_response_success
        ]

        result = wait_for_ollama(max_retries=3, initial_delay=0.1, timeout=1.0)
        assert result is True
        assert mock_sleep.call_count == 2

    @patch("ollama_client.requests.get")
    @patch("ollama_client.time.sleep")
    def test_wait_for_ollama_max_retries_exceeded(self, mock_sleep, mock_get):
        """Test wait_for_ollama fails after max retries."""
        mock_get.side_effect = Exception("Connection refused")

        result = wait_for_ollama(max_retries=3, initial_delay=0.1, timeout=1.0)
        assert result is False
        assert mock_sleep.call_count == 2  # Sleeps between retries, not after last


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
