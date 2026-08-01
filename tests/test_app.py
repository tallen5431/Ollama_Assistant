"""End-to-end tests for the Flask layer.

``app.py`` reads auth and body-size config at import time, so tests that need a
different configuration reload the module under a patched environment.
"""

from __future__ import annotations

import base64
import importlib
import json

import pytest

import app as app_module
import chat_ui

AUTH_ENV = {"CHAT_AUTH_USER": "tj", "CHAT_AUTH_PASSWORD": "hunter2"}


def load_app(monkeypatch, env=None, clear=("CHAT_AUTH", "CHAT_AUTH_USER", "CHAT_AUTH_PASSWORD")):
    """Reimport app.py with a specific environment and return the module."""
    for key in clear:
        monkeypatch.delenv(key, raising=False)
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    return importlib.reload(app_module)


@pytest.fixture
def client(monkeypatch):
    """Client for an app with auth OFF (the documented default)."""
    mod = load_app(monkeypatch)
    return mod.app.test_client()


@pytest.fixture
def auth_client(monkeypatch):
    """Client for an app with Basic Auth ON."""
    mod = load_app(monkeypatch, AUTH_ENV)
    assert mod.AUTH_ENABLED is True
    return mod.app.test_client()


def basic(user="tj", password="hunter2"):
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": "Basic " + token}


class TestHealth:
    def test_healthz_is_plain_ok(self, client):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.get_data(as_text=True) == "ok"

    def test_healthz_tolerates_a_trailing_slash(self, client):
        assert client.get("/healthz/").status_code == 200

    def test_api_health_reports_configuration(self, client):
        data = client.get("/api/health").get_json()
        assert data["status"] == "ok"
        assert data["auth"] is False
        assert "ollama_host" in data and "default_model" in data


class TestAuth:
    def test_ui_requires_credentials(self, auth_client):
        resp = auth_client.get("/")
        assert resp.status_code == 401
        assert "Basic" in resp.headers["WWW-Authenticate"]

    def test_correct_credentials_are_accepted(self, auth_client):
        assert auth_client.get("/", headers=basic()).status_code == 200

    def test_wrong_password_is_rejected(self, auth_client):
        assert auth_client.get("/", headers=basic(password="nope")).status_code == 401

    def test_wrong_user_is_rejected(self, auth_client):
        assert auth_client.get("/", headers=basic(user="mallory")).status_code == 401

    def test_api_routes_are_protected(self, auth_client):
        for path in ("/api/health", "/api/models", "/api/voice/models"):
            assert auth_client.get(path).status_code == 401, path
        assert auth_client.post("/api/chat", json={"prompt": "hi"}).status_code == 401
        assert auth_client.post("/api/transcribe", data=b"x").status_code == 401

    def test_healthz_stays_open_by_design(self, auth_client):
        assert auth_client.get("/healthz").status_code == 200

    def test_no_auth_configured_means_open(self, client):
        assert client.get("/").status_code == 200


class TestCors:
    """The UI is same-origin; no cross-origin access should be granted."""

    def test_no_allow_origin_header_is_returned(self, client):
        resp = client.get("/api/health", headers={"Origin": "https://evil.example"})
        assert "Access-Control-Allow-Origin" not in resp.headers

    def test_preflight_for_chat_is_not_granted(self, client):
        resp = client.options(
            "/api/chat",
            headers={
                "Origin": "https://evil.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert "Access-Control-Allow-Origin" not in resp.headers


class TestBodySizeLimit:
    def test_limit_is_configured(self, client, monkeypatch):
        mod = load_app(monkeypatch)
        assert mod.app.config["MAX_CONTENT_LENGTH"] == 25 * 1024 * 1024

    def test_oversized_upload_is_rejected_as_json(self, monkeypatch):
        mod = load_app(monkeypatch, {"CHAT_MAX_BODY_MB": "1"})
        monkeypatch.setattr(mod.voice, "voice_available", lambda: True)
        client = mod.app.test_client()
        resp = client.post("/api/transcribe", data=b"x" * (2 * 1024 * 1024))
        assert resp.status_code == 413
        assert "too large" in resp.get_json()["error"]

    def test_oversized_chat_body_is_rejected_as_json(self, monkeypatch):
        mod = load_app(monkeypatch, {"CHAT_MAX_BODY_MB": "1"})
        client = mod.app.test_client()
        resp = client.post(
            "/api/chat",
            data=b'{"prompt":"' + b"x" * (2 * 1024 * 1024) + b'"}',
            content_type="application/json",
        )
        assert resp.status_code == 413
        assert "too large" in resp.get_json()["error"]

    def test_limit_is_configurable(self, monkeypatch):
        mod = load_app(monkeypatch, {"CHAT_MAX_BODY_MB": "5"})
        assert mod.app.config["MAX_CONTENT_LENGTH"] == 5 * 1024 * 1024

    def test_invalid_limit_falls_back_to_the_default(self, monkeypatch):
        mod = load_app(monkeypatch, {"CHAT_MAX_BODY_MB": "not-a-number"})
        assert mod.app.config["MAX_CONTENT_LENGTH"] == 25 * 1024 * 1024


class TestChatValidation:
    def test_missing_prompt_and_messages(self, client):
        resp = client.post("/api/chat", json={})
        assert resp.status_code == 400
        assert "Missing" in resp.get_json()["error"]

    def test_empty_messages_list(self, client):
        resp = client.post("/api/chat", json={"messages": []})
        assert resp.status_code == 400

    def test_messages_must_be_a_list(self, client):
        resp = client.post("/api/chat", json={"messages": "hello"})
        assert resp.status_code == 400

    def test_unreachable_ollama_streams_an_error_line_not_a_500(self, client, monkeypatch):
        # Nothing is listening on this port, so the stream fails immediately.
        mod = load_app(monkeypatch, {"OLLAMA_HOST": "http://127.0.0.1:1"})
        resp = mod.app.test_client().post("/api/chat", json={"prompt": "hi"})
        assert resp.status_code == 200
        lines = [l for l in resp.get_data(as_text=True).splitlines() if l.strip()]
        assert "error" in json.loads(lines[-1])

    def test_non_streaming_failure_returns_502(self, monkeypatch):
        mod = load_app(monkeypatch, {"OLLAMA_HOST": "http://127.0.0.1:1"})
        resp = mod.app.test_client().post("/api/chat", json={"prompt": "hi", "stream": False})
        assert resp.status_code == 502
        assert "error" in resp.get_json()


class TestVoiceEndpoints:
    def test_transcribe_rejects_an_empty_body(self, client, monkeypatch):
        monkeypatch.setattr(app_module.voice, "voice_available", lambda: True)
        resp = client.post("/api/transcribe", data=b"")
        assert resp.status_code == 400

    def test_transcribe_rejects_a_traversal_model_id(self, client, monkeypatch):
        monkeypatch.setattr(app_module.voice, "voice_available", lambda: True)
        resp = client.post("/api/transcribe?model=../../etc", data=b"not a wav")
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_download_requires_an_id(self, client, monkeypatch):
        monkeypatch.setattr(app_module.voice, "voice_available", lambda: True)
        resp = client.post("/api/voice/download", json={})
        assert resp.status_code == 400

    def test_download_rejects_an_unknown_id(self, client, monkeypatch):
        monkeypatch.setattr(app_module.voice, "voice_available", lambda: True)
        resp = client.post("/api/voice/download", json={"id": "../../etc"})
        assert resp.status_code == 400

    def test_endpoints_report_501_when_vosk_is_missing(self, client, monkeypatch):
        monkeypatch.setattr(app_module.voice, "voice_available", lambda: False)
        assert client.get("/api/voice/models").status_code == 501
        assert client.post("/api/transcribe", data=b"x").status_code == 501
        assert client.post("/api/voice/download", json={"id": "es"}).status_code == 501


class TestPageRendering:
    def test_title_is_substituted(self):
        page = chat_ui.render_page("My Chat")
        assert "<title>My Chat</title>" in page
        assert "__TITLE__" not in page

    def test_title_is_escaped(self):
        page = chat_ui.render_page("<script>alert(1)</script>")
        assert "<script>alert(1)</script>" not in page
        assert "&lt;script&gt;" in page

    def test_ui_is_served_at_the_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "<!doctype html>" in resp.get_data(as_text=True).lower()


class TestBindHostResolution:
    def test_valid_wildcards_pass_through(self):
        assert app_module._resolve_bind_host("0.0.0.0") == "0.0.0.0"
        assert app_module._resolve_bind_host("::") == "::"

    def test_empty_falls_back(self):
        assert app_module._resolve_bind_host("") == "0.0.0.0"

    def test_localhost_resolves(self):
        assert app_module._resolve_bind_host("127.0.0.1") == "127.0.0.1"

    def test_unresolvable_placeholder_falls_back(self):
        # The bug this guards: a launcher passing an unsubstituted "HOST=PORT".
        assert app_module._resolve_bind_host("HOST=PORT") == "0.0.0.0"
