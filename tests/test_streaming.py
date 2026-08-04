"""Streaming tests against a fake Ollama server.

These exist because of a real failure: Ollama streams chat responses as
``application/x-ndjson`` with no charset, which requests cannot infer an
encoding for. ``iter_lines(decode_unicode=True)`` then silently yields *bytes*
instead of str, and the app blew up with "can't concat str to bytes" the first
time a reply came back. The fake server below reproduces that exact content
type, so a regression is caught here rather than in production.
"""

from __future__ import annotations

import importlib
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import app as app_module
import ollama_client

CHUNKS = [
    {"message": {"content": "Hello"}, "done": False},
    {"message": {"content": " there"}, "done": False},
    {"message": {"content": " 👋"}, "done": False},  # multi-byte, on purpose
    {"message": {"content": ""}, "done": True, "eval_count": 3, "eval_duration": 1_000_000},
]


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.server.last_request = json.loads(body or b"{}")
        self.send_response(200)
        # Exactly what Ollama sends: no charset, so requests infers no encoding.
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        for chunk in CHUNKS:
            self.wfile.write(json.dumps(chunk).encode("utf-8") + b"\n")
            self.wfile.flush()

    def do_GET(self):
        body = json.dumps({"models": [{"name": "codellama:7b"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # keep test output quiet


@pytest.fixture
def fake_ollama(monkeypatch):
    """Run a stand-in Ollama on a free port and point the app at it."""
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("OLLAMA_HOST", f"http://127.0.0.1:{server.server_port}")
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


class TestChatStream:
    def test_yields_str_not_bytes(self, fake_ollama):
        """The regression: every line must be str, whatever requests inferred."""
        lines = list(ollama_client.chat_stream("codellama:7b", [{"role": "user", "content": "hi"}]))
        assert lines, "expected streamed lines"
        assert all(isinstance(line, str) for line in lines)

    def test_lines_are_parseable_ndjson(self, fake_ollama):
        lines = list(ollama_client.chat_stream("codellama:7b", [{"role": "user", "content": "hi"}]))
        parsed = [json.loads(line) for line in lines]
        text = "".join(p.get("message", {}).get("content", "") for p in parsed)
        assert text == "Hello there 👋"
        assert parsed[-1]["done"] is True

    def test_multibyte_survives_the_round_trip(self, fake_ollama):
        joined = "".join(
            json.loads(line).get("message", {}).get("content", "")
            for line in ollama_client.chat_stream("m", [{"role": "user", "content": "hi"}])
        )
        assert "👋" in joined


class TestChatEndpointStreaming:
    def test_endpoint_streams_content_without_an_error_line(self, fake_ollama):
        """End-to-end: the exact path that raised 'can't concat str to bytes'."""
        mod = importlib.reload(app_module)
        resp = mod.app.test_client().post("/api/chat", json={"prompt": "hi"})

        assert resp.status_code == 200
        lines = [json.loads(l) for l in resp.get_data(as_text=True).splitlines() if l.strip()]
        assert not any("error" in obj for obj in lines), lines
        text = "".join(obj.get("message", {}).get("content", "") for obj in lines)
        assert text == "Hello there 👋"

    def test_usage_stats_reach_the_client(self, fake_ollama):
        mod = importlib.reload(app_module)
        resp = mod.app.test_client().post("/api/chat", json={"prompt": "hi"})
        final = [json.loads(l) for l in resp.get_data(as_text=True).splitlines() if l.strip()][-1]
        assert final["eval_count"] == 3
        assert final["done"] is True


class TestListModels:
    def test_models_are_returned(self, fake_ollama):
        assert ollama_client.list_models() == [{"name": "codellama:7b"}]


class TestVisionPassthrough:
    """Images ride along on the message; the app must not strip or reshape them."""

    def test_images_reach_ollama_unchanged(self, fake_ollama):
        mod = importlib.reload(app_module)
        b64 = "iVBORw0KGgoAAAANSUhEUg=="  # stand-in for a real encoded image
        resp = mod.app.test_client().post(
            "/api/chat",
            json={
                "model": "llava:13b",
                "messages": [{"role": "user", "content": "what is this?", "images": [b64]}],
            },
        )
        assert resp.status_code == 200

        sent = fake_ollama.last_request
        assert sent["model"] == "llava:13b"
        assert sent["messages"][0]["images"] == [b64]
        assert sent["messages"][0]["content"] == "what is this?"

    def test_multiple_images_are_preserved_in_order(self, fake_ollama):
        mod = importlib.reload(app_module)
        imgs = ["aaa", "bbb", "ccc"]
        mod.app.test_client().post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "", "images": imgs}]},
        )
        assert fake_ollama.last_request["messages"][0]["images"] == imgs

    def test_history_with_a_past_image_still_streams(self, fake_ollama):
        """A follow-up turn re-sends the earlier image; that must not break."""
        mod = importlib.reload(app_module)
        resp = mod.app.test_client().post(
            "/api/chat",
            json={
                "messages": [
                    {"role": "user", "content": "what is this?", "images": ["zzz"]},
                    {"role": "assistant", "content": "a cat"},
                    {"role": "user", "content": "what colour?"},
                ]
            },
        )
        assert resp.status_code == 200
        assert len(fake_ollama.last_request["messages"]) == 3
