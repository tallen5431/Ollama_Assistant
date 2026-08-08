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
import requests

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
        resp.get_data()          # the turn only reaches Ollama once it is read

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
        ).get_data()
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
        resp.get_data()
        assert len(fake_ollama.last_request["messages"]) == 3


class _ThinkRejectingHandler(BaseHTTPRequestHandler):
    """Ollama before 0.9, which rejects the unknown "think" key outright."""

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        payload = json.loads(body or b"{}")
        self.server.requests.append(payload)
        if "think" in payload and self.server.reject_think:
            out = json.dumps({"error": 'unknown field "think"'}).encode()
            self.send_response(400)
        else:
            out = json.dumps({"message": {"content": "Q: widget 5"}}).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args):
        pass


@pytest.fixture
def json_ollama(monkeypatch):
    """A stand-in Ollama that answers with plain JSON, as /api/chat does when
    stream=false. The NDJSON fixture above cannot serve a non-streaming call."""
    server = HTTPServer(("127.0.0.1", 0), _ThinkRejectingHandler)
    server.requests = []
    server.reject_think = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("OLLAMA_HOST", f"http://127.0.0.1:{server.server_port}")
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


class TestThinkCompatibility:
    """think=False is worth asking for, but not worth failing over.

    Ollama only learned the field in 0.9. Sending it unconditionally to an
    older build would turn every planner call into a hard failure — which the
    planner reports as "could not plan a search".
    """

    def test_a_rejected_think_field_retries_without_it(self, json_ollama):
        reply = ollama_client.chat("m", [{"role": "user", "content": "hi"}], think=False)
        assert reply == "Q: widget 5"
        sent = json_ollama.requests
        assert len(sent) == 2, "expected one rejected call then one retry"
        assert "think" in sent[0] and "think" not in sent[1]

    def test_a_call_that_never_asked_to_think_is_not_retried(self, json_ollama):
        """Only the think field earns a second attempt; other errors stand."""
        ollama_client.chat("m", [{"role": "user", "content": "hi"}])
        assert len(json_ollama.requests) == 1

    def test_a_modern_ollama_gets_think_and_only_one_call(self, json_ollama):
        json_ollama.reject_think = False
        assert ollama_client.chat(
            "m", [{"role": "user", "content": "hi"}], think=False
        ) == "Q: widget 5"
        assert len(json_ollama.requests) == 1
        assert json_ollama.requests[0]["think"] is False

    def test_think_is_absent_when_not_asked_for(self, json_ollama):
        ollama_client.chat("m", [{"role": "user", "content": "hi"}])
        assert "think" not in json_ollama.requests[0]


class TestTimeouts:
    """A generous read budget must not become the connect budget.

    The desktop holding the models sleeps. A sleeping tailnet peer drops the
    SYN rather than refusing it, so a scalar timeout meant the connection
    attempt hung for the whole 300 s before the user was told anything — and a
    web-grounded turn paid it on every call it made.
    """

    def test_connect_and_read_are_separate(self, monkeypatch):
        seen = {}

        def capture(url, **kwargs):
            seen["timeout"] = kwargs.get("timeout")
            raise requests.RequestException("nope")

        monkeypatch.setattr(ollama_client._SESSION, "get", capture)
        ollama_client.invalidate_models_cache()
        with pytest.raises(ValueError):
            ollama_client.get_ollama("/api/tags")
        assert isinstance(seen["timeout"], tuple), "a scalar is also the connect timeout"
        connect, read = seen["timeout"]
        assert connect <= 15, "a sleeping host must be given up on quickly"
        assert read >= 60, "a 30b model legitimately takes minutes to answer"

    def test_the_streaming_call_gets_them_too(self, monkeypatch):
        seen = {}

        def capture(url, **kwargs):
            seen["timeout"] = kwargs.get("timeout")
            raise requests.RequestException("nope")

        monkeypatch.setattr(ollama_client._SESSION, "post", capture)
        with pytest.raises(ValueError):
            list(ollama_client.chat_stream("m", [{"role": "user", "content": "hi"}]))
        assert isinstance(seen["timeout"], tuple)

    def test_the_connect_timeout_is_configurable(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_CONNECT_TIMEOUT", "9")
        import config
        assert config.get_timeouts()[0] == 9

    def test_a_nonsense_setting_falls_back(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_CONNECT_TIMEOUT", "soon")
        import config
        assert config.get_timeouts()[0] == 5


class TestThinkRetryIsNarrow:
    def test_an_unreachable_host_is_not_dialled_twice(self, monkeypatch):
        """Transport failure and a rejected field are both ValueError.

        Retrying on either doubled an already long connect timeout, on the one
        path that runs before the user sees a token.
        """
        calls = []

        def capture(url, **kwargs):
            calls.append(1)
            raise requests.RequestException("timed out")

        monkeypatch.setattr(ollama_client._SESSION, "post", capture)
        with pytest.raises(ValueError):
            ollama_client.chat("m", [{"role": "user", "content": "hi"}], think=False)
        assert len(calls) == 1, f"dialled a dead host {len(calls)} times"

    def test_an_unrelated_http_error_is_not_retried(self, monkeypatch):
        calls = []

        class R:
            ok = False
            status_code = 404
            text = '{"error": "model \'nope\' not found"}'

        monkeypatch.setattr(ollama_client._SESSION, "post", lambda url, **kw: calls.append(1) or R())
        with pytest.raises(ValueError):
            ollama_client.chat("nope", [{"role": "user", "content": "hi"}], think=False)
        assert len(calls) == 1


class TestModelListCaching:
    """A single chat turn asked Ollama for its model list up to five times.

    Three fields of /api/models, then vision routing from inside the streaming
    generator — each its own round trip to a desktop that may be asleep.
    """

    def test_repeated_calls_hit_the_network_once(self, fake_ollama):
        calls = []
        real = ollama_client.get_ollama
        ollama_client.invalidate_models_cache()

        def counting(path, timeout=None):
            calls.append(path)
            return real(path, timeout=timeout)

        try:
            ollama_client.get_ollama = counting
            for _ in range(5):
                assert ollama_client.list_models() == [{"name": "codellama:7b"}]
        finally:
            ollama_client.get_ollama = real
        assert len(calls) == 1, f"asked Ollama {len(calls)} times for one turn"

    def test_fresh_forces_a_call(self, fake_ollama):
        calls = []
        real = ollama_client.get_ollama
        ollama_client.invalidate_models_cache()

        def counting(path, timeout=None):
            calls.append(path)
            return real(path, timeout=timeout)

        try:
            ollama_client.get_ollama = counting
            ollama_client.list_models()
            ollama_client.list_models(fresh=True)
        finally:
            ollama_client.get_ollama = real
        assert len(calls) == 2

    def test_a_failure_is_not_cached(self, monkeypatch):
        """The desktop waking up should show models on the next attempt."""
        monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:1")
        ollama_client.invalidate_models_cache()
        for _ in range(2):
            with pytest.raises(ValueError):
                ollama_client.list_models()

    def test_the_cache_hands_back_shared_dicts(self, fake_ollama):
        """Documenting the hazard, not endorsing it.

        A memoised list returns the same dict objects to every caller, so any
        caller that tags them in place writes into what the next reader sees.
        That is why /api/models copies — see the test below, which is the one
        that would fail if it stopped.
        """
        ollama_client.invalidate_models_cache()
        ollama_client.list_models()[0]["poked"] = True
        assert "poked" in ollama_client.list_models()[0]

    def test_api_models_does_not_tag_the_shared_cache(self, fake_ollama):
        """The actual fix. Tagging in place wrote vision/ocr into the cache
        that every other caller — including vision routing — then read."""
        import importlib
        import app as app_module
        mod = importlib.reload(app_module)
        ollama_client.invalidate_models_cache()

        mod.app.test_client().get("/api/models")
        cached = ollama_client.list_models()
        assert cached, "nothing was cached"
        assert "vision" not in cached[0], (
            "/api/models tagged the memoised dicts in place"
        )
        assert "ocr" not in cached[0]


class TestKeepAlive:
    def test_it_is_absent_unless_asked_for(self, json_ollama):
        ollama_client.chat("m", [{"role": "user", "content": "hi"}])
        assert "keep_alive" not in json_ollama.requests[-1]

    def test_a_helper_can_unload_itself(self, json_ollama):
        ollama_client.chat("m", [{"role": "user", "content": "hi"}], keep_alive="0")
        assert json_ollama.requests[-1]["keep_alive"] == "0"

    def test_the_streaming_call_takes_it_too(self, fake_ollama):
        list(ollama_client.chat_stream("m", [{"role": "user", "content": "hi"}],
                                       keep_alive="30m"))
        assert fake_ollama.last_request["keep_alive"] == "30m"
