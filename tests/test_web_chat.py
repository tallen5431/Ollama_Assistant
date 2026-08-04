"""End-to-end tests for a web-grounded chat turn.

Runs a stand-in Ollama and a stand-in website together, so the whole path is
covered: decide → search → fetch → inject context → stream, plus the progress
and source lines the UI renders.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import app as app_module
import web

ARTICLE = """<!doctype html><html><head><title>Widget 5 released</title></head>
<body><nav>menu</nav><p>Widget 5 shipped on Tuesday with faster startup.</p></body></html>"""


class _Ollama(BaseHTTPRequestHandler):
    """Answers the planner call and the streaming chat call."""

    planner_reply = "SEARCH: widget 5 release"

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        self.server.requests.append(payload)
        if payload.get("stream") is False:
            body = json.dumps({"message": {"content": self.server.planner_reply}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        for chunk in [
            {"message": {"content": "Widget 5"}, "done": False},
            {"message": {"content": " is out."}, "done": True, "eval_count": 4},
        ]:
            self.wfile.write(json.dumps(chunk).encode() + b"\n")
            self.wfile.flush()

    def log_message(self, *args):
        pass


class _Site(BaseHTTPRequestHandler):
    def do_GET(self):
        body = ARTICLE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def _serve(handler):
    server = HTTPServer(("127.0.0.1", 0), handler)
    server.requests = []
    server.planner_reply = _Ollama.planner_reply
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture
def rig(monkeypatch):
    """Fake Ollama + fake site, with the address guard relaxed for localhost."""
    ollama, site = _serve(_Ollama), _serve(_Site)
    monkeypatch.setenv("OLLAMA_HOST", f"http://127.0.0.1:{ollama.server_port}")
    monkeypatch.setenv("WEB_ENABLED", "1")
    monkeypatch.setattr(web, "_is_public", lambda host: True)
    site_url = f"http://127.0.0.1:{site.server_port}/article"
    monkeypatch.setattr(web, "search", lambda q, limit=3: [{"url": site_url, "title": "Widget 5"}])
    try:
        yield {"ollama": ollama, "site_url": site_url,
               "client": app_module.app.test_client()}
    finally:
        for s in (ollama, site):
            s.shutdown()
            s.server_close()


def lines(resp):
    return [json.loads(l) for l in resp.get_data(as_text=True).splitlines() if l.strip()]


class TestSearchGroundedTurn:
    def test_progress_sources_and_answer(self, rig):
        resp = rig["client"].post(
            "/api/chat", json={"messages": [{"role": "user", "content": "what is widget 5?"}], "web": True}
        )
        assert resp.status_code == 200
        out = lines(resp)

        statuses = [o["status"] for o in out if "status" in o]
        assert any("what to search for" in s for s in statuses)
        assert any(s.startswith("Searching:") for s in statuses)
        assert any("Reading" in s for s in statuses)

        sources = [o["sources"] for o in out if "sources" in o]
        assert sources and sources[0][0]["url"] == rig["site_url"]

        text = "".join(o.get("message", {}).get("content", "") for o in out)
        assert text == "Widget 5 is out."

    def test_page_text_is_injected_as_a_system_turn(self, rig):
        resp = rig["client"].post(
            "/api/chat", json={"messages": [{"role": "user", "content": "what is widget 5?"}], "web": True}
        )
        resp.get_data()   # the response streams lazily; consume it to run the turn
        chat_call = [r for r in rig["ollama"].requests if r.get("stream")][-1]
        roles = [m["role"] for m in chat_call["messages"]]
        assert roles == ["system", "user"], "context must sit before the last user turn"

        system = chat_call["messages"][0]["content"]
        assert "Widget 5 shipped on Tuesday" in system
        assert "not instructions" in system      # injection fence survived
        assert "menu" not in system              # nav chrome was stripped

    def test_context_widens_the_window(self, rig):
        resp = rig["client"].post(
            "/api/chat", json={"messages": [{"role": "user", "content": "what is widget 5?"}], "web": True}
        )
        resp.get_data()
        chat_call = [r for r in rig["ollama"].requests if r.get("stream")][-1]
        assert chat_call["options"]["num_ctx"] >= 8192


class TestMultiQueryPlanning:
    def test_every_planned_query_is_searched(self, rig, monkeypatch):
        rig["ollama"].planner_reply = "Q: widget 5 release\nQ: widget 5 changelog\nQ: widget 5 review"
        searched = []

        def fake_search(query, limit=3):
            searched.append(query)
            return [{"url": rig["site_url"] + f"?q={len(searched)}", "title": query}]

        monkeypatch.setattr(web, "search", fake_search)
        resp = rig["client"].post(
            "/api/chat", json={"messages": [{"role": "user", "content": "widget 5?"}], "web": True}
        )
        out = lines(resp)
        assert searched == ["widget 5 release", "widget 5 changelog", "widget 5 review"]
        # Each query contributed a distinct source, up to the document cap.
        sources = [o["sources"] for o in out if "sources" in o][0]
        assert len(sources) == 3
        assert len({s["url"] for s in sources}) == 3

    def test_a_failing_query_does_not_sink_the_others(self, rig, monkeypatch):
        rig["ollama"].planner_reply = "Q: good one\nQ: bad one"

        def fake_search(query, limit=3):
            if query == "bad one":
                raise web.WebError("search backend unreachable")
            return [{"url": rig["site_url"], "title": "Widget 5"}]

        monkeypatch.setattr(web, "search", fake_search)
        resp = rig["client"].post(
            "/api/chat", json={"messages": [{"role": "user", "content": "widget 5?"}], "web": True}
        )
        out = lines(resp)
        assert [o["sources"] for o in out if "sources" in o], "the working query still grounded the answer"
        assert "".join(o.get("message", {}).get("content", "") for o in out) == "Widget 5 is out."

    def test_all_queries_failing_is_reported_not_silent(self, rig, monkeypatch):
        rig["ollama"].planner_reply = "Q: anything"
        monkeypatch.setattr(
            web, "search", lambda q, limit=3: (_ for _ in ()).throw(web.WebError("backend down"))
        )
        resp = rig["client"].post(
            "/api/chat", json={"messages": [{"role": "user", "content": "widget 5?"}], "web": True}
        )
        assert any("backend down" in o.get("status", "") for o in lines(resp))

    def test_planner_failure_is_surfaced(self, rig, monkeypatch):
        monkeypatch.setattr(web, "plan_searches", lambda messages, model, **kw: None)
        resp = rig["client"].post(
            "/api/chat", json={"messages": [{"role": "user", "content": "widget 5?"}], "web": True}
        )
        assert any("Could not plan" in o.get("status", "") for o in lines(resp))

    def test_unreadable_results_are_reported(self, rig, monkeypatch):
        rig["ollama"].planner_reply = "Q: anything"
        monkeypatch.setattr(web, "fetch", lambda url: (_ for _ in ()).throw(web.WebError("403")))
        resp = rig["client"].post(
            "/api/chat", json={"messages": [{"role": "user", "content": "widget 5?"}], "web": True}
        )
        out = lines(resp)
        assert any("couldn't read any" in o.get("status", "") for o in out)
        # Still answers, ungrounded, rather than failing the turn.
        assert "".join(o.get("message", {}).get("content", "") for o in out) == "Widget 5 is out."


class TestUrlInMessage:
    def test_a_pasted_link_is_read_without_searching(self, rig):
        resp = rig["client"].post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": f"summarise {rig['site_url']}"}], "web": True},
        )
        out = lines(resp)
        statuses = [o["status"] for o in out if "status" in o]
        assert any("Reading" in s for s in statuses)
        assert not any("Searching" in s for s in statuses), "a given URL should not trigger a search"
        # Only the streaming call — the planner was never consulted.
        assert not [r for r in rig["ollama"].requests if r.get("stream") is False]


class TestWebOff:
    def test_nothing_is_fetched_when_the_toggle_is_off(self, rig):
        resp = rig["client"].post(
            "/api/chat", json={"messages": [{"role": "user", "content": "what is widget 5?"}]}
        )
        out = lines(resp)
        assert not [o for o in out if "status" in o or "sources" in o]
        chat_call = [r for r in rig["ollama"].requests if r.get("stream")][-1]
        assert [m["role"] for m in chat_call["messages"]] == ["user"]

    def test_kill_switch_overrides_the_toggle(self, rig, monkeypatch):
        monkeypatch.setenv("WEB_ENABLED", "0")
        resp = rig["client"].post(
            "/api/chat", json={"messages": [{"role": "user", "content": "hi"}], "web": True}
        )
        assert not [o for o in lines(resp) if "status" in o]


class TestPlannerDeclines:
    def test_no_search_when_the_model_says_none(self, rig):
        rig["ollama"].planner_reply = "NONE"
        resp = rig["client"].post(
            "/api/chat", json={"messages": [{"role": "user", "content": "write me a haiku"}], "web": True}
        )
        out = lines(resp)
        assert not any("Searching" in o.get("status", "") for o in out)
        assert not [o for o in out if "sources" in o]
        chat_call = [r for r in rig["ollama"].requests if r.get("stream")][-1]
        assert [m["role"] for m in chat_call["messages"]] == ["user"]


class TestRetrievalFailure:
    def test_a_dead_link_reports_and_still_answers(self, rig, monkeypatch):
        monkeypatch.setattr(web, "fetch", lambda url: (_ for _ in ()).throw(web.WebError("boom")))
        resp = rig["client"].post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": f"read {rig['site_url']}"}], "web": True},
        )
        out = lines(resp)
        assert any("boom" in o.get("status", "") for o in out)
        # The reply still arrives — failing to retrieve must not fail the chat.
        text = "".join(o.get("message", {}).get("content", "") for o in out)
        assert text == "Widget 5 is out."
        assert not [o for o in out if "error" in o]
