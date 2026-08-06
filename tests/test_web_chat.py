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

    def test_an_unreadable_page_still_contributes_its_snippet(self, rig, monkeypatch):
        """Paywalls and JS-only pages are the normal case, not the exception.

        The snippet was already paid for by the search; dropping it meant that
        result contributed nothing whatsoever to the answer.
        """
        monkeypatch.setattr(web, "search", lambda q, limit=3: [{
            "url": "https://paywall.example/article",
            "title": "Widget 5 review",
            "snippet": "Widget 5 shipped on Tuesday with a new hinge.",
        }])
        monkeypatch.setattr(web, "fetch",
                            lambda url: (_ for _ in ()).throw(web.WebError("HTTP 403")))
        resp = rig["client"].post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "what is widget 5?"}], "web": True},
        )
        resp.get_data()

        chat_call = [r for r in rig["ollama"].requests if r.get("stream")][-1]
        system = chat_call["messages"][0]["content"]
        assert "new hinge" in system
        # Labelled, so the model doesn't quote a blurb as if it read the page.
        assert "search result summary" in system

    def test_the_model_is_told_todays_date(self, rig):
        resp = rig["client"].post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "what is widget 5?"}], "web": True},
        )
        resp.get_data()
        chat_call = [r for r in rig["ollama"].requests if r.get("stream")][-1]
        assert web.today() in chat_call["messages"][0]["content"]


class TestSearchedButFoundNothing:
    """A failed search must not read like a successful one.

    Without a note the model answers from training data with the same
    confidence as a sourced reply — the exact failure the web button exists to
    prevent.
    """

    def test_the_model_is_told_the_search_came_back_empty(self, rig, monkeypatch):
        monkeypatch.setattr(web, "search", lambda q, limit=3: [])
        resp = rig["client"].post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "what is widget 5?"}], "web": True},
        )
        resp.get_data()
        chat_call = [r for r in rig["ollama"].requests if r.get("stream")][-1]
        system = chat_call["messages"][0]["content"]
        assert "came back with nothing usable" in system
        assert "could not check it against a source" in system

    def test_an_unreadable_page_also_counts_as_empty(self, rig, monkeypatch):
        monkeypatch.setattr(web, "search", lambda q, limit=3: [
            {"url": "https://a.example/", "title": "A", "snippet": ""},
        ])
        monkeypatch.setattr(web, "fetch",
                            lambda url: (_ for _ in ()).throw(web.WebError("HTTP 403")))
        resp = rig["client"].post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "what is widget 5?"}], "web": True},
        )
        resp.get_data()
        chat_call = [r for r in rig["ollama"].requests if r.get("stream")][-1]
        assert "came back with nothing usable" in chat_call["messages"][0]["content"]

    def test_a_planner_that_declined_adds_no_note(self, rig):
        """"No search needed" is a decision, not a failure — stay quiet."""
        rig["ollama"].planner_reply = "NONE"
        resp = rig["client"].post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "say hello"}], "web": True},
        )
        resp.get_data()
        chat_call = [r for r in rig["ollama"].requests if r.get("stream")][-1]
        assert [m["role"] for m in chat_call["messages"]] == ["user"]

    def test_a_successful_search_adds_no_note(self, rig):
        resp = rig["client"].post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "what is widget 5?"}], "web": True},
        )
        resp.get_data()
        chat_call = [r for r in rig["ollama"].requests if r.get("stream")][-1]
        system = chat_call["messages"][0]["content"]
        assert "came back with nothing usable" not in system
        assert "Widget 5 shipped on Tuesday" in system


class TestFetchesDoNotWaitOnStragglers:
    """Six candidates are fetched to keep three, because paywalled and dead
    pages are the normal case. Waiting for the slowest of six after three had
    landed cost up to two full timeouts of dead air before the first token —
    and a pool capped at four ran them in two waves on top of that."""

    def test_the_turn_proceeds_once_enough_pages_have_landed(self):
        import time as clock
        calls = []

        def fetch(url):
            calls.append(url)
            if "slow" in url:
                clock.sleep(5)
            return {"url": url, "title": url, "text": "body"}

        started = clock.monotonic()
        docs, _ = app_module._run_all(
            fetch, ["fast1", "slow1", "fast2", "slow2", "fast3", "slow3"], enough=3)
        elapsed = clock.monotonic() - started

        # Three or four: a better-ranked straggler is given a bounded grace
        # period, and the caller keeps the best few by rank, so anything that
        # arrives in time improves the selection rather than delaying it.
        assert 3 <= len(docs) <= 4
        assert elapsed < 2, f"waited {elapsed:.1f}s for stragglers"

    def test_six_candidates_run_in_one_wave_not_two(self):
        """The pool was capped at four, so URLs 5 and 6 could not start until
        two others finished — a second full timeout on top of the first."""
        import time as clock

        def fetch(url):
            clock.sleep(2)
            return {"url": url, "title": url, "text": "body"}

        started = clock.monotonic()
        docs, _ = app_module._run_all(fetch, [f"u{i}" for i in range(6)])
        elapsed = clock.monotonic() - started

        assert len(docs) == 6
        assert elapsed < 3.5, f"took {elapsed:.1f}s — that is two waves, not one"

    def test_without_a_target_it_still_waits_for_everything(self):
        import time as clock
        docs, _ = app_module._run_all(
            lambda u: {"url": u, "title": u, "text": "b"}, ["a", "b", "c"])
        assert len(docs) == 3

    def test_a_better_ranked_straggler_is_not_lost_to_a_faster_one(self):
        """Selecting purely by completion order means the answer is grounded in
        the fastest pages, not the best-ranked ones the search chose."""
        import time as clock

        def fetch(url):
            clock.sleep(0.6 if url == "rank0" else 0.05)
            return {"url": url, "title": url, "text": "body"}

        started = clock.monotonic()
        docs, _ = app_module._run_all(
            fetch, ["rank0", "rank1", "rank2", "rank3"], enough=3)
        assert "rank0" in [d["url"] for d in docs], "the top-ranked page was dropped"
        assert clock.monotonic() - started < 2

    def test_a_hopeless_straggler_does_not_hold_the_turn(self):
        """The grace period is bounded; a genuinely slow page is still dropped."""
        import time as clock

        def fetch(url):
            clock.sleep(10 if url == "rank0" else 0.05)
            return {"url": url, "title": url, "text": "body"}

        started = clock.monotonic()
        docs, _ = app_module._run_all(
            fetch, ["rank0", "rank1", "rank2", "rank3"], enough=3)
        elapsed = clock.monotonic() - started
        assert "rank0" not in [d["url"] for d in docs]
        assert elapsed < 4, f"waited {elapsed:.1f}s on a hopeless straggler"

    def test_fewer_successes_than_wanted_does_not_hang(self):
        import time as clock
        def fetch(url):
            raise web.WebError("dead")
        started = clock.monotonic()
        docs, errors = app_module._run_all(fetch, ["a", "b"], enough=3)
        assert docs == [] and len(errors) == 2
        assert clock.monotonic() - started < 2


class TestFollowingLinks:
    """A wiki page answers half the question and links to the other half.

    Reading only the page you pasted meant the model said the page did not
    cover something the site covered one click away.
    """

    LINKED = """<html><head><title>Widget 5</title></head><body>
    <p>Widget 5 shipped Tuesday. Its <a href="/hinge">new hinge design</a> is the change,
    and <a href="https://elsewhere.example/x">an outside page</a> mentions it too.</p>
    </body></html>"""

    @pytest.fixture
    def linked_site(self, rig, monkeypatch):
        """Serve a page that links to a second page on the same host."""
        import web as web_module
        base = rig["site_url"].rsplit("/", 1)[0]

        def fake_fetch(url):
            if url.endswith("/hinge"):
                return {"url": url, "requested": url, "title": "Hinge design",
                        "text": "A titanium four-bar linkage rated to 200000 cycles.",
                        "links": []}
            parsed = web_module.html_to_text(self.LINKED, base_url=url)
            return {"url": url, "requested": url, "title": parsed["title"],
                    "text": parsed["text"], "links": parsed["links"]}

        monkeypatch.setattr(web, "fetch", fake_fetch)
        return rig, base

    def ask(self, rig, base, question):
        return rig["client"].post("/api/chat", json={
            "messages": [{"role": "user", "content": f"{question} {base}/article"}],
            "web": True,
        })

    def test_a_relevant_linked_page_is_read_too(self, linked_site, monkeypatch):
        rig, base = linked_site
        monkeypatch.setattr(web, "choose_links",
                            lambda q, links, model, max_links=2, **kw: links[:1])
        out = lines(self.ask(rig, base, "how strong is the hinge?"))
        chat_call = [r for r in rig["ollama"].requests if r.get("stream")][-1]
        system = chat_call["messages"][0]["content"]

        assert "titanium four-bar" in system, "the linked page was not read"
        assert any("Following" in o.get("status", "") for o in out)

    def test_only_same_site_links_are_offered(self, linked_site, monkeypatch):
        """A model picking an arbitrary outbound link is a much larger surface."""
        rig, base = linked_site
        offered = {}
        monkeypatch.setattr(web, "choose_links",
                            lambda q, links, model, max_links=2, **kw: offered.update(links=links) or [])
        self.ask(rig, base, "how strong is the hinge?").get_data()
        assert offered["links"], "nothing was offered at all"
        assert all("elsewhere.example" not in l["url"] for l in offered["links"])

    def test_following_nothing_still_answers_from_the_page(self, linked_site, monkeypatch):
        rig, base = linked_site
        monkeypatch.setattr(web, "choose_links", lambda *a, **k: [])
        out = lines(self.ask(rig, base, "when did it ship?"))
        text = "".join(o.get("message", {}).get("content", "") for o in out)
        assert text == "Widget 5 is out."
        assert not any("Following" in o.get("status", "") for o in out)

    def test_it_can_be_switched_off(self, linked_site, monkeypatch):
        rig, base = linked_site
        monkeypatch.setenv("WEB_FOLLOW_LINKS", "0")
        called = []
        monkeypatch.setattr(web, "choose_links", lambda *a, **k: called.append(1) or [])
        self.ask(rig, base, "how strong is the hinge?").get_data()
        assert not called

    def test_a_picker_failure_does_not_break_the_turn(self, linked_site, monkeypatch):
        rig, base = linked_site
        monkeypatch.setattr(web, "choose_links",
                            lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")))
        out = lines(self.ask(rig, base, "how strong is the hinge?"))
        assert not [o for o in out if "error" in o], "a picker failure broke the turn"
        assert "".join(o.get("message", {}).get("content", "") for o in out) == "Widget 5 is out."

    def test_the_link_map_reaches_the_model_even_without_following(self, linked_site, monkeypatch):
        """So it can say where something is covered rather than guessing."""
        rig, base = linked_site
        monkeypatch.setattr(web, "choose_links", lambda *a, **k: [])
        self.ask(rig, base, "when did it ship?").get_data()
        chat_call = [r for r in rig["ollama"].requests if r.get("stream")][-1]
        system = chat_call["messages"][0]["content"]
        assert "new hinge design" in system
        assert "not fetched" in system

    def test_only_one_hop(self, linked_site, monkeypatch):
        """Depth two explodes; the second page's links are not followed."""
        rig, base = linked_site
        rounds = []
        monkeypatch.setattr(
            web, "choose_links",
            lambda q, links, model, max_links=2, **kw: rounds.append(1) or links[:1])
        self.ask(rig, base, "how strong is the hinge?").get_data()
        assert len(rounds) == 1
