"""Tests for web grounding.

The address guard is the important part: a URL reaching this app can come from
a pasted message, a search result, or a redirect chosen by a remote server, and
none of those are trusted to point somewhere sensible.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import web

PAGE = """<!doctype html>
<html><head><title>  Example   Page </title>
<style>body { color: red; }</style>
<script>var tracker = "should not appear";</script>
</head>
<body>
  <nav>Home About Contact</nav>
  <header>Site banner</header>
  <h1>Main heading</h1>
  <p>First paragraph of real content.</p>
  <p>Second paragraph &amp; an entity.</p>
  <aside>Related links</aside>
  <footer>Copyright notice</footer>
</body></html>"""


class TestAddressGuard:
    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1:11434/api/tags",
            "http://localhost:8070/",
            "http://192.168.1.199/",
            "http://10.0.0.1/",
            "http://172.16.0.1/",
            "http://100.98.112.1:11434/",     # Tailscale CGNAT range
            "http://169.254.169.254/",        # cloud metadata
            "http://[::1]/",
            "http://0.0.0.0/",
        ],
    )
    def test_non_public_addresses_are_refused(self, url):
        with pytest.raises(web.WebError, match="private or local"):
            web.check_url(url)

    @pytest.mark.parametrize("url", ["file:///etc/passwd", "gopher://x/", "ftp://h/f", "javascript:x"])
    def test_non_http_schemes_are_refused(self, url):
        with pytest.raises(web.WebError, match="http"):
            web.check_url(url)

    def test_public_address_passes(self):
        assert web.check_url("http://8.8.8.8/") == "http://8.8.8.8/"

    def test_unresolvable_host_is_refused(self):
        with pytest.raises(web.WebError):
            web.check_url("http://no-such-host.invalid/")

    def test_missing_host_is_refused(self):
        with pytest.raises(web.WebError):
            web.check_url("http:///just-a-path")


class TestFindUrls:
    def test_extracts_and_dedupes(self):
        text = "see https://a.example/x and https://b.example/y and https://a.example/x again"
        assert web.find_urls(text) == ["https://a.example/x", "https://b.example/y"]

    def test_strips_trailing_punctuation(self):
        assert web.find_urls("look at https://a.example/page.") == ["https://a.example/page"]

    def test_ignores_bare_words(self):
        assert web.find_urls("no links here, just example.com text") == []

    def test_respects_the_limit(self):
        text = " ".join(f"https://s{i}.example/" for i in range(10))
        assert len(web.find_urls(text, limit=3)) == 3

    def test_handles_empty(self):
        assert web.find_urls("") == []
        assert web.find_urls(None) == []


class TestHtmlToText:
    def test_drops_scripts_styles_and_chrome(self):
        out = web.html_to_text(PAGE)
        assert "should not appear" not in out["text"]
        assert "color: red" not in out["text"]
        for chrome in ("Home About Contact", "Site banner", "Related links", "Copyright notice"):
            assert chrome not in out["text"]

    def test_keeps_real_content(self):
        text = web.html_to_text(PAGE)["text"]
        assert "Main heading" in text
        assert "First paragraph of real content." in text
        assert "Second paragraph & an entity." in text   # entity decoded

    def test_normalises_the_title(self):
        assert web.html_to_text(PAGE)["title"] == "Example Page"

    def test_survives_malformed_markup(self):
        out = web.html_to_text("<p>text<div><span>more")
        assert "text" in out["text"] and "more" in out["text"]

    def test_paragraphs_do_not_run_together(self):
        text = web.html_to_text("<p>one</p><p>two</p>")["text"]
        assert "onetwo" not in text


class TestBuildContext:
    def test_fences_and_labels_the_material(self):
        ctx = web.build_context([{"url": "https://a.example/", "title": "A", "text": "body"}])
        assert "BEGIN WEB RESULTS" in ctx and "END WEB RESULTS" in ctx
        assert "not instructions" in ctx        # injection warning present
        assert "[1] A" in ctx and "https://a.example/" in ctx and "body" in ctx

    def test_numbers_documents_for_citation(self):
        ctx = web.build_context([
            {"url": "https://a.example/", "title": "A", "text": "x"},
            {"url": "https://b.example/", "title": "B", "text": "y"},
        ])
        assert "[1] A" in ctx and "[2] B" in ctx


class TestWithContext:
    def test_inserts_before_the_last_user_turn(self):
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
        out = web.with_context(msgs, "CTX")
        assert [m["role"] for m in out] == ["user", "assistant", "system", "user"]
        assert out[2]["content"] == "CTX"
        assert out[3]["content"] == "second"

    def test_does_not_mutate_the_original(self):
        msgs = [{"role": "user", "content": "hi"}]
        web.with_context(msgs, "CTX")
        assert len(msgs) == 1

    def test_handles_no_user_turn(self):
        out = web.with_context([{"role": "system", "content": "s"}], "CTX")
        assert out[-1]["content"] == "CTX"


class TestLastUserText:
    def test_returns_the_most_recent_user_turn(self):
        msgs = [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "newest"},
        ]
        assert web.last_user_text(msgs) == "newest"

    def test_empty_when_absent(self):
        assert web.last_user_text([{"role": "assistant", "content": "x"}]) == ""
        assert web.last_user_text([]) == ""

    def test_tolerates_non_dict_entries(self):
        assert web.last_user_text(["junk", {"role": "user", "content": "ok"}]) == "ok"


class _Handler(BaseHTTPRequestHandler):
    """Serves the fixtures the fetch tests need."""

    def do_GET(self):
        if self.path.startswith("/redirect-to-localhost"):
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:11434/api/tags")
            self.end_headers()
            return
        if self.path.startswith("/binary"):
            body = b"\x00\x01\x02"
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/huge"):
            body = ("<p>" + "word " * 20000 + "</p>").encode()
        else:
            body = PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def site(monkeypatch):
    """A local HTTP server, with the address guard stubbed to allow it."""
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    # The guard is exercised directly elsewhere; here we need it out of the way
    # so the fetch/extract path can be tested against a real socket.
    monkeypatch.setattr(web, "_is_public", lambda host: True)
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


class TestFetch:
    def test_returns_title_and_clean_text(self, site):
        doc = web.fetch(site + "/page")
        assert doc["title"] == "Example Page"
        assert "First paragraph of real content." in doc["text"]
        assert "should not appear" not in doc["text"]

    def test_truncates_a_long_page(self, site, monkeypatch):
        monkeypatch.setenv("WEB_MAX_CHARS", "500")
        doc = web.fetch(site + "/huge")
        assert len(doc["text"]) < 700
        assert doc["text"].endswith("…[truncated]")

    def test_rejects_non_text_content(self, site):
        with pytest.raises(web.WebError, match="no text to read"):
            web.fetch(site + "/binary")

    def test_disabled_by_the_kill_switch(self, monkeypatch):
        monkeypatch.setenv("WEB_ENABLED", "0")
        with pytest.raises(web.WebError, match="disabled"):
            web.fetch("https://example.com/")
        with pytest.raises(web.WebError, match="disabled"):
            web.search("anything")


class TestRedirectGuard:
    def test_a_redirect_into_localhost_is_refused(self, monkeypatch):
        """A redirect target is chosen by the remote server, so it is re-checked.

        Without the per-hop check, a public URL could bounce the fetch into
        127.0.0.1 or the LAN and the guard on the original URL would be moot.
        """
        server = HTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()

        # Allow only the first hop, so the redirect meets the real rule.
        calls = {"n": 0}

        def guard(_host):
            calls["n"] += 1
            return calls["n"] == 1

        monkeypatch.setattr(web, "_is_public", guard)
        try:
            with pytest.raises(web.WebError, match="private or local"):
                web.fetch(f"http://127.0.0.1:{server.server_port}/redirect-to-localhost")
            assert calls["n"] >= 2, "the redirect target was never checked"
        finally:
            server.shutdown()
            server.server_close()


def planner(reply):
    """Stub the planner model with a fixed reply."""
    return lambda *a, **k: reply


ASK = [{"role": "user", "content": "what changed in the newest ollama?"}]


class TestPlanSearches:
    def test_parses_multiple_queries(self, monkeypatch):
        monkeypatch.setattr(
            "ollama_client.chat",
            planner("Q: ollama latest release notes\nQ: ollama streaming tool calls"),
        )
        assert web.plan_searches(ASK, "m") == [
            "ollama latest release notes",
            "ollama streaming tool calls",
        ]

    def test_none_means_no_search(self, monkeypatch):
        monkeypatch.setattr("ollama_client.chat", planner("NONE"))
        assert web.plan_searches([{"role": "user", "content": "write a haiku"}], "m") == []

    def test_planner_failure_is_distinguishable_from_no_search(self, monkeypatch):
        """None means "couldn't plan"; [] means "decided not to" — not the same."""
        def boom(*a, **k):
            raise ValueError("model not found")
        monkeypatch.setattr("ollama_client.chat", boom)
        assert web.plan_searches(ASK, "m") is None

    def test_legacy_search_prefix_still_parses(self, monkeypatch):
        monkeypatch.setattr("ollama_client.chat", planner("SEARCH: ollama release notes"))
        assert web.plan_searches(ASK, "m") == ["ollama release notes"]

    def test_tolerates_a_chatty_model(self, monkeypatch):
        monkeypatch.setattr(
            "ollama_client.chat",
            planner("Sure, here are some queries!\nQ: weather boston\nHope that helps"),
        )
        assert web.plan_searches(ASK, "m") == ["weather boston"]

    def test_strips_quotes_bullets_and_trailing_stops(self, monkeypatch):
        monkeypatch.setattr(
            "ollama_client.chat",
            planner('Q: - "python 3.13 release date".\nQ: 1. python 3.13 changelog'),
        )
        assert web.plan_searches(ASK, "m") == [
            "python 3.13 release date",
            "python 3.13 changelog",
        ]

    def test_duplicate_queries_are_dropped(self, monkeypatch):
        monkeypatch.setattr(
            "ollama_client.chat", planner("Q: ollama news\nQ: Ollama News\nQ: ollama blog")
        )
        assert web.plan_searches(ASK, "m") == ["ollama news", "ollama blog"]

    def test_respects_the_query_cap(self, monkeypatch):
        monkeypatch.setattr(
            "ollama_client.chat", planner("\n".join(f"Q: query {i}" for i in range(9)))
        )
        assert len(web.plan_searches(ASK, "m", max_queries=3)) == 3

    def test_unparseable_reply_yields_no_queries(self, monkeypatch):
        monkeypatch.setattr("ollama_client.chat", planner("I think you should look it up"))
        assert web.plan_searches(ASK, "m") == []

    def test_echoed_none_is_not_a_query(self, monkeypatch):
        monkeypatch.setattr("ollama_client.chat", planner("Q: NONE"))
        assert web.plan_searches(ASK, "m") == []

    def test_absurdly_long_query_is_dropped(self, monkeypatch):
        monkeypatch.setattr("ollama_client.chat", planner("Q: " + "x" * 500))
        assert web.plan_searches(ASK, "m") == []

    def test_no_user_turn_means_no_search(self):
        assert web.plan_searches([{"role": "assistant", "content": "x"}], "m") == []

    def test_dedicated_planner_model_is_used_when_set(self, monkeypatch):
        seen = {}

        def capture(model, messages, options=None):
            seen["model"] = model
            seen["options"] = options
            return "Q: something"

        monkeypatch.setattr("ollama_client.chat", capture)
        monkeypatch.setenv("WEB_PLANNER_MODEL", "qwen2.5-coder:0.5b")
        web.plan_searches(ASK, "qwen3-coder:30b")
        assert seen["model"] == "qwen2.5-coder:0.5b"
        # Planning is a routing decision: deterministic and short.
        assert seen["options"]["temperature"] == 0
        assert seen["options"]["num_predict"] <= 128

    def test_falls_back_to_the_answering_model(self, monkeypatch):
        seen = {}

        def capture(model, messages, options=None):
            seen["model"] = model
            return "NONE"

        monkeypatch.setattr("ollama_client.chat", capture)
        monkeypatch.delenv("WEB_PLANNER_MODEL", raising=False)
        web.plan_searches(ASK, "llama3.1:8b")
        assert seen["model"] == "llama3.1:8b"


class TestPlannerInput:
    def test_includes_recent_context_for_follow_ups(self):
        msgs = [
            {"role": "user", "content": "tell me about qwen3 coder"},
            {"role": "assistant", "content": "It is a code model in 30b and 14b sizes."},
            {"role": "user", "content": "what about the 14b one?"},
        ]
        text = web.planner_input(msgs)
        assert "qwen3 coder" in text and "what about the 14b one?" in text

    def test_trims_long_turns(self):
        msgs = [{"role": "user", "content": "x" * 5000}]
        assert len(web.planner_input(msgs)) <= 700

    def test_empty_conversation(self):
        assert web.planner_input([]) == ""


class TestMergeResults:
    def test_interleaves_so_every_query_contributes(self):
        groups = [
            [{"url": "a1"}, {"url": "a2"}, {"url": "a3"}],
            [{"url": "b1"}, {"url": "b2"}],
        ]
        assert [r["url"] for r in web.merge_results(groups, 4)] == ["a1", "b1", "a2", "b2"]

    def test_drops_duplicate_urls_across_queries(self):
        groups = [[{"url": "same"}, {"url": "a2"}], [{"url": "same"}, {"url": "b2"}]]
        assert [r["url"] for r in web.merge_results(groups, 4)] == ["same", "a2", "b2"]

    def test_respects_the_limit(self):
        groups = [[{"url": f"a{i}"} for i in range(5)], [{"url": f"b{i}"} for i in range(5)]]
        assert len(web.merge_results(groups, 3)) == 3

    def test_handles_empty_input(self):
        assert web.merge_results([], 3) == []
        assert web.merge_results([[], []], 3) == []


class TestSearchParsing:
    def test_duckduckgo_results_are_unwrapped(self, monkeypatch):
        html = (
            '<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Freal.example%2Fa">'
            "First result</a>"
            '<a class="result__a" href="https://direct.example/b">Second result</a>'
            '<a class="other" href="https://ignored.example/">Not a result</a>'
        )
        parser = web._DuckLinks()
        parser.feed(html)
        urls = [r["url"] for r in parser.results]
        assert urls == ["https://real.example/a", "https://direct.example/b"]
        assert parser.results[0]["title"] == "First result"

    def test_searxng_is_preferred_when_configured(self, monkeypatch):
        monkeypatch.setenv("SEARXNG_URL", "http://127.0.0.1:8888")
        captured = {}

        def fake_get(url, **kwargs):
            captured["url"] = url
            captured["trusted"] = kwargs.get("trusted")
            payload = json.dumps({"results": [{"url": "https://a.example/", "title": "A"}]})

            class R:
                ok = True
                encoding = "utf-8"
                headers = {"Content-Type": "application/json"}
                def iter_content(self, size):
                    yield payload.encode()
                def __enter__(self):
                    return self
                def __exit__(self, *a):
                    return False
            return R()

        monkeypatch.setattr(web, "_get", fake_get)
        assert web.search("test query") == [{"url": "https://a.example/", "title": "A"}]
        assert captured["url"].startswith("http://127.0.0.1:8888/search")
        # A self-hosted SearXNG is normally on localhost, so it must be exempt
        # from the public-address rule — but only because the operator set it.
        assert captured["trusted"] is True

    def test_blank_query_returns_nothing(self):
        assert web.search("   ") == []


class TestReviewRegressions:
    """Cases found by the code review — each failed before its fix."""

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1:11434\\@example.com/",   # urlparse vs urllib3 split
            "http://example.com@127.0.0.1/",           # userinfo confusion
            "http://user:pw@example.com/",             # embedded credentials
            "http://exa\rmple.com/",                   # control character
            "http://exa\nmple.com/",
            "  ",
        ],
    )
    def test_parser_differential_urls_are_refused(self, url):
        """The guard must check the host requests will actually dial."""
        with pytest.raises(web.WebError):
            web.check_url(url)

    def test_fetch_only_ever_raises_weberror(self, monkeypatch):
        """Callers catch WebError alone; a raw requests error kills the turn."""
        import requests

        def dying_get(*a, **k):
            raise requests.exceptions.ChunkedEncodingError("connection died mid-body")

        monkeypatch.setattr(web, "_is_public", lambda h: True)
        monkeypatch.setattr(web._SESSION, "get", dying_get)
        with pytest.raises(web.WebError):
            web.fetch("http://example.com/")

    def test_a_redirect_is_rechecked_even_from_a_trusted_start(self, monkeypatch):
        """trusted covers the configured endpoint, never where it sends us."""
        calls = []

        class Redirect:
            status_code = 302
            headers = {"Location": "http://127.0.0.1:11434/api/tags"}
            def close(self):
                pass

        monkeypatch.setattr(web._SESSION, "get", lambda url, **k: (calls.append(url), Redirect())[1])
        with pytest.raises(web.WebError, match="private or local"):
            web._get("http://127.0.0.1:8888/search", trusted=True)
        assert len(calls) == 1, "must not follow the redirect"

    def test_deadline_expires(self):
        """A wall-clock budget, so a slow-drip server can't hold a worker."""
        import time as _t

        budget = web._Deadline(5)
        assert not budget.expired()
        budget.started -= 6          # pretend six seconds passed
        assert budget.expired()
        # Still returns a positive timeout so the next call fails rather than hangs.
        assert budget.remaining() == 0.5

    def test_deadline_has_a_floor(self):
        """An absurdly small WEB_TIMEOUT must not make every fetch impossible."""
        assert web._Deadline(0.01).seconds == 1.0
