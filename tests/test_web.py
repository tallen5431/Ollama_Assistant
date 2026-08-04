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


class TestDecideQuery:
    def test_parses_a_search_decision(self, monkeypatch):
        monkeypatch.setattr("ollama_client.chat", lambda *a, **k: "SEARCH: ollama release notes")
        assert web.decide_query("m", [{"role": "user", "content": "what's new?"}]) == "ollama release notes"

    def test_none_means_no_search(self, monkeypatch):
        monkeypatch.setattr("ollama_client.chat", lambda *a, **k: "NONE")
        assert web.decide_query("m", [{"role": "user", "content": "hello"}]) is None

    def test_tolerates_a_chatty_model(self, monkeypatch):
        monkeypatch.setattr("ollama_client.chat", lambda *a, **k: "Sure!\nSEARCH: weather in Boston\nHope that helps")
        assert web.decide_query("m", [{"role": "user", "content": "weather?"}]) == "weather in Boston"

    def test_strips_quotes(self, monkeypatch):
        monkeypatch.setattr("ollama_client.chat", lambda *a, **k: 'SEARCH: "python 3.13 release"')
        assert web.decide_query("m", [{"role": "user", "content": "?"}]) == "python 3.13 release"

    def test_unparseable_reply_means_no_search(self, monkeypatch):
        monkeypatch.setattr("ollama_client.chat", lambda *a, **k: "I think maybe you should look it up")
        assert web.decide_query("m", [{"role": "user", "content": "?"}]) is None

    def test_a_failing_model_does_not_break_the_chat(self, monkeypatch):
        def boom(*a, **k):
            raise ValueError("ollama down")
        monkeypatch.setattr("ollama_client.chat", boom)
        assert web.decide_query("m", [{"role": "user", "content": "?"}]) is None

    def test_no_user_turn_means_no_search(self):
        assert web.decide_query("m", [{"role": "assistant", "content": "x"}]) is None


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

            class R:
                ok = True
                def json(self):
                    return {"results": [{"url": "https://a.example/", "title": "A"}]}
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
