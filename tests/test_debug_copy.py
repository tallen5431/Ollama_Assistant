"""Copy for Claude: one reply, or a whole chat, as a plain-text debug report."""

from __future__ import annotations

import importlib
import json
import subprocess

import pytest

import app as app_module
import chat_ui
from conftest import page_script


@pytest.fixture
def client(monkeypatch):
    for key in ("CHAT_AUTH", "CHAT_AUTH_USER", "CHAT_AUTH_PASSWORD"):
        monkeypatch.delenv(key, raising=False)
    return importlib.reload(app_module).app.test_client()


class TestTheServerSaysWhatItIs:
    def test_health_carries_the_version_and_settings(self, client, monkeypatch):
        monkeypatch.setenv("WEB_PLANNER_MODEL", "qwen3.5:4b")
        monkeypatch.setenv("OLLAMA_NUM_CTX", "16384")
        out = client.get("/api/health").get_json()
        assert out["version"]
        assert out["settings"]["planner"] == "qwen3.5:4b"
        assert out["settings"]["num_ctx"] == 16384
        assert out["settings"]["search_backend"] in ("searxng", "duckduckgo")

    def test_nothing_secret_is_in_the_settings(self, client, monkeypatch):
        monkeypatch.setenv("CHAT_AUTH_PASSWORD", "hunter2")
        monkeypatch.setenv("SEARXNG_URL", "http://user:pass@searx.local")
        text = json.dumps(client.get("/api/health").get_json()["settings"])
        assert "hunter2" not in text and "pass@" not in text

    def test_a_checkout_without_git_says_unknown(self, monkeypatch):
        def broken(*a, **k):
            raise OSError("no git")
        monkeypatch.setattr(subprocess, "run", broken)
        assert app_module._app_version() == "unknown"


def _run(js: str) -> str:
    page = page_script(chat_ui.render_page("t"))
    start = page.index("      function withoutPosition(")
    end = page.index("      function debugHeader(")
    out = subprocess.run(["node", "-e", page[start:end] + "\n" + js],
                         capture_output=True, text=True, check=True, timeout=30)
    return out.stdout.strip()


class TestWhatIsCopied:
    def test_a_photos_position_is_removed(self):
        got = _run('console.log(withoutPosition("Photo: taken at 51.507220, -0.127500, '
                   'see https://www.openstreetmap.org/?mlat=51.5 ok"))')
        assert "51.50" not in got and "openstreetmap" not in got
        assert "[position removed]" in got and "[map link removed]" in got

    def test_ordinary_numbers_survive(self):
        got = _run('console.log(withoutPosition("28.4 BLEU, 8 heads, $19.99, version 1.2"))')
        assert got == "28.4 BLEU, 8 heads, $19.99, version 1.2"

    def test_long_text_is_clipped_and_says_so(self):
        got = _run('console.log(clip("x".repeat(1000), 600))')
        assert got.startswith("x" * 600) and got.endswith("[400 more characters]")


class TestThePage:
    def page(self):
        return chat_ui.render_page("t")

    def test_every_reply_gets_the_button(self):
        page = self.page()
        body = page[page.index("function addReplyCopy"):page.index("function addReplySpeak")]
        assert "addDebugCopy(view);" in body

    def test_it_offers_this_reply_or_the_whole_chat(self):
        page = self.page()
        assert '["This reply", () => debugHeader() + "\\n" + turnReport(view)]' in page
        assert '["Whole chat", chatReport]' in page

    def test_steps_and_sources_are_kept_as_data(self):
        page = self.page()
        assert "(view.stepLog = view.stepLog || []).push(entry);" in page
        assert "view.sourceList = list || [];" in page

    def test_the_report_has_the_version_settings_steps_and_answer(self):
        page = self.page()
        report = page[page.index("function debugHeader"):page.index("function chatReport")]
        for part in ("info.version", "set.planner", "set.num_ctx", '"--- Steps ---"',
                     '"--- Answer ---"', '"--- Sources ---"', "withoutPosition(entry.detail"):
            assert part in report, part
