"""Markdown rendering, exercised the way the browser actually runs it.

The renderer lives inside chat_ui._PAGE as JavaScript, so these extract the
shipped source and run it under node. Skipped where node isn't installed.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest

import chat_ui

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")

SENTINEL = chr(1)   # the code-span placeholder, built without typing it literally


def render(text: str) -> str:
    """Run the shipped renderMarkdown over one input and return its HTML."""
    page_js = re.search(r"<script>(.*?)</script>", chat_ui.render_page("t"), re.S).group(1)
    start = page_js.index("const MD_SENTINEL_RE")
    end = page_js.index("// Painted once")
    script = page_js[start:end] + f"\nprocess.stdout.write(renderMarkdown({json.dumps(text)}));"
    return subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, check=True
    ).stdout


class TestEscaping:
    @pytest.mark.parametrize(
        "payload",
        [
            "<script>alert(1)</script>",
            "<img src=x onerror=alert(1)>",
            '<a href="javascript:alert(1)">x</a>',
            "<iframe src=//evil.example></iframe>",
            "<svg/onload=alert(1)>",
        ],
    )
    def test_markup_in_model_output_is_inert(self, payload):
        out = render(payload)
        for tag in ("<script", "<img", "<iframe", "<svg"):
            assert tag not in out.lower()
        assert "&lt;" in out

    def test_a_forged_code_span_placeholder_cannot_inject(self):
        """The sentinel is stripped on the way in, so it can't forge a slot."""
        out = render(f"literal {SENTINEL}0{SENTINEL} then `real`")
        assert "<code>real</code>" in out
        assert out.count("<code>") == 1


class TestCodeSpans:
    def test_markdown_inside_a_code_span_is_left_alone(self):
        """`*args` must show its asterisks rather than become italics."""
        out = render("pass `*args` and `**kwargs` to it")
        assert "<code>*args</code>" in out
        assert "<code>**kwargs</code>" in out
        assert "<em>" not in out and "<strong>" not in out

    def test_two_code_spans_do_not_leak_into_each_other(self):
        out = render("run `SELECT * FROM a` then `SELECT * FROM b`")
        assert out.count("<code>") == 2
        assert "<em>" not in out

    def test_a_link_inside_a_code_span_stays_literal(self):
        out = render("see `[a](https://x.com)` literally")
        assert "<a " not in out
        assert "[a](https://x.com)" in out


class TestEmphasisFlanking:
    @pytest.mark.parametrize(
        "text",
        [
            "Run SELECT * FROM users, then SELECT * FROM logs.",
            "Delete *.log and *.tmp",
            "3 * 4 * 5 = 60",
        ],
    )
    def test_loose_asterisks_in_prose_survive(self, text):
        """Bare asterisks are routine in a dev chat and must not vanish."""
        out = render(text)
        assert "<em>" not in out, out
        assert out.count("*") == text.count("*"), out

    def test_real_emphasis_still_renders(self):
        out = render("this is *important* and **critical**")
        assert "<em>important</em>" in out
        assert "<strong>critical</strong>" in out


class TestBlocks:
    def test_fenced_code_keeps_its_language_and_a_copy_button(self):
        out = render("Run:\n\n```bash\nsudo systemctl restart x\n```")
        assert 'data-lang="bash"' in out
        assert "sudo systemctl restart x" in out
        assert 'class="copy"' in out

    def test_lists_and_headings(self):
        assert "<ul><li>one</li><li>two</li></ul>" in render("- one\n- two")
        assert "<ol>" in render("1. one\n2. two")
        assert "<h4>Result</h4>" in render("## Result")

    def test_links_get_safe_attributes(self):
        out = render("see [docs](https://ollama.com/docs)")
        assert 'rel="noopener noreferrer"' in out
        assert 'href="https://ollama.com/docs"' in out
