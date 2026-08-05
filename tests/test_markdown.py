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


class TestLineEndings:
    """A stray carriage return defeated every block pattern.

    "." does not match \r in a JS regex, and "$" without the m flag only
    matches end-of-input — so with CRLF output every marker rendered literally.
    """

    def test_crlf_is_handled_like_lf(self):
        assert render("## Title\r\nText\r\n- one\r\n- two") == render("## Title\nText\n- one\n- two")

    def test_a_lone_carriage_return_also_splits(self):
        out = render("## Title\rText")
        assert "<h4>Title</h4>" in out
        assert "##" not in out

    def test_crlf_inside_a_fence_keeps_the_code(self):
        out = render("```bash\r\nls -la\r\necho hi\r\n```")
        assert "ls -la" in out and "echo hi" in out
        assert "\r" not in out


class TestFences:
    """Every one of these swallowed or dropped part of a reply."""

    def test_a_fence_that_opens_and_closes_on_one_line_is_a_code_block(self):
        out = render("To install, run:\n\n```pip install foo```\n\nThen restart.")
        assert "<code>pip install foo</code>" in out
        # Not a language badge — that was the command being painted as one.
        assert "data-lang" not in out
        # And the rest of the reply survived rather than being eaten by an
        # unterminated fence.
        assert "Then restart." in out

    def test_text_after_the_closing_fence_is_kept(self):
        out = render("```\nls\n``` and then reboot")
        assert "<code>ls</code>" in out
        assert "and then reboot" in out

    def test_an_unterminated_fence_still_renders_as_code(self):
        """A cut-off reply is truncated, not malformed — showing it raw is worse."""
        out = render("```python\nx = 1\ny = 2")
        assert 'data-lang="python"' in out
        assert "x = 1" in out and "y = 2" in out


class TestHeadings:
    def test_a_hash_indented_into_code_is_not_a_heading(self):
        """CommonMark stops at 3 spaces; 4 is an indented code block."""
        out = render("    # install deps\n    pip install -r requirements.txt")
        assert "<h3>" not in out
        assert "# install deps" in out

    def test_up_to_three_spaces_still_makes_a_heading(self):
        assert "<h4>Result</h4>" in render("   ## Result")


class TestListContinuation:
    def test_an_indented_detail_line_continues_the_step(self):
        """Ending the list there poisoned every marker after it."""
        out = render("1. one\n2. two\n   extra detail\n3. three")
        assert out == "<ol><li>one</li><li>two extra detail</li><li>three</li></ol>", out

    def test_a_blank_line_still_ends_the_list(self):
        out = render("- one\n- two\n\nseparate paragraph")
        assert "<ul><li>one</li><li>two</li></ul>" in out
        assert "<p>separate paragraph</p>" in out


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
