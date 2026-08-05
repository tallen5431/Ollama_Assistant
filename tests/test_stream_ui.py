"""The browser's side of a streaming turn, run under node.

send() decides what reaches the screen, what reaches messages[], and what
reaches the store — and until now nothing tested it. Two bugs lived there: a
turn that produced no visible text was left on screen and in messages[] but
never written (so a phone and a desktop disagreed, and the next send posted two
user roles), and the reasoning stream was read from the /api/generate field
name so the panel never opened on an Ollama new enough to stream it natively.

These extract the shipped functions out of chat_ui._PAGE — so the backslash
doubling is exactly what a browser sees — and run them against a scripted
stream with only the DOM they touch stubbed.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest

import chat_ui

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def _page() -> str:
    return re.search(r"<script>(.*?)</script>", chat_ui.render_page("t"), re.S).group(1)


def _slice(page: str, start: str, end: str) -> str:
    at = page.index(start)
    return page[at:page.index(end, at)]


_DOM = r"""
function el(tag) {
  return {
    tag, children: [], className: "", textContent: "", innerHTML: "", value: "", hidden: false,
    classList: { add() {}, remove() {} },
    appendChild(c) { this.children.push(c); return c; },
    remove() { this.removed = true; },
    querySelector() { return el("div"); },
    querySelectorAll() { return []; },
    addEventListener() {}, focus() {},
  };
}
const document = { createElement: el, createTextNode: (t) => ({ text: t }),
                   createRange: () => ({ selectNodeContents() {} }) };
const window = { getSelection: () => ({ removeAllRanges() {}, addRange() {} }) };
const navigator = {};
const chatEl = el("div"), inputEl = el("textarea"), hintEl = el("div");
const sendBtn = el("button"), stopBtn = el("button"), modelEl = el("select");
const webEl = { checked: false };
let pendingImages = [], messages = [], busy = false, controller = null, pendingAutoSend = false;
let errors = [], saved = { rows: [], conversation: false };
function renderThumbs() {}
function autosize() {}
function clearPlaceholder() {}
function scrollDown() {}
function setStatus() {}
function markError(m) { errors.push(m); }
function paintMarkdown(e, raw) { e.textContent = raw; }
function fmtUsage() { return ""; }
function showSources() {}
async function ensureConversation() { saved.conversation = true; }
async function saveMessage(role, content) { saved.rows.push(role); }
function refreshConversations() {}
let lastView = null;
function addAssistant() {
  const root = el("div"); chatEl.children.push(root);
  lastView = { root, bubble: el("div"), status: el("div"), meta: el("div"),
               think: el("div"), thinkBody: el("div") };
  return lastView;
}
class AbortController { constructor() { this.signal = {}; } abort() {} }
class TextDecoder { decode(buf) { return buf ? Buffer.from(buf).toString("utf8") : ""; } }

let SCRIPT = [], ABORT_AFTER = -1, TYPE_DURING = null;
async function fetch() {
  let i = 0;
  return { ok: true, body: { getReader: () => ({
    async read() {
      // Whatever the user types while the reply is in flight.
      if (TYPE_DURING !== null) { inputEl.value = TYPE_DURING; TYPE_DURING = null; }
      if (ABORT_AFTER >= 0 && i >= ABORT_AFTER) { const e = new Error("abort"); e.name = "AbortError"; throw e; }
      if (i >= SCRIPT.length) return { done: true, value: undefined };
      return { done: false, value: new TextEncoder().encode(JSON.stringify(SCRIPT[i++]) + "\n") };
    },
    releaseLock() {},
  }) } };
}

async function run(script, abortAfter, typed, typeDuring) {
  SCRIPT = script; ABORT_AFTER = abortAfter; TYPE_DURING = typeDuring;
  messages = []; saved = { rows: [], conversation: false }; errors = [];
  chatEl.children = []; pendingImages = [];
  inputEl.value = typed;
  await send();
  for (let t = 0; t < 8; t++) await Promise.resolve();
  return {
    messageRoles: messages.map(m => m.role),
    savedRoles: saved.rows,
    input: inputEl.value,
    onScreen: chatEl.children.filter(c => !c.removed).length,
    errors, hint: hintEl.textContent,
    panel: lastView ? lastView.thinkBody.textContent : "",
    panelShown: lastView ? lastView.think.hidden === false : false,
    bubble: lastView ? lastView.bubble.textContent : "",
  };
}
"""


def drive(script, abort_after=-1, typed="what is 2+2?", type_during=None):
    """Run the shipped send() against one scripted stream; return what happened."""
    page = _page()
    js = "\n".join([
        _DOM,
        _slice(page, "      function addUser", "      // ---- Image attachments"),
        _slice(page, "      // Split assistant text", "      function fmtUsage"),
        _slice(page, "      async function send()", "      function stop()"),
        f"run({json.dumps(script)}, {abort_after}, {json.dumps(typed)}, "
        f"{json.dumps(type_during)})"
        ".then(r => process.stdout.write(JSON.stringify(r)));",
    ])
    out = subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


class TestTurnBookkeeping:
    """Screen, messages[] and the store must never disagree about a turn."""

    def test_a_normal_reply_is_kept_everywhere(self):
        r = drive([{"message": {"content": "4"}}, {"done": True}])
        assert r["messageRoles"] == ["user", "assistant"]
        assert r["savedRoles"] == ["user", "assistant"]
        assert r["input"] == ""

    def test_stop_after_some_text_keeps_the_partial(self):
        r = drive([{"message": {"content": "The ans"}}, {"message": {"content": "wer"}}], abort_after=2)
        assert r["messageRoles"] == ["user", "assistant"]
        assert r["savedRoles"] == ["user", "assistant"]

    def test_stop_during_reasoning_rolls_the_whole_turn_back(self):
        """For a reasoning model this is the entire scratchpad — the usual case.

        The turn used to stay on screen and in messages[] while never being
        written, so the next send posted two user roles in a row.
        """
        r = drive([{"message": {"content": "", "thinking": "thinking hard"}}], abort_after=1)
        assert r["messageRoles"] == []
        assert r["savedRoles"] == []
        assert r["onScreen"] == 0
        assert r["input"] == "what is 2+2?", "the typed message must come back"
        assert "back in the box" in r["hint"]

    def test_an_empty_reply_rolls_back_and_says_so(self):
        r = drive([{"done": True}])
        assert r["messageRoles"] == []
        assert r["savedRoles"] == []
        assert any("empty reply" in e for e in r["errors"])
        assert r["input"] == "what is 2+2?"

    def test_a_rollback_does_not_clobber_a_newly_typed_message(self):
        """You start typing the next question while the first is in flight."""
        r = drive([{"done": True}], typed="first question",
                  type_during="second question, typed while waiting")
        assert r["input"] == "second question, typed while waiting"
        # The turn is still rolled back — only the restore is skipped.
        assert r["messageRoles"] == []


class TestReasoningStream:
    """Six of the ten installed models emit reasoning."""

    def test_native_thinking_reaches_the_panel(self):
        """/api/chat nests it under message, as it does content.

        Reading the top-level field — the /api/generate shape — meant the panel
        never opened and the bubble sat on its placeholder, looking hung.
        """
        r = drive([
            {"message": {"content": "", "thinking": "Let me work"}},
            {"message": {"content": "", "thinking": " through it."}},
            {"message": {"content": "The answer is 4."}, "done": True},
        ])
        assert r["panel"] == "Let me work through it.", "reasoning was dropped"
        assert r["panelShown"] is True, "the panel never opened"
        assert r["messageRoles"] == ["user", "assistant"]

    def test_the_bubble_stops_looking_hung_as_soon_as_reasoning_arrives(self):
        """With content empty for the whole scratchpad, the reasoning deltas are
        the only sign of life — dropping them left a motionless placeholder."""
        r = drive([{"message": {"content": "", "thinking": "still going"}}], abort_after=1)
        assert r["panel"] == "still going"
        assert r["panelShown"] is True

    def test_inline_tags_still_work_for_an_older_ollama(self):
        r = drive([{"message": {"content": "<think>reasoning</think>The answer is 4."}},
                   {"done": True}])
        assert r["panel"] == "reasoning"
        assert r["bubble"] == "The answer is 4."

    def test_a_reply_that_is_only_reasoning_is_not_recorded_as_an_answer(self):
        r = drive([{"message": {"content": "", "thinking": "hmm"}}, {"done": True}])
        assert r["messageRoles"] == []
        assert r["savedRoles"] == []


class TestSplitThink:
    """Three delimiting shapes, all seen from Ollama."""

    def split(self, raw):
        js = _slice(_page(), "      // Split assistant text", "      function fmtUsage") + \
            f"process.stdout.write(JSON.stringify(splitThink({json.dumps(raw)})));"
        return json.loads(subprocess.run(["node", "-e", js], capture_output=True,
                                         text=True, check=True).stdout)

    def test_a_closed_block_goes_to_the_panel(self):
        out = self.split("<think>reasoning</think>answer")
        assert out["thinking"] == "reasoning"
        assert out["content"] == "answer"

    def test_a_closing_tag_with_no_opening_one_is_still_reasoning(self):
        """Ollama's deepseek-r1 template opens <think> in the prompt."""
        out = self.split("reasoning about it</think>the answer")
        assert out["thinking"] == "reasoning about it"
        assert out["content"] == "the answer"

    def test_a_stray_closing_tag_after_a_real_block_keeps_the_answer(self):
        out = self.split("<think>a</think>real answer </think> tail")
        assert out["thinking"] == "a"
        assert "real answer" in out["content"]

    def test_plain_text_is_untouched(self):
        out = self.split("just an answer")
        assert out["thinking"] == ""
        assert out["content"] == "just an answer"


class TestScrollFollowing:
    """Scrolling up to re-read the question during a long answer used to drag
    you back to the bottom on every token; the only escape was to press Stop."""

    def run_scroll(self, script):
        page = _page()
        js = "\n".join([
            r"""
            let scrolls = [], listener = null;
            const chatEl = {
              scrollHeight: 1000, clientHeight: 200, scrollTop: 800,
              addEventListener(name, fn) { listener = fn; },
            };
            function requestAnimationFrame(fn) { fn(); }
            function scrollTo(top) { chatEl.scrollTop = top; listener(); }
            """,
            _slice(page, "      let scrollPending = false;", "      function clearPlaceholder"),
            script + "process.stdout.write(JSON.stringify(chatEl.scrollTop));",
        ])
        return json.loads(subprocess.run(["node", "-e", js], capture_output=True,
                                         text=True, check=True).stdout)

    def test_a_stream_follows_when_you_are_at_the_bottom(self):
        assert self.run_scroll("chatEl.scrollHeight = 1200; scrollDown();") == 1200

    def test_a_stream_does_not_yank_you_back_after_scrolling_up(self):
        top = self.run_scroll(
            "scrollTo(100);"                      # the user scrolls up to re-read
            "chatEl.scrollHeight = 1200; scrollDown();"   # more tokens arrive
        )
        assert top == 100, "the view was dragged back to the bottom"

    def test_scrolling_back_down_resumes_following(self):
        top = self.run_scroll(
            "scrollTo(100);"
            "scrollTo(chatEl.scrollHeight - chatEl.clientHeight);"   # back to the bottom
            "chatEl.scrollHeight = 1200; scrollDown();"
        )
        assert top == 1200

    def test_your_own_message_always_jumps_to_the_bottom(self):
        top = self.run_scroll("scrollTo(100); chatEl.scrollHeight = 1200; scrollDown(true);")
        assert top == 1200, "sending a message must always show it"
