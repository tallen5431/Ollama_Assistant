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
from conftest import page_script

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def _page() -> str:
    return page_script(chat_ui.render_page("t"))


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
                   createRange: () => ({ selectNodeContents() {} }),
                   addEventListener() {}, visibilityState: "visible" };
const window = { getSelection: () => ({ removeAllRanges() {}, addRange() {} }) };
// No wakeLock and no share: the absence of an optional API must never break a
// turn, which is what these tests would catch.
const navigator = {};
const chatEl = el("div"), inputEl = el("textarea"), hintEl = el("div");
const sendBtn = el("button"), stopBtn = el("button"), modelEl = el("select");
const webEl = { checked: false };
let pendingImages = [], messages = [], busy = false, controller = null, pendingAutoSend = false;
// send() reads this to decide whether a photo's own date goes with it.
let exifOn = true;
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
  const think = el("details");
  think.hidden = true;      // as the real markup ships it: <details ... hidden>
  lastView = { root, bubble: el("div"), status: el("div"), meta: el("div"),
               think, thinkBody: el("div") };
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
        # From the wake-lock helpers, which send() calls, through send() itself.
        _slice(page, "      // Feature-detected, like isSecureContext elsewhere",
               "      function stop()"),
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

    def test_a_closing_tag_alone_on_its_line_is_a_scratchpad_terminator(self):
        """Ollama's deepseek-r1 template opens <think> in the prompt, so the
        reply starts inside the scratchpad and emits only the closing tag."""
        out = self.split("reasoning about it\n</think>\nthe answer")
        assert out["thinking"].strip() == "reasoning about it"
        assert out["content"].strip() == "the answer"

    @pytest.mark.parametrize("reply", [
        "Yes — the reply only carries </think> at the end. That is all.",
        "Use `</think>` to close it. That is the whole answer.",
        "The closing tag is </think> and the opening one is <thing>.",
    ])
    def test_a_reply_that_merely_mentions_the_tag_is_not_truncated(self, reply):
        """This lost everything before the tag — and the truncated text was
        what got pushed into messages[] and written to history, so the loss was
        permanent and invisible."""
        out = self.split(reply)
        assert out["content"] == reply
        assert out["thinking"] == ""

    def test_a_reply_that_is_only_scratchpad_is_shown_whole(self):
        """Better an odd-looking answer than a blank one."""
        out = self.split("I was still thinking\n</think>")
        assert "still thinking" in out["content"]

    def test_a_stray_closing_tag_after_a_real_block_keeps_the_answer(self):
        out = self.split("<think>a</think>real answer </think> tail")
        assert out["thinking"] == "a"
        assert "real answer" in out["content"]

    def test_plain_text_is_untouched(self):
        out = self.split("just an answer")
        assert out["thinking"] == ""
        assert out["content"] == "just an answer"

    def test_a_fenced_example_of_reasoning_output_is_not_truncated(self):
        """Ask a coder model what deepseek-r1 emits and it shows you one.

        The tag is then alone on its line, inside ```, and treating it as a
        real terminator threw away the half of the reply that explained it —
        into a collapsed panel, and out of what reached the history database.
        """
        reply = "Models emit this:\n\n```\nreasoning\n</think>\nanswer\n```\n\nThat is all."
        out = self.split(reply)
        assert out["content"] == reply
        assert out["thinking"] == ""

    def test_a_real_orphan_after_a_closed_fence_still_works(self):
        """The fence check must not disable the rule outright."""
        out = self.split("```\ncode\n```\nreasoning here\n</think>\nthe answer")
        assert out["content"].strip() == "the answer"


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

    def test_a_queued_frame_does_not_yank_you_back(self):
        """A frame already queued when the user scrolls away must not fire.

        Firing it blindly both moved the view and re-armed following, because
        an auto-scroll raises a scroll event of its own.
        """
        page = _page()
        js = "\n".join([
            r"""
            let listener = null, pendingFrame = null;
            const chatEl = {
              scrollHeight: 2000, clientHeight: 200, scrollTop: 1800,
              addEventListener(name, fn) { listener = fn; },
            };
            function requestAnimationFrame(fn) { pendingFrame = fn; }
            function runFrame() { const f = pendingFrame; pendingFrame = null; if (f) f(); }
            function scrollTo(top) { chatEl.scrollTop = top; listener(); }
            """,
            _slice(page, "      let scrollPending = false;", "      function clearPlaceholder"),
            r"""
            scrollDown();      // a frame is queued while we are at the bottom
            scrollTo(0);       // the user scrolls up before it runs
            runFrame();        // the queued frame fires
            process.stdout.write(JSON.stringify(chatEl.scrollTop));
            """,
        ])
        top = json.loads(subprocess.run(["node", "-e", js], capture_output=True,
                                        text=True, check=True).stdout)
        assert top == 0, "a queued frame dragged the view back to the bottom"


class TestRollbackDoesNotReachIntoAnotherConversation:
    """newChat() and loadConversation() abort the in-flight request.

    Pushing the abandoned message and its photos into the composer of a
    different conversation is not a rescue, it is a surprise — and the hint
    claimed a restore that had not happened.
    """

    def drive_with_swap(self, script, abort_after):
        """Swap `messages` mid-stream, as newChat()/loadConversation() do."""
        page = _page()
        js = "\n".join([
            _DOM.replace(
                "      if (TYPE_DURING !== null) { inputEl.value = TYPE_DURING; TYPE_DURING = null; }",
                "      if (TYPE_DURING === 'SWAP') { messages = []; TYPE_DURING = null; }"),
            _slice(page, "      function addUser", "      // ---- Image attachments"),
            _slice(page, "      // Split assistant text", "      function fmtUsage"),
            _slice(page, "      // Feature-detected, like isSecureContext elsewhere",
                   "      function stop()"),
            f"run({json.dumps(script)}, {abort_after}, \"abandoned question\", \"SWAP\")"
            ".then(r => process.stdout.write(JSON.stringify(r)));",
        ])
        out = subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True)
        return json.loads(out.stdout)

    def test_an_abandoned_turn_does_not_repopulate_the_composer(self):
        r = self.drive_with_swap([{"message": {"content": "", "thinking": "hmm"}}], 1)
        assert r["input"] == "", "the discarded message was pushed into a new conversation"

    def test_and_does_not_claim_it_came_back(self):
        r = self.drive_with_swap([{"message": {"content": "", "thinking": "hmm"}}], 1)
        assert "back in the box" not in r["hint"]


class TestTheWaitCounterIsNotSuppressedByAnEmptyStatus:
    """app.py yields {"status": ""} to *clear* the line when the planner
    declines. Latching on that suppressed the counter on exactly the turns
    that are slowest — a cold 30b with the web toggle left on."""

    def owned_after(self, script):
        page = _page()
        js = "\n".join([
            _DOM,
            _slice(page, "      function addUser", "      // ---- Image attachments"),
            _slice(page, "      // Split assistant text", "      function fmtUsage"),
            _slice(page, "      // Feature-detected, like isSecureContext elsewhere",
                   "      function stop()"),
            f"run({json.dumps(script)}, -1, \"hi\", null)"
            ".then(() => process.stdout.write(JSON.stringify(!!lastView.statusOwned)));",
        ])
        return json.loads(subprocess.run(["node", "-e", js], capture_output=True,
                                         text=True, check=True).stdout)

    def test_an_empty_status_does_not_take_over_the_line(self):
        assert self.owned_after([{"status": ""}, {"message": {"content": "hi"}},
                                 {"done": True}]) is False

    def test_a_status_that_is_then_cleared_hands_the_line_back(self):
        """The real sequence when the planner declines: _gather_web always
        yields "Working out what to search for…" first, then "" to clear it.
        Latching on the first left the counter suppressed for the whole of the
        longest wait there is — a cold 30b with the web toggle left on."""
        assert self.owned_after([{"status": "Working out what to search for…"},
                                 {"status": ""},
                                 {"message": {"content": "hi"}}, {"done": True}]) is False

    def test_a_real_status_does(self):
        assert self.owned_after([{"status": "Searching: x"}, {"message": {"content": "hi"}},
                                 {"done": True}]) is True



def _code_only(page: str) -> str:
    """The page's JavaScript with comments, strings and regexes blanked out.

    Scanning the raw source finds "model(" inside a comment and "Option(" inside
    a string, which is noise. Walking it once with a little state costs less
    than an allowlist that grows every time someone writes a sentence.
    """
    out = []
    i, n = 0, len(page)
    # Whether a "/" here starts a regex or is a division, decided by what came
    # before it — the same rule a real tokeniser uses, minus the edge cases that
    # do not appear in this file.
    prev = ""
    while i < n:
        ch = page[i]
        two = page[i:i + 2]
        if two == "//":
            i = page.find("\n", i)
            if i < 0:
                break
            continue
        if two == "/*":
            end = page.find("*/", i + 2)
            i = n if end < 0 else end + 2
            continue
        if ch in "\"'`":
            quote, i = ch, i + 1
            while i < n:
                if page[i] == "\\":
                    i += 2
                    continue
                if page[i] == quote:
                    i += 1
                    break
                i += 1
            out.append('""')
            prev = '"'
            continue
        if ch == "/" and prev in "(,=:[!&|?{};+-*%~^" + "\n":
            i += 1
            while i < n:
                if page[i] == "\\":
                    i += 2
                    continue
                if page[i] == "[":
                    while i < n and page[i] != "]":
                        i += 1 if page[i] != "\\" else 2
                if page[i] == "/":
                    i += 1
                    break
                i += 1
            out.append("RE")
            prev = "E"
            continue
        out.append(ch)
        if not ch.isspace():
            prev = ch
        i += 1
    return "".join(out)


class TestEveryFunctionThePageCallsExists:
    """A deleted helper survived the whole suite once. Only a browser caught it.

    Editing chat_ui.py means splicing text into one long string, and a slice
    that reaches a line too far takes working code with it. 687 tests passed
    with loadBitmap gone — nothing here calls toAttachment, so nothing noticed
    until an image was attached in a real browser.
    """

    # Everything the page legitimately reaches for that it does not define.
    PROVIDED = {
        # Language and standard library
        "Array", "Boolean", "Date", "Error", "Infinity", "Intl", "JSON", "Map",
        "Math", "NaN", "Number", "Object", "Promise", "RegExp", "Set", "String",
        "Symbol", "TextDecoder", "TextEncoder", "Uint8Array", "Int16Array",
        "Float32Array", "ArrayBuffer", "DataView", "Blob", "File", "Option",
        "FileReader", "FormData", "Headers", "Request", "Response", "URL",
        "AbortController", "Image", "Function", "parseInt", "parseFloat",
        "isFinite", "isNaN", "encodeURIComponent", "decodeURIComponent",
        "btoa", "atob", "structuredClone", "queueMicrotask",
        # Browser
        "document", "window", "navigator", "location", "localStorage", "console",
        "fetch", "setTimeout", "clearTimeout", "setInterval", "clearInterval",
        "requestAnimationFrame", "cancelAnimationFrame", "alert", "confirm",
        "prompt", "createImageBitmap", "getComputedStyle", "MediaRecorder",
        "AudioContext", "webkitAudioContext", "SpeechRecognition",
        "webkitSpeechRecognition", "OffscreenCanvas", "ResizeObserver",
        "MutationObserver", "CustomEvent", "Event", "AbortSignal",
        # Reserved words that the scanner sees in call position
        "if", "for", "while", "switch", "catch", "return", "typeof", "function",
        "await", "async", "new", "delete", "void", "in", "of", "do", "else", "try",
    }

    def test_no_call_names_something_that_is_not_defined(self):
        page = _code_only(_page())
        declared = set(re.findall(r"\b(?:function|const|let|var|class)\s+([A-Za-z_$][\w$]*)", page))
        # Parameter names are declarations too.
        for params in re.findall(r"\bfunction\s*[\w$]*\s*\(([^)]*)\)", page):
            declared |= {p.strip().split("=")[0].strip() for p in params.split(",") if p.strip()}
        for params in re.findall(r"\(([^()]*)\)\s*=>", page):
            declared |= {p.strip().split("=")[0].strip() for p in params.split(",") if p.strip()}
        declared |= set(re.findall(r"([A-Za-z_$][\w$]*)\s*=>", page))
        declared |= set(re.findall(r"\bcatch\s*\(\s*([\w$]+)\s*\)", page))

        called = set()
        for match in re.finditer(r"(^|[^.\w$'\"])([A-Za-z_$][\w$]*)\s*\(", page):
            called.add(match.group(2))

        missing = sorted(called - declared - self.PROVIDED)
        assert not missing, (
            f"the page calls {missing}, which nothing in it defines — "
            "most likely an edit spliced over a helper"
        )
