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
import web
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
// The store is the server's job now: /api/chat is given the thread id and
// writes the turn itself, so what this side is responsible for is creating the
// thread before the stream starts and taking it away again if the turn
// produced nothing. `saved` records exactly that.
let errors = [], saved = { conversation: false, dropped: false, sentId: null };
let currentConvoId = null;
let threadEmpty = true;
function renderThumbs() {}
// Asking the next question stops the answer to the last one being read out.
let stopped = 0;
function stopSpeaking() { stopped++; }
const speakEl = { checked: false };
function speakReply() {}
function autosize() {}
function clearPlaceholder() {}
function scrollDown() {}
function setStatus() {}
function markError(m) { errors.push(m); }
function paintMarkdown(e, raw) { e.textContent = raw; }
function fmtUsage() { return ""; }
function showSources() {}
async function ensureConversation() {
  saved.conversation = true;
  if (!currentConvoId) currentConvoId = "convo-1";
}
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
// The connection dying mid-stream, which is not an abort: no AbortError, just
// a read that fails. DROP_AFTER says after how many lines.
let DROP_AFTER = -1;
// What the server says once the page comes back asking. `landedAfter` is how
// many polls it takes for the finished turn to appear in the thread.
let TURN_RUNNING = true, LANDED_AFTER = 1, polls = 0, statusAsks = 0;
// How many messages the thread already held: 0 for a first question,
// 2 for the second, 4 for the third.
let PRIOR_TURNS = 0;
let loadedConvo = null;
async function loadConversation(id) { loadedConvo = id; }
async function fetch(url, opts) {
  // "Is that turn still going?" — the question the page could not ask before,
  // and without which a dropped connection and a dead server look the same.
  if (typeof url === "string" && url.indexOf("api/chat/status") === 0) {
    statusAsks++;
    return { ok: true, json: async () => ({ running: TURN_RUNNING }) };
  }
  // dropIfEmpty() asks the server what is in the thread and deletes it when
  // the answer is nothing. The recovery poll asks the same question and reads
  // the last role.
  if (typeof url === "string" && url.indexOf("api/conversations/") === 0) {
    if (opts && opts.method === "DELETE") { saved.dropped = true; return { ok: true }; }
    if (threadEmpty) return { ok: true, json: async () => ({}) };
    polls++;
    const done = polls >= LANDED_AFTER;
    // Empty until the turn lands, which is how it really is: the server
    // writes the question and the answer together when generation ends, so at
    // the moment a connection dies the thread has nothing in it at all. That
    // is exactly what made dropIfEmpty() delete it out from under the reply.
    // PRIOR_TURNS is what the thread already held before this turn started —
    // zero for the first question in a thread, two for the second, and so on.
    // The stored thread therefore *ends in an assistant message* the whole
    // time this turn is still being generated, which is the case the recovery
    // has to tell apart from its own reply arriving.
    const before = [];
    for (let n = 0; n < PRIOR_TURNS; n++) {
      before.push({ role: "user", content: "earlier q" + n },
                  { role: "assistant", content: "earlier answer " + n });
    }
    return { ok: true, json: async () => ({ messages: done
      ? before.concat([{ role: "user", content: "q" },
                       { role: "assistant", content: "the answer" }])
      : before }) };
  }
  if (opts && opts.body) saved.sentId = JSON.parse(opts.body).conversation_id || null;
  let i = 0;
  return { ok: true, body: { getReader: () => ({
    async read() {
      // Whatever the user types while the reply is in flight.
      if (TYPE_DURING !== null) { inputEl.value = TYPE_DURING; TYPE_DURING = null; }
      if (ABORT_AFTER >= 0 && i >= ABORT_AFTER) { const e = new Error("abort"); e.name = "AbortError"; throw e; }
      if (DROP_AFTER >= 0 && i >= DROP_AFTER) throw new TypeError("Failed to fetch");
      if (i >= SCRIPT.length) return { done: true, value: undefined };
      return { done: false, value: new TextEncoder().encode(JSON.stringify(SCRIPT[i++]) + "\n") };
    },
    releaseLock() {},
  }) } };
}

async function run(script, abortAfter, typed, typeDuring, opts) {
  opts = opts || {};
  SCRIPT = script; ABORT_AFTER = abortAfter; TYPE_DURING = typeDuring;
  DROP_AFTER = opts.dropAfter === undefined ? -1 : opts.dropAfter;
  TURN_RUNNING = opts.running !== false;
  LANDED_AFTER = opts.landedAfter === undefined ? 1 : opts.landedAfter;
  PRIOR_TURNS = opts.priorTurns || 0;
  polls = 0; statusAsks = 0; loadedConvo = null;
  // The live thread starts out holding whatever the stored one does — a
  // follow-up question is asked into a conversation already on the screen.
  messages = [];
  for (let n = 0; n < PRIOR_TURNS; n++) {
    messages.push({ role: "user", content: "earlier q" + n },
                  { role: "assistant", content: "earlier answer " + n });
  }
  errors = [];
  saved = { conversation: false, dropped: false, sentId: null };
  currentConvoId = PRIOR_TURNS ? "convo-1" : null;
  threadEmpty = opts.threadEmpty !== false;
  chatEl.children = []; pendingImages = [];
  inputEl.value = typed;
  await send();
  for (let t = 0; t < 8; t++) await Promise.resolve();
  // The recovery runs on real timers, so give it real time to run.
  if (opts.settleMs) await new Promise(r => setTimeout(r, opts.settleMs));
  const out = {
    messageRoles: messages.map(m => m.role),
    threadMade: saved.conversation,
    threadDropped: saved.dropped,
    sentThreadId: saved.sentId,
    input: inputEl.value,
    onScreen: chatEl.children.filter(c => !c.removed).length,
    errors, hint: hintEl.textContent,
    panel: lastView ? lastView.thinkBody.textContent : "",
    panelShown: lastView ? lastView.think.hidden === false : false,
    bubble: lastView ? lastView.bubble.textContent : "",
    turnStatus: lastView ? lastView.status.textContent : "",
    polls, statusAsks, loadedConvo,
  };
  // Disarm anything still waiting: a pending recovery timer would keep node
  // alive long after the answer has been written out.
  stopWaiting();
  return out;
}
"""


def drive(script, abort_after=-1, typed="what is 2+2?", type_during=None, **opts):
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
        f"{json.dumps(type_during)}, {json.dumps(opts)})"
        ".then(r => { process.stdout.write(JSON.stringify(r)); process.exit(0); });",
    ])
    out = subprocess.run(["node", "-e", js], capture_output=True, text=True,
                         check=True, timeout=30)
    return json.loads(out.stdout)


class TestTurnBookkeeping:
    """Screen, messages[] and the store must never disagree about a turn."""

    def test_a_normal_reply_is_kept_everywhere(self):
        r = drive([{"message": {"content": "4"}}, {"done": True}])
        assert r["messageRoles"] == ["user", "assistant"]
        assert r["threadMade"] and not r["threadDropped"]
        assert r["sentThreadId"] == "convo-1", \
            "the server writes the turn, so it has to be told which thread"
        assert r["input"] == ""

    def test_stop_after_some_text_keeps_the_partial(self):
        r = drive([{"message": {"content": "The ans"}}, {"message": {"content": "wer"}}], abort_after=2)
        assert r["messageRoles"] == ["user", "assistant"]
        assert not r["threadDropped"], "Stop still produced an answer to keep"

    def test_stop_during_reasoning_rolls_the_whole_turn_back(self):
        """For a reasoning model this is the entire scratchpad — the usual case.

        The turn used to stay on screen and in messages[] while never being
        written, so the next send posted two user roles in a row.
        """
        r = drive([{"message": {"content": "", "thinking": "thinking hard"}}], abort_after=1)
        assert r["messageRoles"] == []
        assert r["threadDropped"], "the thread made for this turn has to go too"
        assert r["onScreen"] == 0
        assert r["input"] == "what is 2+2?", "the typed message must come back"
        assert "back in the box" in r["hint"]

    def test_an_empty_reply_rolls_back_and_says_so(self):
        r = drive([{"done": True}])
        assert r["messageRoles"] == []
        assert r["threadDropped"], "an empty reply must not leave an empty thread"
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
        assert r["threadDropped"]


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


class TestTheTwoStrippersAgree:
    """The browser decides what you read; web.strip_thinking decides what gets
    written down. They are two implementations of one rule, in two languages,
    and they had drifted: the server copy had neither the fenced-code guard nor
    the "something has to follow" guard, so a reply explaining reasoning models
    was shown whole and stored truncated, and a reply that was all scratchpad
    was shown whole and stored not at all. Invisible until the thread is
    reopened tomorrow, and by then the text is gone.

    Sharing the code is not available across the language boundary, so this is
    the next best thing: run both over the same replies and require the same
    answer. Anything added to one and not the other fails here.
    """

    CORPUS = [
        "just an answer",
        "<think>reasoning</think>The answer is 4.",
        "<think>ran out of budget mid-thought",
        "reasoning about it\n</think>\nthe answer",
        "I was still thinking\n</think>",
        "Use `</think>` to close it. That is the whole answer.",
        "The closing tag is </think> and the opening one is <thing>.",
        "Models emit this:\n\n```\nreasoning\n</think>\nanswer\n```\n\nThat is all.",
        "```\ncode\n```\nreasoning here\n</think>\nthe answer",
        "<think>a</think>real answer </think> tail",
        "  </think>\nanswer with the tag indented",
        "</think>\nnothing precedes the tag",
    ]

    def browser(self, raw):
        js = _slice(_page(), "      // Split assistant text", "      function fmtUsage") + \
            f"process.stdout.write(JSON.stringify(splitThink({json.dumps(raw)})));"
        out = subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True)
        return json.loads(out.stdout)["content"].strip()

    @pytest.mark.parametrize("reply", CORPUS)
    def test_what_is_shown_is_what_is_stored(self, reply):
        assert web.strip_thinking(reply) == self.browser(reply), reply


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
            // The jump-to-latest button is the same state seen twice, so it
            // lives in this slice and looks itself up from here.
            const document = {
              getElementById: () => ({ hidden: true, addEventListener() {} }),
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
            const document = {
              getElementById: () => ({ hidden: true, addEventListener() {} }),
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
            ".then(r => { process.stdout.write(JSON.stringify(r)); process.exit(0); });",
        ])
        out = subprocess.run(["node", "-e", js], capture_output=True, text=True,
                         check=True, timeout=30)
        return json.loads(out.stdout)

    def test_an_abandoned_turn_does_not_repopulate_the_composer(self):
        r = self.drive_with_swap([{"message": {"content": "", "thinking": "hmm"}}], 1)
        assert r["input"] == "", "the discarded message was pushed into a new conversation"

    def test_and_does_not_claim_it_came_back(self):
        r = self.drive_with_swap([{"message": {"content": "", "thinking": "hmm"}}], 1)
        assert "back in the box" not in r["hint"]


class TestLosingTheConnectionIsNotLosingTheTurn:
    """Reported: "the app returns network error if a tab out while the model is
    thinking". Three things were wrong with that, and only one of them was the
    message.

    Generation is detached from the connection that asked for it — that is what
    _detached_stream is for — and the finished turn is written to the thread
    whether or not anyone is still listening. So a dropped stream is a dropped
    *connection*, and the turn behind it is still coming. The page treated it
    as a failure: pulled the question off the screen, put it back in the
    composer, and deleted the empty thread it had just made. The reply then
    arrived into a conversation that no longer existed.
    """

    SCRIPT = [{"message": {"content": "Thinking"}}, {"message": {"content": " it over"}}]

    def dropped(self, **opts):
        """A turn whose connection dies after the first line."""
        opts.setdefault("landedAfter", 1)
        opts.setdefault("settleMs", 400)
        # There is a thread with the question already in it; that is the whole
        # premise, and it is what the recovery reads.
        opts.setdefault("threadEmpty", False)
        return drive(self.SCRIPT, **dict(opts, dropAfter=1))

    def test_the_question_stays_on_the_screen(self):
        r = self.dropped()
        assert r["input"] == "", "the question was dumped back in the composer"
        assert r["onScreen"] >= 2, "the turn was taken off the screen"

    def test_the_thread_is_not_deleted_out_from_under_the_reply(self):
        """The reply is on its way into that thread, and the thread is empty
        until it gets there — so "delete it if it is empty" deletes exactly the
        thread that is about to be written to. That is the step that turned a
        dropped connection into a lost answer.

        landedAfter=2 so the reply has not arrived at the moment of the drop,
        which is the whole point; with it already there, there is no empty
        thread to delete and nothing to get wrong.
        """
        r = self.dropped(landedAfter=2, settleMs=1500)
        assert not r["threadDropped"]

    def test_it_says_what_happened_rather_than_network_error(self):
        r = self.dropped(landedAfter=99)
        assert "still being written" in r["turnStatus"], r["turnStatus"]
        assert not r["errors"], r["errors"]

    def test_and_then_goes_and_collects_it(self):
        r = self.dropped()
        assert r["loadedConvo"] == "convo-1", \
            "the finished turn was never fetched back"

    def test_the_answer_is_re_read_from_the_thread_not_patched_in(self):
        """It comes back with its thinking, its steps and its sources — the
        same path that shows a phone's turn on the desktop."""
        r = self.dropped()
        assert r["loadedConvo"] == "convo-1"

    def test_it_keeps_looking_until_the_reply_lands(self):
        """Backoff is 700ms, then 1.1s, then 1.8s — so three looks need a
        little over three seconds of patience from the test too."""
        r = self.dropped(landedAfter=3, settleMs=3400)
        assert r["polls"] >= 3, r["polls"]
        assert r["loadedConvo"] == "convo-1"

    def test_it_asks_whether_the_turn_is_still_running(self):
        """Without that question a dead server and a dropped connection look
        identical, and waiting out a dead server is its own kind of broken."""
        r = self.dropped(landedAfter=99, settleMs=600)
        assert r["statusAsks"] >= 1

    def test_a_turn_that_really_died_is_given_up_on(self):
        r = self.dropped(landedAfter=99, running=False, settleMs=400)
        assert "did not survive" in r["turnStatus"], r["turnStatus"]
        assert r["polls"] >= 1

    def test_and_says_the_question_is_still_there(self):
        """It is: this path leaves it on screen rather than in the composer,
        so "send it again" has to be the instruction."""
        r = self.dropped(landedAfter=99, running=False, settleMs=400)
        assert "send it again" in r["turnStatus"].lower()

    def test_the_thread_is_only_asked_about_once_per_look(self):
        """Two fetches per poll against a phone on a bad connection is twice
        the chance of neither arriving."""
        r = self.dropped(landedAfter=1, settleMs=400)
        assert r["polls"] == 1, r["polls"]
        assert r["statusAsks"] == 0, "the status was asked for despite the reply landing"

    def test_a_dropped_follow_up_is_not_mistaken_for_a_landed_one(self):
        """Every test above asks the first question in a thread, where "the
        stored thread ends in an assistant message" can only mean this turn's
        reply. On the second question and every one after it, that is already
        true of the *previous* turn — so the first poll concluded the reply had
        landed, stopped waiting, and re-read the thread from the database:
        the question just asked disappeared off the screen, the thread went
        back to ending at the old answer, and the real reply — still being
        generated — never appeared without a manual reload.

        Unlike the first-turn case this is not a race. It happened every time.
        """
        r = self.dropped(priorTurns=2, landedAfter=3, settleMs=3400)
        assert r["polls"] >= 2, "it stopped at the previous turn's answer"
        assert r["loadedConvo"] == "convo-1", "the real reply was never collected"

    def test_and_it_still_asks_whether_that_turn_is_running(self):
        """Which it could not do before: concluding "landed" on the first poll
        skipped the status call entirely, so a follow-up turn that really had
        died was never distinguished from one still being written."""
        r = self.dropped(priorTurns=2, landedAfter=99, settleMs=600)
        assert r["statusAsks"] >= 1

    def test_and_a_dead_follow_up_is_still_given_up_on(self):
        r = self.dropped(priorTurns=4, landedAfter=99, running=False, settleMs=400)
        assert "did not survive" in r["turnStatus"], r["turnStatus"]

    def test_the_first_question_in_a_thread_is_unaffected(self):
        """storedBefore is 0 there, which is the test that was already being
        made — so this must not have moved."""
        r = self.dropped(priorTurns=0)
        assert r["loadedConvo"] == "convo-1"
        assert r["polls"] == 1

    def test_an_unsaved_conversation_still_reports_the_failure(self):
        """With history off there is no thread to collect anything from, so
        the old behaviour is the only one available and must stay."""
        page = _page()
        at = page.index("} else if (messages === thread && currentConvoId) {")
        assert "waitForTurn(currentConvoId, view, err, storedBefore)" in page[at:at + 1400]
        assert "markError(err.message || String(err))" in page[at:at + 3000], \
            "the no-thread branch lost its error"

    @pytest.mark.parametrize("leaving", [
        "function newChat()", "function stop()", "async function loadConversation",
    ])
    def test_leaving_calls_off_the_wait(self, leaving):
        """A wait that outlives the conversation it belongs to would drop
        someone else's thread on the screen."""
        page = _page()
        at = page.index(leaving)
        assert "stopWaiting();" in page[at:at + 500], leaving

    def test_coming_back_to_the_tab_looks_straight_away(self):
        """The phone has just been unlocked and the radio is awake; sitting
        out the rest of a five-second backoff is the wrong instinct."""
        page = _page()
        at = page.index('if (document.visibilityState !== "visible" || !waiting) return;')
        window = page[at:at + 300]
        assert "waiting.tries = 0" in window
        assert "lookForTurn();" in window


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
        "SpeechSynthesisUtterance",
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
