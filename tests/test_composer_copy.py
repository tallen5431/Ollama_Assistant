"""Getting dictated text out of the page and into another app.

This page transcribes speech offline, which makes it a usable dictation pad for
apps that have no dictation of their own — but only if the words can be taken
out again. Before this the only exits were the Send button, which hands them to
a model rather than to you, and selecting the text by hand on a phone keyboard.

The functions under test ship inside chat_ui._PAGE, so they are sliced out of
the rendered page and run under node against a stand-in DOM. Testing a copy of
them would prove nothing about what the browser is served.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

import chat_ui
from conftest import page_script

pytestmark = pytest.mark.skipif(shutil.which("node") is None,
                                reason="node not installed")


def _page() -> str:
    return page_script(chat_ui.render_page("t"))


def _slice(page: str, start: str, end: str) -> str:
    at = page.index(start)
    return page[at:page.index(end, at)]


# Enough DOM for the three exits to be told apart. `navigator` and the pointer
# media query are what each case varies.
_DOM = r"""
function field(value) {
  return { value, hidden: false, style: {}, scrollHeight: 20,
           selected: false, focused: false,
           focus() { this.focused = true; },
           select() { this.selected = true; },
           addEventListener() {} };
}
const inputEl = field("");
const copyOutBtn = { hidden: true }, clearOutBtn = { hidden: true };
const hintEl = { textContent: "" };
const calls = [];
let SHARE = null, CLIPBOARD = null, COARSE = false;
const navigator = {
  get share() { return SHARE; },
  get clipboard() { return CLIPBOARD; },
};
const window = {
  innerHeight: 800,
  matchMedia: (q) => ({ matches: q.indexOf("coarse") >= 0 ? COARSE : false }),
  getSelection: () => ({ removeAllRanges() {}, addRange() {} }),
};
const document = { createRange: () => ({ selectNodeContents() {} }) };
"""


# Which of navigator.share / navigator.clipboard exist, and how they behave.
# Every case below is one real browser: a phone with a share sheet, a desktop
# without one, a plain-HTTP page with no clipboard at all.
_SETUP = r"""
if (SETUP.coarse) COARSE = true;
if (SETUP.share === "ok") SHARE = async () => { calls.push("share"); };
if (SETUP.share === "abort") SHARE = async () => {
  calls.push("share"); const e = new Error("x"); e.name = "AbortError"; throw e;
};
if (SETUP.share === "fail") SHARE = async () => {
  calls.push("share"); throw new Error("no handler");
};
if (SETUP.clipboard === "ok")
  CLIPBOARD = { writeText: async (t) => { calls.push("clipboard:" + t); } };
if (SETUP.clipboard === "fail")
  CLIPBOARD = { writeText: async () => { throw new Error("denied"); } };
"""


def _run(parts: list) -> dict:
    out = subprocess.run(["node", "-e", "\n".join(parts)], capture_output=True,
                         text=True, check=True, timeout=30)
    return json.loads(out.stdout)


def drive(body: str, **setup) -> dict:
    """Run the shipped helpers against a scripted environment, synchronously."""
    page = _page()
    return _run([
        _DOM,
        _slice(page, "      function selectText", "      const selectCode = selectText;"),
        _slice(page, "      function autosize", "      // Coalesce to one layout"),
        _slice(page, "      async function shareOrCopy", "      // Copying a whole reply"),
        f"const SETUP = {json.dumps(setup)};",
        _SETUP,
        body,
        "process.stdout.write(JSON.stringify(OUT)); process.exit(0);",
    ])


def copy_run(**setup) -> dict:
    """shareOrCopy alone, which is async and so writes its own result."""
    page = _page()
    return _run([
        _DOM,
        _slice(page, "      function selectText", "      const selectCode = selectText;"),
        _slice(page, "      async function shareOrCopy", "      // Copying a whole reply"),
        f"const SETUP = {json.dumps(setup)};",
        _SETUP,
        r"""
        const OUT = {};
        (async () => {
          inputEl.value = SETUP.text === undefined ? "hello there" : SETUP.text;
          const ok = await shareOrCopy(inputEl.value.trim(), inputEl,
                                       (how) => { OUT.how = how; });
          OUT.ok = ok; OUT.calls = calls; OUT.selected = inputEl.selected;
          process.stdout.write(JSON.stringify(OUT)); process.exit(0);
        })();
        """,
    ])


class TestTheWayOutOfTheBox:
    """Three routes, because no single one works everywhere."""

    def test_a_phone_gets_the_share_sheet(self):
        """The good one: it reaches the app you actually want to paste into,
        without a clipboard round trip, and it works on plain HTTP."""
        r = copy_run(coarse=True, share="ok", clipboard="ok")
        assert r["calls"] == ["share"], "the clipboard was used despite a share sheet"
        assert r["how"] == "shared" and r["ok"]

    def test_a_desktop_gets_the_clipboard(self):
        """navigator.share exists on some desktops and opens nothing useful."""
        r = copy_run(coarse=False, share="ok", clipboard="ok")
        assert r["calls"] == ["clipboard:hello there"]
        assert r["how"] == "copied" and r["ok"]

    def test_dismissing_the_share_sheet_is_not_a_failure(self):
        """Falling through to the clipboard made cancel look like an error —
        and worse, quietly copied something the user had just declined to send."""
        r = copy_run(coarse=True, share="abort", clipboard="ok")
        assert r["calls"] == ["share"], "cancelling still reached the clipboard"
        assert r["how"] == "cancelled" and not r["ok"]

    def test_but_a_share_that_breaks_falls_back(self):
        r = copy_run(coarse=True, share="fail", clipboard="ok")
        assert r["calls"] == ["share", "clipboard:hello there"]
        assert r["how"] == "copied" and r["ok"]

    def test_no_clipboard_at_all_selects_the_text_instead(self):
        """navigator.clipboard does not exist off a secure origin, which is
        exactly how this app is served over a LAN. An inert button that looks
        like it worked is worse than telling someone which keys to press."""
        r = copy_run(coarse=False, clipboard=None)
        assert r["selected"], "nothing was selected, so Ctrl+C would copy nothing"
        assert r["how"] == "manual" and not r["ok"]

    def test_a_refused_clipboard_does_the_same(self):
        r = copy_run(coarse=False, clipboard="fail")
        assert r["selected"] and r["how"] == "manual"

    def test_an_empty_box_does_nothing_at_all(self):
        r = copy_run(coarse=True, share="ok", clipboard="ok", text="")
        assert r["calls"] == [] and not r["ok"]


class TestSelectingATextarea:
    """selectText ran a Range over the node's child nodes. A textarea keeps its
    text in .value and has none, so the fallback selected nothing and "Press
    Ctrl+C" copied an empty selection — while looking like it had worked.
    """

    def test_a_form_field_is_selected_through_its_own_api(self):
        r = drive(r"""
          const OUT = {};
          inputEl.value = "dictated words";
          selectText(inputEl);
          OUT.selected = inputEl.selected; OUT.focused = inputEl.focused;
        """)
        assert r["selected"] and r["focused"]

    def test_an_ordinary_node_still_uses_a_range(self):
        r = drive(r"""
          const OUT = {};
          let ranged = false;
          document.createRange = () => ({ selectNodeContents() { ranged = true; } });
          selectText({ tag: "div" });
          OUT.ranged = ranged;
        """)
        assert r["ranged"], "a reply bubble must still be selectable"


class TestTheButtonsAppearWhenThereIsSomethingToActOn:
    """An empty composer is the state this page spends most of its life in."""

    def test_they_are_hidden_while_the_box_is_empty(self):
        r = drive(r"""
          const OUT = {};
          inputEl.value = "";
          autosize();
          OUT.copy = copyOutBtn.hidden; OUT.clear = clearOutBtn.hidden;
        """)
        assert r["copy"] and r["clear"]

    def test_and_appear_once_there_is_text(self):
        r = drive(r"""
          const OUT = {};
          inputEl.value = "something";
          autosize();
          OUT.copy = copyOutBtn.hidden; OUT.clear = clearOutBtn.hidden;
        """)
        assert not r["copy"] and not r["clear"]

    def test_whitespace_is_not_text(self):
        r = drive(r"""
          const OUT = {};
          inputEl.value = "   \n  ";
          autosize();
          OUT.copy = copyOutBtn.hidden;
        """)
        assert r["copy"]

    def test_they_go_again_when_the_box_is_emptied(self):
        """Sending clears the box, and a copy button over an empty composer
        copies nothing."""
        r = drive(r"""
          const OUT = {};
          inputEl.value = "typed"; autosize();
          inputEl.value = ""; autosize();
          OUT.copy = copyOutBtn.hidden; OUT.clear = clearOutBtn.hidden;
        """)
        assert r["copy"] and r["clear"]

    def test_dictation_makes_them_appear_though_it_fires_no_input_event(self):
        """The case the whole thing was built for. Transcription assigns to
        .value, which fires no input event — hanging this off that event meant
        the button never appeared for speech, only for typing."""
        r = drive(r"""
          const OUT = {};
          // Exactly what sendForTranscription does to the box.
          inputEl.value = (inputEl.value ? inputEl.value.trim() + " " : "") + "spoken words";
          autosize();
          OUT.copy = copyOutBtn.hidden; OUT.text = inputEl.value;
        """)
        assert not r["copy"], "dictating left no way to copy the result"
        assert r["text"] == "spoken words"


class TestTheMarkupIsWiredUp:
    """The handlers are attached by id; a renamed button fails silently."""

    def test_both_buttons_exist_and_start_hidden(self):
        page = chat_ui.render_page("t")
        assert 'id="copyOut"' in page and 'id="clearOut"' in page
        for button in ("copyOut", "clearOut"):
            at = page.index(f'id="{button}"')
            assert "hidden" in page[at:page.index(">", at)], \
                f"{button} is visible over an empty composer"

    def test_they_sit_in_the_composer_beside_the_mic(self):
        page = chat_ui.render_page("t")
        composer = page[page.index('<div class="composer">'):page.index("</div>", page.index('<div class="composer">'))]
        for button in ("mic", "copyOut", "clearOut", "send"):
            assert f'id="{button}"' in composer, f"{button} is not in the composer"

    def test_the_icons_they_reference_are_defined(self):
        page = chat_ui.render_page("t")
        for icon in ("i-copy", "i-eraser"):
            assert f'<symbol id="{icon}"' in page, f"{icon} would render as nothing"

    def test_the_handlers_are_bound(self):
        js = page_script(chat_ui.render_page("t"))
        assert 'copyOutBtn.addEventListener("click"' in js
        assert 'clearOutBtn.addEventListener("click"' in js

    def test_copying_does_not_empty_the_box(self):
        """Copying is not a decision to throw the text away — you might copy it
        and then also send it. Clearing has its own button."""
        js = page_script(chat_ui.render_page("t"))
        handler = js[js.index('copyOutBtn.addEventListener("click"'):
                     js.index('clearOutBtn.addEventListener("click"')]
        assert 'inputEl.value = ""' not in handler
