"""Reading a reply aloud.

A reply is written to be read: fences, pipes and asterisks are layout, and a
voice reciting them is unlistenable. Most of this is about turning what is on
screen into something worth hearing — the rest is about it stopping when it
should, which on a phone is the difference between a feature and a nuisance.

The browser's own voices, so nothing to download and nothing to install. It is
the one part of the app not running on the owner's hardware; the text never
leaves the device, but the voices are the operating system's.
"""

from __future__ import annotations

import json
import subprocess

import pytest

import chat_ui
from conftest import page_script


def page() -> str:
    return chat_ui.render_page("t")


def spoken(raw: str) -> str:
    """What the page would actually say, given this markdown."""
    js = page_script(page())
    start = js.index("      function speechText(raw) {")
    end = js.index("      function stopSpeaking() {")
    harness = ("\nconsole.log(JSON.stringify(speechText(" + json.dumps(raw) + ")));")
    out = subprocess.run(["node", "-e", js[start:end] + harness],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def chunks(text: str) -> list:
    js = page_script(page())
    start = js.index("      function speechChunks(text) {")
    end = js.index("      function stopSpeaking() {")
    harness = ("\nconsole.log(JSON.stringify(speechChunks(" + json.dumps(text) + ")));")
    out = subprocess.run(["node", "-e", js[start:end] + harness],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


class TestMarkdownBecomesSomethingWorthHearing:
    def test_a_fenced_block_is_named_not_recited(self):
        """"async function paint open bracket" helps nobody, but silence
        leaves you wondering what you missed."""
        said = spoken("Here you go.\n\n```python\ndef f(x):\n    return x\n```\n\nDone.")
        assert "code block" in said
        assert "def f" not in said and "```" not in said

    def test_a_table_is_read_as_rows_of_cells(self):
        said = spoken("| Leg | Miles |\n|---|---|\n| Out | 41 |\n| Back | 39 |")
        assert "|" not in said
        assert "Leg, Miles" in said and "Out, 41" in said

    def test_rows_do_not_run_into_each_other(self):
        """Without a stop between them it came out "Miles Out, 41 Back"."""
        said = spoken("| Leg | Miles |\n|---|---|\n| Out | 41 |\n| Back | 39 |")
        assert "Miles. Out" in said or "Miles.  Out" in said

    @pytest.mark.parametrize("raw, gone, kept", [
        ("**bold** and *italic*", "*", "bold and italic"),
        ("## A heading", "#", "A heading"),
        ("> quoted line", ">", "quoted line"),
        ("- one\n- two", "-", "one"),
        ("`inline code`", "`", "inline code"),
        ("---", "-", ""),
    ])
    def test_the_markers_go_and_the_words_stay(self, raw, gone, kept):
        said = spoken(raw)
        assert gone not in said
        if kept:
            assert kept in said

    def test_a_link_is_read_as_its_words(self):
        """Nobody wants a URL spelled out."""
        said = spoken("See [the notes](https://example.com/a/b?c=d).")
        assert "the notes" in said and "example.com" not in said

    def test_an_image_is_skipped_entirely(self):
        said = spoken("Look: ![a chart](data:image/png;base64,AAAA)")
        assert "base64" not in said and "AAAA" not in said

    def test_a_paragraph_break_becomes_a_full_stop(self):
        said = spoken("First thought\n\nSecond thought")
        assert said == "First thought. Second thought"

    def test_but_not_where_there_is_one_already(self):
        """Every heading was read "Trip summary.."."""
        said = spoken("Trip summary.\n\nIt was long.")
        assert ".." not in said

    @pytest.mark.parametrize("raw", ["", "   ", "\n\n", "```\n```"])
    def test_nothing_to_say_says_nothing(self, raw):
        assert spoken(raw).strip(" .") == ""

    def test_the_numbers_survive_all_of_it(self):
        said = spoken("It was **68 miles** at `21.7` mph, or $54.20 of fuel.")
        for figure in ("68 miles", "21.7", "$54.20"):
            assert figure in said


class TestALongReplyIsQueuedInPieces:
    """Chrome stops speaking after roughly fifteen seconds of one utterance, so
    a long reply spoken as a single one simply stops part-way through.
    """

    def test_sentences_become_separate_utterances(self):
        assert len(chunks("One. Two. Three.")) == 3

    def test_a_sentence_too_long_for_one_utterance_is_broken_up(self):
        long_one = ", ".join(["a clause of some length"] * 30)
        for part in chunks(long_one):
            assert len(part) <= 240, part

    def test_it_breaks_on_punctuation_rather_than_mid_word(self):
        long_one = ", ".join(["a clause of some length"] * 30)
        for part in chunks(long_one):
            assert not part.startswith("length"), "a word was cut in half"

    def test_blank_pieces_are_dropped(self):
        assert "" not in chunks("One.\n\n\n   \n\nTwo.")

    def test_nothing_in_nothing_out(self):
        assert chunks("   ") == []


class TestItStopsWhenItShould:
    """On a phone, a voice that carries on after you have moved on is not a
    feature.
    """

    def page_js(self):
        return page_script(page())

    @pytest.mark.parametrize("moment, why", [
        ("function newChat()", "starting a new chat"),
        ("function stop()", "pressing Stop"),
    ])
    def test_it_is_silenced_at(self, moment, why):
        js = self.page_js()
        at = js.index(moment)
        assert "stopSpeaking();" in js[at:at + 400], why

    def test_asking_the_next_question_silences_the_last_answer(self):
        js = self.page_js()
        at = js.index("if ((!text && !images.length) || busy) return false;")
        assert "stopSpeaking();" in js[at:at + 300]

    def test_opening_another_conversation_silences_it(self):
        js = self.page_js()
        at = js.index("async function loadConversation")
        assert "stopSpeaking();" in js[at:at + 500]

    def test_leaving_the_tab_silences_it(self):
        """A voice coming out of an app you have switched away from."""
        js = self.page_js()
        assert 'window.addEventListener("pagehide", stopSpeaking);' in js
        at = js.index('if (ttsReady()) {')
        assert 'document.visibilityState !== "visible"' in js[at:at + 900]

    def test_the_same_button_starts_and_stops_it(self):
        """The reply being read is the one thing you want to interrupt, and
        hunting for a separate control to do it is the wrong answer."""
        js = self.page_js()
        at = js.index("function speakReply(view)")
        assert "if (speaking && speaking.view === view) { stopSpeaking(); return; }" \
            in js[at:at + 400]

    def test_a_finished_reply_only_clears_the_state_if_it_is_still_its_own(self):
        """A new reply can take over while the old one's queue is still
        draining; clearing blindly would silence the new one.

        Both the finished handler and the failed one, or the half that is
        unguarded silences the new reply on its own.
        """
        js = self.page_js()
        at = js.index("function speakReply(view)")
        body = js[at:js.index("// ---- Search ----", at)]
        assert body.count("if (speaking === mine) stopSpeaking();") == 2, body
        for handler in ("say.onend =", "say.onerror ="):
            at_handler = body.index(handler)
            assert "if (speaking === mine)" in body[at_handler:at_handler + 80], handler


class TestTheControls:
    def test_the_button_is_offered_on_every_reply(self):
        js = page_script(page())
        assert "addReplySpeak(view);" in js
        assert 'btn.className = "replycopy replyspeak";' in js

    def test_it_is_found_by_class_not_by_label(self):
        """The label changes to "Stop reading" while it is going."""
        js = page_script(page())
        at = js.index("function addReplySpeak")
        assert "replyspeak" in js[at:at + 600]

    def test_on_demand_works_whatever_the_toggle_says(self):
        """Wanting one reply read out is not wanting all of them read out, and
        it is the more common of the two."""
        js = page_script(page())
        at = js.index("function addReplySpeak")
        window = js[at:at + 700]
        assert "speakEl.checked" not in window

    def test_the_toggle_reads_a_finished_reply(self):
        js = page_script(page())
        assert "if (speakEl.checked) speakReply(view);" in js

    def test_it_speaks_on_completion_rather_than_while_streaming(self):
        """Chunking a half-arrived sentence reads it wrong, and re-reading the
        corrected version reads it twice."""
        js = page_script(page())
        at = js.index("if (speakEl.checked) speakReply(view);")
        assert "const finalContent = splitThink(rawContent).content;" in js[:at]

    def test_the_voice_and_the_toggle_are_remembered(self):
        js = page_script(page())
        assert 'remember("ttsVoice", ttsVoiceEl.value)' in js
        assert 'rememberToggle(speakEl, "chatSpeak", false)' in js

    def test_this_page_s_language_is_offered_first(self):
        """A browser ships thirty voices and twenty-eight are not the language
        you are reading in."""
        js = page_script(page())
        at = js.index("function loadTtsVoices")
        assert "navigator.language" in js[at:at + 1200]

    def test_a_voice_name_is_never_written_as_markup(self):
        js = page_script(page())
        at = js.index("function loadTtsVoices")
        assert "opt.textContent = voice.name" in js[at:at + 1200]

    def test_a_browser_with_no_speech_shows_none_of_it(self):
        text = page()
        assert 'id="speakbar" hidden' in text
        assert 'id="ttsVoice" title="Which voice reads replies aloud" hidden' in text
        js = page_script(text)
        assert "function ttsReady()" in js
        at = js.index("if (ttsReady()) {")
        assert "speakBarEl.hidden = false;" in js[at:at + 200]

    def test_voices_that_arrive_late_still_fill_the_picker(self):
        """Chrome populates the list asynchronously; without this the picker is
        empty on the first load of every session."""
        js = page_script(page())
        assert 'addEventListener("voiceschanged", loadTtsVoices)' in js
