"""Finding the thing you half remember.

A month in, the conversation list is a column of titles you no longer
recognise, and what you actually remember is a word from inside the answer —
or a number a routine wrote down. Before this there was no way to look.
"""

from __future__ import annotations

import importlib
import re

import pytest

import app as app_module
import chat_ui
import store
from conftest import page_script


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAT_DB", str(tmp_path / "chat.db"))
    yield


@pytest.fixture
def client(monkeypatch):
    for key in ("CHAT_AUTH", "CHAT_AUTH_USER", "CHAT_AUTH_PASSWORD"):
        monkeypatch.delenv(key, raising=False)
    return importlib.reload(app_module).app.test_client()


def thread(title, *messages):
    convo = store.create(title)["id"]
    for i, text in enumerate(messages):
        store.add_message(convo, "user" if i % 2 == 0 else "assistant", text)
    return convo


class TestWhereItLooks:
    def test_inside_a_reply(self):
        """The whole point: the title said nothing about the A23."""
        thread("Weekend away", "how far?", "About 54 miles, an hour on the A23.")
        found = store.search("A23")["conversations"]
        assert [c["title"] for c in found] == ["Weekend away"]

    def test_inside_a_question(self):
        thread("Untitled", "what does zygote mean?", "The first cell.")
        assert len(store.search("zygote")["conversations"]) == 1

    def test_in_a_title(self):
        thread("Recipe from the pantry photo")
        assert len(store.search("pantry")["conversations"]) == 1

    def test_in_what_a_routine_wrote_down(self):
        store.add_record("🚗 Trip", {"Distance travelled": "54 miles to Brighton"})
        found = store.search("Brighton")["records"]
        assert found[0]["fields"]["Distance travelled"] == "54 miles to Brighton"

    def test_in_a_routine_name(self):
        store.add_record("🚗 Trip", {"distance": "54"})
        assert len(store.search("Trip")["records"]) == 1

    def test_in_what_the_value_said_before_it_was_standardised(self):
        """You search for what you remember writing, and standardising
        rewrote it. "54 miles" is stored as "54 mi" and would be findable by
        neither the words that went in nor the ones anybody would think to
        type — so the original is searched alongside the tidy value."""
        for miles in ("54 miles", "31 miles", "12 miles"):
            store.add_record("🚗 Trip", {"Distance": miles})
        assert store.list_records()[0]["fields"]["Distance"] == "12 mi"
        found = store.search("54 miles")["records"]
        assert [r["fields"]["Distance"] for r in found] == ["54 mi"]

    def test_the_tidy_value_is_still_found_too(self):
        """Searching the original must not cost searching the real one."""
        for miles in ("54 miles", "31 miles", "12 miles"):
            store.add_record("🚗 Trip", {"Distance": miles})
        assert len(store.search("54 mi")["records"]) == 1

    def test_case_does_not_matter(self):
        thread("t", "The Brighton Road")
        assert len(store.search("brighton")["conversations"]) == 1
        assert len(store.search("BRIGHTON")["conversations"]) == 1


class TestWhatComesBack:
    def test_a_thread_is_one_result_however_often_it_matches(self):
        """Nine mentions of one word is one conversation, not nine rows."""
        thread("t", "miles", "miles", "miles", "miles")
        assert len(store.search("miles")["conversations"]) == 1

    def test_the_snippet_shows_where_it_matched(self):
        thread("t", "q", "It is roughly 54 miles, which takes about an hour.")
        hit = store.search("54 miles")["conversations"][0]
        assert "54 miles" in hit["snippet"]

    def test_a_long_message_is_cut_around_the_match(self):
        thread("t", "q", "x" * 500 + " needle " + "y" * 500)
        snippet = store.search("needle")["conversations"][0]["snippet"]
        assert "needle" in snippet and len(snippet) < 200
        assert snippet.startswith("…") and snippet.endswith("…")

    def test_a_title_only_match_has_no_snippet_to_show(self):
        thread("Pantry photo")
        assert store.search("pantry")["conversations"][0]["snippet"] == ""

    def test_newest_first(self):
        thread("older", "needle")
        thread("newer", "needle")
        titles = [c["title"] for c in store.search("needle")["conversations"]]
        assert titles == ["newer", "older"]

    def test_the_limit_is_bounded_whatever_is_asked_for(self):
        for i in range(5):
            thread(f"t{i}", "needle")
        assert len(store.search("needle", limit=2)["conversations"]) == 2
        assert len(store.search("needle", limit=10_000)["conversations"]) == 5


class TestAwkwardQueries:
    @pytest.mark.parametrize("query", ["", " ", "a", "  x "])
    def test_too_short_to_be_a_search(self, query):
        """One letter matches everything, which is the same as no search."""
        thread("t", "anything at all")
        assert store.search(query) == {"conversations": [], "records": []}

    def test_a_percent_sign_is_a_percent_sign(self):
        """Unescaped it is LIKE's own wildcard, so "100%" would also find
        "100 of them" — the decoy is the whole test."""
        thread("literal", "100% of it")
        thread("decoy", "100 of them, no percentage anywhere")
        assert [c["title"] for c in store.search("100%")["conversations"]] == ["literal"]

    def test_an_underscore_is_an_underscore(self):
        """Unescaped it matches any single character, so a_b would find axb."""
        thread("under", "a_b")
        thread("other", "axb")
        assert [c["title"] for c in store.search("a_b")["conversations"]] == ["under"]

    def test_a_backslash_is_a_backslash(self):
        thread("slash", r"C:\\Users\\tj")
        assert len(store.search(r"\\Users")["conversations"]) == 1

    def test_nothing_matches_is_not_an_error(self):
        thread("t", "hello")
        assert store.search("zzzqqq") == {"conversations": [], "records": []}


class TestTheRoute:
    def test_it_answers(self, client):
        thread("Weekend away", "q", "an hour on the A23")
        found = client.get("/api/search?q=A23").get_json()
        assert found["conversations"][0]["title"] == "Weekend away"

    def test_a_missing_query_is_an_empty_answer(self, client):
        assert client.get("/api/search").get_json() == {"conversations": [], "records": []}

    def test_it_is_behind_the_same_guard_as_everything_else(self, monkeypatch):
        monkeypatch.setenv("CHAT_AUTH", "tj:hunter2")
        guarded = importlib.reload(app_module).app.test_client()
        assert guarded.get("/api/search?q=anything").status_code == 401


class TestThePage:
    def page(self):
        return chat_ui.render_page("t")

    def test_the_box_is_above_the_list(self):
        text = self.page()
        assert text.index('id="convoSearch"') < text.index('id="convoList"')

    def test_typing_is_debounced(self):
        """This scans the messages table; one query per keystroke runs it eight
        times to answer the eighth."""
        js = page_script(self.page())
        at = js.index("function onSearchInput")
        assert "clearTimeout(searchTimer)" in js[at:at + 400]

    def test_a_stale_answer_is_dropped(self):
        """Typing moves on while a query is in flight, and the slower one
        landing last would paint results for a query you have left behind."""
        js = page_script(self.page())
        at = js.index("async function runSearch")
        assert "if (searchQuery() !== query) return;" in js[at:at + 700]

    def test_a_refresh_does_not_wipe_the_results(self):
        """A save, a delete and a finished turn all refresh the list, and any
        of them replacing the results reads as the search having failed."""
        js = page_script(self.page())
        at = js.index("async function refreshConversations")
        assert "if (searchQuery()) { runSearch(); return; }" in js[at:at + 500]

    def test_a_match_is_never_written_as_markup(self):
        """A snippet is a slice of a stored message; a title is stored text."""
        js = page_script(self.page())
        at = js.index("function paintSearch")
        window = js[at:js.index("function searchHeading", at)]
        code = "\n".join(line for line in window.split("\n")
                         if not line.strip().startswith("//"))
        writes = re.findall(r"(\w+)\.innerHTML\s*=", code)
        assert writes == ["convoListEl"], writes
        for field in ("title.textContent", "line.textContent", "name.textContent",
                      "vals.textContent"):
            assert field in code, field

    def test_escape_clears_it(self):
        assert 'if (e.key === "Escape") clearSearch();' in self.page()
