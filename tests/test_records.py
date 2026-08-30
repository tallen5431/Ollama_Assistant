"""Records — what a routine run wrote down.

A routine that reads two odometers produces a paragraph. What is still useful a
year later is not the paragraph; it is 68 miles, 3 h 08 min, on the 7th of
August. These cover the three halves of that: pulling fields out of prose,
keeping them, and getting them back out as something another machine can read.
"""

from __future__ import annotations

import csv
import importlib
import io
import json
import re

import pytest

import app as app_module
import chat_ui
import ollama_client
import records
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


@pytest.fixture
def replies(monkeypatch):
    """Script what the extraction model says."""
    said = {"text": "{}"}
    monkeypatch.setattr(ollama_client, "chat", lambda *a, **k: said["text"])
    return said


# --------------------------------------------------------------------------
# Pulling fields out of prose
# --------------------------------------------------------------------------

WANTED = ["distance", "elapsed"]


class TestExtraction:
    def test_a_clean_object(self, replies):
        replies["text"] = '{"distance": "68 miles", "elapsed": "3 h 08 min"}'
        assert records.extract("answer", WANTED, "m") == {
            "distance": "68 miles", "elapsed": "3 h 08 min"}

    @pytest.mark.parametrize("wrapper, why", [
        ('Here is the JSON:\n```json\n{JSON}\n```\nHope that helps.', "fenced with prose"),
        ('{JSON}\n\nLet me know if you need anything else.', "a sentence after"),
        ('Sure!\n{JSON}', "a word before"),
    ])
    def test_however_a_small_model_wraps_it(self, replies, wrapper, why):
        body = '{"distance": "68 miles", "elapsed": "3 h 08 min"}'
        replies["text"] = wrapper.replace("{JSON}", body)
        assert records.extract("answer", WANTED, "m")["distance"] == "68 miles", why

    def test_the_case_of_the_keys_does_not_matter(self, replies):
        replies["text"] = '{"Distance": "68 miles", "ELAPSED": "3 h"}'
        assert records.extract("answer", WANTED, "m") == {
            "distance": "68 miles", "elapsed": "3 h"}

    def test_a_scratch_object_before_the_real_one(self, replies):
        """A model that thinks out loud in JSON writes a throwaway first.

        Taking it gave a row of empty strings that looked like a real record.
        """
        replies["text"] = '{"reasoning": "let me see"} {"distance": "68 miles", "elapsed": "3 h"}'
        assert records.extract("answer", WANTED, "m")["distance"] == "68 miles"

    def test_only_the_fields_that_were_asked_for(self, replies):
        """An invented column would silently widen the table."""
        replies["text"] = '{"distance": "68 miles", "elapsed": "3 h", "vibes": "good"}'
        assert set(records.extract("answer", WANTED, "m")) == set(WANTED)

    def test_the_declared_order_is_kept(self, replies):
        replies["text"] = '{"elapsed": "3 h", "distance": "68 miles"}'
        assert list(records.extract("answer", WANTED, "m")) == WANTED

    @pytest.mark.parametrize("value, kept", [
        (None, ""), (False, ""), (["a", "b"], "a, b"), (12, "12"),
        ("  spaced   out  ", "spaced out"),
    ])
    def test_awkward_values(self, replies, value, kept):
        replies["text"] = json.dumps({"distance": value, "elapsed": "x"})
        assert records.extract("answer", WANTED, "m")["distance"] == kept

    @pytest.mark.parametrize("text", ["I cannot do that.", "", "{not json"])
    def test_no_usable_object_is_not_a_record(self, replies, text):
        replies["text"] = text
        assert records.extract("answer", WANTED, "m") == {}

    def test_a_model_that_cannot_be_reached_costs_a_record_not_the_turn(self, monkeypatch):
        """The answer is already on screen; this is the extra."""
        def boom(*a, **k):
            raise RuntimeError("Ollama is asleep")
        monkeypatch.setattr(ollama_client, "chat", boom)
        assert records.extract("answer", WANTED, "m") == {}

    @pytest.mark.parametrize("answer, fields, model", [
        ("", WANTED, "m"), ("answer", [], "m"), ("answer", WANTED, ""),
    ])
    def test_nothing_to_do(self, replies, answer, fields, model):
        replies["text"] = '{"distance": "x"}'
        assert records.extract(answer, fields, model) == {}

    def test_a_reasoning_block_is_not_restated(self, replies, monkeypatch):
        """It is the model talking to itself; as data it means nothing."""
        seen = {}
        monkeypatch.setattr(ollama_client, "chat",
                            lambda m, msgs, **k: seen.setdefault("p", msgs[-1]["content"])
                            and '{"distance":"x","elapsed":"y"}' or '{"distance":"x","elapsed":"y"}')
        records.extract("<think>hmm, carry the one</think>68 miles.", WANTED, "m")
        assert "carry the one" not in seen["p"]
        assert "68 miles" in seen["p"]


class TestValuesAreWrittenAsData:
    """A field goes into a table cell and a CSV, neither of which renders TeX.

    Asking the model not to emit it is not enough on its own: the instruction
    it is following in the same breath is "copy figures exactly as the answer
    gives them", and the answer had them in LaTeX. Reported from a real Uber
    Trip record whose Elapsed time read
    "3 hours and 8 minutes (or $\\approx 3.13$ hours)".
    """

    @pytest.mark.parametrize("written, plain", [
        (r"3 hours and 8 minutes (or $\approx 3.13$ hours)",
         "3 hours and 8 minutes (or ≈ 3.13 hours)"),
        (r"$\frac{68}{3.13}$ mph", "68 / 3.13 mph"),
        (r"\(21.7\) mph", "21.7 mph"),
        (r"\[68 \times 2\]", "68 × 2"),
        (r"$\text{3 h 08 min}$", "3 h 08 min"),
        (r"$\sqrt{16}$ miles", "√(16) miles"),
        ("68 miles", "68 miles"),
        ("", ""),
    ])
    def test_tex_is_unwrapped(self, written, plain):
        assert records._plain(written) == plain

    @pytest.mark.parametrize("money", ["$54.20", "$5 to $10", "It cost $100,407",
                                       "$5-$10 per mile"])
    def test_money_is_left_alone(self, money):
        """A receipt routine records prices, and a dollar pair around prose is
        two prices — turning that into arithmetic is the worse failure."""
        assert records._plain(money) == money

    def test_it_runs_on_what_the_model_returns(self, replies):
        # The JSON has to carry an escaped backslash, as a model's would.
        replies["text"] = json.dumps({"distance": r"$\approx 68$ miles",
                                      "elapsed": "3 h"})
        assert records.extract("answer", WANTED, "m")["distance"] == "≈ 68 miles"

    def test_the_model_is_asked_for_plain_text_too(self):
        """Belt and braces: stripping is the guarantee, asking is the cheap
        half that also keeps the value readable."""
        assert "No LaTeX, no markdown, no formatting" in records._PROMPT

    def test_records_already_kept_get_tidied_once(self):
        """Written straight to the table, because that is the case this is for:
        a record kept by a version that did not strip LaTeX. Going through
        add_record would prove nothing — it cleans on the way in now, so there
        would be nothing left for the backfill to find."""
        import json as _json, time as _time, uuid as _uuid
        rid = _uuid.uuid4().hex
        with store._connect() as conn:
            conn.execute(
                "INSERT INTO records (id, routine_name, fields, created_at)"
                " VALUES (?, ?, ?, ?)",
                (rid, "🚗 Uber Trip", _json.dumps({
                    "Elapsed time": r"3 hours and 8 minutes (or $\approx 3.13$ hours)",
                    "Fare": "$54.20"}), _time.time()))
        assert records.tidy_stored() == 1
        fields = store.list_records()[0]["fields"]
        assert "\\approx" not in fields["Elapsed time"], "the LaTeX survived"
        assert fields["Fare"] == "$54.20", "money is still money"
        assert records.tidy_stored() == 0, "and it is idempotent"
        assert rid == store.list_records()[0]["id"]

    def test_a_clean_log_is_left_untouched(self):
        store.add_record("T", {"distance": "68 miles"})
        assert records.tidy_stored() == 0

    def test_no_database_is_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CHAT_DB", str(tmp_path / "absent.db"))
        assert records.tidy_stored() == 0


# --------------------------------------------------------------------------
# Keeping them
# --------------------------------------------------------------------------

class TestStorage:
    def test_a_record_round_trips(self):
        kept = store.add_record("🚗 Trip", {"distance": "68 miles"}, "rid", "cid")
        back = store.list_records()[0]
        assert back["id"] == kept["id"]
        # Standardised on the way in — and the original is still here, because
        # a tidy-up that cannot be checked against what it replaced is one you
        # have to take on faith.
        assert back["fields"] == {"distance": "68 mi"}
        assert back["raw"] == {"distance": "68 miles"}
        assert back["routine_name"] == "🚗 Trip"
        assert back["conversation_id"] == "cid"

    def test_an_entirely_empty_extraction_is_not_kept(self):
        """A blank row in a log the owner is meant to trust is worse than none."""
        assert store.add_record("T", {"a": "", "b": None}) is None
        assert store.add_record("T", {}) is None
        assert store.list_records() == []

    def test_one_field_filled_is_enough(self):
        assert store.add_record("T", {"a": "", "b": "yes"}) is not None

    def test_newest_first(self):
        for i in range(3):
            store.add_record("T", {"n": str(i)})
        assert [r["fields"]["n"] for r in store.list_records()] == ["2", "1", "0"]

    def test_one_routine_can_be_singled_out(self):
        store.add_record("🚗 Trip", {"a": "1"})
        store.add_record("📄 Read this", {"b": "2"})
        assert len(store.list_records("🚗 Trip")) == 1
        assert len(store.list_records()) == 2

    def test_deleting_the_routine_leaves_the_log_alone(self):
        """A year of trips outlives the routine that made them."""
        routine = store.create_routine("🚗 Trip", "b", record=["distance"])
        store.add_record("🚗 Trip", {"distance": "68 miles"}, routine["id"])
        assert store.delete_routine(routine["id"]) is True
        assert len(store.list_records()) == 1
        assert store.list_records()[0]["routine_name"] == "🚗 Trip"

    def test_an_edit_merges_rather_than_replaces(self):
        """Correcting one cell must not drop the columns it did not mention."""
        kept = store.add_record("T", {"a": "1", "b": "2", "c": "3"})
        after = store.update_record(kept["id"], {"b": "CHANGED"})
        assert after["fields"] == {"a": "1", "b": "CHANGED", "c": "3"}

    def test_a_field_can_be_cleared(self):
        kept = store.add_record("T", {"a": "1", "b": "2"})
        assert store.update_record(kept["id"], {"b": ""})["fields"]["b"] == ""

    def test_editing_something_that_is_not_there(self):
        assert store.update_record("nope", {"a": "1"}) is None
        assert store.delete_record("nope") is False

    def test_columns_are_the_union_in_first_seen_order(self):
        """Routines change; two runs can disagree about their columns."""
        store.add_record("T", {"a": "1", "b": "2"})
        store.add_record("T", {"b": "3", "c": "4"})
        assert store.record_columns(store.list_records()) == ["b", "c", "a"]

    def test_a_value_is_a_field_not_an_essay(self):
        kept = store.add_record("T", {"a": "x" * 900})
        assert len(kept["fields"]["a"]) == 300


class TestRoutinesDeclareWhatTheyKeep:
    def test_the_field_list_round_trips(self):
        made = store.create_routine("T", "b", record=["distance", "elapsed"])
        assert made["record"] == ["distance", "elapsed"]
        assert store.list_routines()[0]["record"] == ["distance", "elapsed"]

    def test_a_routine_that_keeps_nothing(self):
        assert store.create_routine("T", "b")["record"] == []

    @pytest.mark.parametrize("given, kept", [
        (["a", "a", "b"], ["a", "b"]),                    # deduplicated
        (["  spaced  out ", ""], ["spaced out"]),         # tidied, blanks dropped
        (["x" * 99], ["x" * 32]),                         # capped
        ([f"f{i}" for i in range(20)], [f"f{i}" for i in range(12)]),
        ("not a list", []),
        (None, []),
    ])
    def test_the_list_is_cleaned_up(self, given, kept):
        assert store.create_routine("T", "b", record=given)["record"] == kept

    def test_it_can_be_changed_and_cleared(self):
        made = store.create_routine("T", "b", record=["a"])
        assert store.update_routine(made["id"], {"record": ["b", "c"]})["record"] == ["b", "c"]
        assert store.update_routine(made["id"], {"record": []})["record"] == []


# --------------------------------------------------------------------------
# Getting them back out
# --------------------------------------------------------------------------

class TestTheRoutes:
    def test_a_fresh_install_has_none(self, client):
        assert client.get("/api/records").get_json() == {"records": [], "columns": []}

    def test_a_run_becomes_a_row(self, client, replies):
        replies["text"] = '{"distance": "68 miles", "elapsed": "3 h 08 min"}'
        out = client.post("/api/records", json={
            "answer": "You drove 68 miles in 3 h 08 min.",
            "fields": WANTED, "routine_name": "🚗 Trip"}).get_json()
        assert out["record"]["fields"]["distance"] == "68 mi"
        listed = client.get("/api/records").get_json()
        assert listed["columns"] == WANTED
        assert len(listed["records"]) == 1

    def test_an_answer_nothing_could_be_pulled_from_is_not_an_error(self, client, replies):
        """It is an outcome. The reply it came from is still on screen."""
        replies["text"] = "I have no idea."
        resp = client.post("/api/records", json={"answer": "x", "fields": WANTED})
        assert resp.status_code == 200
        assert resp.get_json()["record"] is None

    @pytest.mark.parametrize("body", [
        {"fields": WANTED}, {"answer": "x"}, {"answer": "  ", "fields": WANTED},
        {"answer": "x", "fields": []}, {"answer": "x", "fields": "nope"},
    ])
    def test_a_request_that_cannot_mean_anything(self, client, body):
        assert client.post("/api/records", json=body).status_code == 400

    @pytest.mark.parametrize("payload", [["a"], "hi", 7, None])
    def test_a_body_that_is_not_an_object_is_a_400_not_a_500(self, client, payload):
        assert client.post("/api/records", json=payload).status_code == 400

    def test_a_cell_can_be_corrected(self, client, replies):
        replies["text"] = '{"distance": "68 miles", "elapsed": "3 h"}'
        made = client.post("/api/records", json={
            "answer": "a", "fields": WANTED, "routine_name": "T"}).get_json()["record"]
        out = client.patch(f"/api/records/{made['id']}",
                           json={"fields": {"distance": "70 miles"}}).get_json()
        # A hand correction is standardised like any other value, so the column
        # stays consistent whoever typed the cell.
        assert out["record"]["fields"] == {"distance": "70 mi", "elapsed": "3h"}
        assert out["record"]["raw"]["distance"] == "70 miles"

    def test_patch_and_delete_on_something_that_is_not_there(self, client):
        assert client.patch("/api/records/nope", json={"fields": {"a": "1"}}).status_code == 404
        assert client.delete("/api/records/nope").status_code == 404

    def test_patch_without_fields(self, client):
        assert client.patch("/api/records/x", json={}).status_code == 400

    def test_csv_is_something_another_machine_can_read(self, client, replies):
        replies["text"] = '{"distance": "68 miles", "elapsed": "3 h 08 min"}'
        client.post("/api/records", json={"answer": "a", "fields": WANTED,
                                          "routine_name": "🚗 Trip"})
        resp = client.get("/api/records.csv")
        assert resp.status_code == 200
        assert resp.mimetype == "text/csv"
        assert "attachment" in resp.headers["Content-Disposition"]
        rows = list(csv.reader(io.StringIO(resp.get_data(as_text=True))))
        # The unit is in the header and the cell holds the bare number, so a
        # spreadsheet sums the column instead of treating it as text.
        assert rows[0] == ["taken_at", "routine",
                           "distance (mi)", "elapsed (hours)"]
        assert rows[1][1:] == ["🚗 Trip", "68", "3.1333"]
        # An ISO timestamp, so a spreadsheet and a database both parse it.
        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$", rows[1][0])

    def test_csv_of_an_empty_log_is_still_valid_csv(self, client):
        rows = list(csv.reader(io.StringIO(
            client.get("/api/records.csv").get_data(as_text=True))))
        assert rows == [["taken_at", "routine"]]

    def test_a_ragged_log_keeps_every_column(self, client):
        """Two runs of a routine that changed its fields between them.

        Taking the columns off the newest row alone silently drops whatever the
        older runs recorded — which is exactly the data you kept them for.
        """
        store.add_record("T", {"a": "1"})
        store.add_record("T", {"b": "2"})
        rows = list(csv.reader(io.StringIO(
            client.get("/api/records.csv").get_data(as_text=True))))
        assert sorted(rows[0][2:]) == ["a", "b"], rows[0]
        assert all(len(r) == len(rows[0]) for r in rows), rows
        assert sorted(r[2] + r[3] for r in rows[1:]) == ["1", "2"]

    def test_the_routes_are_behind_auth_when_it_is_on(self, monkeypatch):
        """The log is as exposed as the history is."""
        monkeypatch.setenv("CHAT_AUTH_USER", "tj")
        monkeypatch.setenv("CHAT_AUTH_PASSWORD", "hunter2")
        guarded = importlib.reload(app_module).app.test_client()
        assert guarded.get("/api/records").status_code == 401
        assert guarded.get("/api/records.csv").status_code == 401
        assert guarded.post("/api/records", json={"answer": "a", "fields": ["b"]}).status_code == 401
        assert guarded.delete("/api/records/x").status_code == 401

    def test_a_broken_database_is_a_503_with_json(self, client, tmp_path, monkeypatch):
        broken = tmp_path / "broken.db"
        broken.write_bytes(b"not a database" * 100)
        monkeypatch.setenv("CHAT_DB", str(broken))
        resp = client.get("/api/records")
        assert resp.status_code == 503
        assert resp.get_json()["history"] is False


# --------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------

class TestThePage:
    def page(self):
        return chat_ui.render_page("t")

    def test_there_is_a_records_pane(self):
        page = self.page()
        assert 'id="tabRecords"' in page and 'id="recordPane"' in page

    def test_a_routine_run_keeps_a_record_without_being_asked(self):
        page = self.page()
        # Anchored to the start of the statement, so a call that has been
        # short-circuited or commented out does not still match.
        assert "\n          keepRecord(routine, last.content, root ? { root: root } : null);" in page

    def test_a_routine_that_keeps_nothing_makes_no_call(self):
        page = self.page()
        at = page.index("async function keepRecord")
        assert "!routine.record || !routine.record.length" in page[at:at + 400]

    def test_the_kept_line_follows_the_declared_order(self):
        """jsonify sorts keys, so the wire order is alphabetical."""
        page = self.page()
        assert "showKept(view, data.record, routine.record)" in page
        assert "order && order.length ? order : Object.keys" in page

    def test_stored_text_never_becomes_markup(self):
        """A routine name and a field value are both stored text."""
        page = self.page()
        at = page.index("function renderRecords")
        body = page[at:page.index("async function editRecord", at)]
        for line in body.splitlines():
            code = line.split("//")[0]
            if "innerHTML" in code:
                assert '= ""' in code, line.strip()
        assert "td.textContent = text;" in body, \
            "every cell goes in as text, whatever it holds"

    def test_the_cells_can_be_corrected(self):
        page = self.page()
        assert 'td.contentEditable = "true";' in page
        assert "editRecord(record.id, name, value)" in page

    def test_both_exports_are_offered(self):
        page = self.page()
        assert '["⤓ CSV", "api/records.csv" + query, "records.csv"]' in page
        assert '["⤓ JSON", "api/records" + query, null]' in page

    def test_the_exports_follow_the_filter(self):
        """Exporting one routine while looking at one routine."""
        page = self.page()
        assert 'const query = recordFilter ? "?routine=" + encodeURIComponent(recordFilter) : "";' in page


class TestTheLayoutFitsWhereItIsShown:
    """A table with eight columns in a 320px drawer is unreadable at any width;
    the header alone wraps to two lines per column. Measured on the owner's
    screenshots — narrow gets cards, wide gets the room a table needs.
    """

    def page(self):
        return chat_ui.render_page("t")

    def test_a_narrow_screen_gets_one_card_per_record(self):
        page = self.page()
        assert "#recordList thead { display:none; }" in page
        assert "#recordList td::before { content:attr(data-label);" in page
        assert 'td.setAttribute("data-label", label);' in page, \
            "the heading has to travel down with the cell"

    def test_the_card_layout_is_only_for_narrow_screens(self):
        """A table is the right shape for a log when there is room for one."""
        page = self.page()
        at = page.index("#recordList .tablewrap { overflow-x:visible; }")
        media = page.rindex("@media", 0, at)
        assert "max-width: 720px" in page[media:at]

    def test_a_wide_table_fills_the_drawer_instead_of_hugging_the_left(self):
        """Four columns in a 64rem drawer left three quarters of it empty."""
        page = self.page()
        assert "#recordList table { border-collapse:collapse; min-width:100%;" in page
        assert "#recordList th, #recordList td { border:1px solid var(--border);" in page

    def test_the_card_rules_come_after_the_table_rules(self):
        """Both selectors weigh the same, so the later one wins. Put the table
        borders after the media block and every phone card grows gridlines.
        """
        page = self.page()
        table = page.index("#recordList th, #recordList td { border:1px solid")
        card = page.index("#recordList td { border:0;")
        assert table < card, "the narrow-screen block has to be able to override"

    def test_the_log_gets_the_window_rather_than_a_drawer(self):
        """A drawer is where you pick something. Records is a table you read,
        filter, correct and export — a destination, and the only part of the
        app that wants the whole width. At 21rem it was eight columns in 320px;
        widened to 56rem it pushed the conversation it came from off screen.
        """
        text = self.page()
        assert 'id="recordPane" class="recordsview"' in text
        assert text.index('</aside>') < text.index('id="recordPane"'), \
            "it has to be out of the drawer, not a pane inside it"
        assert ".recordsview { flex:1 1 auto;" in text

    def test_the_composer_goes_away_while_it_is_open(self):
        """There is nothing to type at, and on a phone it was 150px of the
        screen the table needed."""
        assert "body.records .chatarea, body.records footer { display:none !important; }" \
            in self.page()

    def test_there_is_a_way_back(self):
        page = self.page()
        assert 'id="backToChat"' in page
        js = page_script(page)
        assert "backBtn.addEventListener(\"click\", closeRecords);" in js
        at = js.index('if (e.key !== "Escape") return;')
        assert "if (recordsOpen()) closeRecords();" in js[at:at + 400], "and Escape does it too"

    def test_the_bar_gives_the_conversation_its_name_back(self):
        """Records borrows the title bar; leaving it on "Records" afterwards
        would say you were somewhere you are not."""
        js = page_script(self.page())
        assert "let currentTitle" in js
        assert 'if (!recordsOpen()) currentTitle = text || "";' in js
        at = js.index("function closeRecords")
        assert "setChatTitle(currentTitle);" in js[at:at + 300]

    def test_starting_or_opening_a_conversation_leaves_the_log(self):
        js = page_script(self.page())
        assert js.count("if (recordsOpen()) closeRecords();") >= 3, \
            "new chat, opening a thread, and Escape"

    def test_the_drawer_is_no_longer_widened_for_it(self):
        """Dead once the log stopped living in there."""
        text = self.page()
        assert "drawer.wide" not in text
        assert 'classList.toggle("wide"' not in text

    def test_records_can_be_narrowed_to_one_routine(self):
        """The columns are the union across routines, so three routines with
        four fields each is a twelve-column table. Filtering is the fix."""
        page = self.page()
        assert "let recordFilter = \"\";" in page
        assert "records.filter(r => r.routine_name === recordFilter)" in page

    def test_the_columns_follow_the_filter(self):
        """Otherwise filtering leaves the empty columns of the other routines."""
        page = self.page()
        at = page.index("const shown = recordFilter")
        window = page[at:at + 500]
        assert "for (const record of shown)" in window
        assert "columns.push(name)" in window

    def test_one_routine_is_not_a_choice(self):
        page = self.page()
        assert "if (names.length < 2) return bar;" in page

    def test_the_routine_column_goes_away_when_it_is_redundant(self):
        page = self.page()
        assert 'if (!recordFilter) cell("Routine", record.routine_name);' in page

    def test_the_editor_asks_for_the_field_names(self):
        page = self.page()
        assert 'id="rRecord"' in page
        # Lines, not commas: a declaration can now carry a kind or a formula,
        # and three field names joined by an operator do not fit on a shared
        # line. Commas still separate — fields.py accepts either.
        assert 'record: rRecordEl.value.split("\\n")' in page

    def test_the_editor_explains_the_syntax(self):
        """A box that silently accepts "name: money" and "a = b / c" without
        saying so is a box nobody will ever type them into."""
        page = self.page()
        assert "<code>name: kind</code>" in page
        assert "<code>name = a / b</code>" in page

    def test_a_standardised_cell_shows_what_it_used_to_say(self):
        """The original is stored so the tidy-up can be checked. Storing it and
        never showing it would leave that promise reachable only by curl."""
        page = self.page()
        assert 'td.title = "As it was recorded: " + was;' in page
        assert 'td.classList.add("tidied");' in page
        assert "#recordList .tidied" in page, "the marker has no style"

    def test_the_marker_is_added_after_the_class_is_assigned(self):
        """td.className = "editable" replaces the whole attribute, so adding
        the marker first meant it silently never appeared."""
        page = self.page()
        assert page.index('td.className = "editable";') < \
            page.index('td.classList.add("tidied");')

    @pytest.mark.parametrize("value, expected", [
        ("=cmd|' /C calc'!A1", "'=cmd|' /C calc'!A1"),
        ("+1+1", "'+1+1"),
        ("@SUM(A1)", "'@SUM(A1)"),
        # Standardised first, then defused: a quantity comes out as a bare
        # number, and neither step can turn the other's output into a formula.
        ("68 miles", "68"),
        ("-5 °C", "-5 °C"),          # a reading, not a formula
        ("3 h 08 min", "3.1333"),
        # A formula wearing a unit is still prose to the standardiser — it
        # refuses anything that is not wholly a quantity — so it arrives at the
        # defuser untouched and gets the quote it needs.
        ("=1+1 miles", "'=1+1 miles"),
        ("@SUM(A1) mph", "'@SUM(A1) mph"),
    ])
    def test_a_cell_cannot_become_a_formula(self, client, value, expected):
        """Excel runs a leading =, + or @ when the file is opened.

        "-" is handled separately, by the two tests below: a bare negative
        number is a reading and is left alone, but "-2+3" opens as 1.
        """
        store.add_record("T", {"v": value})
        rows = list(csv.reader(io.StringIO(
            client.get("/api/records.csv").get_data(as_text=True))))
        assert rows[1][2] == expected

    @pytest.mark.parametrize("value", [
        "-2+3",                      # a spreadsheet opens this as 1
        "-1+cmd|'/c calc'!A0",       # and this as a running program
        "-1*7", "-(1)", "-1&\"x\"",
    ])
    def test_a_leading_minus_is_defused_when_it_starts_a_formula(self, value):
        """The rule used to skip "-" entirely, on the stated grounds that a
        spreadsheet only treats it as a formula when what follows is not a
        number. It does: "-2+3" opens as 1. Record values are written by a
        model reading a page or a photo, so they are not the owner's own
        typing."""
        import app
        assert app._csv_safe(value).startswith("'"), value

    @pytest.mark.parametrize("value", [
        "-5 °C", "-40", "-1,250.75", "-2.5", "-5-3", "-",
    ])
    def test_but_a_reading_is_left_exactly_as_it_was(self, value):
        """Which is why the rule is not simply "quote every minus": quoting a
        temperature corrupts the log to prevent nothing."""
        import app
        assert app._csv_safe(value) == value

    def test_a_routine_name_cannot_become_a_formula_either(self, client):
        store.add_record("=HYPERLINK(\"http://evil\")", {"v": "1"})
        rows = list(csv.reader(io.StringIO(
            client.get("/api/records.csv").get_data(as_text=True))))
        assert rows[1][1].startswith("'=")


class TestFieldsKeepTheirDeclaredOrder:
    """A trip reads start, end, distance — not "average speed" first.

    Flask sorts JSON keys by default, so the wire order was alphabetical and
    every consumer of it inherited that: the line under the reply, the table
    columns, and the cards on a phone.
    """

    def test_the_wire_keeps_the_order_the_record_was_built_in(self, client):
        order = ["Start odometer", "End odometer", "Distance", "Average speed"]
        store.add_record("T", {name: str(i) for i, name in enumerate(order)})
        out = client.get("/api/records").get_json()
        assert list(out["records"][0]["fields"]) == order
        assert out["columns"] == order

    def test_the_csv_columns_follow_it_too(self, client):
        order = ["Start odometer", "End odometer", "Distance"]
        store.add_record("T", {name: str(i) for i, name in enumerate(order)})
        rows = list(csv.reader(io.StringIO(
            client.get("/api/records.csv").get_data(as_text=True))))
        assert rows[0][2:] == order

    def test_it_is_switched_off_at_the_app_rather_than_per_route(self):
        """Every response, so a later route cannot quietly reintroduce it."""
        assert app_module.app.json.sort_keys is False


class TestATypedRoutineEndToEnd:
    """The declaration reaches the model, the arithmetic, and the database."""

    DECL = ["Start odometer: distance", "End odometer: distance",
            "Distance traveled = End odometer - Start odometer",
            "Start time: timestamp", "End time: timestamp",
            "Elapsed time = End time - Start time",
            "Total earnings: money",
            "Earnings per hour = Total earnings / Elapsed time"]

    FULL = ('{"Start odometer": "102,072", "End odometer": "102,165",'
            ' "Start time": "Tuesday 25 August 2026 at 20:06 (UTC-04:00)",'
            ' "End time": "Wednesday 26 August 2026 at 00:31 (UTC-04:00)",'
            ' "Total earnings": "$115.94"}')
    NO_TIMES = ('{"Start odometer": "102,072 mi", "End odometer": "102,165 mi",'
                ' "Start time": "", "End time": "", "Total earnings": "$115.94"}')

    def post(self, client, replies, text):
        replies["text"] = text
        return client.post("/api/records", json={
            "answer": "a", "fields": self.DECL, "routine_name": "Trip"}).get_json()

    def test_the_model_is_only_asked_for_the_readings(self, client, replies, monkeypatch):
        asked = {}
        real = records.extract
        monkeypatch.setattr(records, "extract",
                            lambda a, n, m, kinds=None: asked.update(n=n, k=kinds) or real(a, n, m, kinds))
        self.post(client, replies, self.FULL)
        assert asked["n"] == ["Start odometer", "End odometer", "Start time",
                              "End time", "Total earnings"]
        assert asked["k"]["Total earnings"] == "money"

    def test_the_derived_fields_are_worked_out_here(self, client, replies):
        got = self.post(client, replies, self.FULL)["record"]["fields"]
        assert got["Distance traveled"] == "93 mi"
        assert got["Elapsed time"] == "4h 25m"
        assert got["Earnings per hour"] == "$26.25"

    def test_a_rate_with_nothing_to_divide_by_is_left_empty(self, client, replies):
        """This row is the one that used to come back as $23.19 an hour."""
        out = self.post(client, replies, self.NO_TIMES)
        assert out["record"]["fields"]["Earnings per hour"] == ""
        assert any("Elapsed time" in g for g in out["gaps"])

    def test_the_columns_come_out_in_the_order_declared(self, client, replies):
        got = self.post(client, replies, self.FULL)["record"]["fields"]
        assert list(got) == [f.split(":")[0].split(" =")[0] for f in self.DECL]

    def test_the_declared_kind_settles_the_very_first_record(self, client, replies):
        """With one row there is no column to vote, so an undeclared odometer
        would come out bare and every row after it with a unit."""
        got = self.post(client, replies, self.FULL)["record"]["fields"]
        assert got["Start odometer"] == "102072 mi"

    def test_a_routine_keeps_its_declaration(self, client):
        made = client.post("/api/routines", json={
            "name": "Trip", "body": "b", "record": self.DECL}).get_json()
        rid = made["id"]
        listed = client.get("/api/routines").get_json()["routines"]
        kept = [r for r in listed if r["id"] == rid][0]["record"]
        assert "Elapsed time = End time - Start time" in kept
        assert "Total earnings: money" in kept


class TestYouAreToldWhatIsWrongWithADeclaration:
    def test_saving_a_routine_reports_a_typo(self, client):
        out = client.post("/api/routines", json={
            "name": "T", "body": "b",
            "record": ["Fare: money", "Took: duration",
                       "Per hour = Fare / Tooke"]}).get_json()
        assert out["problems"], "a typo saved silently"
        assert 'no field called "Tooke"' in out["problems"][0]

    def test_it_is_still_saved(self, client):
        """The column works; it just does not do what was meant. Refusing the
        save would lose the other nine fields over one typo."""
        out = client.post("/api/routines", json={
            "name": "T", "body": "b",
            "record": ["Fare: money", "Per hour = Fare / Tooke"]}).get_json()
        assert out["id"]
        assert len(client.get("/api/routines").get_json()["routines"]) == 1

    def test_a_sound_routine_reports_nothing(self, client):
        out = client.post("/api/routines", json={
            "name": "T", "body": "b",
            "record": ["Fare: money", "Took: duration",
                       "Per hour = Fare / Took"]}).get_json()
        assert out["problems"] == []

    def test_editing_one_reports_too(self, client):
        made = client.post("/api/routines", json={
            "name": "T", "body": "b", "record": ["Fare: money"]}).get_json()
        out = client.patch(f"/api/routines/{made['id']}", json={
            "record": ["Fare: money", "Per hour = Fare / Nope"]}).get_json()
        assert out["problems"] and "Nope" in out["problems"][0]

    def test_the_editor_shows_them(self):
        page = chat_ui.render_page("t")
        assert "saved.problems" in page

    def test_the_csv_uses_the_declared_kind(self, client, replies):
        """A column declared text and stored as text was exported as a number,
        because the export re-inferred instead of asking."""
        client.post("/api/routines", json={
            "name": "T", "body": "b", "record": ["Code: text"]})
        replies["text"] = '{"Code": "100"}'
        client.post("/api/records", json={"answer": "a", "fields": ["Code: text"],
                                          "routine_name": "T"})
        rows = list(csv.reader(io.StringIO(
            client.get("/api/records.csv").get_data(as_text=True))))
        assert rows[0][2] == "Code", "a text column should carry no unit"
        assert rows[1][2] == "100"

    def test_a_fixed_declaration_stops_reading_as_broken(self):
        """The warning was set on a failed save and never cleared, so fixing
        the fault and saving again left the old complaint on screen."""
        page = page_script(chat_ui.render_page("t"))
        save = page[page.index("async function saveRoutine"):]
        save = save[:save.index("savingRoutine = true;")]
        assert 'routineWarnEl.textContent = "";' in save, \
            "the warning is not cleared before a fresh attempt"
