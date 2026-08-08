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
import sqlite3

import pytest

import app as app_module
import chat_ui
import ollama_client
import records
import store


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


# --------------------------------------------------------------------------
# Keeping them
# --------------------------------------------------------------------------

class TestStorage:
    def test_a_record_round_trips(self):
        kept = store.add_record("🚗 Trip", {"distance": "68 miles"}, "rid", "cid")
        back = store.list_records()[0]
        assert back["id"] == kept["id"]
        assert back["fields"] == {"distance": "68 miles"}
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
        assert out["record"]["fields"]["distance"] == "68 miles"
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
        assert out["record"]["fields"] == {"distance": "70 miles", "elapsed": "3 h"}

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
        assert rows[0] == ["taken_at", "routine", "distance", "elapsed"]
        assert rows[1][1:] == ["🚗 Trip", "68 miles", "3 h 08 min"]
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
        assert "who.textContent = record.routine_name;" in body

    def test_the_cells_can_be_corrected(self):
        page = self.page()
        assert 'cell.contentEditable = "true";' in page
        assert "editRecord(record.id, name, value)" in page

    def test_both_exports_are_offered(self):
        page = self.page()
        assert 'csv.href = "api/records.csv"' in page
        assert 'jsonLink.href = "api/records"' in page

    def test_the_editor_asks_for_the_field_names(self):
        page = self.page()
        assert 'id="rRecord"' in page
        assert "record: rRecordEl.value.split(\",\")" in page

    @pytest.mark.parametrize("value, expected", [
        ("=cmd|' /C calc'!A1", "'=cmd|' /C calc'!A1"),
        ("+1+1", "'+1+1"),
        ("@SUM(A1)", "'@SUM(A1)"),
        ("68 miles", "68 miles"),
        ("-5 °C", "-5 °C"),          # a reading, not a formula
        ("3 h 08 min", "3 h 08 min"),
    ])
    def test_a_cell_cannot_become_a_formula(self, client, value, expected):
        """Excel runs a leading =, + or @ when the file is opened.

        Not "-": a negative number is an ordinary reading, and a spreadsheet
        only treats it as a formula when what follows is not one.
        """
        store.add_record("T", {"v": value})
        rows = list(csv.reader(io.StringIO(
            client.get("/api/records.csv").get_data(as_text=True))))
        assert rows[1][2] == expected

    def test_a_routine_name_cannot_become_a_formula_either(self, client):
        store.add_record("=HYPERLINK(\"http://evil\")", {"v": "1"})
        rows = list(csv.reader(io.StringIO(
            client.get("/api/records.csv").get_data(as_text=True))))
        assert rows[1][1].startswith("'=")
