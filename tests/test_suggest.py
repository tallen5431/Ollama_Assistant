"""Suggesting the declaration a routine's prompt already describes.

A real routine asked for ten plain fields and spelled out, in its prompt, how
the model should work out five of them. The model did that arithmetic in prose
every run, and a fifth of its hourly rates were off by more than rounding.
Everything needed to do the sums exactly was already written down — just not in
the shape the declaration reads. These cover reading it in that shape, and
offering the result where it will be seen.
"""

from __future__ import annotations

import importlib

import pytest

import app as app_module
import chat_ui
import fields
import ollama_client
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


NAMES = ["Start odometer", "End odometer", "Distance traveled", "Start time",
         "End time", "Elapsed time", "Average speed", "Total earnings",
         "Earnings per mile", "Earnings per hour"]

# The routine's own wording, typographic operators and all.
BODY = """Two vehicle dashboard photos are attached.

Total earnings: $115.94

CALCULATIONS:

- Distance traveled = higher odometer − lower odometer
- Elapsed time = exact end camera timestamp − exact start camera timestamp
- Average speed = distance ÷ exact elapsed hours
- Earnings per mile = typed total earnings ÷ distance
- Earnings per hour = typed total earnings ÷ exact elapsed hours
"""

WANTED = ["Start odometer: distance", "End odometer: distance",
          "Distance traveled = End odometer - Start odometer",
          "Start time = earliest photo taken", "End time = latest photo taken",
          "Elapsed time = End time - Start time",
          "Average speed = Distance traveled / Elapsed time",
          "Total earnings: money",
          "Earnings per mile = Total earnings / Distance traveled",
          "Earnings per hour = Total earnings / Elapsed time"]


class TestReadingThePromptsSums:
    def test_the_real_routine_comes_out_as_its_typed_declaration(self):
        got = fields.suggest(NAMES, BODY, photos=2)
        assert got["record"] == WANTED

    def test_each_change_says_where_it_came_from(self):
        why = fields.suggest(NAMES, BODY, photos=2)["why"]
        assert any(w.startswith("Distance traveled = End odometer - Start odometer")
                   and "higher odometer" in w for w in why)
        assert any(w.startswith("Start time = earliest photo taken") for w in why)

    def test_what_it_suggests_reads_back_cleanly(self):
        got = fields.suggest(NAMES, BODY, photos=2)["record"]
        assert fields.problems(fields.parse(got), got) == []

    def test_a_routine_already_declared_that_way_gets_nothing(self):
        assert fields.suggest(WANTED, BODY, photos=2) is None

    def test_without_photos_the_times_stay_read_fields(self):
        got = fields.suggest(NAMES, BODY, photos=0)["record"]
        assert "Start time" in got and "End time" in got

    def test_the_log_decides_a_readings_kind(self):
        got = fields.suggest(["Fare", "Tip", "Total"], "Total = fare + tip", 0,
                             seen={"Fare": "money", "Tip": "money"})
        assert got["record"] == ["Fare: money", "Tip: money", "Total = Fare + Tip"]

    @pytest.mark.parametrize("said", ["Total = fare plus tip", "1. Total = Fare + Tip",
                                      "* Total = fare + tip."])
    def test_however_the_sum_is_written(self, said):
        got = fields.suggest(["Fare", "Tip", "Total"], said, 0)
        assert "Total = Fare + Tip" in got["record"]

    def test_a_formula_typed_into_the_declaration_is_read_too(self):
        """Kept as one long field name, which the model was then asked for."""
        theirs = NAMES[:6] + ["Average speed = distance ÷ exact elapsed hours"] + NAMES[7:]
        got = fields.suggest(theirs, "", photos=0)["record"]
        assert "Average speed = Distance traveled / Elapsed time" in got


class TestItKeepsQuietWhenItIsNotSure:
    def test_a_description_that_fits_two_fields_is_no_answer(self):
        assert fields.suggest(["Odometer A", "Odometer B", "Miles"],
                              "Miles = odometer - odometer") is None

    def test_a_sum_with_no_meaning_is_not_suggested(self):
        assert fields.suggest(["Fare", "Odometer", "Net"], "Net = fare - odometer", 0,
                              {"Fare": "money", "Odometer": "distance"}) is None

    def test_prose_that_only_looks_like_a_sum(self):
        assert fields.suggest(["Mood"], "Describe the weather = how it felt today.") is None

    def test_a_hyphen_in_a_name_is_not_a_subtraction(self):
        """The declaration grammar could not read it back, so it is not offered."""
        assert fields.suggest(["Check-in time", "Check-out time", "Stay"],
                              "Stay = check-out time − check-in time") is None

    def test_no_declaration_no_suggestion(self):
        assert fields.suggest([], BODY, photos=2) is None


class TestItIsOffered:
    def make(self):
        return store.create_routine("Trip", BODY, photos=2, record=NAMES)

    def test_the_editor_can_ask_for_it(self, client):
        rid = self.make()["id"]
        got = client.get(f"/api/routines/{rid}/suggestion").get_json()["suggestion"]
        assert got["record"] == WANTED
        assert got["routine_id"] == rid

    def test_a_routine_with_nothing_to_suggest(self, client):
        rid = store.create_routine("Trip", "b", record=WANTED)["id"]
        assert client.get(f"/api/routines/{rid}/suggestion").get_json() == {"suggestion": None}

    def test_an_unknown_routine(self, client):
        assert client.get("/api/routines/nope/suggestion").get_json() == {"suggestion": None}

    def test_it_comes_back_with_the_record_under_the_reply(self, client, monkeypatch):
        rid = self.make()["id"]
        monkeypatch.setattr(ollama_client, "chat", lambda *a, **k:
                            '{"Start odometer": "105739", "End odometer": "105811"}')
        out = client.post("/api/records", json={
            "answer": "a", "fields": NAMES, "routine_name": "Trip",
            "routine_id": rid}).get_json()
        assert out["record"] and out["suggest"]["record"] == WANTED

    def test_a_failure_costs_only_the_suggestion(self, client, monkeypatch):
        rid = self.make()["id"]
        monkeypatch.setattr(store, "routine_suggestion",
                            lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
        assert client.get(f"/api/routines/{rid}/suggestion").get_json() == {"suggestion": None}

    def test_saving_it_reworks_the_log(self, client):
        """Accepting the suggestion is a save, and a save is a recheck."""
        rid = self.make()["id"]
        store.add_record("Trip", {"Start odometer": "105739 mi", "End odometer": "105811 mi",
                                  "Distance traveled": "70 mi"})
        out = client.patch(f"/api/routines/{rid}", json={"record": WANTED}).get_json()
        assert out["problems"] == []
        assert out["rechecked"]["values"] == 1
        assert store.list_records("Trip")[0]["fields"]["Distance traveled"] == "72 mi"


class TestThePage:
    def page(self):
        return chat_ui.render_page("t")

    def test_the_editor_asks_when_it_opens(self):
        page = self.page()
        body = page[page.index("function openRoutineEditor"):]
        assert "if (routine) loadSuggestion(routine.id);" in body[:body.index("\n      }\n")]
        assert '"api/routines/" + id + "/suggestion"' in page

    def test_it_is_offered_not_applied(self):
        """It changes what the model is asked for, so a person looks first:
        the button fills the box, and only Save changes anything."""
        page = self.page()
        use = page[page.index('getElementById("rSuggestUse")'):]
        use = use[:use.index("});")]
        assert "rRecordEl.value = suggestedRecord.join" in use
        assert "fetch(" not in use

    def test_a_stale_answer_does_not_land_on_another_routine(self):
        assert "if (suggestion && id !== editingRoutineId) return;" in self.page()

    def test_the_reason_goes_in_as_text(self):
        page = self.page()
        assert "rSuggestWhyEl.textContent =" in page
        assert "rSuggestWhyEl.innerHTML" not in page

    def test_the_reply_offers_it_once_per_routine(self):
        page = self.page()
        assert "if (data.suggest) offerFix(view, data.suggest);" in page
        assert "offeredFix[suggestion.routine_id] = true;" in page
