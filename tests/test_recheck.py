"""Rechecking a log against its own readings.

A routine's sums used to be done by a model, and a real log of 27 trips showed
what that costs: one hourly rate 17% high because the model called a 2h 20m
trip "2 hours", one speed left blank, one per-mile rate rounded down. Every
reading in that log was right. These cover working the sums out again from
those readings — at startup, when the sums are declared, and when a reading is
corrected — without losing what the model wrote and without blanking anything
that cannot be worked out.
"""

from __future__ import annotations

import importlib

import pytest

import app as app_module
import fields
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


DECL = ["Start odometer: distance", "End odometer: distance",
        "Distance traveled = End odometer - Start odometer",
        "Start time: timestamp", "End time: timestamp",
        "Elapsed time = End time - Start time",
        "Total earnings: money",
        "Average speed = Distance traveled / Elapsed time",
        "Earnings per mile = Total earnings / Distance traveled",
        "Earnings per hour = Total earnings / Elapsed time"]

# Rows as the real log kept them, readings and the model's arithmetic both.
SEP_26 = {"Start odometer": "105739 mi", "End odometer": "105811 mi",
          "Distance traveled": "72 mi", "Start time": "2026-09-26 19:22",
          "End time": "2026-09-26 21:42", "Elapsed time": "2h",
          "Average speed": "36 mph", "Total earnings": "$75.71",
          "Earnings per mile": "$1.05", "Earnings per hour": "$37.86"}
# Clock times with no date: the elapsed time cannot be worked out again, so
# the recorded one has to stand, and the speed and rate are checked against it.
OCT_21 = {"Start odometer": "107298 mi", "End odometer": "107375 mi",
          "Distance traveled": "77 mi", "Start time": "18:47",
          "End time": "22:51", "Elapsed time": "4h 4m",
          "Average speed": "18.17 mph", "Total earnings": "$82.06",
          "Earnings per mile": "$1.065", "Earnings per hour": "$19.28"}
OCT_22 = {"Start odometer": "107423 mi", "End odometer": "107558 mi",
          "Distance traveled": "135 mi", "Start time": "13:49",
          "End time": "21:14", "Elapsed time": "7h 25m",
          "Average speed": "", "Total earnings": "$177.61",
          "Earnings per mile": "$1.31", "Earnings per hour": "$23.95"}
# Right to the precision shown, and written more precisely than this app would.
AUG_25 = {"Start odometer": "102072 mi", "End odometer": "102165 mi",
          "Distance traveled": "93 mi",
          "Start time": "2026-08-25 20:06 UTC-04:00",
          "End time": "2026-08-26 00:31 UTC-04:00", "Elapsed time": "4h 25m",
          "Average speed": "21.1 mph", "Total earnings": "$115.94",
          "Earnings per mile": "$1.2467", "Earnings per hour": "$26.25"}


def recompute(row):
    return fields.recompute(fields.parse(DECL), row)[0]


class TestWorkingARowOutAgain:
    def test_a_rounded_elapsed_time_and_everything_built_on_it(self):
        assert recompute(SEP_26) == {"Elapsed time": "2h 20m",
                                     "Average speed": "30.9 mph",
                                     "Earnings per hour": "$32.45"}

    def test_a_value_that_cannot_be_worked_out_is_used_as_recorded(self):
        """18:47 to 22:51 has no date, so 4h 4m stands — and the speed checked
        against it is 18.9, not the 18.17 the model wrote."""
        got = recompute(OCT_21)
        assert "Elapsed time" not in got
        assert got["Average speed"] == "18.9 mph"
        assert got["Earnings per hour"] == "$20.18"
        assert got["Earnings per mile"] == "$1.07"

    def test_a_blank_is_filled_when_its_readings_are_there(self):
        assert recompute(OCT_22)["Average speed"] == "18.2 mph"

    def test_a_correct_row_is_left_exactly_as_written(self):
        """$1.2467 is the same figure as $1.25 written more precisely."""
        assert recompute(AUG_25) == {}

    @pytest.mark.parametrize("row", [SEP_26, OCT_21, OCT_22, AUG_25])
    def test_twice_changes_nothing_the_second_time(self, row):
        once = dict(row, **recompute(row))
        assert recompute(once) == {}

    def test_nothing_is_ever_emptied(self):
        """A row with no readings at all has nothing to correct with."""
        row = dict(SEP_26, **{"Start odometer": "", "End odometer": "",
                              "Total earnings": ""})
        got = recompute(row)
        assert all(v for v in got.values())
        assert "Distance traveled" not in got

    def test_a_declaration_with_no_sums_changes_nothing(self):
        assert fields.recompute(fields.parse(["Fare: money"]), {"Fare": "$3"})[0] == {}


class TestRecheckingTheStore:
    def keep(self, *rows):
        store.create_routine("Trip", "b", record=DECL)
        # Kept as the model left them: no declaration to work from at the time.
        return [store.add_record("Trip", row, declared={}) for row in rows]

    def test_wrong_values_are_replaced_and_the_originals_kept(self):
        kept = self.keep(SEP_26)[0]
        done = store.recheck_records("Trip")
        assert (done["records"], done["values"]) == (1, 3)
        got = [r for r in store.list_records("Trip") if r["id"] == kept["id"]][0]
        assert got["fields"]["Earnings per hour"] == "$32.45"
        assert got["raw"]["Earnings per hour"] == "$37.86"
        assert got["raw"]["Elapsed time"] == "2h"
        assert sorted(got["checked"]) == ["Average speed", "Earnings per hour",
                                          "Elapsed time"]

    def test_the_readings_are_never_touched(self):
        self.keep(SEP_26)
        store.recheck_records("Trip")
        got = store.list_records("Trip")[0]["fields"]
        assert got["Start time"] == "2026-09-26 19:22"
        assert got["Total earnings"] == "$75.71"

    def test_a_right_log_is_not_written_to(self):
        self.keep(AUG_25)
        assert store.recheck_records("Trip")["records"] == 0
        assert store.list_records("Trip")[0]["checked"] == []

    def test_a_second_recheck_finds_nothing(self):
        self.keep(SEP_26, OCT_21, OCT_22, AUG_25)
        first = store.recheck_records("Trip")
        assert first["records"] == 3
        assert store.recheck_records("Trip")["records"] == 0

    def test_the_first_original_survives_a_second_correction(self):
        """What the model wrote is the thing worth remembering, not whatever
        the app worked out in between."""
        kept = self.keep(SEP_26)[0]
        store.recheck_records("Trip")
        store.update_record(kept["id"], {"End odometer": "105821 mi"})
        got = store.list_records("Trip")[0]
        assert got["raw"]["Earnings per hour"] == "$37.86"

    def test_a_blank_that_was_filled_says_so(self):
        self.keep(OCT_22)
        store.recheck_records("Trip")
        got = store.list_records("Trip")[0]
        assert got["fields"]["Average speed"] == "18.2 mph"
        assert "Average speed" in got["checked"]
        assert "Average speed" not in got["raw"], "there was nothing to keep"

    def test_every_routine_with_sums_at_once(self):
        self.keep(SEP_26)
        store.create_routine("Notes", "b", record=["Mood"])
        store.add_record("Notes", {"Mood": "fine"})
        assert store.recheck_all() == {"records": 1, "values": 3}

    def test_a_routine_with_no_sums_is_left_alone(self):
        store.create_routine("Notes", "b", record=["Mood"])
        store.add_record("Notes", {"Mood": "fine"})
        assert store.recheck_records("Notes")["records"] == 0


class TestCorrectingAReadingCorrectsWhatIsBuiltOnIt:
    def test_a_fixed_odometer_moves_the_distance_and_the_rates(self):
        store.create_routine("Trip", "b", record=DECL)
        kept = store.add_record("Trip", AUG_25)
        got = store.update_record(kept["id"], {"End odometer": "102193 mi"})
        assert got["fields"]["Distance traveled"] == "121 mi"
        assert got["fields"]["Earnings per mile"] == "$0.96"
        assert "Distance traveled" in got["checked"]

    def test_a_value_typed_in_is_no_longer_marked_as_worked_out(self):
        store.create_routine("Trip", "b", record=DECL)
        kept = store.add_record("Trip", OCT_22, declared={})
        store.recheck_records("Trip")
        got = store.update_record(kept["id"], {"Total earnings": "$190.00"})
        assert got["fields"]["Earnings per hour"] == "$25.62"
        got = store.update_record(kept["id"], {"Elapsed time": "7h 30m"})
        assert "Elapsed time" not in got["checked"]


class TestItHappensWithoutBeingAsked:
    def test_saving_the_sums_corrects_the_log_kept_before_them(self, client):
        store.create_routine("Trip", "b", record=["Start odometer"])
        store.add_record("Trip", SEP_26)
        rid = store.list_routines()[0]["id"]
        out = client.patch("/api/routines/" + rid, json={"record": DECL}).get_json()
        assert out["rechecked"]["records"] == 1
        assert {"field": "Earnings per hour", "was": "$37.86", "now": "$32.45"} in [
            {k: c[k] for k in ("field", "was", "now")} for c in out["rechecked"]["changes"]]

    def test_renaming_a_routine_does_not_recheck(self, client):
        made = client.post("/api/routines", json={"name": "Trip", "body": "b",
                                                  "record": DECL}).get_json()
        out = client.patch("/api/routines/" + made["id"], json={"body": "c"}).get_json()
        assert out["rechecked"]["records"] == 0

    def test_a_new_routine_reports_a_recheck_too(self, client):
        store.add_record("Trip", SEP_26)
        out = client.post("/api/routines", json={"name": "Trip", "body": "b",
                                                 "record": DECL}).get_json()
        assert out["rechecked"]["records"] == 1

    def test_the_app_starting_rechecks_every_log(self, client):
        store.create_routine("Trip", "b", record=DECL)
        store.add_record("Trip", SEP_26, declared={})
        importlib.reload(app_module)
        assert store.list_records("Trip")[0]["fields"]["Earnings per hour"] == "$32.45"

    def test_a_failed_recheck_costs_the_recheck_not_the_save(self, client, monkeypatch):
        def broken(*a, **k):
            raise RuntimeError("boom")
        monkeypatch.setattr(store, "recheck_records", broken)
        out = client.post("/api/routines", json={"name": "Trip", "body": "b",
                                                 "record": DECL})
        assert out.status_code == 200
        assert out.get_json()["rechecked"]["records"] == 0

    def test_an_edit_returns_the_row_as_worked_out(self, client):
        store.create_routine("Trip", "b", record=DECL)
        kept = store.add_record("Trip", AUG_25)
        out = client.patch("/api/records/" + kept["id"],
                           json={"fields": {"End odometer": "102193 mi"}}).get_json()
        assert out["record"]["fields"]["Distance traveled"] == "121 mi"


class TestTheEditorSaysSo:
    def page(self):
        import chat_ui
        return chat_ui.render_page("t")

    def test_saving_the_sums_says_how_much_of_the_log_changed(self):
        page = self.page()
        assert "const re = saved.rechecked || {};" in page
        assert "worked out again from their readings" in page

    def test_a_new_routine_with_a_warning_is_not_made_twice(self):
        """The editor stays open to show the warning; saving again from it
        has to update the routine it just made, not create another."""
        page = self.page()
        body = page[page.index("async function saveRoutine"):]
        assert body.index("editingRoutineId = kept.id;") < \
            body.index('routineWarnEl.textContent = "Saved. " + notes.join(" ");')
