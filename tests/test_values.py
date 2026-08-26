"""Standardising a recorded value without changing what it says.

The samples here are real: they come from a log of the same Uber routine, where
one trip was recorded twice five minutes apart and the two rows agreed about
every fact and about none of the formatting.
"""

from __future__ import annotations

import json
import time
import uuid

import pytest

import store
import values


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Its own database per test, like every other module that writes one."""
    monkeypatch.setenv("CHAT_DB", str(tmp_path / "chat.db"))
    yield


class TestReadingWhatTheModelWrote:
    @pytest.mark.parametrize("text, kind", [
        ("$115.94", values.MONEY),
        ("$1.2465", values.MONEY),
        ("$1.24 per mile", values.MONEY),
        ("$0.55 per mile ($36.00 / 66 mi)", values.MONEY),
        ("≈ $0.89", values.MONEY),
        ("102,072 mi", values.DISTANCE),
        ("66 miles", values.DISTANCE),
        ("21.70 MPH", values.SPEED),
        ("≈ 23.21 mph", values.SPEED),
        ("4 hours 25 minutes", values.DURATION),
        ("3 hours and 8 minutes (or ≈ 3.13 hours)", values.DURATION),
        ("Friday, 07 August 2026 at 13:37", values.TIMESTAMP),
        ("102,072", values.NUMBER),
    ])
    def test_it_knows_what_it_is_looking_at(self, text, kind):
        found = values.parse(text)
        assert found and found.kind == kind

    @pytest.mark.parametrize("text", [
        "", "   ", "no idea", "the driver said it was busy", "n/a",
    ])
    def test_and_says_so_when_it_is_not_a_quantity(self, text):
        assert values.parse(text) is None


class TestPresentationChangesAndNumbersDoNot:
    """The rule the whole module works to. A record is worth keeping only if it
    still says in a year exactly what it said today."""

    @pytest.mark.parametrize("text, expected", [
        ("$1.2465", "$1.2465"),      # not rounded to $1.25
        ("$36.00", "$36.00"),        # not shortened to $36
        ("21.70 MPH", "21.70 mph"),  # not shortened to 21.7
        ("102,072", "102072"),
    ])
    def test_every_digit_survives(self, text, expected):
        assert values.canonical(text) == expected

    @pytest.mark.parametrize("text, expected", [
        ("$1.24 per mile", "$1.24"),
        ("$0.55 per mile ($36.00 / 66 mi)", "$0.55"),
        ("≈ $0.89", "$0.89"),
        ("100,409 miles", "100409 mi"),
        ("≈ 23.21 mph", "23.21 mph"),
        ("3 hours and 21 minutes (or 3.35 hours)", "3h 21m"),
        ("4 hours 25 minutes", "4h 25m"),
    ])
    def test_only_the_decoration_goes(self, text, expected):
        assert values.canonical(text) == expected

    def test_the_working_is_dropped_not_the_answer(self):
        """"$0.55 per mile ($36.00 / 66 mi)" — the bracket is a second copy of
        the inputs, and a naive first-number match takes $36.00 out of it."""
        assert values.canonical("$0.55 per mile ($36.00 / 66 mi)") == "$0.55"


class TestProseIsLeftAlone:
    """A value that merely contains a number is a sentence, not a quantity.
    Rewriting "54 miles to Brighton" as "54 mi" deletes where the trip went —
    and takes the word Brighton out of the search index with it."""

    @pytest.mark.parametrize("text", [
        "54 miles to Brighton",
        "drove 54 miles",
        "about 3 hours of waiting at the airport",
        "went 21.70 MPH on the freeway",
        "Trip on 12 August 2026",
        "$40 in tips, $8 of it cash",
    ])
    def test_a_sentence_survives_intact(self, text):
        assert values.canonical(text) == text

    def test_even_when_the_column_says_otherwise(self):
        assert values.canonical("54 miles to Brighton", values.DISTANCE) == \
            "54 miles to Brighton"


class TestTimestamps:
    @pytest.mark.parametrize("text, expected", [
        ("Wednesday, August 12, 2026, at 15:41", "2026-08-12 15:41"),
        ("Friday, 07 August 2026 at 13:37", "2026-08-07 13:37"),
        ("Aug 3, 2026 at 7:05 pm", "2026-08-03 19:05"),
        ("2026-08-25T20:06", "2026-08-25 20:06"),
        ("12 August 2026", "2026-08-12"),
    ])
    def test_one_shape_out_of_many(self, text, expected):
        assert values.canonical(text, values.TIMESTAMP) == expected

    def test_the_timezone_is_kept(self):
        """It arrives in brackets, like the workings do, and dropping it made
        every row logged away from home quietly wrong: 20:06 in an unstated
        zone is not the same fact as 20:06 UTC-04:00."""
        assert values.canonical(
            "Tuesday 25 August 2026 at 20:06 (evening) (UTC-04:00)",
            values.TIMESTAMP) == "2026-08-25 20:06 UTC-04:00"

    def test_a_missing_time_is_not_invented(self):
        """Filling in 00:00 would put a trip at midnight and look like a
        reading rather than a gap."""
        assert values.canonical("12 August 2026", values.TIMESTAMP) == "2026-08-12"

    def test_they_sort(self):
        stamps = [values.canonical(t, values.TIMESTAMP) for t in [
            "Friday, 07 August 2026 at 13:37",
            "Wednesday, August 12, 2026, at 15:41",
            "Saturday, August 8, 2026, at 18:52"]]
        assert sorted(stamps) == [
            "2026-08-07 13:37", "2026-08-08 18:52", "2026-08-12 15:41"]


class TestTheColumnDecides:
    """Per value it cannot be decided: "102,072" is a bare number and
    "102,072 mi" is a distance, and they are the same odometer written by the
    same routine a minute apart."""

    ODOMETER = ["102,072", "102,072 mi", "100,473 mi", "100,409 miles"]

    def test_a_column_of_odometers_is_distance(self):
        assert values.column_kind(self.ODOMETER) == values.DISTANCE

    def test_so_a_bare_number_in_it_gains_the_unit(self):
        assert values.canonical("93", values.DISTANCE) == "93 mi"

    def test_and_the_whole_column_comes_out_alike(self):
        assert values.canonical_column(self.ODOMETER) == [
            "102072 mi", "102072 mi", "100473 mi", "100409 mi"]

    def test_a_column_that_is_mostly_prose_stays_prose(self):
        column = ["went well", "quiet night", "busy, lots of surge", "68 miles"]
        assert values.column_kind(column) == values.TEXT
        assert values.canonical_column(column) == column

    def test_an_empty_column_is_text(self):
        assert values.column_kind(["", "", ""]) == values.TEXT

    def test_naming_a_unit_outvotes_omitting_one(self):
        """One value saying "mi" is evidence; three saying nothing is not."""
        assert values.column_kind(["1", "2", "3 mi"]) == values.DISTANCE


class TestTheSpreadsheetGetsNumbers:
    """A cell reading "$115.94" is text: it will not sum, and it sorts
    "100 mi" before "93 mi"."""

    @pytest.mark.parametrize("text, kind, expected", [
        ("$115.94", values.MONEY, "115.94"),
        ("93 mi", values.DISTANCE, "93"),
        ("4h 25m", values.DURATION, "4.4167"),
        ("21.2 mph", values.SPEED, "21.2"),
    ])
    def test_the_unit_comes_off(self, text, kind, expected):
        assert values.number_of(text, kind) == expected

    def test_a_timestamp_stays_readable(self):
        """A number would be a worse answer than the stamp itself."""
        assert values.number_of("2026-08-12 15:41", values.TIMESTAMP) == \
            "2026-08-12 15:41"

    def test_prose_is_handed_over_untouched(self):
        assert values.number_of("busy night", values.TEXT) == "busy night"

    @pytest.mark.parametrize("kind, seen, expected", [
        (values.MONEY, ["$1.00"], "USD"),
        (values.DISTANCE, ["66 miles"], "mi"),
        (values.SPEED, ["21 mph"], "mph"),
        (values.DURATION, ["3h"], "hours"),
        (values.TEXT, ["busy"], ""),
    ])
    def test_the_unit_goes_in_the_header(self, kind, seen, expected):
        assert values.unit_label(kind, seen) == expected


class TestRewritingTheLogAlreadyKept:
    """Records written before any of this existed are the ones that need it
    most: they are the whole log."""

    def kept(self, fields, routine="🚗 Uber Trip", at=None):
        """A record written the old way, straight into the table."""
        rid = uuid.uuid4().hex
        with store._connect() as conn:
            conn.execute(
                "INSERT INTO records (id, routine_name, fields, created_at)"
                " VALUES (?, ?, ?, ?)",
                (rid, routine, json.dumps(fields), at or time.time()))
        return rid

    def test_it_standardises_what_is_already_there(self):
        self.kept({"Start odometer": "102,072", "Distance traveled": "93"})
        self.kept({"Start odometer": "100,409 miles", "Distance traveled": "66 miles"})
        assert store.normalise_stored()["records"] == 2
        rows = [r["fields"] for r in store.list_records()]
        assert {r["Distance traveled"] for r in rows} == {"93 mi", "66 mi"}
        assert {r["Start odometer"] for r in rows} == {"102072 mi", "100409 mi"}

    def test_nothing_is_lost(self):
        """The constraint the whole change was made under."""
        self.kept({"Elapsed time": "3 hours and 8 minutes (or ≈ 3.13 hours)"})
        store.normalise_stored()
        record = store.list_records()[0]
        assert record["fields"]["Elapsed time"] == "3h 08m"
        assert record["raw"]["Elapsed time"] == \
            "3 hours and 8 minutes (or ≈ 3.13 hours)"

    def test_it_is_idempotent(self):
        self.kept({"Distance traveled": "66 miles"})
        assert store.normalise_stored()["records"] == 1
        assert store.normalise_stored()["records"] == 0

    def test_and_running_it_twice_keeps_the_first_original(self):
        """Not what the first pass made of it — that would launder the very
        thing `raw` exists to preserve."""
        self.kept({"Distance traveled": "66 miles"})
        store.normalise_stored()
        store.normalise_stored()
        assert store.list_records()[0]["raw"]["Distance traveled"] == "66 miles"

    def test_a_tidy_log_is_untouched(self):
        self.kept({"Distance traveled": "66 mi"})
        assert store.normalise_stored()["records"] == 0
        assert store.list_records()[0]["raw"] == {}

    def test_each_routine_is_judged_on_its_own_columns(self):
        """Two routines can both have a "Total" and mean different things."""
        self.kept({"Total": "$40.00"}, routine="Trip")
        self.kept({"Total": "12 miles"}, routine="Walk")
        store.normalise_stored()
        by_routine = {r["routine_name"]: r["fields"]["Total"]
                      for r in store.list_records()}
        assert by_routine == {"Trip": "$40.00", "Walk": "12 mi"}

    def test_prose_records_are_not_rewritten(self):
        self.kept({"Notes": "54 miles to Brighton, quiet all evening"})
        store.normalise_stored()
        assert store.list_records()[0]["fields"]["Notes"] == \
            "54 miles to Brighton, quiet all evening"

    def test_an_empty_database_is_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CHAT_DB", str(tmp_path / "nothing.db"))
        assert store.normalise_stored() == {"records": 0, "values": 0}


class TestNewRecordsArriveStandardised:
    def test_the_column_teaches_the_next_row(self):
        """The first row says "miles", so the second row's bare 66 is miles."""
        store.add_record("Trip", {"Distance": "93 miles"})
        second = store.add_record("Trip", {"Distance": "66"})
        assert second["fields"]["Distance"] == "66 mi"

    def test_the_original_is_kept_for_whatever_changed(self):
        made = store.add_record("Trip", {"Distance": "93 miles", "Note": "busy"})
        assert made["fields"]["Distance"] == "93 mi"
        assert made["raw"] == {"Distance": "93 miles"}, \
            "only the value that changed should be remembered"

    def test_the_first_record_of_a_routine_still_works(self):
        made = store.add_record("Brand new", {"Fare": "$12.50"})
        assert made["fields"]["Fare"] == "$12.50"


def _checker():
    """tools/check_records.py, imported by path like the script it is."""
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "tools" / "check_records.py"
    spec = importlib.util.spec_from_file_location("check_records", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTheThingsStandardisingWillNotFix:
    """Putting a value in one shape is safe, so the app does it on its own.
    Changing a number is not, so it is reported and left alone."""

    COLUMNS = ["Start odometer", "End odometer", "Distance traveled",
               "Elapsed time", "Total earnings", "Earnings per mile",
               "Earnings per hour"]

    def row(self, **kw):
        base = dict(zip(self.COLUMNS, [""] * len(self.COLUMNS)))
        base.update(kw)
        return base

    def test_it_works_out_which_column_is_which(self):
        roles = _checker().find_roles(self.COLUMNS)
        assert roles["start_odo"] == "Start odometer"
        assert roles["end_odo"] == "End odometer"
        assert roles["earnings"] == "Total earnings"
        assert roles["per_hour"] == "Earnings per hour"
        assert roles["per_mile"] == "Earnings per mile"

    def test_an_unrecognised_log_is_not_guessed_at(self):
        roles = _checker().find_roles(["Mood", "Weather", "Notes"])
        assert "earnings" not in roles

    def test_a_rate_worked_out_from_nothing_is_caught(self, capsys):
        """The real one: $23.19/hour on a row with no elapsed time in it."""
        check = _checker()
        rows = [self.row(**{"Total earnings": "$115.94",
                            "Distance traveled": "93",
                            "Earnings per hour": "$23.19 per hour"})]
        found = check.check_arithmetic(rows, check.find_roles(self.COLUMNS))
        assert found == 1
        assert "no elapsed time" in capsys.readouterr().out

    def test_a_rate_that_does_not_divide_is_caught(self, capsys):
        check = _checker()
        rows = [self.row(**{"Total earnings": "$68.21",
                            "Distance traveled": "68 miles",
                            "Elapsed time": "3 hours and 8 minutes",
                            "Earnings per hour": "$21.71 per hour"})]
        assert check.check_arithmetic(rows, check.find_roles(self.COLUMNS)) == 1
        assert "21.77" in capsys.readouterr().out, "the computed figure is shown"

    def test_a_row_that_adds_up_is_left_alone(self, capsys):
        check = _checker()
        rows = [self.row(**{"Total earnings": "$36.00",
                            "Start odometer": "100,473", "End odometer": "100,539",
                            "Distance traveled": "66 miles",
                            "Elapsed time": "3 hours and 21 minutes",
                            "Earnings per mile": "$0.55 per mile",
                            "Earnings per hour": "$10.75 per hour"})]
        assert check.check_arithmetic(rows, check.find_roles(self.COLUMNS)) == 0
        assert "adds up" in capsys.readouterr().out

    def test_the_same_trip_recorded_twice_is_found(self, capsys):
        """It hid behind the formatting: one capture wrote "102,072" and the
        other "102,072 mi", so a plain comparison saw two different trips."""
        check = _checker()
        rows = [self.row(**{"Start odometer": "102,072", "End odometer": "102,165",
                            "Total earnings": "$115.94"}),
                self.row(**{"Start odometer": "102,072 mi", "End odometer": "102,165 mi",
                            "Total earnings": "$115.94"})]
        assert check.check_duplicates(rows, check.find_roles(self.COLUMNS)) == 1
        assert "appears 2 times" in capsys.readouterr().out

    def test_two_different_trips_are_not_a_duplicate(self):
        check = _checker()
        rows = [self.row(**{"Start odometer": "102,072", "End odometer": "102,165",
                            "Total earnings": "$115.94"}),
                self.row(**{"Start odometer": "100,409", "End odometer": "100,469",
                            "Total earnings": "$53.54"})]
        assert check.check_duplicates(rows, check.find_roles(self.COLUMNS)) == 0
