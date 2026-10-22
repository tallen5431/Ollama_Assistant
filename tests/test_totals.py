"""Totals by week, by month and all time.

The two ways a spreadsheet gets these wrong are the two things tested hardest
here: adding up readings that are not amounts (odometers, clock times), and
averaging rates instead of dividing totals.
"""

from __future__ import annotations

import importlib
import time
from datetime import date, datetime

import pytest

import app as app_module
import chat_ui
import fields
import ollama_client
import store
import totals


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
        "Earnings per hour = Total earnings / Elapsed time"]
PARSED = fields.parse(DECL)
COLUMNS = fields.names(PARSED)
KINDS = dict(fields.kinds(PARSED), **{
    "Distance traveled": "distance", "Elapsed time": "duration",
    "Average speed": "speed", "Earnings per hour": "money"})


def trip(start, odo, miles, hours, money, rid=None):
    """One kept run, as storage hands it over."""
    end = datetime.fromtimestamp(datetime.fromisoformat(start).timestamp() + hours * 3600)
    return {"id": rid or start, "created_at": time.time(), "fields": {
        "Start odometer": f"{odo} mi", "End odometer": f"{odo + miles} mi",
        "Distance traveled": f"{miles} mi", "Start time": start,
        "End time": end.strftime("%Y-%m-%d %H:%M"), "Elapsed time": f"{hours}h",
        "Total earnings": f"${money:.2f}"}}


def summary(rows, today=date(2026, 10, 22)):
    return totals.summarise(rows, PARSED, KINDS, COLUMNS, today)


class TestWhatAddsUp:
    def test_readings_are_never_added(self):
        """Two odometers summed is a number of no meaning at all."""
        how = {c["name"]: c["how"] for c in totals.plan(PARSED, KINDS, COLUMNS)}
        assert "Start odometer" not in how and "End odometer" not in how
        assert "Start time" not in how and "End time" not in how

    def test_amounts_are_added_and_rates_are_divided(self):
        how = {c["name"]: c["how"] for c in totals.plan(PARSED, KINDS, COLUMNS)}
        assert how == {"Distance traveled": "sum", "Elapsed time": "sum",
                       "Total earnings": "sum", "Average speed": "rate",
                       "Earnings per hour": "rate"}

    def test_a_routine_with_no_sums_declared_still_totals_its_amounts(self):
        plain = fields.parse(["Odometer", "Fare", "Tip", "Pay per hour"])
        kinds = {"Odometer": "distance", "Fare": "money", "Tip": "money",
                 "Pay per hour": "money"}
        how = totals.plan(plain, kinds, fields.names(plain))
        assert [c["name"] for c in how] == ["Fare", "Tip"]


class TestTheArithmetic:
    def test_a_rate_is_total_over_total_not_an_average_of_rates(self):
        """$100 in one hour and $100 in four is $40 an hour over five hours,
        not the $62.50 that averaging $100/h and $25/h would claim."""
        got = summary([trip("2026-10-20 18:00", 1000, 20, 1, 100, "a"),
                       trip("2026-10-21 18:00", 1020, 80, 4, 100, "b")])["all"]
        assert got["values"]["Earnings per hour"] == "$40.00"
        assert got["values"]["Average speed"] == "20 mph"
        assert got["values"]["Total earnings"] == "$200.00"
        assert got["values"]["Elapsed time"] == "5h"
        assert got["values"]["Distance traveled"] == "100 mi"

    def test_a_rate_counts_only_the_runs_that_have_both_halves(self):
        """Earnings with no hours recorded would otherwise raise the rate."""
        broken = trip("2026-10-21 18:00", 1020, 80, 4, 300, "b")
        broken["fields"]["Elapsed time"] = ""
        got = summary([trip("2026-10-20 18:00", 1000, 20, 2, 50, "a"), broken])["all"]
        assert got["values"]["Earnings per hour"] == "$25.00"
        assert got["values"]["Total earnings"] == "$350.00", "the amount still counts"

    def test_miles_and_kilometres_are_not_added_together(self):
        a = trip("2026-10-20 18:00", 1000, 20, 1, 10, "a")
        b = trip("2026-10-21 18:00", 1020, 30, 1, 10, "b")
        b["fields"]["Distance traveled"] = "30 km"
        assert "Distance traveled" not in summary([a, b])["all"]["values"]


class TestPeriods:
    ROWS = [trip("2026-10-20 18:00", 1000, 20, 1, 10, "a"),    # this week (Tue)
            trip("2026-10-19 09:00", 900, 10, 1, 10, "b"),     # this week (Mon)
            trip("2026-10-18 22:00", 800, 10, 1, 10, "c"),     # last week (Sun)
            trip("2026-09-30 18:00", 700, 10, 1, 10, "d")]

    def test_weeks_start_on_monday_and_are_newest_first(self):
        weeks = summary(self.ROWS)["weeks"]
        assert [(w["label"], w["count"]) for w in weeks] == [
            ("This week", 2), ("Last week", 1), ("Week of 28 Sep 2026", 1)]

    def test_months(self):
        months = summary(self.ROWS)["months"]
        assert [(m["label"], m["count"]) for m in months] == [
            ("October 2026", 3), ("September 2026", 1)]

    def test_a_run_belongs_to_the_day_it_started_not_the_day_it_was_kept(self):
        late = trip("2026-10-18 22:00", 800, 10, 1, 10, "c")
        late["created_at"] = datetime(2026, 10, 19, 6, 3).timestamp()
        assert totals.day_of(late, KINDS, COLUMNS) == (date(2026, 10, 18), "Start time")

    def test_a_time_with_no_date_falls_back_to_when_it_was_kept(self):
        row = trip("2026-10-18 22:00", 800, 10, 1, 10, "c")
        row["fields"]["Start time"] = "18:47"
        row["fields"]["End time"] = "22:51"
        row["created_at"] = datetime(2026, 10, 21, 6, 1).timestamp()
        assert totals.day_of(row, KINDS, COLUMNS) == (date(2026, 10, 21), "")

    def test_the_week_a_new_run_fell_in(self):
        got = totals.week_of(self.ROWS[0], self.ROWS, PARSED, KINDS, COLUMNS,
                             date(2026, 10, 22))
        assert got["label"] == "This week" and got["count"] == 2
        assert got["values"]["Total earnings"] == "$20.00"


class TestTheRoutes:
    def keep(self):
        store.create_routine("Trip", "b", record=DECL)
        for row in TestPeriods.ROWS:
            store.add_record("Trip", row["fields"])

    def test_the_summary(self, client):
        self.keep()
        got = client.get("/api/records/summary?routine=Trip").get_json()["summary"]
        assert got["all"]["count"] == 4
        assert got["all"]["values"]["Total earnings"] == "$40.00"
        assert [m["key"] for m in got["months"]] == ["2026-10", "2026-09"]

    def test_a_routine_with_nothing_kept(self, client):
        assert client.get("/api/records/summary?routine=Nope").get_json() == {"summary": None}

    def test_naming_the_routine_is_required(self, client):
        assert client.get("/api/records/summary").status_code == 400

    def test_the_week_comes_back_with_a_kept_run(self, client, monkeypatch):
        store.create_routine("Trip", "b", record=DECL)
        monkeypatch.setattr(ollama_client, "chat", lambda *a, **k: (
            '{"Start odometer": "1000", "End odometer": "1020", "Total earnings": "$30",'
            ' "Start time": "2026-10-20 18:00", "End time": "2026-10-20 19:00"}'))
        out = client.post("/api/records", json={
            "answer": "a", "fields": DECL, "routine_name": "Trip"}).get_json()
        assert out["week"]["count"] == 1
        assert out["week"]["values"]["Earnings per hour"] == "$30.00"

    def test_a_failure_costs_only_the_week(self, client, monkeypatch):
        store.create_routine("Trip", "b", record=DECL)
        monkeypatch.setattr(ollama_client, "chat", lambda *a, **k: '{"Total earnings": "$30"}')
        monkeypatch.setattr(store, "week_so_far",
                            lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
        out = client.post("/api/records", json={
            "answer": "a", "fields": DECL, "routine_name": "Trip"})
        assert out.status_code == 200
        assert out.get_json()["record"] and out.get_json()["week"] is None


class TestThePage:
    def page(self):
        return chat_ui.render_page("t")

    def test_the_drawer_shows_totals_for_one_routine(self):
        page = self.page()
        assert "fillTotals(holder, one);" in page
        assert '"api/records/summary?routine=" +' in page

    def test_the_only_routine_counts_as_one_routine(self):
        """With a single routine there is no filter to pick, and the totals
        were the thing that log most wanted."""
        assert "shown.every(\n          r => r.routine_name === shown[0].routine_name)" \
            in self.page()

    def test_a_rate_says_how_it_was_worked_out(self):
        assert '", over the runs that have both"' in self.page()

    def test_the_week_line_under_the_reply(self):
        page = self.page()
        assert "if (data.week) showWeek(view, data.week);" in page
        body = page[page.index("function showWeek"):]
        body = body[:body.index("\n      }\n")]
        assert "line.textContent =" in body and "innerHTML" not in body


class TestTheOdometerOnlyGoesUp:
    LOG = [trip("2026-10-20 18:34", 107381, 33, 3, 71.11, "a"),
           trip("2026-10-20 14:00", 107298, 77, 4, 82.06, "b")]

    def check(self, start, end):
        new = trip("2026-10-22 13:49", start, end - start, 7, 177.61, "new")
        return totals.continuity(new, self.LOG, PARSED, KINDS, COLUMNS)

    def test_a_run_that_starts_where_the_last_ended_says_how_far_between(self):
        got = self.check(107423, 107558)
        assert got == {"overlaps": [],
                       "since": "9 mi on the odometer since the run kept for 20 Oct"}

    def test_an_overlap_is_named_with_both_runs_readings(self):
        got = self.check(107400, 107450)
        assert len(got["overlaps"]) == 1
        assert "107,400" in got["overlaps"][0] and "107,381 to 107,414" in got["overlaps"][0]
        assert got["since"] == "", "no gap measured from a run that may be the misread one"

    def test_the_same_run_kept_twice(self):
        got = self.check(107381, 107414)
        assert got["overlaps"] == ["Start odometer and End odometer: the same readings "
                                   "as the run kept for 20 Oct — kept twice?"]

    def test_a_run_from_before_the_log_began_is_fine(self):
        assert self.check(107000, 107100) is None

    def test_runs_end_to_end_do_not_overlap(self):
        assert self.check(107414, 107500)["overlaps"] == []

    def test_a_routine_with_no_odometer_pair_has_nothing_to_check(self):
        plain = fields.parse(["Fare: money"])
        row = {"id": "x", "created_at": 0, "fields": {"Fare": "$3"}}
        assert totals.continuity(row, [], plain, {"Fare": "money"}, ["Fare"]) is None

    def test_it_comes_back_with_a_kept_run(self, client, monkeypatch):
        store.create_routine("Trip", "b", record=DECL)
        for row in self.LOG:
            store.add_record("Trip", row["fields"])
        monkeypatch.setattr(ollama_client, "chat", lambda *a, **k: (
            '{"Start odometer": "107400", "End odometer": "107450", "Total earnings": "$30",'
            ' "Start time": "2026-10-22 13:49", "End time": "2026-10-22 15:00"}'))
        out = client.post("/api/records", json={
            "answer": "a", "fields": DECL, "routine_name": "Trip"}).get_json()
        assert len(out["continuity"]["overlaps"]) == 1

    def test_the_page_shows_an_overlap_with_the_other_gaps(self):
        page = chat_ui.render_page("t")
        assert ".concat(fit.overlaps || [])" in page
        assert "if (fit.since) showSince(view, fit.since);" in page
