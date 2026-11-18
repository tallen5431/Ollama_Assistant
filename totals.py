#!/usr/bin/env python3
"""Across a routine's log: totals by week, by month and all time, and
whether a new run's odometer readings fit with the rest.

The question a log of trips is kept to answer is "how did this week go", and
the arithmetic for it has two traps that a spreadsheet's SUM and AVERAGE both
fall into:

* **Not everything adds up.** An odometer is a reading, not an amount: the sum
  of two of them is a number of no meaning at all, and so is the sum of two
  clock times. Only the amounts a run *produced* — miles driven, hours worked,
  money earned — are summed.
* **A rate is not averaged.** Four trips at $30 an hour and one at $10 is not
  $26 an hour if the $10 trip was the long one. A rate over a period is the
  period's total over the period's total, so an hourly rate is worked out
  again from the summed earnings and the summed hours — and only across the
  runs that have both, or a run with earnings and no recorded hours would
  quietly raise it.

Pure functions over rows already kept; storage hands them in.
"""

from __future__ import annotations

import re
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import fields
import values

# Kinds that are amounts when they are not readings: what a run produced.
_AMOUNTS = (values.MONEY, values.DISTANCE, values.DURATION)
_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})\b")


def plan(declaration: List[fields.Field], kinds: Dict[str, str],
         columns: List[str]) -> List[Dict[str, str]]:
    """Which columns are totalled, and how: ``sum`` or ``rate``.

    A declared sum says what it is built from, so a declaration settles it:
    the two sides of "End odometer - Start odometer" are readings, and
    "Total earnings / Elapsed time" is a rate of two amounts. Without one, a
    column of money, distance or time is an amount unless its name says it is
    a reading ("odometer") or a rate ("per hour").
    """
    by_name = {f.name: f for f in declaration}
    readings = set()
    for field in declaration:
        # Two distances taken from one another are two odometer readings far
        # more often than they are two amounts; money and time taken from one
        # another are amounts (gross less fees, shift less breaks).
        if field.computed and field.op == "-":
            for side in (field.left, field.right):
                if kinds.get(side) in (values.DISTANCE, values.NUMBER, ""):
                    readings.add(side)
    out: List[Dict[str, str]] = []
    sums: Dict[str, str] = {}
    for name in columns:
        kind = kinds.get(name, "")
        field = by_name.get(name)
        if kind not in _AMOUNTS or name in readings or values.rate_of(name):
            continue
        if field is not None and (field.source or (field.computed and field.op == "/")):
            continue
        if not (field and field.computed) and re.search(r"odometer|reading", name, re.I):
            continue
        sums[name] = kind
        out.append({"name": name, "how": "sum", "kind": kind})
    for name in columns:
        field = by_name.get(name)
        if field is None or not field.computed or field.op != "/":
            continue
        if field.left in sums and field.right in sums:
            kind = fields.result_kind(sums[field.left], "/", sums[field.right])
            if kind and kind != values.NUMBER:
                out.append({"name": name, "how": "rate", "kind": kind,
                            "left": field.left, "right": field.right})
    # In the routine's own column order, so the totals read like its table.
    order = {name: at for at, name in enumerate(columns)}
    return sorted(out, key=lambda c: order.get(c["name"], len(order)))


_DATE_TIME = re.compile(r"^(\d{4})-(\d{2})-(\d{2})[ T](\d{1,2}):(\d{2})")
_CLOCK = re.compile(r"^(\d{1,2}):(\d{2})\b")
NO_DATE = "no date in it — the day comes from when it was kept"


def trip_start(record: Dict[str, Any], kinds: Dict[str, str],
               columns: List[str]) -> Tuple[Optional[datetime], date, str]:
    """When a run started — ``(start, day, where that came from)``.

    From the run's first timestamp column that holds anything: a full date and
    time gives both; a date alone gives the day. A clock time with no date
    ("18:47", "16:53 (UTC-05:00)") is put on the day the run was kept — or the
    day before, when it was kept earlier in the day than it started, which is
    a trip driven in the evening and saved after midnight. With nothing to go
    on, the day it was kept. ``start`` is wall-clock time as written.
    """
    kept = datetime.fromtimestamp(float(record.get("created_at") or 0))
    for name in columns:
        if kinds.get(name) != values.TIMESTAMP:
            continue
        text = str(record["fields"].get(name, "")).strip()
        if not text:
            continue
        full = _DATE_TIME.match(text)
        if full:
            try:
                start = datetime(*(int(g) for g in full.groups()))
                return start, start.date(), name
            except ValueError:
                continue
        day_only = _DATE.match(text)
        if day_only:
            try:
                return None, date(*(int(g) for g in day_only.groups())), name
            except ValueError:
                continue
        clock = _CLOCK.match(text)
        if clock and int(clock.group(1)) < 24 and int(clock.group(2)) < 60:
            start = kept.replace(hour=int(clock.group(1)), minute=int(clock.group(2)),
                                 second=0, microsecond=0)
            if start > kept:
                start -= timedelta(days=1)
            return start, start.date(), f"{name} ({NO_DATE})"
    return None, kept.date(), ""


def day_of(record: Dict[str, Any], kinds: Dict[str, str],
           columns: List[str]) -> Tuple[date, str]:
    """The day a run happened, and where that came from ("" for the day kept).

    A trip that started at 18:34 and was logged at 06:03 the next morning
    belongs to the day it was driven.
    """
    _, day, source = trip_start(record, kinds, columns)
    return day, source


def _week(day: date) -> date:
    return day - timedelta(days=day.weekday())


def _total(rows: List[Dict[str, Any]], how: List[Dict[str, str]],
           kinds: Dict[str, str]) -> Dict[str, Any]:
    """One period's figures: each sum, and each rate from the sums."""
    out: Dict[str, str] = {}
    parsed = [{c["name"]: values.parse_as(str(r["fields"].get(c["name"], "")),
                                          c["kind"], c["name"])
               for c in how if c["how"] == "sum"} for r in rows]
    for column in how:
        name, kind = column["name"], column["kind"]
        if column["how"] == "sum":
            found = [p[name] for p in parsed if p.get(name)]
            units = {v.unit for v in found if v.unit}
            if not found or len(units) > 1:
                continue        # nothing to add, or miles beside kilometres
            unit = units.pop() if units else ""
            number = sum(v.number for v in found)
            out[name] = values.render(kind, number, unit or "$", unit)
            continue
        # A rate, from the runs that have both halves of it.
        pairs = [(p[column["left"]], p[column["right"]]) for p in parsed
                 if p.get(column["left"]) and p.get(column["right"])]
        if not pairs:
            continue
        top = sum(a.number for a, _ in pairs)
        bottom = sum(b.number for _, b in pairs)
        if not bottom:
            continue
        left, right = pairs[0]
        unit = fields._result_unit(kind, left, right)
        out[name] = values.render(kind, top / bottom, unit if kind == values.MONEY else "$",
                                  unit)
    return {"count": len(rows), "values": out}


def summarise(records: List[Dict[str, Any]], declaration: List[fields.Field],
              kinds: Dict[str, str], columns: List[str],
              today: Optional[date] = None) -> Dict[str, Any]:
    """Every week and month in the log, newest first, and the whole of it."""
    how = plan(declaration, kinds, columns)
    today = today or datetime.fromtimestamp(time.time()).date()
    weeks: Dict[date, List[Dict[str, Any]]] = {}
    months: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    dated_by = ""
    for record in records:
        day, source = day_of(record, kinds, columns)
        dated_by = dated_by or source
        weeks.setdefault(_week(day), []).append(record)
        months.setdefault((day.year, day.month), []).append(record)
    this_week, last_week = _week(today), _week(today) - timedelta(days=7)

    def week_label(start: date) -> str:
        if start == this_week:
            return "This week"
        if start == last_week:
            return "Last week"
        return "Week of " + f"{start.day} {start:%b %Y}"

    return {
        "columns": how,
        "dated_by": dated_by,
        "all": dict(_total(records, how, kinds), label="All time"),
        "weeks": [dict(_total(weeks[k], how, kinds), label=week_label(k), key=k.isoformat())
                  for k in sorted(weeks, reverse=True)],
        "months": [dict(_total(months[k], how, kinds),
                        label=date(k[0], k[1], 1).strftime("%B %Y"),
                        key=f"{k[0]:04d}-{k[1]:02d}")
                   for k in sorted(months, reverse=True)],
    }


def week_of(record: Dict[str, Any], records: List[Dict[str, Any]],
            declaration: List[fields.Field], kinds: Dict[str, str],
            columns: List[str], today: Optional[date] = None) -> Optional[Dict[str, Any]]:
    """The totals for the week one run fell in — the line under its reply."""
    how = plan(declaration, kinds, columns)
    if not how:
        return None
    today = today or datetime.fromtimestamp(time.time()).date()
    start = _week(day_of(record, kinds, columns)[0])
    same = [r for r in records if _week(day_of(r, kinds, columns)[0]) == start]
    if record["id"] not in {r["id"] for r in same}:
        same.append(record)
    label = ("This week" if start == _week(today)
             else "Week of " + f"{start.day} {start:%b %Y}")
    return dict(_total(same, how, kinds), label=label, columns=how)


def _spans(declaration: List[fields.Field], kinds: Dict[str, str]) -> List[Tuple[str, str]]:
    """``(start, end)`` reading pairs: the two sides of an "End - Start" distance."""
    out = []
    for field in declaration:
        if field.computed and field.op == "-" and \
                kinds.get(field.left) == kinds.get(field.right) == values.DISTANCE:
            out.append((field.right, field.left))
    return out


def _span(record: Dict[str, Any], start: str, end: str) -> Optional[Tuple[float, float, str]]:
    a = values.parse_as(str(record["fields"].get(start, "")), values.DISTANCE, start)
    b = values.parse_as(str(record["fields"].get(end, "")), values.DISTANCE, end)
    if not a or not b or a.number > b.number or (a.unit and b.unit and a.unit != b.unit):
        return None
    return a.number, b.number, b.unit or a.unit


def continuity(record: Dict[str, Any], records: List[Dict[str, Any]],
               declaration: List[fields.Field], kinds: Dict[str, str],
               columns: List[str]) -> Optional[Dict[str, Any]]:
    """How a new run's odometer readings sit against every other run's.

    An odometer only goes up, so two runs cannot share a mile: if this one
    starts before another one ends and ends after it starts, one of the two
    was misread, or the same trip was kept twice. That is checked when the run
    is kept, while the photos are still on screen to check it against —
    rather than found months later in a column of distances that add up to
    more than the car has driven.

    And the gap back to the run before is said too: miles the odometer moved
    between two runs were driven, just not on one, and for a log of paid
    trips that is the personal or commuting mileage.
    """
    out: Dict[str, Any] = {"overlaps": [], "since": ""}
    for start, end in _spans(declaration, kinds):
        mine = _span(record, start, end)
        if not mine:
            continue
        before: Optional[Tuple[float, Dict[str, Any]]] = None
        for other in records:
            if other["id"] == record["id"]:
                continue
            theirs = _span(other, start, end)
            if not theirs or theirs[2] != mine[2]:
                continue
            day = day_of(other, kinds, columns)[0]
            when = f"{day.day} {day:%b}"
            if (theirs[0], theirs[1]) == (mine[0], mine[1]):
                out["overlaps"].append(
                    f"{start} and {end}: the same readings as the run kept for "
                    f"{when} — kept twice?")
            elif mine[0] < theirs[1] and theirs[0] < mine[1]:
                out["overlaps"].append(
                    f"{start} {_reading(mine[0])} to {end} "
                    f"{_reading(mine[1])} overlaps the run kept for {when} "
                    f"({_reading(theirs[0])} to {_reading(theirs[1])}) "
                    "— an odometer only goes up, so one of them is misread")
            elif theirs[1] <= mine[0] and (before is None or theirs[1] > before[0]):
                before = (theirs[1], other)
        # Not beside an overlap: a gap measured from a run that may be the
        # misread one is a figure built on the thing in question.
        if before and not out["since"] and not out["overlaps"]:
            day = day_of(before[1], kinds, columns)[0]
            gap = mine[0] - before[0]
            out["since"] = (f"{values.render(values.DISTANCE, gap, '$', mine[2])} on the "
                            f"odometer since the run kept for {day.day} {day:%b}")
    return out if out["overlaps"] or out["since"] else None


def _reading(number: float) -> str:
    return f"{number:,.0f}" if number == int(number) else f"{number:,}"


def _odometers(declaration: List[fields.Field], kinds: Dict[str, str],
               columns: List[str]) -> List[Tuple[str, str]]:
    """The start/end odometer pair: from a declared "End - Start" distance, or
    failing that from two distance columns named start … and end … ."""
    found = _spans(declaration, kinds)
    if found:
        return found
    starts = [c for c in columns if kinds.get(c) == values.DISTANCE and
              re.search(r"\bstart", c, re.I)]
    ends = [c for c in columns if kinds.get(c) == values.DISTANCE and
            re.search(r"\bend", c, re.I)]
    return [(starts[0], ends[0])] if starts and ends else []


def _when(moment: datetime) -> str:
    return f"{moment.day} {moment:%b} {moment:%H:%M}"


def date_checks(records: List[Dict[str, Any]], declaration: List[fields.Field],
                kinds: Dict[str, str], columns: List[str]) -> Dict[str, List[Dict[str, str]]]:
    """What looks wrong with each run's date, by record id, as
    ``{"level": "error" | "note", "text"}``.

    Three things a log can show by itself, without anyone's help:

    * a start time **after** the run was kept — impossible, so the date is
      wrong (a real log had a trip "starting" seventeen hours after it was
      saved, which put it in the wrong week);
    * a start time with **no date**, so its week is a best guess;
    * runs whose **odometers and start times disagree about the order** —
      the odometer only goes up, so the run with the higher reading came later.
    """
    out: Dict[str, List[Dict[str, str]]] = {}
    starts: Dict[str, Optional[datetime]] = {}
    for record in records:
        start, day, source = trip_start(record, kinds, columns)
        starts[record["id"]] = start if source and NO_DATE not in source else None
        kept = datetime.fromtimestamp(float(record.get("created_at") or 0))
        notes: List[Dict[str, str]] = []
        if start is not None and NO_DATE not in source and start > kept + timedelta(minutes=30):
            notes.append({"level": "error",
                          "text": f"Starts {_when(start)}, after it was kept ({_when(kept)}) — "
                                  "the date is probably wrong"})
        if NO_DATE in source:
            notes.append({"level": "note",
                          "text": f"Start time has no date, so it is counted on {day.day} "
                                  f"{day:%b} from when it was kept — add the date to be sure"})
        if notes:
            out[record["id"]] = notes
    for start_col, end_col in _odometers(declaration, kinds, columns):
        ordered = []
        for record in records:
            span = _span(record, start_col, end_col)
            if span and starts.get(record["id"]):
                ordered.append((span[0], span[1], starts[record["id"]], record))
        ordered.sort(key=lambda item: item[0])
        for before, after in zip(ordered, ordered[1:]):
            if after[2] < before[2] - timedelta(minutes=30):
                out.setdefault(after[3]["id"], []).append({
                    "level": "error",
                    "text": f"The odometer puts this after the {_when(before[2])} run "
                            f"({before[0]:,.0f}–{before[1]:,.0f}), but its start time is "
                            "earlier — one of the two dates is wrong"})
        break
    return out
