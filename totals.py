#!/usr/bin/env python3
"""Totals across a routine's log: by week, by month, and all time.

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


def day_of(record: Dict[str, Any], kinds: Dict[str, str],
           columns: List[str]) -> Tuple[date, str]:
    """The day a run happened, and where that came from.

    The run's own first timestamp where it has one with a date in it: a trip
    that started at 18:34 and was logged at 06:03 the next morning belongs to
    the day it was driven. Otherwise the day it was kept.
    """
    for name in columns:
        if kinds.get(name) != values.TIMESTAMP:
            continue
        match = _DATE.match(str(record["fields"].get(name, "")).strip())
        if match:
            try:
                return date(*(int(g) for g in match.groups())), name
            except ValueError:
                continue
    return datetime.fromtimestamp(float(record.get("created_at") or 0)).date(), ""


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
