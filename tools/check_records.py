#!/usr/bin/env python3
"""Look over a log of records for the things standardising deliberately won't fix.

Putting values into one shape is safe, so the app does it on its own. Changing
a *number* is not, so nothing here does — it reports, and you decide.

Three things it looks for:

* **Rows that disagree with their own arithmetic.** A routine's derived fields
  are written by a model doing sums in prose, and a model that has been given
  no duration will still produce an hourly rate. Found in a real log: the same
  trip recorded twice, once as $26.23/hour and once as $23.19/hour. The second
  had no start or end time in it at all.
* **The same run recorded twice.** Two captures of one screen, minutes apart.
* **Values still worth tidying**, as a preview of what the app would do — so
  you can see the change before it happens rather than after.

    .venv/bin/python tools/check_records.py
    .venv/bin/python tools/check_records.py --csv records.csv
    .venv/bin/python tools/check_records.py --routine "🚗 Uber Trip"
"""

from __future__ import annotations

import csv as csvlib
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import values  # noqa: E402

OK, BAD, WARN, INFO = "  ✅", "  ❌", "  ⚠️ ", "     ·"

# Which column plays which part, recognised by name because that is all there
# is to go on. Deliberately conservative: a check is only run when every input
# it needs was identified, so an unrecognised log is reported as unrecognised
# rather than quietly half-checked.
ROLES = {
    "start_odo": ("start odometer", "odometer start", "starting odometer"),
    "end_odo": ("end odometer", "odometer end", "ending odometer", "final odometer"),
    "distance": ("distance traveled", "distance travelled", "distance", "miles", "mileage"),
    "start_time": ("start time", "started", "start"),
    "end_time": ("end time", "ended", "finish time", "end"),
    "elapsed": ("elapsed time", "elapsed", "duration", "time worked", "hours worked"),
    "speed": ("average speed", "avg speed", "speed"),
    "earnings": ("total earnings", "earnings", "total", "fare", "pay", "income"),
    "per_mile": ("earnings per mile", "per mile", "$/mile", "rate per mile"),
    "per_hour": ("earnings per hour", "per hour", "$/hour", "hourly", "rate per hour"),
}


def find_roles(columns: List[str]) -> Dict[str, str]:
    """Map each role to the column that plays it, best match first."""
    found: Dict[str, str] = {}
    lowered = {c: " ".join(c.lower().split()) for c in columns}
    for role, names in ROLES.items():
        for wanted in names:                    # most specific name first
            for column, low in lowered.items():
                if column in found.values():
                    continue
                if low == wanted or wanted in low:
                    found[role] = column
                    break
            if role in found:
                break
    return found


def _num(row: Dict[str, str], column: str, kind: str = "") -> float:
    """One cell as a number, or NaN when it isn't one."""
    if not column:
        return float("nan")
    found = values.parse_as(row.get(column, ""), kind) if kind else values.parse(row.get(column, ""))
    return found.number if found and found.kind != values.TIMESTAMP else float("nan")


def check_arithmetic(rows: List[Dict[str, str]], roles: Dict[str, str]) -> int:
    """Recompute what can be recomputed and report where the row disagrees."""
    print("\n2. Do the rows agree with their own arithmetic?")
    if not roles.get("earnings"):
        print(f"{WARN} no earnings column recognised, so there is nothing to check")
        return 0

    problems = 0
    for i, row in enumerate(rows, 1):
        earn = _num(row, roles.get("earnings", ""), values.MONEY)
        dist = _num(row, roles.get("distance", ""), values.DISTANCE)
        hours = _num(row, roles.get("elapsed", ""), values.DURATION)
        d0 = _num(row, roles.get("start_odo", ""), values.DISTANCE)
        d1 = _num(row, roles.get("end_odo", ""), values.DISTANCE)
        label = row.get("__label__") or f"row {i}"

        def say(what: str, stated: float, computed: float, unit: str = "") -> None:
            nonlocal problems
            problems += 1
            print(f"{BAD} {label}: {what} says {unit}{stated:g}, "
                  f"the row's own figures give {unit}{computed:.4g}")

        if d0 == d0 and d1 == d1 and dist == dist and abs((d1 - d0) - dist) > 0.5:
            say("distance", dist, d1 - d0)
        if earn == earn and dist == dist and dist > 0:
            stated = _num(row, roles.get("per_mile", ""), values.MONEY)
            if stated == stated and abs(stated - earn / dist) > 0.015:
                say("earnings per mile", stated, earn / dist, "$")
        if earn == earn and hours == hours and hours > 0:
            stated = _num(row, roles.get("per_hour", ""), values.MONEY)
            if stated == stated and abs(stated - earn / hours) > 0.05:
                say("earnings per hour", stated, earn / hours, "$")
        if dist == dist and hours == hours and hours > 0:
            stated = _num(row, roles.get("speed", ""), values.SPEED)
            if stated == stated and abs(stated - dist / hours) > 0.5:
                say("average speed", stated, dist / hours)

        # A rate with nothing to divide by was not measured, it was guessed —
        # which is exactly how one real row came to be 13% out.
        for role, needs, what in (("per_hour", hours, "elapsed time"),
                                  ("per_mile", dist, "distance")):
            stated = _num(row, roles.get(role, ""), values.MONEY)
            if stated == stated and needs != needs:
                problems += 1
                print(f"{BAD} {label}: {roles[role]} is ${stated:g}, but the row "
                      f"has no {what} to have worked it out from")

    if not problems:
        print(f"{OK} every row that can be checked adds up")
    return problems


def check_duplicates(rows: List[Dict[str, str]], roles: Dict[str, str]) -> int:
    """Two captures of one screen. Identical readings are one run, not two."""
    print("\n3. Is anything recorded twice?")
    keys = [roles.get(r, "") for r in ("start_odo", "end_odo", "earnings")]
    keys = [k for k in keys if k]
    if len(keys) < 2:
        print(f"{WARN} not enough recognised columns to tell two runs apart")
        return 0
    # Standardised before comparing, or the two captures of one trip never
    # match: one wrote "102,072" and the other "102,072 mi", which is the very
    # inconsistency that hid the duplicate in the first place.
    kinds = {k: values.column_kind([r.get(k, "") for r in rows]) for k in keys}
    seen: Dict[tuple, List[str]] = {}
    for i, row in enumerate(rows, 1):
        key = tuple(values.canonical(row.get(k, ""), kinds[k]) for k in keys)
        if not any(key):
            continue
        seen.setdefault(key, []).append(row.get("__label__") or f"row {i}")
    repeats = {k: v for k, v in seen.items() if len(v) > 1}
    for key, where in repeats.items():
        print(f"{BAD} {' / '.join(str(k) for k in key)} appears "
              f"{len(where)} times: {', '.join(where)}")
    if not repeats:
        print(f"{OK} no run appears twice")
    return len(repeats)


def check_shape(rows: List[Dict[str, str]], columns: List[str]) -> int:
    """What standardising would change, as a preview."""
    print("\n4. Values not yet in the standard shape")
    pending = 0
    for column in columns:
        seen = [r.get(column, "") for r in rows]
        kind = values.column_kind(seen)
        changes = [(v, values.canonical(v, kind)) for v in seen
                   if v and values.canonical(v, kind) != v]
        if not changes:
            continue
        unit = values.unit_label(kind, seen)
        print(f"{INFO} {column} — {kind}{' (' + unit + ')' if unit else ''}")
        for was, now in changes[:4]:
            print(f"        {was!r} → {now!r}")
        if len(changes) > 4:
            print(f"        …and {len(changes) - 4} more")
        pending += len(changes)
    if not pending:
        print(f"{OK} everything is already in shape")
    else:
        print(f"{INFO} the app does this itself on the next start; the value it "
              "replaces is kept")
    return pending


def load_csv(path: str) -> tuple:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csvlib.DictReader(handle))
    if not rows:
        return [], []
    skip = {"taken_at", "routine"}
    columns = [c for c in rows[0] if c and c not in skip]
    for row in rows:
        row["__label__"] = row.get("taken_at") or ""
    return rows, columns


def load_db(routine: str) -> tuple:
    import store
    records = store.list_records(routine)
    rows = []
    for record in records:
        row = dict(record["fields"])
        row["__label__"] = record["id"][:8]
        rows.append(row)
    return rows, store.record_columns(records)


def main() -> int:
    args = sys.argv[1:]
    routine = ""
    path = ""
    if "--csv" in args:
        path = args[args.index("--csv") + 1]
    if "--routine" in args:
        routine = args[args.index("--routine") + 1]

    rows, columns = load_csv(path) if path else load_db(routine)
    print(f"Checking {len(rows)} record(s) from {path or 'the database'}"
          + (f", routine {routine!r}" if routine else ""))
    if not rows:
        print(f"{WARN} nothing to check")
        return 0

    print("\n1. Which column is which")
    roles = find_roles(columns)
    for role, column in roles.items():
        print(f"{INFO} {role:<11} {column}")
    missing = [r for r in ROLES if r not in roles]
    if missing:
        print(f"{INFO} not recognised: {', '.join(missing)} — checks needing "
              "these are skipped rather than guessed")

    problems = check_arithmetic(rows, roles) + check_duplicates(rows, roles)
    check_shape(rows, columns)

    print("\n" + "-" * 62)
    if problems:
        print(f"{problems} thing(s) above need a person: nothing here edits a")
        print("number, because a log that corrects itself is not a log.")
        return 1
    print("Nothing needs deciding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
