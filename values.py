#!/usr/bin/env python3
"""Put a recorded value into a standard shape without changing what it says.

A routine's fields are written by whichever model answered, in whatever words
it reached for that day. One log of the same trip, twice, five minutes apart:

    "102,072"      "102,072 mi"       "100,409 miles"
    93             93 mi              66 miles
    $1.2465        $1.24 per mile     $0.55 per mile ($36.00 / 66 mi)
    21.2 mph       ≈ 23.21 mph        21.70 MPH

Every one of those is correct. None of them sorts, sums, or compares against
the row above it, which is what a log is for.

**The rule this module works to: normalise the presentation, never the number.**
`$1.2465` does not become `$1.25`, and `$36.00` does not become `$36`. Rounding
is a change to the data, and the whole point of a record is that it still says
in a year what it said today. What gets removed is only ever decoration — a
thousands separator, a repeated unit, an "approximately", a parenthetical
showing the working — and the caller keeps the original string alongside, so
even the decoration is recoverable.

Nothing here asks a model anything. It is ordinary parsing over values a model
already produced, which is the right tool once the shapes are this small and
this repetitive — and unlike the extraction step it cannot invent a figure.
"""

from __future__ import annotations

import re
from typing import Dict, List, NamedTuple, Optional

# The kinds a column can be. "text" is the honest fallback and the default:
# a value this cannot read is passed through untouched rather than mangled.
MONEY, DISTANCE, SPEED, DURATION, TIMESTAMP, NUMBER, TEXT = (
    "money", "distance", "speed", "duration", "timestamp", "number", "text")

KINDS = (MONEY, DISTANCE, SPEED, DURATION, TIMESTAMP, NUMBER, TEXT)

# What each kind is measured in, for a column header. Money is deliberately
# absent: the symbol is part of the value and there is no reason to assume USD.
UNITS = {DISTANCE: "mi", SPEED: "mph", DURATION: "hours"}


class Value(NamedTuple):
    """One parsed value.

    ``digits`` is the number exactly as it was written, minus separators — it
    is what gets rendered back out, so that precision survives a round trip.
    ``number`` is the same thing as a float, for arithmetic and for sorting,
    where the extra digits do not matter and being a number does.
    """
    kind: str
    digits: str
    number: float
    unit: str = ""
    text: str = ""          # the canonical rendering
    approximate: bool = False


# A model showing its working: "$0.55 per mile ($36.00 / 66 mi)". The answer is
# outside the bracket and the bracket is a second, differently-formatted copy of
# the inputs — which is exactly what a naive "first number" match would grab.
#
# Except a timezone, which arrives in brackets too: "at 20:06 (evening)
# (UTC-04:00)". Stripping that threw away the one part of a timestamp that
# cannot be recovered by looking at it again — a trip at 20:06 in an unstated
# zone is not the same fact as one at 20:06 UTC-04:00, and every row logged
# while travelling would have been quietly wrong.
_WORKING = re.compile(r"\s*\((?![^()]*(?:UTC|GMT))[^()]*\)", re.I)

# "or 3.35 hours", "≈", "about". The value is not less true without them.
_HEDGE = re.compile(r"(?:≈|~|\bapprox(?:\.|imately)?\b|\babout\b|\baround\b)\s*", re.I)

# A number, whole or fractional, with or without thousands separators, and
# with or without a leading minus. Both of the optional parts were missing at
# first and both silently changed the figure rather than failing to read it:
# "-$12.50" came out as "$12.50" — a refund recorded as income — and ".5 mi"
# came out as "5 mi", ten times the distance. A parser that cannot read a value
# must leave it alone; one that reads it wrongly is worse than not having one.
_NUM = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+"
_SIGNED = r"(-?\s*(?:" + _NUM + r"))"

# The sign may sit either side of the symbol: -$12.50 and $-12.50 both occur.
_MONEY_RE = re.compile(r"(-)?\s*([$£€])\s*(-)?\s*(" + _NUM + r")")
_DISTANCE_RE = re.compile(_SIGNED + r"\s*(mi\b|miles?\b|km\b|kilometres?\b|kilometers?\b)", re.I)
_SPEED_RE = re.compile(_SIGNED + r"\s*(mph\b|mi/h\b|km/h\b|kph\b)", re.I)
_BARE_RE = re.compile(r"^\s*" + _SIGNED + r"\s*$")

# "4 hours 25 minutes", "3 hours and 8 minutes", "3.35 hours", "45 minutes".
_HOURS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:h\b|hr?s?\b|hours?\b)", re.I)
_MINS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:m\b|mins?\b|minutes?\b)", re.I)
_CLOCK_RE = re.compile(r"^\s*(\d{1,3}):([0-5]\d)\s*$")

_MONTHS = ("january february march april may june july august september "
           "october november december").split()
_MONTH_AT = {name[:3]: i + 1 for i, name in enumerate(_MONTHS)}

_WEEKDAY = re.compile(
    r"\b(mon|tues?|wed(?:nes)?|thur?s?|fri|sat(?:ur)?|sun)(?:day)?\b[,\s]*", re.I)
# "(evening)", "(morning)" — a label for the time already given.
_DAYPART = re.compile(r"\s*\((?:early |late )?(?:morning|afternoon|evening|night|noon|midnight)\)", re.I)
_OFFSET = re.compile(r"\(?\b(?:UTC|GMT)\s*([+-]\d{1,2})(?::?(\d{2}))?\b\)?", re.I)

_DMY = re.compile(r"\b(\d{1,2})\s+([A-Za-z]{3,9})\.?\s+(\d{4})\b")
_MDY = re.compile(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b")
# No trailing \b: an ISO stamp runs "2026-08-25T20:06", and there is no word
# boundary between a digit and a T, so the whole value failed to parse.
_ISO_DATE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})")
# Same reason, plus a guard against starting mid-clock: without the colon in
# the lookbehind, "20:06" could also match at "0:06".
_TIME = re.compile(r"(?<![\d:])(\d{1,2}):(\d{2})(?::\d{2})?\s*(am|pm)?", re.I)
_ISO_T = re.compile(r"(\d)T(\d)")


def _clean(text: str) -> str:
    """Whitespace collapsed, working and hedging removed."""
    flat = " ".join(str(text or "").split())
    return _HEDGE.sub("", _WORKING.sub("", flat)).strip()


# What may be left over once the quantity has been read out of a value, and
# still leave it a quantity. Anything else means the value is prose that merely
# *contains* a number — "54 miles to Brighton" — and prose must be left exactly
# as it was found. Rewriting that one to "54 mi" is not tidying, it is deleting
# where the trip went, and it took the word Brighton out of search with it.
_RESIDUE = re.compile(
    r"^(?:"
    r"per\s+(?:mile|mi|hour|hr|km|gallon|gal|day|week|trip)|"
    r"/\s*(?:mile|mi|hour|hr|km)|"
    r"an?\s+(?:mile|hour)|each|total|and|or|at|on|of|"
    r"[\s,.;:/()]"
    r")*$", re.I)


def _left_over(text: str, spans: List[tuple]) -> str:
    """What remains of ``text`` once the parts that were understood are cut."""
    keep, at = [], 0
    for start, end in sorted(spans):
        keep.append(text[at:start])
        at = max(at, end)
    keep.append(text[at:])
    return " ".join("".join(keep).split())


def _only_a_quantity(text: str, spans: List[tuple]) -> bool:
    """Whether the value is the quantity, rather than a sentence containing one."""
    return bool(_RESIDUE.match(_left_over(text, spans)))


def _digits(body: str) -> str:
    """A matched number, separators gone and every digit kept.

    ".5" is written back as "0.5". The zero is not a change to the figure and
    it stops a value being read back as 5 by anything less careful than this.
    """
    body = " ".join(str(body or "").split()).replace(" ", "").replace(",", "")
    if body.startswith("."):
        return "0" + body
    if body.startswith("-."):
        return "-0" + body[1:]
    return body


def _money(text: str) -> Optional[Value]:
    match = _MONEY_RE.search(text)
    if not match or not _only_a_quantity(text, [match.span()]):
        return None
    sign = "-" if (match.group(1) or match.group(3)) else ""
    digits = _digits(sign + match.group(4))
    symbol = match.group(2)
    # The sign goes outside the symbol, which is how a person writes it.
    body = digits[1:] if digits.startswith("-") else digits
    return Value(MONEY, digits, float(digits), symbol,
                 f"{'-' if digits.startswith('-') else ''}{symbol}{body}")


def _distance(text: str) -> Optional[Value]:
    match = _DISTANCE_RE.search(text)
    if not match or not _only_a_quantity(text, [match.span()]):
        return None
    unit = "km" if match.group(2).lower().startswith(("km", "kilom")) else "mi"
    digits = _digits(match.group(1))
    return Value(DISTANCE, digits, float(digits), unit, f"{digits} {unit}")


def _speed(text: str) -> Optional[Value]:
    match = _SPEED_RE.search(text)
    if not match or not _only_a_quantity(text, [match.span()]):
        return None
    unit = "km/h" if match.group(2).lower() in ("km/h", "kph") else "mph"
    digits = _digits(match.group(1))
    return Value(SPEED, digits, float(digits), unit, f"{digits} {unit}")


def _duration(text: str) -> Optional[Value]:
    clock = _CLOCK_RE.match(text)
    if clock:
        total = int(clock.group(1)) * 60 + int(clock.group(2))
        return Value(DURATION, f"{total / 60:.4f}".rstrip("0").rstrip("."),
                     total / 60, "hours", _hm(total))
    hours = _HOURS_RE.search(text)
    mins = _MINS_RE.search(text)
    if not hours and not mins:
        return None
    if not _only_a_quantity(text, [m.span() for m in (hours, mins) if m]):
        return None
    total = 0.0
    if hours:
        total += float(hours.group(1)) * 60
    if mins:
        total += float(mins.group(1))
    if total <= 0:
        return None
    as_hours = total / 60
    return Value(DURATION, f"{as_hours:.4f}".rstrip("0").rstrip("."),
                 as_hours, "hours", _hm(round(total)))


def _hm(minutes: int) -> str:
    """A duration the way a person says one: 4h 25m, or 45m under the hour."""
    hours, mins = divmod(int(minutes), 60)
    if not hours:
        return f"{mins}m"
    return f"{hours}h {mins:02d}m" if mins else f"{hours}h"


def _timestamp(text: str) -> Optional[Value]:
    """A date and time as ISO-ish, keeping only what was actually stated.

    No guessing: a value with a date and no time comes back as a date, and one
    with neither comes back as nothing at all. Inventing 00:00 for a missing
    time would put a trip at midnight and look like a reading rather than a gap.
    """
    # The T of an ISO stamp is a separator, not a word. Left in, it is residue
    # with a letter in it, and the whole value reads as prose about a date.
    body = _ISO_T.sub(r"\1 \2", text)
    body = _DAYPART.sub("", _WEEKDAY.sub("", body))
    offset = ""
    off = _OFFSET.search(body)
    if off:
        offset = f"UTC{int(off.group(1)):+03d}:{off.group(2) or '00'}"
        body = _OFFSET.sub("", body)

    day = month = year = None
    iso = _ISO_DATE.search(body)
    dmy = _DMY.search(body)
    mdy = _MDY.search(body)
    if iso:
        year, month, day = int(iso.group(1)), int(iso.group(2)), int(iso.group(3))
    elif dmy and _MONTH_AT.get(dmy.group(2)[:3].lower()):
        day, month, year = (int(dmy.group(1)),
                            _MONTH_AT[dmy.group(2)[:3].lower()], int(dmy.group(3)))
    elif mdy and _MONTH_AT.get(mdy.group(1)[:3].lower()):
        month, day, year = (_MONTH_AT[mdy.group(1)[:3].lower()],
                            int(mdy.group(2)), int(mdy.group(3)))
    if not year or not 1 <= (month or 0) <= 12 or not 1 <= (day or 0) <= 31:
        return None
    # The date and the time are cut out of what is left after the weekday and
    # the offset have gone; anything still standing means this is a sentence
    # about a date rather than a date.
    spans = [m.span() for m in (iso, dmy, mdy, _TIME.search(body)) if m]
    if not _only_a_quantity(body, spans):
        return None

    stamp = f"{year:04d}-{month:02d}-{day:02d}"
    clock = _TIME.search(body)
    if clock:
        hour, minute = int(clock.group(1)), int(clock.group(2))
        meridiem = (clock.group(3) or "").lower()
        if meridiem == "pm" and hour < 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        if hour < 24 and minute < 60:
            stamp += f" {hour:02d}:{minute:02d}"
    if offset:
        stamp += f" {offset}"
    # Sorts as a string, which is the whole reason for this shape.
    return Value(TIMESTAMP, stamp, 0.0, offset, stamp)


def _bare(text: str) -> Optional[Value]:
    match = _BARE_RE.match(text)
    if not match:
        return None
    digits = _digits(match.group(1))
    return Value(NUMBER, digits, float(digits), "", digits)


# Order matters. Money is first because "$1.24 per mile" also contains a bare
# number; speed before distance because "21 mph" must not read as 21 miles;
# timestamp before bare because a lone year would otherwise be a number.
_READERS = (_money, _speed, _distance, _duration, _timestamp, _bare)


def parse(text: str) -> Optional[Value]:
    """Read one value, or None when it is prose rather than a quantity."""
    body = _clean(text)
    if not body:
        return None
    approximate = bool(_HEDGE.search(" ".join(str(text or "").split())))
    for reader in _READERS:
        found = reader(body)
        if found:
            return found._replace(approximate=approximate)
    return None


def parse_as(text: str, kind: str) -> Optional[Value]:
    """Read one value as a known kind, which is what a column gives us.

    A bare "93" in a column of miles is 93 miles — a fact only the column
    knows, and the single most common shape in a real log, because a model
    that has just written "Distance traveled" does not repeat the unit.
    """
    body = _clean(text)
    if not body or kind in (TEXT, ""):
        return None
    direct = {MONEY: _money, DISTANCE: _distance, SPEED: _speed,
              DURATION: _duration, TIMESTAMP: _timestamp}.get(kind)
    found = direct(body) if direct else None
    if not found:
        bare = _bare(body)
        if not bare:
            # Not this column's kind, and not a bare number either. Left alone
            # rather than re-read as whatever else it might be: a "20:06" in a
            # column of start times came back as a duration of twenty hours,
            # because a clock and a stopwatch are written identically and only
            # the column knows which one this is.
            return None
        if kind == MONEY:
            found = Value(MONEY, bare.digits, bare.number, "$", f"${bare.digits}")
        elif kind in (DISTANCE, SPEED):
            unit = UNITS[kind]
            found = Value(kind, bare.digits, bare.number, unit,
                          f"{bare.digits} {unit}")
        elif kind == DURATION:
            found = Value(DURATION, bare.digits, bare.number, "hours",
                          _hm(round(bare.number * 60)))
        else:
            found = bare
    approximate = bool(_HEDGE.search(" ".join(str(text or "").split())))
    return found._replace(approximate=approximate)


def column_kind(values: List[str]) -> str:
    """One kind for a whole column, decided by what most of it looks like.

    Per value this cannot be done: "102,072" is a number and "102,072 mi" is a
    distance, and they are the same odometer written twice. A column is the
    unit of meaning here, so the column votes — and a bare number never
    outvotes a value that named its own unit, since naming it is evidence and
    omitting it is not.
    """
    votes: Dict[str, int] = {}
    filled = 0
    for raw in values:
        if not str(raw or "").strip():
            continue
        filled += 1
        found = parse(raw)
        if found:
            votes[found.kind] = votes.get(found.kind, 0) + 1
    if not votes:
        return TEXT
    # A column of notes with one stray quantity in it is a column of notes.
    # Measured against how much of it did not parse *at all*, not against the
    # other quantities: a bare number sitting beside "3 mi" is the same column
    # written twice, while "quiet night" beside it is not.
    if filled - sum(votes.values()) > sum(votes.values()):
        return TEXT
    named = {k: n for k, n in votes.items() if k != NUMBER}
    if named:
        # Naming a unit is evidence; omitting one is not, so a single "3 mi"
        # decides a column of bare numbers rather than being outvoted by them.
        return max(named, key=lambda k: (named[k], -KINDS.index(k)))
    return NUMBER if votes.get(NUMBER) else TEXT


def canonical(text: str, kind: str = "") -> str:
    """The standard rendering of one value; unchanged when it cannot be read."""
    flat = " ".join(str(text or "").split())
    if not flat:
        return ""
    found = parse_as(flat, kind) if kind else parse(flat)
    return found.text if found else flat


def canonical_column(values: List[str], kind: str = "") -> List[str]:
    """Every value in a column, rendered the same way as its neighbours."""
    kind = kind or column_kind(values)
    return [canonical(v, kind) for v in values]


def number_of(text: str, kind: str = "") -> str:
    """The bare number, for a spreadsheet cell. "" when there isn't one.

    A cell reading "$115.94" is text to a spreadsheet and will not sum; one
    reading 115.94 under a column headed "Total earnings (USD)" is a number
    and says exactly as much.
    """
    found = parse_as(text, kind) if kind else parse(text)
    if not found or found.kind == TIMESTAMP:
        return canonical(text, kind) if found else " ".join(str(text or "").split())
    return found.digits


def unit_label(kind: str, values: List[str]) -> str:
    """What to put in a column header, or "" when the kind carries no unit."""
    if kind == MONEY:
        symbols = {parse(v).unit for v in values if parse(v)} - {""}
        return "USD" if symbols <= {"$"} else "/".join(sorted(symbols))
    if kind in (DISTANCE, SPEED):
        units = {parse_as(v, kind).unit for v in values if parse_as(v, kind)} - {""}
        return "/".join(sorted(units)) or UNITS[kind]
    return UNITS.get(kind, "")
