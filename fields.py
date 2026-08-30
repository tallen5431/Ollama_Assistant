#!/usr/bin/env python3
"""What a routine records: the fields, their kinds, and which are worked out.

A routine used to declare a bare list of names, and a model was asked to
restate its own answer as all of them — including the ones that are arithmetic
over the others. That is the wrong job to give a language model, and a real log
shows why: one trip recorded twice, five minutes apart, came back as $26.23 an
hour and as $23.19 an hour. The second row had no start or end time in it at
all, so its rate had been worked out from nothing. The model was not asked to
divide; it was asked what the answer said, and it obliged.

So a field can now say what it is, and a field that is arithmetic can say so:

    Start odometer: distance
    End odometer: distance
    Distance traveled = End odometer - Start odometer
    Start time: timestamp
    End time: timestamp
    Elapsed time = End time - Start time
    Total earnings: money
    Earnings per mile = Total earnings / Distance traveled
    Earnings per hour = Total earnings / Elapsed time

Only the four plain fields are asked for. The rest are computed here, from
those, in Python — so a rate with nothing to divide by comes out **empty**
rather than invented, which is the whole point.

A bare name still means what it always did: read it from the answer, and let
the kind be inferred from the column. Every routine written before this reads
exactly as it did, because a list of names is a list of untyped fields.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import values

# Only these two shapes, and only two operands. A field is a reading or one
# arithmetic step over two readings; anything more belongs in a spreadsheet,
# and a small grammar is one whose failures are all visible.
_DECLARE = re.compile(r"^(?P<name>[^:=]+?)\s*:\s*(?P<kind>[A-Za-z]+)$")
_COMPUTE = re.compile(r"^(?P<name>[^=]+?)\s*=\s*(?P<left>.+?)\s*"
                      r"(?P<op>[-+*/])\s*(?P<right>.+)$")

# A value taken from the photo's own file rather than from anything a model
# said about it. The app already has these exactly — the browser reads the EXIF
# before the image is re-encoded — and it was rendering them to prose, asking a
# model to read that prose, and parsing the model's prose back. Four hops for a
# figure that started out exact, and the middle two can invent.
#
# Worse than lossy: the block labels its lines "Image 1" and "Image 2" while
# the photos themselves carry no labels at all, so the model has to align two
# lists across two messages by position. That join is what fails, and it fails
# on large models too, because it is a binding problem rather than a hard one.
_SOURCE = re.compile(
    r"^(?P<name>[^=]+?)\s*=\s*(?:"
    r"photo\s+(?P<index>\d{1,2})\s+taken"
    r"|(?P<order>earliest|first|latest|last)\s+photo\s+taken"
    r")$", re.I)

OPS = "-+*/"

# What comes out of combining two kinds. Absent means "a number", which is the
# honest answer for arithmetic nobody has taught this about.
_RESULT = {
    (values.DISTANCE, "-", values.DISTANCE): values.DISTANCE,
    (values.DISTANCE, "+", values.DISTANCE): values.DISTANCE,
    (values.TIMESTAMP, "-", values.TIMESTAMP): values.DURATION,
    (values.DURATION, "-", values.DURATION): values.DURATION,
    (values.DURATION, "+", values.DURATION): values.DURATION,
    (values.MONEY, "-", values.MONEY): values.MONEY,
    (values.MONEY, "+", values.MONEY): values.MONEY,
    (values.MONEY, "/", values.DISTANCE): values.MONEY,
    (values.MONEY, "/", values.DURATION): values.MONEY,
    (values.MONEY, "/", values.NUMBER): values.MONEY,
    (values.MONEY, "*", values.NUMBER): values.MONEY,
    (values.DISTANCE, "/", values.DURATION): values.SPEED,
    (values.NUMBER, "-", values.NUMBER): values.NUMBER,
    (values.NUMBER, "+", values.NUMBER): values.NUMBER,
}

# A timestamp subtracts as seconds, so the difference has to be put back into
# the units a duration is counted in.
_SECONDS_TO_HOURS = {(values.TIMESTAMP, "-", values.TIMESTAMP): 1 / 3600.0}


class Field(NamedTuple):
    """One column of a routine's log."""
    name: str
    kind: str = ""            # "" means "work it out from the column"
    left: str = ""            # the other two are empty unless this is computed
    op: str = ""
    right: str = ""
    source: str = ""          # "photo 2 taken", "earliest photo taken"

    @property
    def computed(self) -> bool:
        return bool(self.op)

    @property
    def derived(self) -> bool:
        """Whether this comes from somewhere other than the model's answer."""
        return bool(self.op or self.source)

    def declaration(self) -> str:
        """The text this was written as, which is also how it is stored."""
        if self.source:
            return f"{self.name} = {self.source}"
        if self.computed:
            return f"{self.name} = {self.left} {self.op} {self.right}"
        return f"{self.name}: {self.kind}" if self.kind else self.name


def parse(raw: Any) -> List[Field]:
    """Read a declaration, however it arrives.

    A list of plain names — every routine written before this — comes back as
    untyped read fields, which is exactly what it used to mean.
    """
    out: List[Field] = []
    seen = set()
    for line in _lines(raw):
        field = _one(line)
        if not field or field.name.lower() in seen:
            continue
        seen.add(field.name.lower())
        out.append(field)
    return out


def _lines(raw: Any) -> List[str]:
    """Declarations, split on newlines or commas.

    Both, because the editor is one box and people use whichever separator is
    to hand — and a formula with spaces in its field names reads far better on
    a line of its own.
    """
    if isinstance(raw, str):
        parts: List[str] = [raw]
    elif isinstance(raw, (list, tuple)):
        parts = [str(item) for item in raw if item is not None]
    else:
        return []
    out: List[str] = []
    for part in parts:
        for line in re.split(r"[\n,]", str(part)):
            line = " ".join(line.split())
            if line:
                out.append(line)
    return out


def _one(line: str) -> Optional[Field]:
    """One declaration, or None when there is nothing usable in it."""
    taken = _SOURCE.match(line)
    if taken:
        name = taken.group("name").strip()
        if not name:
            return None
        where = (f"photo {int(taken.group('index'))} taken" if taken.group("index")
                 else f"{taken.group('order').lower()} photo taken")
        # Always a timestamp: it is a camera's record of when the shutter went.
        return Field(name, values.TIMESTAMP, source=where)
    computed = _COMPUTE.match(line)
    if computed:
        name = computed.group("name").strip()
        left = computed.group("left").strip()
        right = computed.group("right").strip()
        if name and left and right:
            return Field(name, "", left, computed.group("op"), right)
        return Field(name) if name else None
    declared = _DECLARE.match(line)
    if declared:
        kind = declared.group("kind").strip().lower()
        name = declared.group("name").strip()
        # An unknown kind is a typo, and guessing which one was meant is worse
        # than ignoring it: the column still works, it just infers as before.
        return Field(name, kind if kind in values.KINDS else "") if name else None
    return Field(line) if line else None


def problems(fields: List[Field], raw: Any = None) -> List[str]:
    """What is wrong with a declaration, in words, or [] when nothing is.

    Checked when a routine is *saved*, because the alternative is finding out
    much later and indirectly. A formula naming a field that does not exist —
    one typo, "Elapsed tme" — is not an error anywhere: it computes to nothing,
    every run, forever, and an empty column looks exactly like a run where
    there was no data. This is the difference between a mistake you fix in
    seconds and one you notice in a month of records.
    """
    known = {f.name.lower() for f in fields}
    out: List[str] = []
    # Said once per field. A self-referential formula is also, technically, a
    # cycle, and reporting it twice in different words reads as two faults.
    complained: set = set()

    for field in fields:
        if not field.computed:
            continue
        for side, operand in (("left", field.left), ("right", field.right)):
            if any(op in operand for op in OPS):
                # "Net = Gross - Fees - Tax". The grammar is two operands and
                # one operator, so the tail became part of a field name that
                # can never exist — a column permanently empty for a reason
                # nothing announced.
                out.append(
                    f"{field.name}: \"{operand}\" is more than one sum. This "
                    "understands one operator over two fields; give the middle "
                    "step a field of its own and build on it.")
            elif operand.lower() not in known:
                out.append(f"{field.name}: there is no field called \"{operand}\".")
            elif operand.lower() == field.name.lower():
                out.append(f"{field.name}: works itself out from itself.")
            else:
                continue
            complained.add(field.name.lower())

    # A line that was written as a formula and did not parse as one. It still
    # becomes a plain field, so the column works — but it does not do what was
    # typed, and silently keeping the name while dropping the rest is how you
    # end up with a field called "x" and no idea why.
    for line in _lines(raw if raw is not None else []):
        if "=" not in line or _SOURCE.match(line):
            continue
        match = _COMPUTE.match(line)
        if not match or not (match.group("left").strip()
                             and match.group("right").strip()):
            out.append(f"\"{line}\" is not a sum this understands, so it was "
                       "kept as an ordinary field name.")

    # Cycles, and anything else that cannot be reached. Resolution is by
    # dependency now, so the only computed fields left unresolvable are ones
    # that need each other.
    resolved = {f.name.lower() for f in fields if not f.computed}
    pending = [f for f in fields if f.computed]
    while pending:
        ready = [f for f in pending
                 if f.left.lower() in resolved and f.right.lower() in resolved]
        if not ready:
            break
        resolved |= {f.name.lower() for f in ready}
        pending = [f for f in pending if f not in ready]
    for field in pending:
        if field.name.lower() in complained:
            continue
        if all(o.lower() in known for o in (field.left, field.right)):
            out.append(f"{field.name}: this and \"{field.left}\" or "
                       f"\"{field.right}\" each wait on the other.")
    return out


def names(fields: List[Field]) -> List[str]:
    """Every column, in order, computed ones included."""
    return [f.name for f in fields]


def to_ask(fields: List[Field]) -> List[Field]:
    """The fields a model has to supply, which is the ones nothing can derive."""
    return [f for f in fields if not f.derived]


def declarations(fields: List[Field]) -> List[str]:
    """Back to the list of strings a routine stores."""
    return [f.declaration() for f in fields]


def kinds(fields: List[Field], rows: Optional[List[Dict[str, str]]] = None) -> Dict[str, str]:
    """The kind of each column: declared where it was, inferred where it wasn't.

    A declaration always wins. Inference is a good guess across a column and a
    good guess is still a guess, so when someone has said what a column holds,
    that is the answer.
    """
    out: Dict[str, str] = {}
    for field in fields:
        if field.kind:
            out[field.name] = field.kind
    if rows is not None:
        for field in fields:
            if field.name not in out:
                seen = [str(r.get(field.name, "")) for r in rows]
                out[field.name] = values.column_kind(seen)
    return out


# EXIF writes "2026:08:07 13:37:12"; the colons in the date are the giveaway.
_EXIF_TAKEN = re.compile(r"^\s*(\d{4})[:-](\d{2})[:-](\d{2})[ T](\d{2}):(\d{2})")
_EXIF_OFFSET = re.compile(r"^[+-]\d{2}:?\d{2}$")


def photo_time(meta: Any) -> str:
    """One photo's capture time, as a standard timestamp. "" if it has none."""
    if not isinstance(meta, dict):
        return ""
    match = _EXIF_TAKEN.match(" ".join(str(meta.get("taken") or "").split()))
    if not match:
        return ""
    stamp = (f"{match.group(1)}-{match.group(2)}-{match.group(3)} "
             f"{match.group(4)}:{match.group(5)}")
    offset = " ".join(str(meta.get("offset") or "").split())
    # Where the camera recorded one. Without it the time is still the time, it
    # just cannot be compared against a photo taken in another zone.
    if _EXIF_OFFSET.match(offset):
        stamp += f" UTC{offset}"
    return values.canonical(stamp, values.TIMESTAMP)


def from_photos(fields: List[Field],
                photos: Optional[List[Any]]) -> Dict[str, str]:
    """Fill the fields that come from a photo's own file rather than a model.

    This is the whole answer to "why does it get the times wrong": it does not
    have to read them. The browser read the EXIF before the image was
    re-encoded, so the exact capture time is already here — rendering it to
    prose and asking a model to read it back was three chances to lose it and
    one to invent it.

    ``earliest`` and ``latest`` sort by the recorded instant, which is what a
    trip actually needs: the photos are attached in whatever order they were
    picked, and the later one is the end of the trip whichever that was.
    """
    wanted = [f for f in fields if f.source]
    if not wanted or not photos:
        return {}
    times = [photo_time(meta) for meta in photos]
    # Sorted by the instant, not by the text: a stamp with an offset and one
    # without do not compare as strings in any useful way.
    ordered = sorted(
        (t for t in times if t),
        key=lambda t: (values.parse_as(t, values.TIMESTAMP) or
                       values.Value("", "", float("inf"))).number)
    out: Dict[str, str] = {}
    for field in wanted:
        match = re.match(r"^photo (\d+) taken$", field.source)
        if match:
            index = int(match.group(1)) - 1
            out[field.name] = times[index] if 0 <= index < len(times) else ""
        elif field.source.startswith(("earliest", "first")):
            out[field.name] = ordered[0] if ordered else ""
        else:
            out[field.name] = ordered[-1] if ordered else ""
    return out


def result_kind(left: str, op: str, right: str) -> str:
    """What kind the arithmetic produces."""
    return _RESULT.get((left, op, right), values.NUMBER)


def compute(fields: List[Field], row: Dict[str, str],
            known: Optional[Dict[str, str]] = None) -> Tuple[Dict[str, str], List[str]]:
    """Fill in the computed fields of one row.

    Returns the values and a note for each one that could not be worked out.
    A field whose inputs are missing is left **empty** — never guessed, which
    is the failure this whole idea exists to prevent — and the note says which
    input was missing, so a gap in the log reads as a gap rather than a bug.

    Computed fields can build on one another and are resolved by what each
    needs, not by the order they were written in — an hourly rate may divide by
    an elapsed time declared below it.
    """
    known = dict(known or {})
    known.update(kinds(fields))
    filled: Dict[str, str] = {}
    notes: List[str] = []
    # Readings first, so a computed field can be built from another one.
    numbers: Dict[str, values.Value] = {}
    for field in fields:
        if field.computed:
            continue
        found = values.parse_as(row.get(field.name, ""), known.get(field.name, ""))
        if found:
            numbers[field.name] = found

    # Resolved by what each one needs rather than by the order they were
    # written in. Declaration order used to decide it, so an hourly rate
    # written above the elapsed time it divides by came out empty while the
    # elapsed time right below it computed perfectly — a silent, order-dependent
    # blank, which is the least debuggable kind.
    pending = [f for f in fields if f.computed]
    while pending:
        stuck = []
        for field in pending:
            left, right = numbers.get(field.left), numbers.get(field.right)
            if not left or not right:
                stuck.append(field)
                continue
            _one_sum(field, left, right, numbers, filled, notes)
        if len(stuck) == len(pending):
            break                      # nothing moved; the rest cannot be done
        pending = stuck

    for field in pending:
        missing = [n for n in (field.left, field.right) if n not in numbers]
        filled[field.name] = ""
        notes.append(f"{field.name}: nothing recorded for "
                     + " or ".join(missing or [field.left, field.right]))
    return filled, notes


def _one_sum(field: Field, left: values.Value, right: values.Value,
             numbers: Dict[str, values.Value], filled: Dict[str, str],
             notes: List[str]) -> None:
    """Work out one computed field, or record why it could not be."""
    kind = result_kind(left.kind, field.op, right.kind)
    try:
        number = _apply(left.number, field.op, right.number)
    except ZeroDivisionError:
        filled[field.name] = ""
        notes.append(f"{field.name}: {field.right} is zero")
        return
    # A duration between two clock times that carry no zone is right only if
    # the clock did not move between them. EXIF very often records no offset,
    # so this is the ordinary case rather than the exotic one — and the warning
    # used to live in the routine's prompt, which is no longer where the
    # arithmetic happens.
    if (left.kind, field.op, right.kind) == (values.TIMESTAMP, "-", values.TIMESTAMP) \
            and not (left.unit and right.unit):
        notes.append(f"{field.name}: worked out from clock times with no time "
                     "zone recorded — out by whole hours if the clock moved "
                     "between them")
    number *= _SECONDS_TO_HOURS.get((left.kind, field.op, right.kind), 1.0)
    if number != number:
        # One of the inputs had no number in it — a date with no time of day,
        # most often. Not a failure worth shouting about, but not a figure
        # either.
        filled[field.name] = ""
        notes.append(f"{field.name}: {field.left} or {field.right} has no "
                     "value to work with")
        return
    text = values.render(kind, number, left.unit if kind == values.MONEY else "$")
    filled[field.name] = text
    numbers[field.name] = values.Value(kind, text, number, "", text)


def _apply(left: float, op: str, right: float) -> float:
    if op == "-":
        return left - right
    if op == "+":
        return left + right
    if op == "*":
        return left * right
    if right == 0:
        raise ZeroDivisionError
    return left / right


def mismatches(fields: List[Field], row: Dict[str, str]) -> Dict[str, str]:
    """Values that are not the kind their column was declared to hold.

    Reported, never corrected. A model that answered "unknown" where a price
    was wanted has told you something; overwriting it with a blank, or coercing
    it into a number, throws that away and leaves a log that looks complete.
    """
    out: Dict[str, str] = {}
    for field in fields:
        if not field.kind or field.computed:
            continue
        text = str(row.get(field.name, "")).strip()
        if not text:
            continue
        if not values.parse_as(text, field.kind):
            out[field.name] = (f"declared {field.kind}, but "
                               f"{text!r} does not read as one")
    return out
