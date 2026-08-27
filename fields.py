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

    @property
    def computed(self) -> bool:
        return bool(self.op)

    def declaration(self) -> str:
        """The text this was written as, which is also how it is stored."""
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


def names(fields: List[Field]) -> List[str]:
    """Every column, in order, computed ones included."""
    return [f.name for f in fields]


def to_ask(fields: List[Field]) -> List[Field]:
    """The fields a model has to supply, which is the ones nothing can derive."""
    return [f for f in fields if not f.computed]


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

    Computed fields are resolved in the order declared and can build on one
    another: an elapsed time computed from two timestamps is available to the
    hourly rate declared below it.
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

    for field in fields:
        if not field.computed:
            continue
        left, right = numbers.get(field.left), numbers.get(field.right)
        missing = [n for n, v in ((field.left, left), (field.right, right)) if not v]
        if missing:
            filled[field.name] = ""
            notes.append(f"{field.name}: nothing recorded for "
                         + " or ".join(missing))
            continue
        kind = result_kind(left.kind, field.op, right.kind)
        try:
            number = _apply(left.number, field.op, right.number)
        except ZeroDivisionError:
            filled[field.name] = ""
            notes.append(f"{field.name}: {field.right} is zero")
            continue
        number *= _SECONDS_TO_HOURS.get((left.kind, field.op, right.kind), 1.0)
        if number != number:
            # One of the inputs had no number in it — a date with no time of
            # day, most often. Not a failure worth shouting about, but not a
            # figure either.
            filled[field.name] = ""
            notes.append(f"{field.name}: {field.left} or {field.right} has no "
                         "value to work with")
            continue
        text = values.render(kind, number, left.unit if kind == values.MONEY else "$")
        filled[field.name] = text
        numbers[field.name] = values.Value(kind, text, number, "", text)
    return filled, notes


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
