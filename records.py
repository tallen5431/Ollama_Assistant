#!/usr/bin/env python3
"""Turn a routine's answer into fields worth keeping.

A routine that reads two odometers produces a paragraph. What survives being
useful a year later is not the paragraph — it is 68 miles, 3 h 08 min, on the
7th of August. This asks the model that just answered to restate its own answer
as the fields the routine declared, and nothing else.

Extraction rather than parsing, because the answers are prose written by a
language model and no regex survives contact with the next one. The model is
local and the call is short: temperature 0, a hard cap on tokens, and the
result is thrown away rather than guessed at when it does not parse.

It is deliberately not perfect, and the app does not pretend otherwise: what it
produces is editable in the records table, because a log you cannot correct is
one you stop trusting.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from config import logger

# Long enough for a dozen short fields and nothing like long enough for prose.
_MAX_TOKENS = 400
# The answer is what is being restated; a very long one is trimmed from the
# front because a routine's conclusions come at the end.
_MAX_ANSWER_CHARS = 6000

_PROMPT = (
    "You restate an answer as data. You are given an answer that was just "
    "written, and a list of field names. Reply with one JSON object and "
    "nothing else: no prose, no explanation, no code fence.\n\n"
    "Use exactly the field names you were given, all of them, in that order. "
    "Every value is a string. Where the answer does not state something, use "
    "an empty string — never a guess, never \"unknown\", never a placeholder. "
    "Copy figures exactly as the answer gives them, units included. Do not "
    "calculate anything the answer did not calculate, and do not correct it.\n\n"
    "Write every value as plain text. No LaTeX, no markdown, no formatting: "
    "a value goes into a spreadsheet column, so write \"3.13 hours\" rather "
    "than \"$\\approx 3.13$ hours\"."
)


# The same job chat_ui.js does for a reply on screen, done here for a value on
# its way into the database — because a record is read back as a table cell, in
# a CSV, and by whatever the CSV is loaded into, none of which render TeX.
# Asking the model not to emit it is not enough on its own: the instruction it
# is following in the same breath is "copy figures exactly as the answer gives
# them", and the answer had them in LaTeX.
_TEX_SYMBOLS = (
    (r"\\approx", "≈"), (r"\\times", "×"), (r"\\cdot", "·"),
    (r"\\div", "÷"), (r"\\pm", "±"), (r"\\mp", "∓"),
    (r"\\leq?\b", "≤"), (r"\\geq?\b", "≥"), (r"\\neq", "≠"),
    (r"\\rightarrow", "→"), (r"\\to\b", "→"), (r"\\infty", "∞"),
    (r"\\degree|\\deg\b", "°"),
)
_TEX_SPAN = re.compile(
    r"\$\$(.+?)\$\$|\$([^$\n]+?)\$|\\\((.+?)\\\)|\\\[(.+?)\\\]", re.S)


def _tex_to_text(body: str) -> str:
    """A TeX fragment as the words a person would have written."""
    out = body
    for _ in range(4):        # these nest
        out = re.sub(r"\\(?:text|mathrm|mathbf|mathit|operatorname)\s*\{([^{}]*)\}",
                     r"\1", out)
        out = re.sub(r"\\(?:d|t)?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}",
                     r"(\1) / (\2)", out)
        out = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"√(\1)", out)
    for pattern, symbol in _TEX_SYMBOLS:
        out = re.sub(pattern, symbol, out)
    out = re.sub(r"\\left|\\right", "", out)
    out = re.sub(r"\\[,;:!]", " ", out)
    out = re.sub(r"\\\\", " ", out)
    out = re.sub(r"\\([%$&#_{}])", r"\1", out)
    # Brackets the fraction rule added, dropped again where neither side needs
    # them: "(68) / (3.13)" is worse than "68 / 3.13".
    out = re.sub(r"\(([^()\s]+)\) / \(([^()\s]+)\)", r"\1 / \2", out)
    return " ".join(out.split())


def _plain(value: str) -> str:
    """Strip TeX wrapping from a field value, leaving money alone.

    A dollar pair is only treated as maths when it actually contains a TeX
    command. "$54.20" and "$5 to $10" are what a receipt routine records, and
    turning those into arithmetic would be a far worse failure than leaving a
    stray dollar sign in a rare one.
    """
    def unwrap(match: "re.Match[str]") -> str:
        dollars = match.group(1) if match.group(1) is not None else match.group(2)
        if dollars is not None:
            if "\\" not in dollars:
                return match.group(0)
            return _tex_to_text(dollars)
        # \( \) and \[ \] say what they are; nothing else uses them.
        body = match.group(3) if match.group(3) is not None else match.group(4)
        return _tex_to_text(body)

    return " ".join(_TEX_SPAN.sub(unwrap, value).split())


def _objects(text: str) -> List[Dict[str, Any]]:
    """Every top-level JSON object in a reply, in order.

    Small models fence their JSON, preface it with "Here is the JSON:", add a
    sentence afterwards, or think out loud in a first object before writing the
    real one. Finding the braces is more reliable than asking them again, and
    it costs nothing when the reply was clean to begin with.
    """
    found: List[Dict[str, Any]] = []
    if not text:
        return found
    start = text.find("{")
    while start >= 0:
        depth, in_string, escaped = 0, False, False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start:index + 1])
                    except ValueError:
                        break
                    if isinstance(parsed, dict):
                        found.append(parsed)
                    break
        start = text.find("{", start + 1)
    return found


def extract(answer: str, fields: List[str], model: str) -> Dict[str, str]:
    """Restate ``answer`` as ``fields``. Empty dict when it could not be done.

    Never raises: a failed extraction costs a record, and a record is worth
    less than the answer the user already has on screen.
    """
    from ollama_client import chat   # local import keeps this module standalone

    wanted = [f for f in (fields or []) if f]
    if not wanted or not (answer or "").strip() or not model:
        return {}

    # Strip a reasoning block if the answering model left one in: it is the
    # model talking to itself, and restating it as data is meaningless.
    body = re.sub(r"<think>.*?</think>", "", answer, flags=re.S | re.I).strip()
    if len(body) > _MAX_ANSWER_CHARS:
        body = body[-_MAX_ANSWER_CHARS:]

    prompt = (
        "Field names, in order:\n" + "\n".join(f"- {name}" for name in wanted) +
        "\n\n----- BEGIN ANSWER -----\n" + body + "\n----- END ANSWER -----"
    )
    try:
        reply = chat(
            model,
            [{"role": "system", "content": _PROMPT},
             {"role": "user", "content": prompt}],
            options={"temperature": 0, "num_predict": _MAX_TOKENS},
            # This runs straight after the answer, on the same model, and the
            # user may well send another message. Leave it loaded.
            think=False,
        )
    except Exception as exc:  # noqa: BLE001 - never let this break a turn
        logger.warning("Record extraction (%s) failed: %s", model, exc)
        return {}

    candidates = _objects(reply or "")
    if not candidates:
        logger.info("Record extraction returned no JSON object")
        return {}
    # The first object that actually holds one of the fields asked for. A model
    # that thinks out loud in JSON writes a scratch object first, and taking it
    # gave a row of empty strings that looked like a real extraction.
    wanted_lower = {name.lower() for name in wanted}
    found = next((c for c in candidates
                  if any(str(k).strip().lower() in wanted_lower for k in c)),
                 candidates[0])

    # Only the fields that were asked for, in the order they were asked for —
    # a model that invents an extra column would silently widen the table.
    out: Dict[str, str] = {}
    lowered = {str(k).strip().lower(): v for k, v in found.items()}
    for name in wanted:
        value = found.get(name, lowered.get(name.lower(), ""))
        if value is None or value is False:
            value = ""
        elif isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value)
        out[name] = _plain(" ".join(str(value).split()))
    return out


def tidy_stored(limit: int = 2000) -> int:
    """Rewrite kept values that still carry the LaTeX they were written in.

    Records made before extraction learned to strip it are already in the
    database, in the CSV export and in whatever the CSV was loaded into. Run
    once at startup: idempotent, and it only rewrites a value that actually
    changes, so a clean log is a no-op.
    """
    import store

    if not store.db_path().exists():
        return 0
    changed = 0
    for record in store.list_records(limit=limit):
        fields = record.get("fields") or {}
        fixed = {name: _plain(value) for name, value in fields.items()}
        if fixed != fields:
            store.update_record(record["id"], fixed)
            changed += 1
    return changed
