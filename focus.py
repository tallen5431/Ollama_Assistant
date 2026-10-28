#!/usr/bin/env python3
"""Cutting a long page down to the parts that bear on the question.

A page used to be cut by length: its first WEB_MAX_CHARS characters were kept
and the rest dropped. On a long article that keeps the introduction and the
navigation and loses the answer, which is usually further down — the release
note's third section, the manual's page 40.

Here a page is split into sections, each section is scored against the
question, and the best ones are kept up to the budget, in the order they
appear, with a marker where something was left out. The opening section is
always kept, because it says what the page is.

Scoring is by keywords unless WEB_EMBED_MODEL names an Ollama embedding model,
in which case it is by meaning, so a section that answers in different words
than the question used is still found. Keywords need nothing installed and an
embedding model that fails falls back to them.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Callable, Dict, List, Optional, Sequence

logger = logging.getLogger("ollama-chat")

GAP = "\n[…]\n"
_SECTION_CHARS = 900
_WORD = re.compile(r"[a-z0-9]+")
_PAGE = re.compile(r"\[page \d+\]")
# What a kept section costs beyond its own text: the gap marker before it and
# the "[page 12] (continued)" label it may be given.
_EXTRA = len(GAP) + len("[page 9999] (continued)\n")
_URL = re.compile(r"https?://\S+|\bwww\.\S+", re.I)
# How much a question word counts once a kept section already has it. Without
# this, the first part of a two-part question fills the budget with sections
# that all answer the same part — "28.4 BLEU" six ways and no "8 heads".
_ALREADY = 0.35
_STOP = frozenset("""
a an and are as at be but by can do does for from has have how i if in into is
it its me my of on or our so that the their them then there these this those to
was we what when where which who why will with would you your about please tell
find show give explain summarise summarize summary page article this that
""".split())

Embed = Callable[[List[str]], Optional[List[List[float]]]]


def sections(text: str, size: int = _SECTION_CHARS) -> List[str]:
    """The page in pieces of roughly ``size`` characters, split at paragraphs.

    A paragraph longer than ``size`` is split at sentence ends, so one huge
    block of PDF text does not become a single all-or-nothing section.
    """
    pieces: List[str] = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        while len(block) > size * 1.5:
            cut = max(block.rfind(". ", 0, size), block.rfind("\n", 0, size))
            cut = cut + 1 if cut > size // 3 else size
            pieces.append(block[:cut].strip())
            block = block[cut:].strip()
        if block:
            pieces.append(block)
    out: List[str] = []
    for piece in pieces:
        if out and len(out[-1]) + len(piece) < size:
            out[-1] = out[-1] + "\n\n" + piece
        else:
            out.append(piece)
    return out


def terms(text: str) -> List[str]:
    words = _WORD.findall(str(text or "").lower())
    return [w for w in words if w not in _STOP and (len(w) > 2 or w.isdigit())]


def _keyword_weights(question: str, parts: Sequence[str]) -> List[Dict[str, float]]:
    """Per section, how much each question word counts there (BM25-ish).

    A rare question word in a section counts for more than a common one.
    """
    wanted = set(terms(question))
    if not wanted:
        return [{} for _ in parts]
    bags = [terms(p) for p in parts]
    n = len(parts)
    df = {w: sum(1 for bag in bags if w in bag) for w in wanted}
    out: List[Dict[str, float]] = []
    for bag in bags:
        weights: Dict[str, float] = {}
        for word in wanted:
            tf = bag.count(word) if bag else 0
            if tf:
                idf = math.log(1 + (n - df[word] + 0.5) / (df[word] + 0.5))
                weights[word] = idf * (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * len(bag) / 120))
        out.append(weights)
    return out


def _keyword_scores(question: str, parts: Sequence[str]) -> List[float]:
    return [sum(w.values()) for w in _keyword_weights(question, parts)]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _meaning_scores(question: str, parts: Sequence[str],
                    embed: Embed) -> Optional[List[float]]:
    try:
        vectors = embed([question] + list(parts))
    except Exception as exc:  # noqa: BLE001 - fall back to keywords
        logger.warning("Embedding for page focus failed: %s", exc)
        return None
    if not vectors or len(vectors) != len(parts) + 1:
        return None
    return [_cosine(vectors[0], v) for v in vectors[1:]]


def focus(text: str, question: str, budget: int,
          embed: Optional[Embed] = None) -> str:
    """``text`` cut to ``budget`` characters, keeping what answers ``question``.

    A page within budget comes back untouched. A question with no words worth
    matching — "summarise this" — keeps the front of the page, which is the
    old behaviour and the right one for that request.
    """
    text = str(text or "")
    if len(text) <= budget:
        return text
    # A link in the question names the page, not what is wanted from it. Its
    # pieces — "arxiv", "org", "pdf", "1706" — matched every entry in a
    # paper's reference list, and those filled the budget in place of the
    # section that said how many attention heads the model used.
    question = _URL.sub(" ", str(question or ""))
    # Finer pieces for a tighter budget, so the best one always fits.
    parts = sections(text, size=min(_SECTION_CHARS, max(200, budget // 5)))
    if len(parts) < 2:
        return _front(text, budget)
    if not terms(question):
        # Nothing to match — a bare link, "summarise this". The front alone was
        # a manual's cover, legal notice and safety warnings, and asked about
        # the air fryer the model invented its capacity. A skim across the
        # whole document at least shows what is in it.
        return _skim(parts, budget)
    meaning = _meaning_scores(question, parts, embed) if embed else None
    weights = _keyword_weights(question, parts) if meaning is None else []
    if meaning is None and not any(weights):
        return _front(text, budget)
    # The opening is kept, because it says what the page is — but only its
    # start. A paper's first section is its author list, and keeping a third
    # of the budget for e-mail addresses is a third not spent on the answer.
    lead = max(120, budget // 8)
    if len(parts[0]) > lead:
        parts[0] = _front(parts[0], lead)
    keep = {0}
    used = len(parts[0])
    if meaning is not None:
        ranked = [i for i in sorted(range(1, len(parts)), key=lambda i: -meaning[i])
                  if meaning[i] > 0]
    else:
        ranked = _covering(weights, parts, budget - used)
    for index in ranked:
        cost = len(parts[index]) + _EXTRA
        if used + cost > budget:
            continue
        keep.add(index)
        used += cost
    return _assemble(parts, sorted(keep), budget)


def _assemble(parts: List[str], keep: List[int], budget: int) -> str:
    """The kept sections in page order, with gaps marked and pages named."""
    # Which page each section is on, for a document that marks its pages (a
    # PDF does: "[page 8]"). Only the section a page starts with carries the
    # marker, so a section kept from the middle of a page arrived with no page
    # at all — and a model citing it guessed: 28.4 BLEU "on page 6", from a
    # table on page 8.
    pages: List[str] = []
    current = ""
    for part in parts:
        found = _PAGE.findall(part)
        pages.append(current if not part.startswith("[page ") else (found[0] if found else current))
        if found:
            current = found[-1]
    out: List[str] = []
    last = -1
    for index in sorted(keep):
        if out and index != last + 1:
            out.append(GAP.strip())
        text = parts[index]
        if pages[index] and not text.startswith("[page "):
            text = f"{pages[index]} (continued)\n{text}"
        out.append(text)
        last = index
    if last != len(parts) - 1:
        out.append(GAP.strip())
    result = "\n\n".join(out)
    return result if len(result) <= budget else _front(result, budget)


def _skim(parts: List[str], budget: int) -> str:
    """The start of the document, then evenly spaced sections from the rest."""
    lead = max(120, budget // 8)
    first = _front(parts[0], lead) if len(parts[0]) > lead else parts[0]
    room = budget - len(first)
    rest = list(range(1, len(parts)))
    # As many as fit at their own length, spread evenly from first to last.
    average = max(1, sum(len(parts[i]) + _EXTRA for i in rest) // max(1, len(rest)))
    wanted = max(1, min(len(rest), room // average))
    step = len(rest) / wanted
    chosen = sorted({rest[min(len(rest) - 1, int(k * step))] for k in range(wanted)})
    keep = [0]
    for index in chosen:
        cost = len(parts[index]) + _EXTRA
        if cost <= room:
            keep.append(index)
            room -= cost
    parts = [first] + parts[1:]
    return _assemble(parts, keep, budget)


def _covering(weights: List[Dict[str, float]], parts: Sequence[str],
              room: int) -> List[int]:
    """Sections in the order to keep them, so every part of the question is covered.

    Greedy: each pick is the section adding most, counting a question word
    for less once a kept section already has it. A section on the second half
    of the question then beats a fourth section on the first half.
    """
    seen: Dict[str, int] = {}
    left = set(range(1, len(parts)))
    order: List[int] = []
    while left and room > 0:
        def gain(i: int) -> float:
            return sum(v * (_ALREADY ** seen.get(w, 0)) for w, v in weights[i].items())
        fits = [i for i in left if len(parts[i]) + _EXTRA <= room]
        if not fits:
            break
        best = max(fits, key=lambda i: (gain(i), -i))
        if gain(best) <= 0:
            break
        order.append(best)
        left.discard(best)
        room -= len(parts[best]) + _EXTRA
        for word in weights[best]:
            seen[word] = seen.get(word, 0) + 1
    return order


def _front(text: str, budget: int) -> str:
    return text[:budget].rsplit(" ", 1)[0] + " …[truncated]"
