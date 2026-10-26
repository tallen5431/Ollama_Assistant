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
from typing import Callable, List, Optional, Sequence

logger = logging.getLogger("ollama-chat")

GAP = "\n[…]\n"
_SECTION_CHARS = 900
_WORD = re.compile(r"[a-z0-9]+")
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


def _keyword_scores(question: str, parts: Sequence[str]) -> List[float]:
    """BM25-ish: a rare question word in a section counts for more."""
    wanted = set(terms(question))
    if not wanted:
        return [0.0] * len(parts)
    bags = [terms(p) for p in parts]
    n = len(parts)
    df = {w: sum(1 for bag in bags if w in bag) for w in wanted}
    out = []
    for bag in bags:
        if not bag:
            out.append(0.0)
            continue
        score = 0.0
        for word in wanted:
            tf = bag.count(word)
            if tf:
                idf = math.log(1 + (n - df[word] + 0.5) / (df[word] + 0.5))
                score += idf * (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * len(bag) / 120))
        out.append(score)
    return out


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
    # Finer pieces for a tighter budget, so the best one always fits.
    parts = sections(text, size=min(_SECTION_CHARS, max(200, budget // 5)))
    if len(parts) < 2 or not terms(question):
        return _front(text, budget)
    scores = (_meaning_scores(question, parts, embed) if embed else None) \
        or _keyword_scores(question, parts)
    if not any(scores):
        return _front(text, budget)
    # The opening is kept, but never at the cost of the answer: a page whose
    # first section is most of the budget keeps only the start of it.
    lead = budget // 3
    if len(parts[0]) > lead:
        parts[0] = _front(parts[0], lead)
    keep = {0}
    used = len(parts[0])
    ranked = sorted(range(1, len(parts)), key=lambda i: -scores[i])
    for index in ranked:
        if scores[index] <= 0:
            break
        cost = len(parts[index]) + len(GAP)
        if used + cost > budget:
            continue
        keep.add(index)
        used += cost
    out: List[str] = []
    last = -1
    for index in sorted(keep):
        if out and index != last + 1:
            out.append(GAP.strip())
        out.append(parts[index])
        last = index
    if last != len(parts) - 1:
        out.append(GAP.strip())
    result = "\n\n".join(out)
    return result if len(result) <= budget else _front(result, budget)


def _front(text: str, budget: int) -> str:
    return text[:budget].rsplit(" ", 1)[0] + " …[truncated]"
