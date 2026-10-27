#!/usr/bin/env python3
"""Research recipes: a routine that also says how to search.

A routine already says what to ask. A recipe adds how the web part of the turn
should go, for the kinds of research that want something different from the
defaults:

* **sites** — look only on these sites ("docs.python.org, github.com"). The
  queries carry ``site:`` and anything else that comes back is dropped.
* **fresh** — only results from the past day, week, month or year, for news
  and prices. Passed to the search engine as its own time filter.
* **rounds** — how many extra searches are allowed when the pages fall short,
  in place of WEB_SEARCH_ROUNDS.
* **pages** — how many pages to read, in place of WEB_MAX_DOCS.
* **format** — how the answer should be laid out: a table, bullet points, a
  short answer, steps, or a comparison.

Everything here is validated, because it is stored text that ends up in a
search query and a prompt. An empty recipe changes nothing, so every routine
written before this behaves exactly as it did.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

FRESH = ("day", "week", "month", "year")
MAX_SITES = 5
MAX_ROUNDS = 3
MAX_PAGES = 6

# What the answering model is told, per format. Said as an instruction about
# layout only: the rules about sticking to the pages stay where they are.
FORMATS: Dict[str, Tuple[str, str]] = {
    "table": ("Table",
              "Lay the answer out as a Markdown table, one row per item and one "
              "column per attribute the question asks about, with a Source column "
              "giving the page each row came from. Leave a cell empty rather than "
              "guessing. After the table, add at most two sentences."),
    "bullets": ("Bullet points",
                "Answer as short bullet points, most important first, each ending "
                "with the source it came from in brackets."),
    "brief": ("Short answer",
              "Answer in at most three sentences. Lead with the direct answer, "
              "then the one source it rests on."),
    "steps": ("Step by step",
              "Answer as numbered steps someone can follow in order, quoting exact "
              "commands, settings or values from the pages."),
    "compare": ("Compare sources",
                "Say what each source says on the question, one short paragraph per "
                "source, then say plainly where they agree and where they disagree. "
                "Do not pick a winner unless one source is clearly more authoritative "
                "or more recent, and say which reason applied."),
}

_HOST = re.compile(r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$")


def site_of(text: str) -> str:
    """A bare hostname from whatever was typed: a URL, "www.x.com/path", "x.com"."""
    text = str(text or "").strip().lower()
    if not text:
        return ""
    if "://" not in text:
        text = "https://" + text
    try:
        host = urlparse(text).hostname or ""
    except ValueError:
        return ""
    host = host[4:] if host.startswith("www.") else host
    return host if _HOST.match(host) else ""


def clean(raw: Any) -> Dict[str, Any]:
    """A recipe with only valid settings in it; ``{}`` when it sets nothing."""
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Any] = {}
    sites = raw.get("sites")
    if isinstance(sites, str):
        sites = re.split(r"[\s,]+", sites)
    if isinstance(sites, list):
        kept: List[str] = []
        for site in sites:
            host = site_of(site)
            if host and host not in kept:
                kept.append(host)
        if kept:
            out["sites"] = kept[:MAX_SITES]
    fresh = str(raw.get("fresh") or "").strip().lower()
    if fresh in FRESH:
        out["fresh"] = fresh
    for key, top, low in (("rounds", MAX_ROUNDS, 0), ("pages", MAX_PAGES, 1)):
        value = raw.get(key)
        if value is None or value == "":
            continue
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        out[key] = max(low, min(top, number))
    fmt = str(raw.get("format") or "").strip().lower()
    if fmt in FORMATS:
        out["format"] = fmt
    return out


def query_for(query: str, recipe: Dict[str, Any], site: Optional[str] = None) -> str:
    """The query as sent to the engine: with ``site:`` where the recipe has sites."""
    sites = [site] if site else recipe.get("sites") or []
    if not sites:
        return query
    if len(sites) == 1:
        return f"{query} site:{sites[0]}"
    return f"{query} (" + " OR ".join(f"site:{s}" for s in sites) + ")"


def allowed(url: str, recipe: Dict[str, Any]) -> bool:
    """Whether a result is on one of the recipe's sites (or it names none)."""
    sites = recipe.get("sites") or []
    if not sites:
        return True
    host = site_of(url)
    return any(host == s or host.endswith("." + s) for s in sites)


def planner_note(recipe: Dict[str, Any]) -> str:
    """Added to the search planner's instructions, so its queries fit the recipe."""
    notes = []
    if recipe.get("sites"):
        notes.append("The searches are limited to " + ", ".join(recipe["sites"])
                     + " already, so do not put site names or site: in the queries.")
    if recipe.get("fresh"):
        notes.append(f"Only results from the past {recipe['fresh']} will be returned, "
                     "so write queries for current information.")
    return ("\n\n" + " ".join(notes)) if notes else ""


def answer_note(recipe: Dict[str, Any]) -> str:
    """The layout instruction for the answer, or ""."""
    fmt = recipe.get("format")
    return f"Answer format for this request: {FORMATS[fmt][1]}" if fmt in FORMATS else ""


def describe(recipe: Dict[str, Any]) -> str:
    """One line for the panel under the reply."""
    parts = []
    if recipe.get("sites"):
        parts.append("only " + ", ".join(recipe["sites"]))
    if recipe.get("fresh"):
        parts.append("past " + recipe["fresh"])
    if "rounds" in recipe:
        parts.append(f"{recipe['rounds']} extra round(s)")
    if "pages" in recipe:
        parts.append(f"{recipe['pages']} page(s)")
    if recipe.get("format"):
        parts.append(FORMATS[recipe["format"]][0].lower())
    return " · ".join(parts)


# Shipped examples, installed from the Routines pane on request. Each is
# (name, body, recipe). All search the web, none takes a photo.
STARTERS: Tuple[Tuple[str, str, Dict[str, Any]], ...] = (
    (
        "🛒 Compare prices",
        "Find current prices for the following from several different shops. For "
        "each, give the shop, the exact product or variant, the price with "
        "currency, whether it is in stock, and the page it is on. Say if a price "
        "excludes tax or shipping. If two shops list different variants, say so "
        "rather than comparing them as equals.\n\nProduct: ",
        {"fresh": "month", "rounds": 2, "pages": 5, "format": "table"},
    ),
    (
        "📚 Check the docs",
        "Answer from the official documentation only, quoting the exact setting, "
        "command or function name it uses and the version it applies to. If the "
        "documentation does not say, say that instead of answering from memory.\n\n"
        "Question: ",
        {"rounds": 2, "pages": 4, "format": "steps"},
    ),
    (
        "📰 Latest news",
        "What is the latest on the following? Give what happened and when, from at "
        "least two different outlets, with dates. Separate confirmed facts from "
        "claims and reports, and say where outlets disagree.\n\nTopic: ",
        {"fresh": "week", "rounds": 1, "pages": 5, "format": "compare"},
    ),
    (
        "📄 Explain this page",
        "Read the page at the link below and explain it: what it is, the main "
        "points, and anything a reader would need to act on — dates, prices, "
        "requirements. Keep the page's own figures exactly as written.\n\nLink: ",
        {"rounds": 0, "format": "bullets"},
    ),
)
