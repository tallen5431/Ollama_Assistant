#!/usr/bin/env python3
"""Check the web-access plumbing against the real internet.

The test suite covers this against stand-in servers; this exercises the two
things a stand-in cannot: whether DuckDuckGo's HTML still parses, and whether
link extraction finds anything useful on a real page. Nothing here touches
Ollama, so it works whether or not the desktop is awake.

    .venv/bin/python tools/check_web.py
    .venv/bin/python tools/check_web.py https://en.wikipedia.org/wiki/Ada_Lovelace
    .venv/bin/python tools/check_web.py https://en.wikipedia.org/wiki/Ada_Lovelace "analytical engine"

A question can be given as the second argument: the link list is ranked against
it, and the ranking is most of what makes the list worth showing a model, so
checking it with a question the page can actually answer says more than the
default does.
"""

from __future__ import annotations

import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("WEB_ENABLED", "1")

import web  # noqa: E402

DEFAULT_PAGE = "https://en.wikipedia.org/wiki/Ada_Lovelace"
DEFAULT_QUESTION = "what was the analytical engine?"
OK, BAD, WARN = "  ✅", "  ❌", "  ⚠️ "


def check_search() -> bool:
    """Does the configured search backend still return usable results?"""
    from config import get_search_url
    backend = get_search_url() or "DuckDuckGo (no SEARXNG_URL set)"
    print(f"\n1. Search backend — {backend}")
    started = time.monotonic()
    try:
        results = web.search("ollama release notes", limit=3)
    except web.WebError as exc:
        print(f"{BAD} search failed: {exc}")
        print("     If this is DuckDuckGo, it may be rate-limiting you. Self-hosting")
        print("     SearXNG and setting SEARXNG_URL is the sturdier option.")
        return False

    if not results:
        print(f"{BAD} the backend returned no results at all — its HTML has probably changed")
        return False

    print(f"{OK} {len(results)} results in {time.monotonic() - started:.1f}s")
    with_snippets = [r for r in results if r.get("snippet")]
    for r in results:
        print(f"     · {r['title'][:64]}")
        print(f"       {r['url'][:78]}")
        print(f"       snippet: {(r.get('snippet') or '(none)')[:66]}")
    if not with_snippets:
        print(f"{WARN} no snippets parsed. Not fatal — a page that cannot be fetched")
        print("     just contributes nothing, as it did before — but the class names")
        print("     have probably changed and the fallback is doing nothing.")
    return True


def check_fetch(url: str) -> dict:
    """Can a real page be fetched, decoded, and stripped to readable text?"""
    print(f"\n2. Fetching — {url}")
    started = time.monotonic()
    try:
        doc = web.fetch(url)
    except web.WebError as exc:
        print(f"{BAD} {exc}")
        return {}
    print(f"{OK} {len(doc['text'])} characters in {time.monotonic() - started:.1f}s")
    print(f"     title: {doc['title'][:70]}")
    body = " ".join(doc["text"].split())
    print(f"     text : {body[:110]}…")
    if "Ã" in body or "â€" in body:
        print(f"{BAD} that looks like mojibake — the charset was misread")
    return doc


def check_links(doc: dict, question: str) -> None:
    """Did link extraction find topic links, and does ranking float them up?"""
    print(f"\n3. Links found in the readable body — ranked against {question!r}")
    links = doc.get("links") or []
    if not links:
        print(f"{WARN} none. Following will be skipped and the model gets no link map.")
        print("     Fine for a page that genuinely has no links; suspicious on a wiki.")
        return

    same = [l for l in links if web.same_site(doc["url"], l["url"])]
    followable = "same-site only" if not web.followable(
        doc, {"url": "https://elsewhere.example/"}) else "any site (WEB_FOLLOW_SCOPE=any)"
    print(f"{OK} {len(links)} links, {len(same)} on the same site; following is {followable}")

    ranked = web.rank_links(links, question, here=doc["url"])
    print("     the order the model is shown them in, best first:")
    for i, link in enumerate(ranked[:8], 1):
        away = "" if web.same_site(doc["url"], link["url"]) else " (external)"
        print(f"     {i}. {(link['text'][:44] + away)[:48]:50} {link['url'][:52]}")

    # Said carefully. An unchanged order does not mean the ranking did nothing
    # useful — it also happens when the page already listed its links best
    # first — so this reports what it saw rather than diagnosing why.
    if [l["url"] for l in ranked] == [l["url"] for l in links]:
        print(f"{WARN} the order is unchanged from the page's own. Either nothing")
        print("     matched the question, or the page already led with the best")
        print("     link. Re-run with a question using words from a section title")
        print("     to tell the two apart.")
    else:
        print(f"{OK} ranking moved the question's links up the list")

    junk = [l for l in ranked[:8] if l["text"].lower() in
            {"edit", "read", "view history", "talk", "main page", "contents"}]
    if junk:
        print(f"{WARN} navigation leaked through: {[l['text'] for l in junk][:5]}")
    else:
        print(f"{OK} no obvious navigation or chrome near the top of the list")


def check_context(doc: dict, question: str) -> None:
    """Does the assembled context stay inside its budget, and does it number
    the links the same way a fetch request is resolved?"""
    from config import get_num_ctx
    print("\n4. Context assembly")
    budget = web.context_budget(get_num_ctx())
    ids: dict = {}
    context = web.build_context([doc], char_budget=budget, question=question,
                                link_ids=ids, may_fetch=True)
    ratio = len(context) / budget if budget else 0
    verdict = OK if ratio <= 1.15 else BAD
    print(f"{verdict} {len(context)} characters against a {budget} budget ({ratio:.2f}x)")
    print(f"     OLLAMA_NUM_CTX={get_num_ctx()}, leaving the rest for the conversation")
    if "Other pages linked from this one" in context:
        print(f"{OK} the link map is in the context the model will see ({len(ids)} numbered)")
    elif doc.get("links"):
        print(f"{WARN} the budget left no room for the link map at this num_ctx")

    # The one failure here that is worse than no links at all: a request for
    # [1.3] resolving to a different page than the one listed as [1.3].
    listed = dict(re.findall(r"^\[(\d+\.\d+)\][^—]*— (\S+)$", context, re.M))
    if listed == {k: v["url"] for k, v in ids.items()}:
        print(f"{OK} every number in the list resolves to the page shown beside it")
    else:
        print(f"{BAD} the numbering shown and the numbering resolved disagree —")
        print("     a request to read a link would fetch the wrong page")

    if context.count("----- END WEB RESULTS -----") != 1:
        print(f"{BAD} the page forged an end-of-results marker — that should be impossible")


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PAGE
    question = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_QUESTION
    print("Checking web access against the real internet.")
    print("Nothing here talks to Ollama, so a sleeping desktop does not matter.")

    searched = check_search()
    doc = check_fetch(url)
    if doc:
        check_links(doc, question)
        check_context(doc, question)

    print("\n" + "-" * 62)
    if searched and doc:
        print("Both halves work. Try the UI checks next — see the README.")
        return 0
    print("Something above failed; the detail is next to the ❌.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
