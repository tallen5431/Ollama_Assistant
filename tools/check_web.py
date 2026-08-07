#!/usr/bin/env python3
"""Check the web-access plumbing against the real internet.

The test suite covers this against stand-in servers; this exercises the two
things a stand-in cannot: whether DuckDuckGo's HTML still parses, and whether
link extraction finds anything useful on a real page. Nothing here touches
Ollama, so it works whether or not the desktop is awake.

    .venv/bin/python tools/check_web.py
    .venv/bin/python tools/check_web.py https://en.wikipedia.org/wiki/Ada_Lovelace
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("WEB_ENABLED", "1")

import web  # noqa: E402

DEFAULT_PAGE = "https://en.wikipedia.org/wiki/Ada_Lovelace"
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


def check_links(doc: dict) -> None:
    """Did link extraction find topic links rather than navigation?"""
    print("\n3. Links found in the readable body")
    links = doc.get("links") or []
    if not links:
        print(f"{WARN} none. Following will be skipped and the model gets no link map.")
        print("     Fine for a page that genuinely has no links; suspicious on a wiki.")
        return

    same = [l for l in links if web.same_site(doc["url"], l["url"])]
    print(f"{OK} {len(links)} links, {len(same)} on the same site (only these are followed)")
    for link in same[:8]:
        print(f"     · {link['text'][:48]:50} {link['url'][:52]}")

    junk = [l for l in same if l["text"].lower() in
            {"edit", "read", "view history", "talk", "main page", "contents"}]
    if junk:
        print(f"{WARN} navigation leaked through: {[l['text'] for l in junk][:5]}")
    else:
        print(f"{OK} no obvious navigation or chrome in the list")


def check_context(doc: dict) -> None:
    """Does the assembled context stay inside its budget?"""
    from config import get_num_ctx
    print("\n4. Context assembly")
    budget = web.context_budget(get_num_ctx())
    context = web.build_context([doc], char_budget=budget)
    ratio = len(context) / budget if budget else 0
    verdict = OK if ratio <= 1.15 else BAD
    print(f"{verdict} {len(context)} characters against a {budget} budget ({ratio:.2f}x)")
    print(f"     OLLAMA_NUM_CTX={get_num_ctx()}, leaving the rest for the conversation")
    if "Other pages linked from this one" in context:
        print(f"{OK} the link map is in the context the model will see")
    if context.count("----- END WEB RESULTS -----") != 1:
        print(f"{BAD} the page forged an end-of-results marker — that should be impossible")


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PAGE
    print("Checking web access against the real internet.")
    print("Nothing here talks to Ollama, so a sleeping desktop does not matter.")

    searched = check_search()
    doc = check_fetch(url)
    if doc:
        check_links(doc)
        check_context(doc)

    print("\n" + "-" * 62)
    if searched and doc:
        print("Both halves work. Try the UI checks next — see the README.")
        return 0
    print("Something above failed; the detail is next to the ❌.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
