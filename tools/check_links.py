#!/usr/bin/env python3
"""Check the link-following path against real models and a real page.

`check_web.py` deliberately never talks to Ollama, so it works with a sleeping
desktop — but that leaves the half of this feature that only a real model can
answer completely unchecked. The test suite drives stand-in servers and scripted
token streams, which proves the plumbing and proves nothing at all about whether
a 3b model actually replies "1\\n3" when shown a two-column list of links, or
replies with a bare "FETCH: [2.3]" when it is told it may.

Those are the two questions that decide whether this feature does anything on
your hardware, and neither can be answered offline. So this one does wake the
desktop, and says which model gave which answer.

    .venv/bin/python tools/check_links.py
    .venv/bin/python tools/check_links.py URL "a question the page does not answer"
    .venv/bin/python tools/check_links.py URL "question" --model qwen3:8b

Pick the question deliberately. The interesting case is one the page *mentions*
but does not answer, where a link plainly would — that is what the feature is
for. If the page answers it outright then a direct answer is the correct result
and not a failure, and this tool says so rather than marking it wrong.
"""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("WEB_ENABLED", "1")

import web  # noqa: E402

DEFAULT_PAGE = "https://en.wikipedia.org/wiki/Ada_Lovelace"
DEFAULT_QUESTION = "what was the analytical engine and how did it work?"
OK, BAD, WARN, INFO = "  ✅", "  ❌", "  ⚠️ ", "  ·"


def pick_models(override: str) -> tuple:
    """(answering, picker) model names, or ("", "") if Ollama can't be reached."""
    from ollama_client import list_models, model_name
    from config import get_default_model, get_planner_model

    print("\n0. Models")
    try:
        installed = [model_name(m) for m in list_models(fresh=True)]
    except Exception as exc:  # noqa: BLE001 - the whole point is to report this
        print(f"{BAD} could not reach Ollama: {exc}")
        print("     OLLAMA_HOST is "
              f"{os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434')}. Everything")
        print("     below needs a real model, so nothing else can run.")
        return "", ""

    if not installed:
        print(f"{BAD} Ollama answered but has no models installed.")
        return "", ""

    answering = override or get_default_model()
    if answering not in installed:
        print(f"{WARN} {answering} is not installed; using {installed[0]} instead")
        answering = installed[0]
    picker = get_planner_model() or answering
    if picker not in installed:
        print(f"{WARN} WEB_PLANNER_MODEL={picker} is not installed; "
              f"the answering model will pick")
        picker = answering

    print(f"{OK} answering with {answering}, picking links with {picker}")
    if picker == answering:
        print(f"{INFO} both jobs on one model. Setting WEB_PLANNER_MODEL to")
        print("     something small is usually the better arrangement.")
    return answering, picker


def check_ranking(doc: dict, question: str) -> None:
    """Did ranking move the links that bear on the question to the top?"""
    print(f"\n1. Ranking — {len(doc.get('links') or [])} links found")
    links = doc.get("links") or []
    if not links:
        print(f"{WARN} the page has no links in its readable body. Nothing below")
        print("     can be checked; try a wiki or documentation page.")
        return

    where = {l["url"]: i for i, l in enumerate(links, 1)}
    ranked = web.rank_links(links, question, here=doc["url"])
    print("     what the model will see, best first (· = its place on the page):")
    for i, link in enumerate(ranked[:8], 1):
        away = "" if web.same_site(doc["url"], link["url"]) else " (external)"
        print(f"     {i}. {(link['text'][:40] + away)[:44]:46} "
              f"· was #{where[link['url']]}")

    shown = min(8, len(ranked))
    moved = sum(1 for i, l in enumerate(ranked[:8], 1) if where[l["url"]] > i)
    if moved:
        was = "was" if moved == 1 else "were"
        print(f"{OK} {moved} of the top {shown} {was} promoted from further "
              "down the page")
    else:
        print(f"{WARN} nothing was promoted. Either the page already led with its")
        print("     best links, or no link matched the question — try a question")
        print("     using words from a section title to tell those apart.")


def check_picker(doc: dict, question: str, picker: str, answering: str) -> None:
    """Does the real picker model return link numbers in the documented shape?

    The listing gained a second column and the picker is a small model, so this
    is the most likely place for the feature to quietly do nothing: choose_links
    swallows a malformed reply and returns [], which is indistinguishable from
    a considered "none of these".
    """
    print(f"\n2. The picker — {picker}")
    candidates = [l for l in (doc.get("links") or []) if web.followable(doc, l)]
    if not candidates:
        print(f"{WARN} no followable links (WEB_FOLLOW_SCOPE="
              f"{os.getenv('WEB_FOLLOW_SCOPE', 'site')}), so nothing to pick from")
        return

    # The raw reply, captured from the real call rather than reconstructed:
    # choose_links folds "chose nothing" and "replied with nonsense" into the
    # same empty list, and telling those apart is the whole point here.
    import ollama_client
    raw = {}
    real_chat = ollama_client.chat

    def capture(model, messages, **kw):
        reply = real_chat(model, messages, **kw)
        raw["reply"] = reply
        return reply

    ollama_client.chat = capture
    try:
        chosen = web.choose_links(question, candidates, picker,
                                  answering_model=answering)
    finally:
        ollama_client.chat = real_chat

    reply = web.strip_thinking(raw.get("reply") or "").strip()
    print(f"     it replied: {reply[:120]!r}")
    if chosen:
        print(f"{OK} parsed {len(chosen)} link(s) — the format was understood:")
        for link in chosen:
            print(f"     · {link['text'][:44]:46} {link['url'][:52]}")
    elif web._NONE_RE.search(reply):
        print(f"{OK} it said none of these are worth opening, in the documented")
        print("     shape. Correct if the page already covers the question — try")
        print("     one it does not to see a link actually chosen.")
    elif not reply:
        print(f"{BAD} it returned nothing at all. If this is a reasoning model,")
        print("     use a plain one for WEB_PLANNER_MODEL — the whole budget goes")
        print("     into thinking and the answer is truncated away.")
    else:
        print(f"{BAD} the reply parsed to no links and is not a 'none' either, so")
        print("     following will silently never happen on this model. Expected")
        print("     bare numbers, one per line. Try a different WEB_PLANNER_MODEL.")


def check_fetch_request(doc: dict, question: str, answering: str) -> None:
    """Does the answering model ask for a link in the shape the app parses?"""
    from config import get_num_ctx
    from ollama_client import chat

    print(f"\n3. Asking to read a link — {answering}")
    ids: dict = {}
    context = web.build_context([doc], char_budget=web.context_budget(get_num_ctx()),
                                question=question, link_ids=ids, may_fetch=True)
    if not ids:
        print(f"{WARN} no numbered links survived the context budget at num_ctx="
              f"{get_num_ctx()}, so there is nothing to ask for.")
        return

    # The failure that would silently fetch the wrong page. Cheap, so always.
    listed = dict(re.findall(r"^\[(\d+\.\d+)\][^—]*— (\S+)$", context, re.M))
    if listed == {k: v["url"] for k, v in ids.items()}:
        print(f"{OK} {len(ids)} links numbered, and every number resolves to the")
        print("     page shown beside it")
    else:
        print(f"{BAD} the numbering shown and the numbering resolved disagree —")
        print("     a request would fetch the wrong page. Stop and fix this.")
        return

    try:
        reply = chat(answering,
                     [{"role": "system", "content": context},
                      {"role": "user", "content": question}],
                     options={"num_ctx": get_num_ctx()}, think=False)
    except Exception as exc:  # noqa: BLE001
        print(f"{BAD} the model could not be asked: {exc}")
        return

    reply = web.strip_thinking(reply or "").strip()
    wanted = web.fetch_request(reply)
    print(f"     it replied: {reply[:160]!r}")

    if wanted and wanted in ids:
        print(f"{OK} a well-formed request for [{wanted}] → {ids[wanted]['url']}")
        print("     The app would fetch that and ask again with it in front of it.")
    elif wanted:
        print(f"{WARN} it asked for [{wanted}], which is not on the list it was")
        print("     shown. The app refuses and re-asks, so this costs a hop.")
    elif re.search(r"\bFETCH\b", reply, re.I):
        print(f"{BAD} it tried to ask but not in the shape the app parses — the")
        print("     request must be the entire reply, with no prose around it.")
        print("     The user would see the marker instead of an answer.")
    else:
        print(f"{OK} it answered directly rather than asking for a link.")
        print("     Correct when the page already answers the question. To see a")
        print("     request, re-run with something the page only points at.")


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--model"]
    override = ""
    if "--model" in sys.argv:
        at = sys.argv.index("--model")
        override = sys.argv[at + 1] if at + 1 < len(sys.argv) else ""
        args = [a for a in args if a != override]

    url = args[0] if args else DEFAULT_PAGE
    question = args[1] if len(args) > 1 else DEFAULT_QUESTION

    print("Checking the link path against real models and a real page.")
    print(f"Page:     {url}")
    print(f"Question: {question!r}")

    answering, picker = pick_models(override)
    if not answering:
        return 1

    print(f"\n   Fetching {url}")
    try:
        doc = web.fetch(url)
    except web.WebError as exc:
        print(f"{BAD} {exc}")
        return 1
    print(f"{OK} {len(doc['text'])} characters, {len(doc.get('links') or [])} links")

    check_ranking(doc, question)
    check_picker(doc, question, picker, answering)
    check_fetch_request(doc, question, answering)

    print("\n" + "-" * 62)
    print("Read the ❌ and ⚠️  lines above; a ✅ on step 2 and step 3 means the")
    print("models on this box speak the two formats the feature depends on.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
