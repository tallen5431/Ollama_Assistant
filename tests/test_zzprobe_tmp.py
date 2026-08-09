"""Temporary probe — deleted after the investigation."""

from __future__ import annotations

import json

import pytest

import web
from test_web_chat import rig, two_pages, lines  # noqa: F401


def _ask(rig, question="what is widget 5?"):
    resp = rig["client"].post("/api/chat", json={
        "messages": [{"role": "user", "content": question}], "web": True})
    return lines(resp)


def _system_turn(rig):
    chat_call = [r for r in rig["ollama"].requests if r.get("stream")][-1]
    return chat_call["messages"][0]["content"]


def test_probe_how_many_pages_the_rig_distills(rig, monkeypatch):
    monkeypatch.setenv("WEB_DISTILLER_MODEL", "small:1b")
    calls = []

    def flaky(question, doc, model, **kw):
        calls.append(doc["url"])
        raise RuntimeError("that one fell over")

    monkeypatch.setattr(web, "distil", flaky)
    out = _ask(rig)
    print("\nPROBE rig distiller calls:", calls)
    steps = [o["debug"] for o in out if "debug" in o]
    print("PROBE Distilled step:",
          [s for s in steps if s.get("step") == "Distilled"])
    assert len(calls) == 1, f"expected one page, saw {calls}"


def test_probe_two_pages_one_raises_one_succeeds(two_pages, monkeypatch):
    rig = two_pages
    monkeypatch.setenv("WEB_DISTILLER_MODEL", "small:1b")
    calls = []

    def mixed(question, doc, model, **kw):
        calls.append(doc["url"])
        if doc["url"].endswith("/one"):
            raise RuntimeError("that one fell over")
        return "SECOND PAGE ONLY."

    monkeypatch.setattr(web, "distil", mixed)
    out = _ask(rig)
    system = _system_turn(rig)
    print("\nPROBE two_pages calls:", calls)
    print("PROBE contains SECOND PAGE ONLY:", "SECOND PAGE ONLY." in system)
    print("PROBE contains page one text:", "page one is about apples" in system)
    print("PROBE count SECOND:", system.count("SECOND PAGE ONLY."))
    assert len(calls) == 2
    assert "SECOND PAGE ONLY." in system
    assert "page one is about apples" in system
    assert system.count("SECOND PAGE ONLY.") == 1
    assert not [o for o in out if "error" in o]
