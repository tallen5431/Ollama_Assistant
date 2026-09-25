"""What the turn actually did, on the record.

A reply that says "I cannot access image metadata" while the photo-details
toggle is on is either the feature failing or the model ignoring it, and from
the outside those look identical. So does a web answer that says it could not
check anything: the search may never have been planned, may have been planned
badly, or may have run and found nothing. This is the record that tells them
apart — collapsed, because it is only wanted when an answer looks wrong.
"""

from __future__ import annotations

import importlib
import json

import pytest

import app as app_module
import chat_ui
from conftest import page_script


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAT_DB", str(tmp_path / "chat.db"))
    yield


@pytest.fixture
def app(monkeypatch):
    for key in ("CHAT_AUTH", "CHAT_AUTH_USER", "CHAT_AUTH_PASSWORD"):
        monkeypatch.delenv(key, raising=False)
    return importlib.reload(app_module)


def steps(resp) -> list:
    """The debug entries out of an NDJSON reply."""
    out = []
    for raw in resp.get_data(as_text=True).splitlines():
        if not raw.strip():
            continue
        obj = json.loads(raw)
        if isinstance(obj, dict) and "debug" in obj:
            out.append(obj["debug"])
    return out


def named(entries, name):
    return next((e for e in entries if e["step"] == name), None)


def a_reply(model, messages, **kw):
    yield json.dumps({"message": {"content": "an answer"}})
    yield json.dumps({"done": True})


class TestItSaysWhatHappenedToThePhoto:
    """The reported case: photo details on, and the model announcing it cannot
    read metadata. Either it was never sent or it was ignored, and only one of
    those is the app's fault.
    """

    def send(self, app, monkeypatch, messages):
        monkeypatch.setattr(app, "chat_stream", a_reply)
        return steps(app.app.test_client().post(
            "/api/chat", json={"model": "m", "messages": messages}))

    def test_details_that_were_read_are_recorded_with_their_text(self, app, monkeypatch):
        found = self.send(app, monkeypatch, [{
            "role": "user", "content": "when was this?", "images": ["b64"],
            "image_meta": [{"taken": "2026:07:14 18:42:07", "lat": 51.51, "lon": -0.1275}]}])
        entry = named(found, "Photo details")
        assert entry is not None
        assert "carried their own record" in entry["detail"]
        assert "51.51" in entry["text"], "the exact text the model was given"

    def test_a_photo_with_nothing_in_it_says_so_rather_than_nothing(self, app, monkeypatch):
        """Silence here is what made the two failures indistinguishable."""
        found = self.send(app, monkeypatch, [
            {"role": "user", "content": "what is this?", "images": ["b64"]}])
        entry = named(found, "Photo details")
        assert entry is not None
        assert "none" in entry["detail"]
        assert "screenshot" in entry["detail"], "and names the usual reason"

    def test_a_turn_with_no_photo_gets_no_photo_step(self, app, monkeypatch):
        found = self.send(app, monkeypatch, [{"role": "user", "content": "hello"}])
        assert named(found, "Photo details") is None
        assert named(found, "Images") is None

    def test_it_says_whether_the_model_can_see(self, app, monkeypatch):
        found = self.send(app, monkeypatch, [
            {"role": "user", "content": "what is this?", "images": ["b64"]}])
        entry = named(found, "Images")
        assert entry is not None
        assert "1 in the thread" in entry["detail"]
        assert "cannot see" in entry["detail"] or "reads them itself" in entry["detail"]


class TestItSaysWhatWasPutInFront:
    def test_every_turn_records_what_was_sent(self, app, monkeypatch):
        monkeypatch.setattr(app, "chat_stream", a_reply)
        found = steps(app.app.test_client().post(
            "/api/chat", json={"model": "m", "messages": [
                {"role": "user", "content": "hello"}]}))
        entry = named(found, "Sent to the model")
        assert entry is not None
        assert "num_ctx" in entry["detail"]

    def test_the_injected_context_is_shown_verbatim(self, app, monkeypatch):
        """The scaffolding is what shapes the answer, so it is the thing worth
        being able to read afterwards."""
        monkeypatch.setattr(app, "chat_stream", a_reply)
        found = steps(app.app.test_client().post(
            "/api/chat", json={"model": "m", "messages": [{
                "role": "user", "content": "when?", "images": ["b64"],
                "image_meta": [{"taken": "2026:07:14 18:42:07"}]}]}))
        entry = named(found, "Sent to the model")
        assert entry["system"], "the system turns are the scaffolding"
        assert any("camera recorded" in block for block in entry["system"])


class TestItSaysWhatTheWebDid:
    def _web_on(self, app, monkeypatch):
        monkeypatch.setattr(app, "chat_stream", a_reply)
        monkeypatch.setattr(app, "web_enabled", lambda: True)

    def test_a_planner_that_declines_is_recorded(self, app, monkeypatch):
        self._web_on(app, monkeypatch)
        monkeypatch.setattr(app.web, "plan_searches", lambda *a, **k: [])
        found = steps(app.app.test_client().post("/api/chat", json={
            "model": "m", "web": True,
            "messages": [{"role": "user", "content": "what is 2+2?"}]}))
        entry = named(found, "Planned searches")
        assert entry is not None and "would not help" in entry["detail"]

    def test_a_planner_that_cannot_be_reached_is_recorded(self, app, monkeypatch):
        self._web_on(app, monkeypatch)
        monkeypatch.setattr(app.web, "plan_searches", lambda *a, **k: None)
        found = steps(app.app.test_client().post("/api/chat", json={
            "model": "m", "web": True,
            "messages": [{"role": "user", "content": "hello"}]}))
        assert "could not be reached" in named(found, "Planned searches")["detail"]

    def test_the_queries_are_recorded(self, app, monkeypatch):
        self._web_on(app, monkeypatch)
        monkeypatch.setattr(app.web, "plan_searches",
                            lambda *a, **k: ["hosyond 3.5 screen", "arduino tft"])
        monkeypatch.setattr(app.web, "search", lambda q, **k: [])
        found = steps(app.app.test_client().post("/api/chat", json={
            "model": "m", "web": True,
            "messages": [{"role": "user", "content": "what is this?"}]}))
        assert "hosyond 3.5 screen" in named(found, "Planned searches")["detail"]

    def test_a_search_that_finds_nothing_is_recorded(self, app, monkeypatch):
        """The reported case: the answer said it could not verify anything, and
        this is what tells you the search ran and came back empty."""
        self._web_on(app, monkeypatch)
        monkeypatch.setattr(app.web, "plan_searches", lambda *a, **k: ["anything"])
        monkeypatch.setattr(app.web, "search", lambda q, **k: [])
        found = steps(app.app.test_client().post("/api/chat", json={
            "model": "m", "web": True,
            "messages": [{"role": "user", "content": "what is this?"}]}))
        entry = named(found, "Search results")
        assert entry is not None and entry["detail"].startswith("0 from")

    def test_the_pages_that_were_read_are_recorded(self, app, monkeypatch):
        self._web_on(app, monkeypatch)
        monkeypatch.setattr(app.web, "plan_searches", lambda *a, **k: ["q"])
        monkeypatch.setattr(app.web, "search", lambda q, **k: [
            {"url": "https://example.com/a", "title": "A", "snippet": "s"}])
        monkeypatch.setattr(app.web, "fetch", lambda url, **k: {
            "url": url, "title": "A", "text": "the page text"})
        found = steps(app.app.test_client().post("/api/chat", json={
            "model": "m", "web": True,
            "messages": [{"role": "user", "content": "what is this?"}]}))
        assert "https://example.com/a" in named(found, "Search results")["urls"]
        assert named(found, "Pages read")["detail"].startswith("1 of 1")
        assert named(found, "Context from the web")["detail"].startswith("1 document")

    def test_a_backend_that_broke_is_named_rather_than_silent(self, app, monkeypatch):
        """The reported case: "0 from 0 query group(s)" with nothing beside it.
        A backend that answers 200 with an empty page now says so, and the
        panel is where you read it."""
        self._web_on(app, monkeypatch)
        monkeypatch.setattr(app.web, "plan_searches", lambda *a, **k: ["q"])

        def broken(query, **kw):
            raise app.web.WebError("html.duckduckgo.com returned a page with "
                                   "no results in it. Set SEARXNG_URL…")
        monkeypatch.setattr(app.web, "search", broken)
        found = steps(app.app.test_client().post("/api/chat", json={
            "model": "m", "web": True,
            "messages": [{"role": "user", "content": "what is this?"}]}))
        entry = named(found, "Search results")
        assert entry is not None
        assert "no results in it" in entry["detail"]
        assert "SEARXNG_URL" in entry["detail"]

    def test_none_of_it_appears_when_the_web_is_off(self, app, monkeypatch):
        monkeypatch.setattr(app, "chat_stream", a_reply)
        found = steps(app.app.test_client().post("/api/chat", json={
            "model": "m", "messages": [{"role": "user", "content": "hello"}]}))
        assert named(found, "Planned searches") is None


class TestThePanel:
    def page(self):
        return chat_ui.render_page("t")

    def test_it_is_its_own_panel_beside_the_thinking_one(self):
        text = self.page()
        assert 'details class="think steps" hidden' in text
        assert "Show what it did" in text
        js = page_script(text)
        # The thinking panel must still find its own element, not this one.
        assert 'wrap.querySelector("details.think:not(.steps)")' in js

    def test_it_stays_shut_until_asked(self):
        """It is noise on every turn that went fine, which is most of them."""
        text = self.page()
        at = text.index('class="think steps"')
        assert "open" not in text[at:text.index("</details>", at)]

    def test_a_step_is_never_written_as_markup(self):
        """A step carries a page title, a search query and a model name, none
        of which this app wrote."""
        js = page_script(self.page())
        at = js.index("function addStep")
        window = js[at:js.index("// Split assistant text", at)]
        assert "innerHTML" not in window
        for field in ("name.textContent", "detail.textContent", "body.textContent"):
            assert field in window, field

    def test_the_long_parts_fold_away_again(self):
        js = page_script(self.page())
        at = js.index("function addStep")
        window = js[at:at + 2200]
        assert 'more.className = "stepmore"' in window
        assert 'sum.textContent = "show"' in window

    def test_the_revealed_text_cannot_run_off_the_bubble(self):
        text = self.page()
        assert ".stepmore[open] { flex:1 1 100%; }" in text
        assert "width:100%; box-sizing:border-box;" in text

    def test_a_line_that_is_not_a_step_is_left_alone(self):
        """status, sources, content and thinking all share this channel."""
        js = page_script(self.page())
        at = js.index("if (obj.debug)")
        assert "addStep(view, obj.debug); scrollDown(); continue;" in js[at:at + 120]


class TestThePanelDoesNotMisleadAboutWhatItIsShowing:
    """Both of these were found by being misled by them.

    The panel is the only view of what the model was actually handed, so a
    view that quietly omits the part you are looking for is worse than no
    view — you draw a conclusion from it and go on to act on the conclusion.
    """

    def test_a_clipped_context_says_it_is_clipped(self, app, monkeypatch):
        """It clipped at 2000 characters in silence, and the per-page link
        lists live at the *end* of a block five times that long — so the panel
        showed a context with no numbered links in it whether or not there
        were any."""
        monkeypatch.setattr(app, "chat_stream", a_reply)
        monkeypatch.setattr(app, "web_enabled", lambda: True)
        monkeypatch.setattr(app.web, "plan_searches", lambda *a, **k: ["q"])
        monkeypatch.setattr(app.web, "search",
                            lambda *a, **k: [{"url": "https://e.test/a", "title": "A",
                                              "snippet": "s"}])
        monkeypatch.setattr(app.web, "fetch", lambda url, **k: {
            "url": url, "title": "A", "text": "long page. " * 900, "links": []})
        found = steps(app.app.test_client().post("/api/chat", json={
            "model": "m", "web": True,
            "messages": [{"role": "user", "content": "what is a?"}]}))
        blocks = named(found, "Sent to the model")["system"]
        clipped = [b for b in blocks if "characters shown" in b]
        assert clipped, "a clipped block must say so"
        assert "of" in clipped[0], "and say how much of how much"
        assert "link lists come after" in clipped[0], \
            "and say what is missing, which is the whole reason it matters"

    def test_a_short_context_is_not_labelled_as_clipped(self, app, monkeypatch):
        monkeypatch.setattr(app, "chat_stream", a_reply)
        found = steps(app.app.test_client().post(
            "/api/chat", json={"model": "m", "messages": [
                {"role": "user", "content": "hello"}]}))
        for block in named(found, "Sent to the model")["system"] or []:
            assert "characters shown" not in block


class TestItSaysWhetherTheModelWasInvitedToFollowALink:
    """WEB_MAX_HOPS is the app following a link on its own judgement;
    WEB_FETCH_HOPS is the model asking for one. A turn where the app followed
    two links and the model was never invited looked exactly like a turn where
    it was invited and declined."""

    def _turn(self, app, monkeypatch):
        monkeypatch.setattr(app, "chat_stream", a_reply)
        monkeypatch.setattr(app, "web_enabled", lambda: True)
        monkeypatch.setattr(app.web, "plan_searches", lambda *a, **k: ["q"])
        monkeypatch.setattr(app.web, "search", lambda *a, **k: [
            {"url": "https://e.test/a", "title": "A", "snippet": "s"}])
        monkeypatch.setattr(app.web, "fetch", lambda url, **k: {
            "url": url, "title": "A", "text": "the page", "links": []})
        return steps(app.app.test_client().post("/api/chat", json={
            "model": "m", "web": True,
            "messages": [{"role": "user", "content": "what is a?"}]}))

    def test_a_turn_that_never_touched_the_web_says_nothing_about_it(
            self, app, monkeypatch):
        """A line about a setting that could not have applied is clutter, and
        the panel is worth reading only while everything in it is this turn."""
        monkeypatch.setattr(app, "chat_stream", a_reply)
        found = steps(app.app.test_client().post(
            "/api/chat", json={"model": "m", "messages": [
                {"role": "user", "content": "hello"}]}))
        assert named(found, "Asking to read a link") is None

    def test_off_says_which_setting_turned_it_off(self, app, monkeypatch):
        monkeypatch.setattr(app, "get_web_fetch_hops", lambda: 0)
        entry = named(self._turn(app, monkeypatch), "Asking to read a link")
        assert entry is not None
        assert "WEB_FETCH_HOPS" in entry["detail"]

    def test_and_names_the_setting_it_is_confused_with(self, app, monkeypatch):
        monkeypatch.setattr(app, "get_web_fetch_hops", lambda: 0)
        entry = named(self._turn(app, monkeypatch), "Asking to read a link")
        assert "WEB_MAX_HOPS" in entry["detail"], \
            "the confusable setting is named, because it is confusable"

    def test_on_but_nothing_to_name_is_a_different_answer(self):
        note = app_module._fetch_offer_note(1, {})
        assert "no numbered links" in note and "budget" in note

    def test_on_with_links_says_how_many(self):
        note = app_module._fetch_offer_note(2, {"1.1": {}, "1.2": {}, "2.1": {}})
        assert "3" in note and "2" in note


class TestWhyNoLinkWasFollowed:
    """Absence was ambiguous. A turn that followed nothing showed no row at
    all, and four quite different things looked identical: the pages carried
    no links, the links were all off-site, the picker declined, or the picker
    could not be reached. The same class of silence that sent an OCR
    diagnosis in the wrong direction for two rounds.
    """

    def _turn(self, app, monkeypatch, links, chooses):
        monkeypatch.setattr(app, "chat_stream", a_reply)
        monkeypatch.setattr(app, "web_enabled", lambda: True)
        monkeypatch.setattr(app.web, "plan_searches", lambda *a, **k: ["q"])
        monkeypatch.setattr(app.web, "search", lambda *a, **k: [
            {"url": "https://bbc.test/news", "title": "N", "snippet": "s"}])
        monkeypatch.setattr(app.web, "fetch", lambda url, **k: {
            "url": url, "title": "N", "text": "page", "links": links})
        monkeypatch.setattr(app.web, "choose_links", chooses)
        return steps(app.app.test_client().post("/api/chat", json={
            "model": "m", "web": True,
            "messages": [{"role": "user", "content": "what is a?"}]}))

    SAME = [{"url": f"https://bbc.test/s/{i}", "text": f"Story {i}"} for i in range(5)]
    OFF = [{"url": f"https://other.test/{i}", "text": f"Other {i}"} for i in range(5)]

    def test_no_links_on_the_page_at_all(self, app, monkeypatch):
        found = self._turn(app, monkeypatch, [], lambda *a, **k: [])
        assert "none to follow" in named(found, "Followed links")["detail"]

    def test_links_that_are_all_off_site_name_the_setting(self, app, monkeypatch):
        found = self._turn(app, monkeypatch, self.OFF, lambda *a, **k: [])
        detail = named(found, "Followed links")["detail"]
        assert "5 link(s)" in detail, "say there were links, not that there were none"
        assert "WEB_FOLLOW_SCOPE" in detail

    def test_a_picker_that_declined_says_so(self, app, monkeypatch):
        found = self._turn(app, monkeypatch, self.SAME, lambda *a, **k: [])
        detail = named(found, "Followed links")["detail"]
        assert "chose none" in detail and "5 offered" in detail

    def test_a_picker_that_could_not_be_reached_is_not_the_same_thing(
            self, app, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("Ollama is asleep")
        found = self._turn(app, monkeypatch, self.SAME, boom)
        assert "could not be asked" in named(found, "Followed links")["detail"]

    def test_and_a_turn_that_followed_something_still_says_what(
            self, app, monkeypatch):
        found = self._turn(app, monkeypatch, self.SAME,
                           lambda *a, **k: [self.SAME[0]])
        assert "1 chosen" in named(found, "Followed links")["detail"]
