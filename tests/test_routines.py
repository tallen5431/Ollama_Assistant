"""Routines — saved prompts you tap instead of typing.

A routine is a name, a body of prompt text, and three things it declares about
the turn it is used on: how many photos it expects, and whether web access and
photo details should be forced on or off for that turn.

The forcing pair is three-state on purpose, and most of the storage tests here
exist because ``bool(row["web"])`` collapses NULL into False — which silently
turns "no opinion about the web" into "force it off" on the first read.
"""

from __future__ import annotations

import importlib
import re
import sqlite3

import pytest

import app as app_module
import chat_ui
import store


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Point the store at a throwaway database for every test."""
    monkeypatch.setenv("CHAT_DB", str(tmp_path / "chat.db"))
    yield


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("CHAT_AUTH", raising=False)
    monkeypatch.delenv("CHAT_AUTH_USER", raising=False)
    monkeypatch.delenv("CHAT_AUTH_PASSWORD", raising=False)
    return importlib.reload(app_module).app.test_client()


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------

class TestLifecycle:
    def test_create_list_update_delete(self):
        made = store.create_routine("🚗 Trip", "Read both odometers.", photos=2)
        assert store.list_routines()[0]["id"] == made["id"]

        changed = store.update_routine(made["id"], {"name": "Trip"})
        assert changed["name"] == "Trip"
        assert changed["body"] == "Read both odometers.", "an untouched field must survive"

        assert store.delete_routine(made["id"]) is True
        assert store.list_routines() == []
        assert store.delete_routine(made["id"]) is False

    def test_a_routine_that_is_not_there(self):
        assert store.update_routine("nope", {"name": "x"}) is None
        assert store.delete_routine("nope") is False

    def test_nothing_to_change_is_not_an_update(self):
        made = store.create_routine("x", "y")
        assert store.update_routine(made["id"], {}) is None
        assert store.update_routine(made["id"], {"unknown": 1}) is None

    def test_the_strip_keeps_the_order_things_were_made_in(self):
        """Not "recently used" — that moves a tap target a thumb has learned."""
        for name in ("one", "two", "three"):
            store.create_routine(name, "body")
        store.update_routine(store.list_routines()[0]["id"], {"name": "one again"})
        assert [r["name"] for r in store.list_routines()] == ["one again", "two", "three"]


class TestTheThreeStateToggles:
    """NULL, 0 and 1 are three different answers and must stay that way."""

    @pytest.mark.parametrize("value", [None, True, False])
    def test_it_round_trips(self, value):
        made = store.create_routine("r", "b", web=value, photo_meta=value)
        back = store.list_routines()[0]
        assert back["web"] is value
        assert back["photo_meta"] is value
        assert made["web"] is value

    def test_none_is_not_false(self):
        no_opinion = store.create_routine("a", "b", web=None)
        forced_off = store.create_routine("c", "d", web=False)
        assert store.list_routines()[0]["web"] is None
        assert store.list_routines()[1]["web"] is False
        assert no_opinion["web"] is not forced_off["web"]

    def test_an_update_can_clear_a_forcing(self):
        made = store.create_routine("r", "b", web=True)
        assert store.update_routine(made["id"], {"web": None})["web"] is None

    def test_updating_one_leaves_the_other_alone(self):
        made = store.create_routine("r", "b", web=True, photo_meta=False)
        after = store.update_routine(made["id"], {"name": "renamed"})
        assert after["web"] is True and after["photo_meta"] is False


class TestLimits:
    @pytest.mark.parametrize("photos, kept", [(9, 4), (-1, 0), (2, 2), (None, 0), ("", 0)])
    def test_the_photo_count_is_clamped_to_what_the_composer_accepts(self, photos, kept):
        assert store.create_routine("r", "b", photos=photos)["photos"] == kept
        made = store.create_routine("s", "b")
        assert store.update_routine(made["id"], {"photos": photos})["photos"] == kept

    def test_a_long_name_is_cut_to_a_chip(self):
        made = store.create_routine("A" * 200, "b")
        assert len(made["name"]) == 40
        assert len(store.list_routines()[0]["name"]) == 40

    def test_a_name_is_one_line(self):
        """It goes in a chip; a newline would break the strip."""
        assert store.create_routine("two\nlines", "b")["name"] == "two lines"

    def test_a_body_that_would_own_the_context_window_is_cut(self):
        made = store.create_routine("r", "x" * 9000)
        assert len(made["body"]) == 4000
        assert len(store.list_routines()[0]["body"]) == 4000


class TestTheStarters:
    def test_installing_them_twice_gives_one_of_each(self):
        first = store.create_starters()
        assert len(first) == 4
        assert store.create_starters() == []
        assert len(store.list_routines()) == 4

    def test_they_survive_a_rename_of_one(self):
        store.create_starters()
        store.update_routine(store.list_routines()[0]["id"], {"name": "mine"})
        # The renamed one no longer claims the shipped name, so it comes back.
        assert [r["name"] for r in store.create_starters()] == ["🚗 Trip"]

    def test_the_trip_routine_says_what_the_metadata_turn_actually_says(self):
        """The prompt leans on labels nothing else explains to the model.

        web._metadata_lines numbers by attachment position ("Image 1"), and
        describe_images enumerates the same batch ("[image 1]"). The routine is
        what tells the model those are the same two photos, so if either label
        ever changes this has to change with it.
        """
        trip = {r["name"]: r for r in store.create_starters()}["🚗 Trip"]
        assert '"Image 1"' in trip["body"] and '"Image 2"' in trip["body"]
        assert '"[image 1]"' in trip["body"] and '"[image 2]"' in trip["body"]
        assert trip["photos"] == 2
        assert trip["photo_meta"] is True, "without this there are no capture times"

    def test_the_trip_routine_refuses_rather_than_estimating(self):
        trip = {r["name"]: r for r in store.create_starters()}["🚗 Trip"]
        assert "Do not estimate" in trip["body"]
        assert "time zone" in trip["body"], "capture times carry none"

    def test_no_shipped_routine_forces_web_access_on(self):
        """Photos plus a forced search is the one path that sends a position out.

        A routine with photos and web on hands the photo's coordinates to the
        search planner, whose queries go to a search engine. That can be the
        owner's deliberate choice; it must never be something that arrived by
        pressing "add the starters".
        """
        for routine in store.create_starters():
            assert routine["web"] is not True, routine["name"]

    def test_every_starter_is_usable_as_shipped(self):
        for routine in store.create_starters():
            assert routine["name"].strip() and routine["body"].strip()
            assert len(routine["name"]) <= 40
            assert len(routine["body"]) <= 4000
            assert 0 <= routine["photos"] <= 4


class TestItDoesNotBreakTheDatabase:
    def test_history_still_reports_itself_available(self):
        """Seeding must not leave a transaction open.

        A write inside _connect() would leave one, and available()'s BEGIN
        IMMEDIATE then raises "cannot start a transaction within a transaction"
        — which contains neither "locked" nor "busy", so it is read as a broken
        database and the entire history UI disappears.
        """
        store.create_starters()
        assert store.available() is True

    def test_routines_and_conversations_share_the_file_without_interfering(self):
        convo = store.create("a chat")
        store.create_starters()
        assert store.get(convo["id"]) is not None
        assert len(store.list_routines()) == 4
        assert store.delete(convo["id"]) is True
        assert len(store.list_routines()) == 4, "deleting a chat is not deleting routines"

    def test_an_older_database_gains_the_table(self, tmp_path, monkeypatch):
        """A chat.db written before routines existed."""
        path = tmp_path / "old.db"
        monkeypatch.setenv("CHAT_DB", str(path))
        conn = sqlite3.connect(str(path))
        before = re.sub(r"CREATE TABLE IF NOT EXISTS routines.*?\);", "",
                        store._SCHEMA, flags=re.S)
        before = re.sub(r"CREATE INDEX IF NOT EXISTS idx_routines_position.*?;", "",
                        before, flags=re.S)
        assert "routines" not in before
        conn.executescript(before)
        conn.commit()
        conn.close()

        assert store.list_routines() == []
        made = store.create_routine("🚗 Trip", "b", photos=2)
        assert store.list_routines()[0]["id"] == made["id"]


# --------------------------------------------------------------------------
# The API
# --------------------------------------------------------------------------

class TestTheRoutes:
    def test_a_fresh_install_has_none(self, client):
        assert client.get("/api/routines").get_json() == {"routines": []}

    def test_create_then_list(self, client):
        made = client.post("/api/routines", json={
            "name": "🚗 Trip", "body": "Read both.", "photos": 2,
            "web": False, "photo_meta": True}).get_json()
        assert made["id"] and made["photos"] == 2
        assert made["web"] is False and made["photo_meta"] is True
        assert client.get("/api/routines").get_json()["routines"][0]["id"] == made["id"]

    @pytest.mark.parametrize("body, missing", [
        ({"body": "x"}, "name"),
        ({"name": "x"}, "body"),
        ({"name": "   ", "body": "x"}, "name"),
        ({"name": "x", "body": "   "}, "body"),
    ])
    def test_a_routine_needs_both_halves(self, client, body, missing):
        resp = client.post("/api/routines", json=body)
        assert resp.status_code == 400
        assert missing in resp.get_json()["error"]

    @pytest.mark.parametrize("payload", [["a"], "hi", 7, None])
    def test_a_body_that_is_not_an_object_is_a_400_not_a_500(self, client, payload):
        assert client.post("/api/routines", json=payload).status_code == 400

    def test_patch_returns_the_whole_normalised_record(self, client):
        made = client.post("/api/routines", json={"name": "r", "body": "b"}).get_json()
        out = client.patch(f"/api/routines/{made['id']}", json={"photos": 9}).get_json()
        assert out["ok"] is True
        assert out["routine"]["photos"] == 4, "the client re-renders from this"
        assert out["routine"]["name"] == "r"

    def test_an_absent_field_and_an_explicit_null_are_different(self, client):
        """The whole reason PATCH tests membership rather than reading .get()."""
        made = client.post("/api/routines", json={
            "name": "r", "body": "b", "web": True}).get_json()
        untouched = client.patch(f"/api/routines/{made['id']}",
                                 json={"name": "r2"}).get_json()["routine"]
        assert untouched["web"] is True, "not mentioning it must leave it alone"
        cleared = client.patch(f"/api/routines/{made['id']}",
                               json={"web": None}).get_json()["routine"]
        assert cleared["web"] is None, "naming it as null must clear it"

    def test_patch_with_nothing_in_it(self, client):
        made = client.post("/api/routines", json={"name": "r", "body": "b"}).get_json()
        assert client.patch(f"/api/routines/{made['id']}", json={}).status_code == 400

    def test_patch_cannot_empty_a_routine(self, client):
        made = client.post("/api/routines", json={"name": "r", "body": "b"}).get_json()
        assert client.patch(f"/api/routines/{made['id']}", json={"name": ""}).status_code == 400
        assert client.patch(f"/api/routines/{made['id']}", json={"body": ""}).status_code == 400

    def test_patch_and_delete_on_something_that_is_not_there(self, client):
        assert client.patch("/api/routines/nope", json={"name": "x"}).status_code == 404
        assert client.delete("/api/routines/nope").status_code == 404

    def test_delete_is_not_idempotently_successful(self, client):
        made = client.post("/api/routines", json={"name": "r", "body": "b"}).get_json()
        assert client.delete(f"/api/routines/{made['id']}").get_json() == {"ok": True}
        assert client.delete(f"/api/routines/{made['id']}").status_code == 404

    def test_the_starters_route_is_idempotent(self, client):
        assert len(client.post("/api/routines/starters").get_json()["routines"]) == 4
        assert client.post("/api/routines/starters").get_json()["routines"] == []
        assert len(client.get("/api/routines").get_json()["routines"]) == 4

    def test_a_broken_database_is_a_503_with_json(self, client, tmp_path, monkeypatch):
        """Not Flask's HTML error page, which the UI would parse as nothing."""
        broken = tmp_path / "broken.db"
        broken.write_bytes(b"this is not a database" * 100)
        monkeypatch.setenv("CHAT_DB", str(broken))
        resp = client.get("/api/routines")
        assert resp.status_code == 503
        assert resp.get_json()["history"] is False
        assert "could not be written" in resp.get_json()["error"]


class TestTheRoutesAreBehindAuth:
    def test_every_routines_route_needs_a_password_when_auth_is_on(self, monkeypatch):
        """A routine is an instruction delivered as though the owner typed it."""
        monkeypatch.setenv("CHAT_AUTH_USER", "tj")
        monkeypatch.setenv("CHAT_AUTH_PASSWORD", "hunter2")
        mod = importlib.reload(app_module)
        assert mod.AUTH_ENABLED is True
        guarded = mod.app.test_client()
        assert guarded.get("/api/routines").status_code == 401
        assert guarded.post("/api/routines", json={"name": "a", "body": "b"}).status_code == 401
        assert guarded.post("/api/routines/starters").status_code == 401
        assert guarded.patch("/api/routines/x", json={"name": "a"}).status_code == 401
        assert guarded.delete("/api/routines/x").status_code == 401


# --------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------

class TestThePage:
    def page(self):
        return chat_ui.render_page("t")

    def test_the_strip_sits_above_the_photos_and_the_composer(self):
        """Read downwards: pick a routine, attach its photos, write the message."""
        page = self.page()
        assert page.index('id="routinebar"') < page.index('id="thumbs"')
        assert page.index('id="thumbs"') < page.index('class="composer"')

    def test_each_drawer_pane_owns_its_own_add_button(self):
        page = self.page()
        assert page.index('id="convoPane"') < page.index('id="drawerNew"')
        assert page.index('id="drawerNew"') < page.index('id="routinePane"')
        assert page.index('id="routinePane"') < page.index('id="routineNew"')

    def test_send_goes_through_the_photo_count_guard(self):
        page = self.page()
        assert 'sendBtn.addEventListener("click", trySend)' in page
        assert "e.preventDefault(); trySend();" in page

    def test_the_routine_state_is_cleared_everywhere_a_thread_changes(self):
        """newChat, loadConversation, a finished send, and tapping the lit chip."""
        page = self.page()
        assert page.count("clearRoutine();") >= 4

    def test_a_routine_never_reaches_the_page_as_markup(self):
        """Names and bodies are stored text anyone on the LAN could have written."""
        page = self.page()
        for line in page.splitlines():
            code = line.split("//")[0]
            if "innerHTML" in code and "routine" in code.lower():
                # Clearing the container is the only legitimate use.
                assert '= ""' in code, line.strip()

    def test_the_editor_warns_where_the_choice_is_made(self):
        page = self.page()
        assert "WEB_SHARE_LOCATION" in page
        assert "routineWarnUpdate" in page

    def test_the_thumbnail_says_whether_a_date_was_read(self):
        page = self.page()
        assert ".thumb .stamp" in page
        assert 'stamp.textContent = "🕘"' in page
