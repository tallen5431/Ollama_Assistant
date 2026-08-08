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

    def test_a_double_tap_still_installs_one_set(self):
        """Two taps on a phone arrive together often enough to matter.

        Reading what is already there on one connection and inserting on
        another leaves a window in which both requests find the table empty.
        """
        import threading
        store.create("a chat")           # so the schema exists before the race
        counts = []
        threads = [threading.Thread(target=lambda: counts.append(len(store.create_starters())))
                   for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sorted(counts) == [0, 0, 0, 0, 0, 4], counts
        assert len(store.list_routines()) == 4
        assert store.available() is True, "the race must not leave a transaction open"


    def test_two_routines_saved_at_once_keep_distinct_positions(self):
        """Otherwise their order on the strip falls to the created_at tiebreak."""
        import threading
        store.create("warm the schema")
        threads = [threading.Thread(target=store.create_routine, args=(f"r{i}", "b"))
                   for i in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        positions = [r["position"] for r in store.list_routines()]
        assert len(positions) == 12
        assert len(set(positions)) == 12, positions
        assert store.available() is True

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

    @pytest.mark.parametrize("table", ["routines", "records"])
    def test_an_older_database_gains_a_table_it_never_had(self, tmp_path,
                                                          monkeypatch, table):
        """A chat.db written before that table existed.

        No migration needed for a whole table — executescript runs the schema
        on every connection — but that is only true while the schema really is
        run every time, which is what this holds down.
        """
        path = tmp_path / "old.db"
        monkeypatch.setenv("CHAT_DB", str(path))
        before = re.sub(rf"CREATE TABLE IF NOT EXISTS {table} \(.*?\);", "",
                        store._SCHEMA, flags=re.S)
        before = re.sub(rf"CREATE INDEX IF NOT EXISTS idx_{table}_\w+.*?;", "",
                        before, flags=re.S)
        assert f"TABLE IF NOT EXISTS {table}" not in before
        conn = sqlite3.connect(str(path))
        conn.executescript(before)
        conn.commit()
        conn.close()

        assert store.list_routines() == []
        assert store.list_records() == []
        made = store.create_routine("🚗 Trip", "b", photos=2)
        assert store.list_routines()[0]["id"] == made["id"]
        kept = store.add_record("🚗 Trip", {"distance": "68 miles"})
        assert store.list_records()[0]["id"] == kept["id"]


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

    @pytest.mark.parametrize("photos", [float("inf"), float("-inf"), float("nan"),
                                        "lots", None, [], {"a": 1}])
    def test_a_photo_count_that_is_not_a_number_never_raises(self, client, photos):
        """json.loads accepts Infinity and NaN, and int(inf) raises."""
        resp = client.post("/api/routines",
                           json={"name": "r", "body": "b", "photos": photos})
        assert resp.status_code == 200, resp.get_data(as_text=True)[:200]
        assert resp.get_json()["photos"] == 0

    def test_the_same_on_a_patch(self, client):
        made = client.post("/api/routines", json={"name": "r", "body": "b"}).get_json()
        resp = client.patch(f"/api/routines/{made['id']}", json={"photos": float("inf")})
        assert resp.status_code == 200
        assert resp.get_json()["routine"]["photos"] == 0

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
        assert len(client.post("/api/routines/starters", json={}).get_json()["routines"]) == 4
        assert client.post("/api/routines/starters", json={}).get_json()["routines"] == []
        assert len(client.get("/api/routines").get_json()["routines"]) == 4

    def test_the_starters_route_cannot_be_posted_by_a_plain_form(self, client):
        """It is the only routines route with no body to require.

        A form post is a "simple" request and is never preflighted, so without
        this a page on any other site could POST here from the owner's browser.
        Every other route already needs JSON or a method a form cannot send.
        """
        assert client.post("/api/routines/starters", data={"x": "y"}).status_code == 415
        assert client.post("/api/routines/starters",
                           data="x=y",
                           content_type="text/plain").status_code == 415
        assert client.get("/api/routines").get_json()["routines"] == []

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

    def test_attaching_a_photo_explains_a_missing_position(self):
        """The sentence has to actually be reached, not merely exist."""
        page = self.page()
        at = page.index("function addAttachment")
        body = page[at:page.index("\n      }", at)]
        assert "noteMissingPlace(att);" in body, \
            "nothing calls it, so nobody is ever told why"

    def test_the_thumbnail_says_what_was_actually_read(self):
        """"Photo details is on" and "this photo has a position" differ.

        Until the badge said which, the difference only surfaced in the answer.
        """
        page = self.page()
        assert ".thumb .stamp" in page
        assert '["🕘", "date read from the photo"]' in page
        assert '["📍", "location read from the photo"]' in page
        assert '["📍̸", "the camera tried but had no GPS fix"]' in page, \
            "a GPS block with no fix is its own answer, not silence"

    def test_a_routine_is_only_spent_on_a_turn_that_went(self):
        """send() refuses an empty message and one that arrives mid-stream.

        Disarming there put the toggles back and dropped the photo count while
        the message was still sitting in the composer waiting to be sent.
        """
        page = self.page()
        assert "if (!(await send())) return;\n        clearRoutine();" in page
        assert "|| busy) return false;" in page, "send() must report a refusal"

    def test_clearing_takes_its_own_text_back_out(self):
        """Otherwise picking a second routine stacks the two prompts."""
        page = self.page()
        assert "pendingRoutine.inserted = routine.body;" in page
        assert "inputEl.value.indexOf(inserted) === 0" in page, \
            "and only while it is still there verbatim — edited text is the user's"

    def test_the_voice_retry_goes_through_the_guard(self):
        """A dictated follow-up while a routine is armed still owes it photos."""
        page = self.page()
        assert "setTimeout(trySend, 0)" in page
        assert "setTimeout(send, 0)" not in page

    def test_the_photo_details_toggle_is_read_at_send_time(self):
        """Turning it off after attaching has to mean the details do not go."""
        page = self.page()
        assert "if (exifOn && meta.some(Boolean))" in page

    def test_a_rolled_back_turn_does_not_spend_the_routine(self):
        """Stop hands the message back, so the routine has to still be armed.

        Reporting "the turn went, however it ended" also had clearRoutine strip
        the restored prompt straight back out of the composer, because it was
        still a verbatim prefix — Stop deleted 1500 characters without trace.
        """
        page = self.page()
        assert "let went = true" in page
        assert page.count("went = false;") == 3, \
            "empty reply, stopped-with-nothing, and errored-with-nothing"
        assert "return went;" in page

    def test_dictation_goes_through_the_guard_on_both_paths(self):
        """A dictated turn is still a send, and still owes a routine its photos."""
        page = self.page()
        assert "if (autoSendEl.checked && !busy) { trySend(); return; }" in page
        assert "setTimeout(trySend, 0)" in page
        assert "{ send(); return; }" not in page

    def test_the_queued_dictation_does_not_return_out_of_finally(self):
        """A return inside finally replaces the value the function was giving back.

        It reported every queued dictation as a turn that never went, so the
        routine stayed armed with its own message already sent.
        """
        page = self.page()
        at = page.index("if (pendingAutoSend) {")
        assert "return;" not in page[at:page.index("return went;", at)], \
            "nothing may return out of send()'s finally"

    def test_the_armed_chip_is_drawn_from_what_the_guard_uses(self):
        """Editing a routine while armed had the chip and Send disagree."""
        page = self.page()
        assert "saved.id === armed ? pendingRoutine.routine : saved" in page

    def test_the_prompt_box_stops_where_the_store_does(self):
        page = self.page()
        assert 'id="rBody" rows="8" spellcheck="false" maxlength="4000"' in page
        assert "the prompt was trimmed to" in page, "and says so if it still trimmed"

    def test_saving_is_single_flight(self):
        """editingRoutineId is not set until after the await, so two taps create two."""
        page = self.page()
        assert "if (savingRoutine) return;" in page
        assert "rSaveBtn.disabled = true;" in page

    def test_deleting_only_disarms_once_it_is_gone(self):
        page = self.page()
        at = page.index("async function deleteRoutine")
        body = page[at:page.index("async function addStarters", at)]
        assert body.index("resp.ok") < body.index("clearRoutine()"), \
            "a failed delete must not revert the toggles of a routine that still exists"

    def test_the_count_hint_only_clears_its_own_message(self):
        """hintEl is shared; blanking it wiped the history-unavailable warning."""
        page = self.page()
        assert 'hintEl.textContent.indexOf(" attached.") > 0' in page


class TestTheFooterLeavesRoomToRead:
    """Measured on a 390x844 phone before this: the footer was 293px idle and
    291px with the keyboard up, leaving 42px of conversation. You could not see
    the message you were replying to while replying to it.
    """

    def page(self):
        return chat_ui.render_page("t")

    def test_the_options_collapse_when_the_keyboard_is_up(self):
        page = self.page()
        assert "window.visualViewport" in page, \
            "innerHeight does not move when the keyboard opens on Android"
        assert 'document.body.classList.toggle("typing", tight)' in page
        assert "body.typing #togglebar" in page and "body.typing #routinebar" in page

    def test_the_composer_itself_is_never_collapsed(self):
        """The point is to give the conversation its screen, not to hide the box."""
        page = self.page()
        at = page.index("body.typing #voicebar")
        rules = page[at:page.index("/* Phones:", at)]
        assert ".composer" not in rules and "#thumbs" not in rules

    def test_it_does_nothing_where_there_is_no_keyboard(self):
        """Desktop has no visualViewport shrink, and older browsers have none."""
        page = self.page()
        at = page.index("function fitFooter")
        assert "if (!vv) return;" in page[at:at + 300]

    def test_every_toggle_shares_one_scrolling_row(self):
        """Four wrapping rows is what turned five checkboxes into 128px."""
        page = self.page()
        assert 'id="togglebar"' in page
        for control in ('id="web"', 'id="exif"', 'id="headset"',
                        'id="autosend"', 'id="continuous"'):
            assert page.index('id="togglebar"') < page.index(control) < page.index('id="routinebar"'), control
        assert "#routinebar, #togglebar { flex-wrap:nowrap; overflow-x:auto;" in page

    def test_the_dictation_toggles_hide_without_a_microphone(self):
        """Otherwise they are three settings for a button that is not there."""
        page = self.page()
        assert 'querySelectorAll(".voicesetting")' in page
        assert "el.hidden = !data.voice;" in page

    def test_the_tap_targets_meet_the_platform_minimum(self):
        """Every control measured under 44px; the checkbox rows were 14px tall."""
        page = self.page()
        for rule in (".chip { padding:0.6rem 0.8rem; min-height:44px;",
                     ".voicebar-check { min-height:44px; }",
                     ".composer button { min-height:44px;",
                     ".composer .iconbtn { width:44px; height:44px; }",
                     # The header is three icon buttons in a row, and .iconbtn
                     # is sized for a mouse: measured 34px square on a phone.
                     "#menu, #theme, #newChat { min-height:40px; min-width:40px;"):
            assert rule in page, rule
