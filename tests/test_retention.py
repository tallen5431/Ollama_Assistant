"""Photos expire; words do not.

Attached photos are stored base64 alongside the message. The browser caps a
re-encode at roughly 900KB, which is about 1.2MB of text in the row, so a
two-photo routine run every day is most of a gigabyte a year. What is worth
having a year later is the reading that came off the picture — which is in the
reply, and for a routine also in the records — not the picture.
"""

from __future__ import annotations

import importlib
import time

import pytest

import app as app_module
import chat_ui
import store
from conftest import page_script


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAT_DB", str(tmp_path / "chat.db"))
    monkeypatch.setattr(store, "_last_sweep", 0.0, raising=False)
    yield


@pytest.fixture
def client(monkeypatch):
    for key in ("CHAT_AUTH", "CHAT_AUTH_USER", "CHAT_AUTH_PASSWORD"):
        monkeypatch.delenv(key, raising=False)
    return importlib.reload(app_module).app.test_client()


def photo_message(days_ago=0.0, meta=True):
    """A message with a photo, optionally backdated."""
    convo = store.create("Odometer")["id"]
    store.add_message(convo, "user", "read this", images=["A" * 5000],
                      image_meta=[{"taken": "2026:06:01 09:00:00"}] if meta else None)
    store.add_message(convo, "assistant", "It reads 10,500 miles.")
    if days_ago:
        with store._connect() as conn:
            conn.execute("UPDATE messages SET created_at = ? WHERE conversation_id = ?",
                         (time.time() - days_ago * 86400, convo))
    return convo


def images_of(convo):
    return [m["images"] for m in store.get(convo)["messages"]]


class TestWhatGoesAndWhatStays:
    def test_an_old_photo_goes(self):
        convo = photo_message(days_ago=60)
        done = store.forget_images(30)
        assert done["messages"] == 1 and done["bytes"] > 4000
        assert images_of(convo) == [None, None]

    def test_a_recent_one_stays(self):
        convo = photo_message(days_ago=2)
        assert store.forget_images(30) == {"messages": 0, "bytes": 0}
        assert images_of(convo)[0] == ["A" * 5000]

    def test_every_word_survives(self):
        """The reading is the point; the picture was only how it got here."""
        convo = photo_message(days_ago=60)
        store.forget_images(30)
        said = [m["content"] for m in store.get(convo)["messages"]]
        assert said == ["read this", "It reads 10,500 miles."]

    def test_and_so_does_what_the_photo_said_about_itself(self):
        """When and where it was taken is a different column, and it is the
        part still worth having."""
        convo = photo_message(days_ago=60)
        store.forget_images(30)
        assert store.get(convo)["messages"][0]["image_meta"][0]["taken"] \
            == "2026:06:01 09:00:00"

    def test_records_are_untouched(self):
        store.add_record("🚗 Trip", {"Distance travelled": "68 miles"})
        photo_message(days_ago=60)
        store.forget_images(30)
        assert store.list_records()[0]["fields"]["Distance travelled"] == "68 miles"

    @pytest.mark.parametrize("days", [0, -1, None])
    def test_zero_means_keep_them_for_good(self, days):
        convo = photo_message(days_ago=3650)
        assert store.forget_images(days) == {"messages": 0, "bytes": 0}
        assert images_of(convo)[0] == ["A" * 5000]

    def test_it_is_safe_to_run_twice(self):
        photo_message(days_ago=60)
        store.forget_images(30)
        assert store.forget_images(30) == {"messages": 0, "bytes": 0}

    def test_nothing_stored_is_not_an_error(self):
        assert store.forget_images(30) == {"messages": 0, "bytes": 0}


class TestTheSweep:
    def test_it_runs_once_and_then_waits(self):
        """Hung off ordinary traffic, so it must not run on every request."""
        photo_message(days_ago=60)
        first = store.maybe_forget_images(30)
        photo_message(days_ago=60)
        second = store.maybe_forget_images(30)
        assert first["messages"] == 1
        assert second == {"messages": 0, "bytes": 0}, "it swept again immediately"

    def test_a_broken_database_costs_the_sweep_not_the_request(self, monkeypatch):
        def boom(*a, **k):
            raise store.sqlite3.OperationalError("database is locked")
        monkeypatch.setattr(store, "forget_images", boom)
        assert store.maybe_forget_images(30) == {"messages": 0, "bytes": 0}

    def test_listing_conversations_is_what_drives_it(self, client, monkeypatch):
        ran = []
        monkeypatch.setattr(store, "maybe_forget_images",
                            lambda days: ran.append(days) or {"messages": 0, "bytes": 0})
        client.get("/api/conversations")
        assert ran == [30.0], "the default cutoff should have been passed"


class TestTheRoute:
    def test_applying_it_now(self, client):
        convo = photo_message(days_ago=60)
        done = client.post("/api/photos/forget", json={"days": 30}).get_json()
        assert done["messages"] == 1
        assert images_of(convo) == [None, None]

    def test_it_falls_back_to_the_configured_cutoff(self, client):
        convo = photo_message(days_ago=60)
        assert client.post("/api/photos/forget").get_json()["messages"] == 1
        assert images_of(convo) == [None, None]

    @pytest.mark.parametrize("body", [{"days": "soon"}, {"days": None}, [], "nope"])
    def test_a_nonsense_cutoff_falls_back_rather_than_failing(self, client, body):
        photo_message(days_ago=60)
        resp = client.post("/api/photos/forget", json=body)
        assert resp.status_code == 200

    def test_a_negative_cutoff_cannot_delete_todays_photos(self, client):
        """A cutoff in the future would be every photo you have. Guarded in
        both the route and the store, so this holds if either one moves."""
        convo = photo_message(days_ago=0)
        client.post("/api/photos/forget", json={"days": -5})
        assert images_of(convo)[0] == ["A" * 5000]

    def test_the_policy_is_reported_with_the_cost(self, client):
        stats = client.get("/api/conversations/stats").get_json()
        assert stats["photo_keep_days"] == 30.0
        assert "bytes" in stats

    def test_it_is_behind_the_same_guard_as_everything_else(self, monkeypatch):
        monkeypatch.setenv("CHAT_AUTH", "tj:hunter2")
        guarded = importlib.reload(app_module).app.test_client()
        assert guarded.post("/api/photos/forget").status_code == 401


class TestTheSetting:
    def test_the_default_is_a_month(self, monkeypatch):
        monkeypatch.delenv("PHOTO_KEEP_DAYS", raising=False)
        import config
        assert config.get_photo_keep_days() == 30.0

    @pytest.mark.parametrize("value, days", [
        ("7", 7.0), ("0", 0.0), ("0.5", 0.5), ("-3", 0.0), ("nonsense", 30.0),
    ])
    def test_it_can_be_set(self, monkeypatch, value, days):
        monkeypatch.setenv("PHOTO_KEEP_DAYS", value)
        import config
        assert config.get_photo_keep_days() == days


class TestThePage:
    def page(self):
        return chat_ui.render_page("t")

    def test_the_policy_is_stated_where_the_number_is(self):
        """A size on disk with no lever next to it is just bad news."""
        js = page_script(self.page())
        at = js.index("async function showHistoryCost")
        window = js[at:at + 2000]
        assert '"Photos are kept for "' in window
        assert "on disk" in window

    def test_and_it_says_when_they_are_kept_for_good(self):
        js = page_script(self.page())
        assert 'Photos are kept for good (PHOTO_KEEP_DAYS=0).' in js

    def test_freeing_space_asks_first_and_says_what_it_keeps(self):
        js = page_script(self.page())
        at = js.index("async function forgetPhotosNow")
        window = js[at:at + 900]
        assert "confirm(" in window
        assert "Every message, reply and record is kept" in window

    def test_there_is_no_button_when_nothing_ever_expires(self):
        js = page_script(self.page())
        at = js.index("async function showHistoryCost")
        window = js[at:at + 2000]
        assert "if (days <= 0) return;" in window

    def test_an_expired_photo_is_said_rather_than_left_as_a_gap(self):
        js = page_script(self.page())
        at = js.index("function expiredPhotoNote")
        assert "passed the keep-for period" in js[at:at + 500]
        assert "col.appendChild(expiredPhotoNote(msg.image_meta.length));" in js
