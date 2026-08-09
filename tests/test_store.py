"""Tests for conversation storage."""

from __future__ import annotations

import base64
import os
import sqlite3
import threading
import time

import pytest

import store


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Point the store at a throwaway database for every test."""
    monkeypatch.setenv("CHAT_DB", str(tmp_path / "chat.db"))
    yield


class TestLifecycle:
    def test_create_list_get_delete(self):
        convo = store.create("Hello there", model="llama3.1:8b")
        assert store.list_conversations()[0]["id"] == convo["id"]

        assert store.add_message(convo["id"], "user", "Hello there") is True
        assert store.add_message(convo["id"], "assistant", "Hi!") is True

        full = store.get(convo["id"])
        assert [m["role"] for m in full["messages"]] == ["user", "assistant"]
        assert full["messages"][0]["content"] == "Hello there"

        assert store.delete(convo["id"]) is True
        assert store.get(convo["id"]) is None
        assert store.delete(convo["id"]) is False

    def test_messages_keep_their_order(self):
        convo = store.create()
        for i in range(12):
            store.add_message(convo["id"], "user", f"message {i}")
        contents = [m["content"] for m in store.get(convo["id"])["messages"]]
        assert contents == [f"message {i}" for i in range(12)]

    def test_deleting_a_conversation_takes_its_messages(self):
        convo = store.create()
        store.add_message(convo["id"], "user", "x")
        store.delete(convo["id"])
        assert store.stats()["messages"] == 0

    def test_message_to_a_missing_conversation_is_rejected(self):
        assert store.add_message("nope", "user", "x") is False


class TestTitles:
    def test_first_user_message_names_an_untitled_thread(self):
        convo = store.create()
        assert convo["title"] == "New chat"
        store.add_message(convo["id"], "user", "How do I restart a systemd unit?")
        assert store.get(convo["id"])["title"] == "How do I restart a systemd unit?"

    def test_a_named_thread_is_not_renamed_by_later_messages(self):
        convo = store.create("Deployment notes")
        store.add_message(convo["id"], "user", "something else entirely")
        assert store.get(convo["id"])["title"] == "Deployment notes"

    def test_long_titles_are_truncated(self):
        convo = store.create("x" * 300)
        assert len(convo["title"]) <= 61 and convo["title"].endswith("…")

    def test_whitespace_is_collapsed(self):
        assert store.create("  a\n\n  b  ")["title"] == "a b"

    def test_rename(self):
        convo = store.create("first")
        assert store.rename(convo["id"], "second") is True
        assert store.get(convo["id"])["title"] == "second"
        assert store.rename("nope", "x") is False


class TestAttachments:
    def test_images_and_sources_round_trip(self):
        convo = store.create()
        store.add_message(convo["id"], "user", "what is this?", images=["QUJD"])
        store.add_message(convo["id"], "assistant", "a cat",
                          sources=[{"url": "https://a.example/", "title": "A"}])
        msgs = store.get(convo["id"])["messages"]
        assert msgs[0]["images"] == ["QUJD"]
        assert msgs[1]["sources"] == [{"url": "https://a.example/", "title": "A"}]

    def test_absent_attachments_come_back_as_none(self):
        convo = store.create()
        store.add_message(convo["id"], "user", "plain")
        msg = store.get(convo["id"])["messages"][0]
        assert msg["images"] is None and msg["sources"] is None


class TestOrderingAndStats:
    def test_most_recently_updated_sorts_first(self):
        a = store.create("older")
        store.create("newer")
        # b was created last, but a is the one just touched — activity wins.
        store.add_message(a["id"], "user", "bump a")
        listed = store.list_conversations()
        assert listed[0]["id"] == a["id"]
        assert len(listed) == 2

    def test_message_count_is_reported(self):
        convo = store.create()
        for _ in range(3):
            store.add_message(convo["id"], "user", "x")
        assert store.list_conversations()[0]["message_count"] == 3

    def test_stats(self):
        convo = store.create()
        store.add_message(convo["id"], "user", "x")
        s = store.stats()
        assert s["conversations"] == 1 and s["messages"] == 1 and s["bytes"] > 0


class TestConcurrency:
    def test_writes_from_several_threads_all_land(self):
        """Waitress serves on a thread pool, so this is the real access pattern."""
        import threading

        convo = store.create()
        errors = []

        def writer(n):
            try:
                for i in range(10):
                    store.add_message(convo["id"], "user", f"{n}-{i}")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, errors
        assert len(store.get(convo["id"])["messages"]) == 40

    def test_and_each_one_gets_its_own_place_in_the_thread(self):
        """"What is the next seq" ran outside any transaction. Python's sqlite3
        opens one on the first *write*, so two appends racing both read the
        same answer and both used it — measured, eight at once produced two
        distinct values between them. get() orders by seq, so the thread came
        back in an order nobody chose, and nothing in the schema stops it:
        there is no unique key on (conversation_id, seq).
        """
        convo = store.create()["id"]
        gate = threading.Barrier(8)

        def writer(n):
            gate.wait()
            store.add_message(convo, "user", f"m{n}")

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        with store._connect() as conn:
            seqs = [r["seq"] for r in conn.execute(
                "SELECT seq FROM messages WHERE conversation_id = ?", (convo,))]
        assert sorted(seqs) == list(range(1, 9)), seqs


class TestUnusableDatabase:
    """A bad CHAT_DB must degrade to history-off, not break /api/health."""

    def test_a_file_where_the_directory_should_be(self, tmp_path, monkeypatch):
        blocker = tmp_path / "afile"
        blocker.write_text("not a directory")
        monkeypatch.setenv("CHAT_DB", str(blocker / "chat.db"))
        # mkdir raises OSError, not sqlite3.Error — the case that returned a 500.
        assert store.available() is False

    def test_a_path_that_cannot_exist(self, monkeypatch):
        monkeypatch.setenv("CHAT_DB", "/proc/nope/deeper/chat.db")
        assert store.available() is False

    def test_a_non_database_file(self, tmp_path, monkeypatch):
        junk = tmp_path / "junk.db"
        junk.write_bytes(b"this is definitely not sqlite")
        monkeypatch.setenv("CHAT_DB", str(junk))
        assert store.available() is False

    def test_health_stays_json_when_history_is_unusable(self, tmp_path, monkeypatch):
        """/api/health is what the whole UI bootstraps from."""
        import importlib
        import app as app_module

        blocker = tmp_path / "afile"
        blocker.write_text("x")
        monkeypatch.setenv("CHAT_DB", str(blocker / "chat.db"))
        mod = importlib.reload(app_module)
        resp = mod.app.test_client().get("/api/health")
        assert resp.status_code == 200
        assert resp.get_json()["history"] is False


class TestAvailabilityIsAWriteProbe:
    """Opening a database proves almost nothing.

    On an existing file sqlite allocates no page, so a corrupt database opened
    cleanly and available() returned True forever while every write failed —
    reported as history: true on /api/health, with no reload recovering. After
    a power cut on a small always-on box, SQLITE_CORRUPT is the realistic
    trigger, not a full disk.
    """

    def test_a_corrupt_database_is_reported_unavailable(self, tmp_path, monkeypatch):
        bad = tmp_path / "chat.db"
        bad.write_bytes(b"SQLite format 3\x00" + b"\x00" * 200)   # header-shaped, junk body
        monkeypatch.setenv("CHAT_DB", str(bad))
        assert store.available() is False

    def test_a_healthy_database_is_available(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CHAT_DB", str(tmp_path / "chat.db"))
        assert store.available() is True

    def test_the_probe_leaves_nothing_behind(self, tmp_path, monkeypatch):
        path = tmp_path / "chat.db"
        monkeypatch.setenv("CHAT_DB", str(path))
        store.available()
        names = {r[0] for r in sqlite3.connect(str(path)).execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert not any(n.startswith("_probe") for n in names), names

    def test_a_concurrent_write_is_not_mistaken_for_a_failure(self, tmp_path, monkeypatch):
        """Locked means something is writing to it — a sign of health."""
        path = tmp_path / "chat.db"
        monkeypatch.setenv("CHAT_DB", str(path))
        store.available()
        blocker = sqlite3.connect(str(path), timeout=1)
        blocker.execute("BEGIN EXCLUSIVE")
        try:
            started = time.monotonic()
            assert store.available() is True
            # And it must not stall the page load it runs on.
            assert time.monotonic() - started < 3
        finally:
            blocker.rollback()
            blocker.close()


class TestStoreFailuresReachTheClient:
    def test_a_database_error_is_json_not_an_html_500(self, tmp_path, monkeypatch):
        """The client parsed the HTML error page as a failed fetch and ignored
        it, so history silently stopped accumulating while chat looked fine."""
        import importlib
        import app as app_module

        bad = tmp_path / "chat.db"
        bad.write_bytes(b"SQLite format 3\x00" + b"\x00" * 200)
        monkeypatch.setenv("CHAT_DB", str(bad))
        mod = importlib.reload(app_module)

        resp = mod.app.test_client().post("/api/conversations", json={"title": "x"})
        assert resp.status_code == 503
        assert resp.headers["Content-Type"].startswith("application/json")
        body = resp.get_json()
        assert body["history"] is False
        assert "could not be written" in body["error"]

    def test_health_reports_history_as_unavailable(self, tmp_path, monkeypatch):
        import importlib
        import app as app_module

        bad = tmp_path / "chat.db"
        bad.write_bytes(b"SQLite format 3\x00" + b"\x00" * 200)
        monkeypatch.setenv("CHAT_DB", str(bad))
        mod = importlib.reload(app_module)
        assert mod.app.test_client().get("/api/health").get_json()["history"] is False


class TestDiskIsActuallyReclaimed:
    """The README says deleting a conversation reclaims space; make it true.

    SQLite leaves freed pages on the freelist, so deleting twenty image-heavy
    threads left an 18 MB file at 18 MB. And in WAL mode a VACUUM writes the
    compacted database into the WAL — the main file is not truncated until a
    checkpoint, so VACUUM alone appeared to succeed while changing nothing.
    """

    def big_thread(self, n=3):
        img = base64.b64encode(os.urandom(200_000)).decode()
        convo = store.create("image heavy", "m")
        for _ in range(n):
            store.add_message(convo["id"], "user", "look at this", [img], None)
        return convo["id"]

    def test_deleting_image_heavy_threads_shrinks_the_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CHAT_DB", str(tmp_path / "chat.db"))
        ids = [self.big_thread() for _ in range(6)]
        full = (tmp_path / "chat.db").stat().st_size
        assert full > 3 * 1024 * 1024, "fixture did not create enough data"

        for convo_id in ids:
            store.delete(convo_id)

        after = (tmp_path / "chat.db").stat().st_size
        assert after < full / 4, f"{full} -> {after}: the space was not given back"

    def test_a_small_delete_does_not_rewrite_the_database(self, tmp_path, monkeypatch):
        """VACUUM rewrites everything; doing it per delete would be silly."""
        monkeypatch.setenv("CHAT_DB", str(tmp_path / "chat.db"))
        self.big_thread(n=6)                      # bulk that must not be rewritten
        convo = store.create("tiny", "m")
        store.add_message(convo["id"], "user", "hi", None, None)
        before = (tmp_path / "chat.db").stat().st_size

        store.delete(convo["id"])
        assert (tmp_path / "chat.db").stat().st_size == before

    def test_compact_is_safe_on_an_empty_database(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CHAT_DB", str(tmp_path / "chat.db"))
        store.available()
        assert store.compact() == 0

    def test_compact_never_raises_on_a_broken_database(self, tmp_path, monkeypatch):
        bad = tmp_path / "chat.db"
        bad.write_bytes(b"SQLite format 3\x00" + b"\x00" * 200)
        monkeypatch.setenv("CHAT_DB", str(bad))
        assert store.compact() == 0

    def test_the_data_survives_a_compaction(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CHAT_DB", str(tmp_path / "chat.db"))
        keep = store.create("keep me", "m")
        store.add_message(keep["id"], "user", "important", None, None)
        for convo_id in [self.big_thread() for _ in range(6)]:
            store.delete(convo_id)

        got = store.get(keep["id"])
        assert got["title"] == "keep me"
        assert [m["content"] for m in got["messages"]] == ["important"]


class TestStatsAreReachable:
    """stats() existed, was tested, and had no caller anywhere."""

    def test_the_endpoint_reports_what_history_costs(self, tmp_path, monkeypatch):
        import importlib
        import app as app_module

        monkeypatch.setenv("CHAT_DB", str(tmp_path / "chat.db"))
        mod = importlib.reload(app_module)
        convo = store.create("t", "m")
        store.add_message(convo["id"], "user", "hi", None, None)

        data = mod.app.test_client().get("/api/conversations/stats").get_json()
        assert data["conversations"] == 1
        assert data["messages"] == 1
        assert data["bytes"] > 0


class TestOneBadRowCannotCostYouAConversation:
    """Found by poking at the edges of what the API accepts. Each of these
    makes a thread permanently unopenable, which is a far bigger loss than the
    attachment that caused it.
    """

    def test_a_column_that_will_not_parse_is_ignored_not_raised(self):
        """get() raising 500s the conversation route, and then there is no way
        to reach the messages either side of the bad one."""
        convo = store.create("t")["id"]
        store.add_message(convo, "user", "hello", images=["b64"])
        with store._connect() as conn:
            conn.execute("UPDATE messages SET images = '{not json'")
        got = store.get(convo)
        assert [m["role"] for m in got["messages"]] == ["user"]
        assert got["messages"][0]["images"] is None
        assert got["messages"][0]["content"] == "hello", "the words are what matter"

    @pytest.mark.parametrize("value", ["notalist", 42, {"a": 1}, ""])
    def test_only_a_list_reaches_a_list_column(self, value):
        """The browser iterates these. A string comes back out as a string,
        .map is not a function, and every later open of that thread is a page
        error rather than a conversation."""
        convo = store.create("t")["id"]
        store.save_turn(convo, {"content": "q", "images": value,
                                "image_meta": value}, "a")
        first = store.get(convo)["messages"][0]
        assert first["images"] is None and first["image_meta"] is None

    @pytest.mark.parametrize("value", ["notalist", 42, {"a": 1}])
    def test_add_message_is_guarded_the_same_way(self, value):
        convo = store.create("t")["id"]
        store.add_message(convo, "user", "hello", images=value, sources=value)
        first = store.get(convo)["messages"][0]
        assert first["images"] is None and first["sources"] is None

    def test_a_real_list_still_goes_through(self):
        convo = store.create("t")["id"]
        store.save_turn(convo, {"content": "q", "images": ["b64"],
                                "image_meta": [{"taken": "x"}]}, "a")
        first = store.get(convo)["messages"][0]
        assert first["images"] == ["b64"]
        assert first["image_meta"] == [{"taken": "x"}]

    @pytest.mark.parametrize("query", [123, None, object()])
    def test_a_search_for_something_that_is_not_a_phrase(self, query):
        assert store.search(query) == {"conversations": [], "records": []}

class TestUpgradingWhileTwoDevicesReconnect:
    """The first moment after an update is the one where both devices come
    back at once, and it is the only moment the migration has anything to do.

    Measured before the fix: 12 threads opening an un-migrated database, four
    raised OperationalError("duplicate column name: image_meta"). That is a
    sqlite3.Error, which app.py's handler reports as "conversations are no
    longer being saved" — a successful migration announcing itself as a broken
    database.
    """

    OLD_SCHEMA = """
    CREATE TABLE conversations (id TEXT PRIMARY KEY, title TEXT NOT NULL,
      model TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL);
    CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT,
      conversation_id TEXT NOT NULL, seq INTEGER NOT NULL, role TEXT NOT NULL,
      content TEXT NOT NULL, images TEXT, sources TEXT, created_at REAL NOT NULL);
    CREATE TABLE routines (id TEXT PRIMARY KEY, name TEXT NOT NULL,
      body TEXT NOT NULL, photos INTEGER NOT NULL DEFAULT 0, web INTEGER,
      photo_meta INTEGER, position REAL NOT NULL, created_at REAL NOT NULL,
      updated_at REAL NOT NULL);
    """

    def old_database(self, tmp_path, monkeypatch):
        path = tmp_path / "old.db"
        conn = sqlite3.connect(str(path))
        conn.executescript(self.OLD_SCHEMA)
        conn.execute("INSERT INTO conversations VALUES ('c1','kept','m',0,0)")
        conn.commit()
        conn.close()
        monkeypatch.setenv("CHAT_DB", str(path))
        return path

    def test_a_crowd_of_them_does_not_collide(self, tmp_path, monkeypatch):
        """The scenario, end to end. It asserts on the collision specifically
        rather than on "no errors at all": sixteen threads contending for one
        sqlite file can also hit an ordinary busy timeout, which is a different
        condition the app handles elsewhere, and folding the two together made
        this fail about one run in five under load."""
        self.old_database(tmp_path, monkeypatch)
        errors = []
        ready = threading.Barrier(12)

        def hit():
            ready.wait()
            try:
                store.list_conversations()
            except Exception as exc:            # noqa: BLE001
                errors.append(str(exc))

        threads = [threading.Thread(target=hit) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        collided = [e for e in errors if "duplicate column" in e.lower()]
        assert not collided, collided

    def test_and_the_columns_really_did_arrive(self, tmp_path, monkeypatch):
        path = self.old_database(tmp_path, monkeypatch)
        store.list_conversations()
        conn = sqlite3.connect(str(path))
        try:
            names = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
        finally:
            conn.close()
        assert {"image_meta", "thinking", "steps"} <= names

    def test_and_nothing_that_was_there_was_lost(self, tmp_path, monkeypatch):
        self.old_database(tmp_path, monkeypatch)
        assert [c["title"] for c in store.list_conversations()] == ["kept"]

    def test_losing_the_race_deterministically_is_not_an_error(self, tmp_path, monkeypatch):
        """The sixteen-thread test above only *usually* collides, so on its own
        it would let the fix be removed without complaint. This one is the
        losing thread exactly: another connection adds the column in the gap
        between this one's check and its ALTER.
        """
        path = self.old_database(tmp_path, monkeypatch)

        class Racy(sqlite3.Connection):
            def execute(self, sql, *args, **kwargs):
                out = super().execute(sql, *args, **kwargs)
                if "table_info(messages)" in str(sql):
                    # Read the rows first. A sqlite cursor is lazy, so leaving
                    # it unread meant the check below saw the column the racing
                    # connection was about to add — and skipped the ALTER, which
                    # is the collision this is trying to cause.
                    rows = out.fetchall()
                    other = sqlite3.connect(str(path))
                    try:
                        other.execute("ALTER TABLE messages ADD COLUMN image_meta TEXT")
                        other.commit()
                    except sqlite3.OperationalError:
                        pass          # a later pass; already added
                    finally:
                        other.close()
                    return rows
                return out

        conn = sqlite3.connect(str(path), factory=Racy)
        conn.row_factory = sqlite3.Row
        try:
            store._migrate(conn)          # must not raise
            conn.commit()
        finally:
            conn.close()

    def test_but_a_real_migration_failure_is_still_raised(self, tmp_path, monkeypatch):
        """Swallowing "duplicate column" is the point; swallowing a full disk
        or a read-only file would hide the thing worth knowing."""
        self.old_database(tmp_path, monkeypatch)
        # A column sqlite will refuse for a different reason entirely.
        # sqlite refuses to add a UNIQUE column to an existing table, which is
        # an OperationalError that is emphatically not "duplicate column".
        monkeypatch.setattr(store, "_ADDED_COLUMNS",
                            (("messages", "nope", "TEXT UNIQUE"),))
        with pytest.raises(sqlite3.OperationalError) as caught:
            store.list_conversations()
        assert "duplicate column" not in str(caught.value).lower()
