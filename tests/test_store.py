"""Tests for conversation storage."""

from __future__ import annotations

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
