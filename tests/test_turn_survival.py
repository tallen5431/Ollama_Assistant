"""A reply must outlive the tab that asked for it.

The browser used to write the turn down once the stream finished. That is fine
right up until the tab is not there when it finishes — a phone that locks past
its wake lock, an app switch, a handoff from wifi to cellular — and then two
minutes of a 30b model's work existed nowhere at all.

Two halves to the fix, and both are load-bearing. The server writes the turn,
because it is still there. And it generates on a thread rather than inside the
response, because a WSGI response is *pulled*: with nobody pulling, the old
arrangement kept only the first sentence.
"""

from __future__ import annotations

import importlib
import json
import socket
import threading
import time

import pytest

import app as app_module
import chat_ui
import store
from conftest import page_script


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAT_DB", str(tmp_path / "chat.db"))
    yield


@pytest.fixture
def client(monkeypatch):
    for key in ("CHAT_AUTH", "CHAT_AUTH_USER", "CHAT_AUTH_PASSWORD"):
        monkeypatch.delenv(key, raising=False)
    return importlib.reload(app_module).app.test_client()


# --------------------------------------------------------------------------
# The store: both halves or neither
# --------------------------------------------------------------------------

class TestSaveTurn:
    def test_a_turn_is_written_whole(self):
        convo = store.create("")["id"]
        assert store.save_turn(convo, {"content": "how far?"}, "68 miles.") is True
        roles = [m["role"] for m in store.get(convo)["messages"]]
        assert roles == ["user", "assistant"]

    def test_the_question_keeps_its_photos_and_their_details(self):
        convo = store.create("")["id"]
        store.save_turn(convo, {"content": "read this", "images": ["b64"],
                                "image_meta": [{"taken": "2026:08:08 09:00:00"}]},
                        "It says 10,500.")
        first = store.get(convo)["messages"][0]
        assert first["images"] == ["b64"]
        assert first["image_meta"][0]["taken"] == "2026:08:08 09:00:00"

    def test_an_empty_reply_writes_nothing(self):
        """Half a turn reads as an unanswered question, not a dropped line."""
        convo = store.create("")["id"]
        assert store.save_turn(convo, {"content": "hi"}, "   ") is False
        assert store.get(convo)["messages"] == []

    def test_an_unknown_conversation_writes_nothing(self):
        assert store.save_turn("nope", {"content": "hi"}, "there") is False

    def test_it_names_an_untitled_thread(self):
        convo = store.create("")["id"]
        store.save_turn(convo, {"content": "how far did I drive?"}, "68 miles.")
        assert store.get(convo)["title"] == "how far did I drive?"

    def test_it_leaves_a_named_thread_alone(self):
        convo = store.create("🚗 Trip")["id"]
        store.save_turn(convo, {"content": "how far?"}, "68 miles.")
        assert store.get(convo)["title"] == "🚗 Trip"

    def test_turns_keep_arriving_in_order(self):
        convo = store.create("t")["id"]
        for i in range(3):
            store.save_turn(convo, {"content": f"q{i}"}, f"a{i}")
        said = [m["content"] for m in store.get(convo)["messages"]]
        assert said == ["q0", "a0", "q1", "a1", "q2", "a2"]

    def test_is_empty_reports_what_the_browser_needs_to_tidy_up(self):
        convo = store.create("t")["id"]
        assert store.is_empty(convo) is True
        store.save_turn(convo, {"content": "q"}, "a")
        assert store.is_empty(convo) is False


class TestWhatGetsStored:
    def test_a_reasoning_block_is_not_the_answer(self):
        """It is the model talking to itself; a thread reopened tomorrow should
        read the way it read today."""
        convo = store.create("t")["id"]
        app_module._keep_turn(convo, {"content": "2+2?"},
                              "<think>carry the one</think>It is 4.", [])
        kept = store.get(convo)["messages"][1]["content"]
        assert kept == "It is 4."

    def test_a_reply_that_was_only_reasoning_is_not_a_reply(self):
        convo = store.create("t")["id"]
        app_module._keep_turn(convo, {"content": "q"}, "<think>hmm</think>", [])
        assert store.get(convo)["messages"] == []

    def test_no_conversation_means_no_write(self):
        app_module._keep_turn(None, {"content": "q"}, "a", [])   # must not raise

    def test_a_broken_store_costs_the_record_not_the_turn(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("disk is full")
        monkeypatch.setattr(app_module.store, "save_turn", boom)
        app_module._keep_turn("cid", {"content": "q"}, "a", [])  # must not raise

    def test_the_sources_go_with_it(self):
        convo = store.create("t")["id"]
        app_module._keep_turn(convo, {"content": "q"}, "a",
                              [{"url": "https://example.com", "title": "Ex"}])
        assert store.get(convo)["messages"][1]["sources"][0]["url"] == "https://example.com"

    @pytest.mark.parametrize("line, text", [
        ('{"message": {"content": "hi"}}', "hi"),
        ('{"message": {"thinking": "hmm"}}', ""),
        ('{"done": true}', ""),
        ("not json at all", ""),
        ('{"message": "a string"}', ""),
    ])
    def test_only_reply_text_is_accumulated(self, line, text):
        assert app_module._content_of(line) == text


# --------------------------------------------------------------------------
# The route
# --------------------------------------------------------------------------

class _FakeOllama:
    """A model that answers slowly, so there is a window to disappear inside."""

    def __init__(self, chunks, delay=0.0):
        self.chunks = chunks
        self.delay = delay
        self.produced = 0

    def __call__(self, model, messages, **kw):
        for chunk in self.chunks:
            self.produced += 1
            time.sleep(self.delay)
            yield json.dumps({"message": {"content": chunk}})
        yield json.dumps({"done": True})


class TestTheRouteWritesTheTurn:
    def test_a_finished_reply_is_stored(self, client, monkeypatch):
        monkeypatch.setattr(app_module, "chat_stream", _FakeOllama(["68 ", "miles."]))
        convo = store.create("")["id"]
        resp = client.post("/api/chat", json={
            "model": "m", "messages": [{"role": "user", "content": "how far?"}],
            "conversation_id": convo})
        assert resp.status_code == 200
        resp.get_data()
        said = [m["content"] for m in store.get(convo)["messages"]]
        assert said == ["how far?", "68 miles."]

    def test_without_a_thread_nothing_is_stored(self, client, monkeypatch):
        """An API client and a history-less browser both take this path."""
        monkeypatch.setattr(app_module, "chat_stream", _FakeOllama(["hi"]))
        resp = client.post("/api/chat", json={
            "model": "m", "messages": [{"role": "user", "content": "q"}]})
        resp.get_data()
        assert store.list_conversations() == []

    def test_a_failure_mid_stream_still_keeps_what_was_said(self, client, monkeypatch):
        def half(model, messages, **kw):
            yield json.dumps({"message": {"content": "It is "}})
            raise RuntimeError("the GPU fell over")
        monkeypatch.setattr(app_module, "chat_stream", half)
        convo = store.create("t")["id"]
        client.post("/api/chat", json={
            "model": "m", "messages": [{"role": "user", "content": "q"}],
            "conversation_id": convo}).get_data()
        said = [m["content"] for m in store.get(convo)["messages"]]
        assert said == ["q", "It is"], "a half-finished sentence beats nothing"


class TestCancel:
    def test_stop_names_the_thread_and_the_flag_is_set(self, client):
        flag = app_module._start_turn("cid-1")
        assert client.post("/api/chat/cancel",
                           json={"conversation_id": "cid-1"}).get_json()["cancelled"]
        assert flag.is_set()

    def test_stopping_something_that_is_not_running_is_not_an_error(self, client):
        assert client.post("/api/chat/cancel",
                           json={"conversation_id": "nope"}).get_json()["cancelled"] is False
        assert client.post("/api/chat/cancel", json={}).status_code == 200
        assert client.post("/api/chat/cancel", data="[]",
                           content_type="application/json").status_code == 200

    def test_a_second_turn_supersedes_the_first(self, client):
        """Two tabs on one thread; the older producer must not outlive it."""
        first = app_module._start_turn("cid-2")
        app_module._start_turn("cid-2")
        assert first.is_set()

    def test_finishing_clears_the_slot(self, client):
        flag = app_module._start_turn("cid-3")
        app_module._end_turn("cid-3", flag)
        assert client.post("/api/chat/cancel",
                           json={"conversation_id": "cid-3"}).get_json()["cancelled"] is False


# --------------------------------------------------------------------------
# The part a test client cannot show: nobody is pulling
# --------------------------------------------------------------------------

class TestItKeepsGoingWithNobodyListening:
    """The measured failure. Flask's test client drains the whole response, so
    this needs a real socket that reads two lines and then hangs up.
    """

    def _serve(self, monkeypatch, chunks, delay):
        from werkzeug.serving import make_server
        fake = _FakeOllama(chunks, delay)
        monkeypatch.setattr(app_module, "chat_stream", fake)
        srv = make_server("127.0.0.1", 0, app_module.app, threaded=True)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv, fake

    def test_the_whole_reply_survives_a_client_that_walks_away(self, monkeypatch):
        convo = store.create("t")["id"]
        srv, fake = self._serve(monkeypatch, [f"[{i}]" for i in range(8)], 0.08)
        body = json.dumps({"model": "m", "conversation_id": convo,
                           "messages": [{"role": "user", "content": "q"}]}).encode()
        sock = socket.create_connection(("127.0.0.1", srv.port))
        sock.sendall(b"POST /api/chat HTTP/1.1\r\nHost: x\r\n"
                     b"Content-Type: application/json\r\n"
                     b"Content-Length: %d\r\nConnection: close\r\n\r\n" % len(body) + body)
        sock.recv(400)          # headers and the first chunk or two
        sock.close()            # the phone locks

        for _ in range(60):     # the model keeps going without us
            time.sleep(0.1)
            messages = store.get(convo)["messages"]
            if len(messages) == 2:
                break
        messages = store.get(convo)["messages"]
        assert [m["role"] for m in messages] == ["user", "assistant"]
        assert messages[1]["content"] == "".join(f"[{i}]" for i in range(8)), \
            "only the part already sent survived — the work is still tied to the socket"
        srv.shutdown()

    def test_stop_really_stops_it(self, monkeypatch):
        """Detaching the work from the connection means an abort no longer ends
        it, so Stop has to say so out loud or a 30b model runs on for a minute.
        """
        convo = store.create("t")["id"]
        srv, fake = self._serve(monkeypatch, [f"[{i}]" for i in range(40)], 0.05)
        body = json.dumps({"model": "m", "conversation_id": convo,
                           "messages": [{"role": "user", "content": "q"}]}).encode()
        sock = socket.create_connection(("127.0.0.1", srv.port))
        sock.sendall(b"POST /api/chat HTTP/1.1\r\nHost: x\r\n"
                     b"Content-Type: application/json\r\n"
                     b"Content-Length: %d\r\nConnection: close\r\n\r\n" % len(body) + body)
        sock.recv(400)
        time.sleep(0.3)
        app_module.app.test_client().post("/api/chat/cancel",
                                          json={"conversation_id": convo})
        sock.close()
        time.sleep(1.5)
        assert fake.produced < 40, "cancelling did not reach the producer"
        kept = store.get(convo)["messages"]
        assert len(kept) == 2, "Stop should still keep the half you read"
        srv.shutdown()


# --------------------------------------------------------------------------
# The browser's side of the bargain
# --------------------------------------------------------------------------

class TestTheBrowserHandsOver:
    def page(self):
        return chat_ui.render_page("t")

    def test_the_thread_exists_before_the_stream_starts(self):
        """The server has to have somewhere to write it."""
        js = page_script(self.page())
        at = js.index("async function send()")
        window = js[at:js.index("const view = addAssistant();", at)]
        assert "await ensureConversation(text);" in window

    def test_the_thread_id_goes_with_the_request(self):
        assert "conversation_id: currentConvoId || undefined," in self.page()

    def test_the_browser_no_longer_writes_messages_itself(self):
        """Two writers meant two copies of every turn."""
        assert "saveMessage(" not in page_script(self.page())

    def test_a_turn_that_produced_nothing_takes_its_thread_with_it(self):
        js = page_script(self.page())
        at = js.index("async function dropIfEmpty")
        window = js[at:at + 700]
        assert 'method: "DELETE"' in window
        assert "if (!createdThread || !currentConvoId) return;" in window

    def test_a_thread_that_was_already_there_is_never_deleted(self):
        js = page_script(self.page())
        assert "const createdThread = !hadThread && !!currentConvoId;" in js

    def test_stop_tells_the_server_too(self):
        js = page_script(self.page())
        at = js.index("function stop()")
        window = js[at:at + 700]
        assert '"api/chat/cancel"' in window
        assert "keepalive: true" in window
        assert "controller.abort();" in window
