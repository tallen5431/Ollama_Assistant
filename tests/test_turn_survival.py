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
        assert app_module._message_field(line, "content") == text

    @pytest.mark.parametrize("line, text", [
        ('{"message": {"thinking": "hmm"}}', "hmm"),
        ('{"message": {"content": "hi"}}', ""),
        ("not json at all", ""),
    ])
    def test_and_the_reasoning_is_read_the_same_way(self, line, text):
        assert app_module._message_field(line, "thinking") == text


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


class TestAskingWhetherATurnIsStillRunning:
    """A phone that goes into a pocket mid-reply leaves a turn running that it
    can no longer see. Without a way to ask, the page cannot tell that from a
    server that has died — both are a stream that stopped arriving — and the
    safe reading of the second (discard the turn) throws away the first.
    """

    def running(self, client, convo_id):
        return client.get("/api/chat/status",
                          query_string={"conversation_id": convo_id}).get_json()["running"]

    def test_it_says_yes_while_one_is_going(self, client):
        app_module._start_turn("cid-live")
        assert self.running(client, "cid-live") is True

    def test_and_no_once_it_has_finished(self, client):
        flag = app_module._start_turn("cid-done")
        app_module._end_turn("cid-done", flag)
        assert self.running(client, "cid-done") is False

    def test_a_thread_that_never_had_one_is_not_running(self, client):
        assert self.running(client, "never-heard-of-it") is False

    def test_and_neither_is_no_thread_at_all(self, client):
        """A page with history off has no id to ask about, and asking anyway
        must not be an error."""
        resp = client.get("/api/chat/status")
        assert resp.status_code == 200
        assert resp.get_json()["running"] is False

    def test_cancelling_leaves_it_running_until_the_turn_actually_ends(self, client):
        """Stop sets the flag; the producer notices at its next line. Saying
        "not running" the instant Stop is pressed would have the page conclude
        the reply died, when it is about to be saved."""
        app_module._start_turn("cid-cancelled")
        client.post("/api/chat/cancel", json={"conversation_id": "cid-cancelled"})
        assert self.running(client, "cid-cancelled") is True

    def test_it_never_says_no_before_the_reply_is_readable(self, client, monkeypatch):
        """The pair a reconnecting page reads as "the reply did not survive":
        the thread has nothing in it *and* nothing is running. Deregistering
        the turn before committing it opened exactly that window — small, but
        the page polls straight into it, and the reply is in the database
        milliseconds later.

        The save is made slow so the window, if there is one, is wide enough to
        observe rather than raced for.
        """
        convo = store.create("")["id"]
        real = store.save_turn
        seen = []

        def slow(*a, **kw):
            # Where a reconnecting page would look, at the worst moment.
            seen.append((self.running(client, convo),
                         len(store.get(convo)["messages"])))
            time.sleep(0.2)
            return real(*a, **kw)

        monkeypatch.setattr(app_module.store, "save_turn", slow)
        monkeypatch.setattr(app_module, "chat_stream", _FakeOllama(["68 miles."]))
        resp = client.post("/api/chat", json={
            "model": "m", "messages": [{"role": "user", "content": "how far?"}],
            "conversation_id": convo})
        resp.get_data()
        assert seen, "the save never ran"
        for running, stored in seen:
            assert running or stored, \
                "asked mid-save: nothing running and nothing stored — the page " \
                "reads that as a lost reply"

    def test_it_is_behind_the_same_guard_as_everything_else(self, monkeypatch):
        monkeypatch.setenv("CHAT_AUTH", "tj:hunter2")
        guarded = importlib.reload(app_module).app.test_client()
        assert guarded.get("/api/chat/status?conversation_id=x").status_code == 401


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


class TestThePanelsSurviveTheDevice:
    """Reported: a chat done on the phone, opened on the desktop, had the reply
    and neither panel. They were live stream state only — and they are exactly
    what you go looking for when an answer reads oddly, which is usually later
    and somewhere else.
    """

    def test_both_are_stored_with_the_reply(self):
        convo = store.create("t")["id"]
        store.save_turn(convo, {"content": "q"}, "a",
                        thinking="weighing it up",
                        steps=[{"step": "Photo details", "detail": "none"}])
        kept = store.get(convo)["messages"][1]
        assert kept["thinking"] == "weighing it up"
        assert kept["steps"] == [{"step": "Photo details", "detail": "none"}]

    def test_a_turn_with_neither_stores_neither(self):
        convo = store.create("t")["id"]
        store.save_turn(convo, {"content": "q"}, "a")
        kept = store.get(convo)["messages"][1]
        assert kept["thinking"] is None and kept["steps"] is None

    def test_a_runaway_scratchpad_is_capped(self):
        """A reasoning model on a hard question, times a thread of them, is
        otherwise the database."""
        convo = store.create("t")["id"]
        store.save_turn(convo, {"content": "q"}, "a", thinking="x" * 60_000)
        assert len(store.get(convo)["messages"][1]["thinking"]) == store._THINKING_MAX

    def test_the_route_collects_them_from_the_stream(self, client, monkeypatch):
        def reasoning(model, messages, **kw):
            yield json.dumps({"message": {"thinking": "let me see. "}})
            yield json.dumps({"message": {"thinking": "right."}})
            yield json.dumps({"message": {"content": "the answer"}})
            yield json.dumps({"done": True})
        monkeypatch.setattr(app_module, "chat_stream", reasoning)
        convo = store.create("t")["id"]
        client.post("/api/chat", json={
            "model": "m", "conversation_id": convo,
            "messages": [{"role": "user", "content": "q", "images": ["b64"]}]}).get_data()
        for _ in range(60):
            time.sleep(0.05)
            if len(store.get(convo)["messages"]) == 2:
                break
        kept = store.get(convo)["messages"][1]
        assert kept["thinking"] == "let me see. right."
        assert [s["step"] for s in kept["steps"]] == \
            ["Photo details", "Images", "Sent to the model"]

    def test_a_step_added_later_is_kept_without_anyone_remembering_to(self):
        """Recorded where the lines are relayed, not at each of the thirteen
        yields — so the next one is kept for free."""
        js = page_script(chat_ui.render_page("t"))
        assert "for (const entry of msg.steps || []) addStep(view, entry);" in js

    def test_reopening_a_thread_paints_them_back(self):
        js = page_script(chat_ui.render_page("t"))
        at = js.index("if (msg.thinking) {")
        window = js[at:at + 300]
        assert "view.thinkBody.textContent = msg.thinking;" in window
        assert "view.think.hidden = false;" in window


class TestTheDetachedStreamItself:
    """One helper now carries both the chat turn and the voice download, so its
    edges are worth pinning down rather than rediscovering per caller.

    Its other contract — that a producer's own ``finally`` completes before the
    stream closes, which is what lets a turn be saved before the browser is
    told the reply ended — is enforced by
    ``TestTheRouteWritesTheTurn::test_a_finished_reply_is_stored``. Asserting it
    here would only restate the try/finally that provides it.
    """

    def emit_lines(self, produce):
        import app as mod
        with mod.app.test_request_context("/"):
            resp = mod._detached_stream(produce, "test")
            return b"".join(x.encode() if isinstance(x, str) else x
                            for x in resp.response).decode()

    def test_what_is_emitted_is_what_arrives(self):
        assert self.emit_lines(lambda emit: [emit("a\n"), emit("b\n")]) == "a\nb\n"

    def test_a_producer_that_raises_says_so_rather_than_stopping_dead(self):
        """A thread has nowhere to raise to: without this the client sees a
        stream that simply ends, which reads as a reply that was cut short."""
        def boom(emit):
            emit("first\n")
            raise RuntimeError("producer died")
        out = self.emit_lines(boom)
        assert out.startswith("first\n")
        assert "error" in out.splitlines()[-1]

    def test_producing_nothing_is_an_empty_stream_not_a_hang(self):
        assert self.emit_lines(lambda emit: None) == ""

    def test_a_quiet_producer_still_writes_something(self, monkeypatch):
        """Measured: six turns against an Ollama that accepts and never answers
        took all four of waitress's worker threads, and closing all six clients
        gave back none of them — /healthz stopped answering for the length of
        the Ollama timeout, which the server manager's card reads as the app
        being down. A departed client cannot be detected under waitress except
        by writing to it, so the relay writes when the producer is quiet and
        lets the write do the detecting.
        """
        import app as mod
        monkeypatch.setattr(mod, "_HEARTBEAT_SECONDS", 0.05)

        def slow(emit):
            time.sleep(0.3)
            emit("late\n")

        out = self.emit_lines(slow)
        assert out.endswith("late\n")
        assert out.count("\n") > 1, "nothing was written while the producer was quiet"
        assert out.replace("\n", "") == "late", "the filler was not blank lines"

    def test_and_the_filler_is_something_the_page_already_ignores(self):
        """A blank line, because both NDJSON readers in the page skip one
        explicitly — adding a message shape instead would mean a client that
        has to learn about it."""
        script = page_script(chat_ui.render_page("t"))
        assert script.count("if (!line) continue;") + script.count("if (!raw) continue;") >= 2

    def test_a_talkative_producer_is_never_padded(self):
        """An ordinary turn streams tokens continuously, so the heartbeat must
        not put anything in the middle of one."""
        assert self.emit_lines(lambda emit: [emit("a\n"), emit("b\n")]) == "a\nb\n"


class TestTurnsDoNotTreadOnEachOther:
    def test_three_at_once_still_alternate(self, client, monkeypatch):
        """One conversation, three tabs. Half a turn, or two questions in a
        row, is what a missing transaction looks like from the drawer."""
        def slow(model, messages, **kw):
            for i in range(4):
                time.sleep(0.02)
                yield json.dumps({"message": {"content": f"[{i}]"}})
            yield json.dumps({"done": True})
        monkeypatch.setattr(app_module, "chat_stream", slow)
        convo = store.create("race")["id"]

        def fire(tag):
            client.post("/api/chat", json={
                "model": "m", "conversation_id": convo,
                "messages": [{"role": "user", "content": tag}]}).get_data()

        threads = [threading.Thread(target=fire, args=(f"q{i}",)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for _ in range(80):
            time.sleep(0.05)
            if len(store.get(convo)["messages"]) >= 6:
                break
        roles = [m["role"] for m in store.get(convo)["messages"]]
        assert len(roles) % 2 == 0, f"half a turn was written: {roles}"
        assert roles == ["user", "assistant"] * (len(roles) // 2), roles

    def test_the_cancel_registry_does_not_grow(self):
        before = len(app_module._TURNS)
        for i in range(50):
            flag = app_module._start_turn(f"c{i}")
            app_module._end_turn(f"c{i}", flag)
        assert len(app_module._TURNS) == before

    def test_an_old_turn_ending_leaves_its_replacement_alone(self):
        """Otherwise the second tab's Stop button stops nothing."""
        first = app_module._start_turn("shared")
        second = app_module._start_turn("shared")
        app_module._end_turn("shared", first)
        assert app_module._TURNS.get("shared") is second
        app_module._end_turn("shared", second)
        assert "shared" not in app_module._TURNS
