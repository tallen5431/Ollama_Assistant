"""Managing models from the page: what is loaded, downloading, deleting, unloading.

Downloading fills the desktop's disk and deleting cannot be undone, so most of
what is tested here is who may do it, and that a refusal is a clear answer
rather than a half-done change.
"""

from __future__ import annotations

import importlib
import json
import threading

import pytest

import app as app_module
import chat_ui
import ollama_client


@pytest.fixture
def client(monkeypatch):
    for key in ("CHAT_AUTH", "CHAT_AUTH_USER", "CHAT_AUTH_PASSWORD", "MODEL_ADMIN"):
        monkeypatch.delenv(key, raising=False)
    module = importlib.reload(app_module)
    module._PULLS.clear()
    return module.app.test_client()


@pytest.fixture
def ollama(monkeypatch):
    """A stand-in for the model calls, recording what was asked."""
    calls = {"deleted": [], "unloaded": [], "pulled": []}
    state = {"loaded": [{"name": "qwen3-coder:30b", "size": 19_000, "size_vram": 19_000,
                         "expires_at": "2026-10-26T18:45:00Z"}],
             "pull": [{"status": "pulling manifest"},
                      {"status": "pulling abc", "total": 200, "completed": 100},
                      {"status": "pulling abc", "total": 200, "completed": 200},
                      {"status": "success"}]}

    def pull(name):
        calls["pulled"].append(name)
        for line in state["pull"]:
            if "error" in line:
                raise ValueError(line["error"])
            yield line

    def delete(name):
        if name == "missing:1b":
            raise ValueError("missing:1b is not installed")
        calls["deleted"].append(name)

    monkeypatch.setattr(ollama_client, "loaded_models", lambda: list(state["loaded"]))
    monkeypatch.setattr(ollama_client, "pull_model", pull)
    monkeypatch.setattr(ollama_client, "delete_model", delete)
    monkeypatch.setattr(ollama_client, "unload_model", lambda name: calls["unloaded"].append(name))
    return {"calls": calls, "state": state}


LAN = {"REMOTE_ADDR": "192.168.1.20"}
TAILNET = {"REMOTE_ADDR": "100.101.102.103"}
INTERNET = {"REMOTE_ADDR": "8.8.8.8"}


def post(client, path, body, env=LAN, **kw):
    return client.post(path, json=body, environ_base=env, **kw)


def lines(resp):
    return [json.loads(l) for l in resp.get_data(as_text=True).splitlines() if l.strip()]


class TestNames:
    @pytest.mark.parametrize("name", ["qwen3:30b", "llama3.2", "hf.co/user/repo:Q4_K_M",
                                      "library/qwen3:4b", "nomic-embed-text"])
    def test_real_names(self, name):
        assert ollama_client.valid_model_name(name)

    @pytest.mark.parametrize("name", ["", "a b", "../x", "x;rm -rf", "a:b:c", "-flag",
                                      "x" * 201, "a//b"])
    def test_not_names(self, name):
        assert not ollama_client.valid_model_name(name)

    def test_a_bad_name_is_refused_before_ollama_hears_it(self, client, ollama):
        resp = post(client, "/api/models/pull", {"name": "a b"})
        assert resp.status_code == 400
        assert ollama["calls"]["pulled"] == []


class TestWhoMayChangeThings:
    @pytest.mark.parametrize("env", [LAN, TAILNET, {"REMOTE_ADDR": "127.0.0.1"}])
    def test_this_machine_the_lan_and_the_tailnet(self, client, ollama, env):
        assert post(client, "/api/models/unload", {"name": "qwen3:4b"}, env).status_code == 200

    def test_not_the_internet(self, client, ollama):
        resp = post(client, "/api/models/delete", {"name": "qwen3:4b"}, INTERNET)
        assert resp.status_code == 403
        assert "Basic Auth" in resp.get_json()["error"]
        assert ollama["calls"]["deleted"] == []

    def test_an_address_reported_by_the_proxy_counts(self, client, ollama):
        """Behind the reverse proxy every request arrives from 127.0.0.1; the
        visitor is the one in X-Forwarded-For."""
        resp = client.post("/api/models/delete", json={"name": "qwen3:4b"},
                           environ_base={"REMOTE_ADDR": "127.0.0.1"},
                           headers={"X-Forwarded-For": "8.8.8.8"})
        assert resp.status_code == 403

    def test_anyone_once_they_have_logged_in(self, client, ollama, monkeypatch):
        monkeypatch.setattr(app_module, "AUTH_ENABLED", True)
        assert app_module._may_manage_models.__module__  # the gate reads the flag live
        with app_module.app.test_request_context(environ_base=INTERNET):
            assert app_module._may_manage_models()[0]

    def test_off_means_nobody(self, client, ollama, monkeypatch):
        monkeypatch.setenv("MODEL_ADMIN", "off")
        resp = post(client, "/api/models/pull", {"name": "qwen3:4b"})
        assert resp.status_code == 403 and "MODEL_ADMIN=off" in resp.get_json()["error"]

    def test_on_means_anyone(self, client, ollama, monkeypatch):
        monkeypatch.setenv("MODEL_ADMIN", "on")
        assert post(client, "/api/models/unload", {"name": "x"}, INTERNET).status_code == 200

    def test_only_json(self, client, ollama):
        """A plain form on another site can POST cross-origin without asking."""
        resp = client.post("/api/models/delete", data={"name": "qwen3:4b"}, environ_base=LAN)
        assert resp.status_code == 415
        assert ollama["calls"]["deleted"] == []

    def test_the_panel_is_told_whether_it_may(self, client, ollama):
        assert client.get("/api/models/status", environ_base=LAN).get_json()["manage"] is True
        out = client.get("/api/models/status", environ_base=INTERNET).get_json()
        assert out["manage"] is False and out["manage_reason"]


class TestDownloading:
    def test_progress_then_done(self, client, ollama):
        out = lines(post(client, "/api/models/pull", {"name": "qwen3:4b"}))
        percents = [o.get("percent") for o in out if o.get("percent") is not None]
        assert percents == [50, 100]
        assert out[-1] == {"done": True, "name": "qwen3:4b"}
        assert ollama["calls"]["pulled"] == ["qwen3:4b"]

    def test_ollamas_reason_is_passed_on(self, client, ollama):
        ollama["state"]["pull"] = [{"error": "pull model manifest: file does not exist"}]
        out = lines(post(client, "/api/models/pull", {"name": "nope:1b"}))
        assert out[-1] == {"error": "pull model manifest: file does not exist"}
        status = client.get("/api/models/status", environ_base=LAN).get_json()
        assert status["pulls"][0]["error"].startswith("pull model manifest")

    def test_one_already_under_way_is_not_started_twice(self, client, ollama, monkeypatch):
        gate = threading.Event()

        def slow(name):
            yield {"status": "pulling", "total": 10, "completed": 1}
            gate.wait(5)
        monkeypatch.setattr(ollama_client, "pull_model", slow)
        first = post(client, "/api/models/pull", {"name": "qwen3:4b"})
        try:
            for _ in range(50):
                status = client.get("/api/models/status", environ_base=LAN).get_json()
                if status["pulls"] and status["pulls"][0].get("percent") == 10:
                    break
                threading.Event().wait(0.05)
            assert status["pulls"][0]["name"] == "qwen3:4b", "progress is visible elsewhere"
            again = post(client, "/api/models/pull", {"name": "qwen3:4b"})
            assert again.status_code == 409
        finally:
            gate.set()
            first.get_data()


class TestDeletingAndUnloading:
    def test_delete(self, client, ollama):
        resp = post(client, "/api/models/delete", {"name": "qwen3:4b"})
        assert resp.status_code == 200 and ollama["calls"]["deleted"] == ["qwen3:4b"]

    def test_deleting_what_is_not_there(self, client, ollama):
        resp = post(client, "/api/models/delete", {"name": "missing:1b"})
        assert resp.status_code == 502 and "not installed" in resp.get_json()["error"]

    def test_unload_one(self, client, ollama):
        post(client, "/api/models/unload", {"name": "qwen3-coder:30b"})
        assert ollama["calls"]["unloaded"] == ["qwen3-coder:30b"]

    def test_free_everything(self, client, ollama):
        ollama["state"]["loaded"].append({"name": "llama3.1:8b"})
        out = post(client, "/api/models/unload", {"all": True}).get_json()
        assert out["unloaded"] == ["qwen3-coder:30b", "llama3.1:8b"]
        assert ollama["calls"]["unloaded"] == ["qwen3-coder:30b", "llama3.1:8b"]

    def test_free_everything_is_gated_too(self, client, ollama):
        assert post(client, "/api/models/unload", {"all": True}, INTERNET).status_code == 403
        assert ollama["calls"]["unloaded"] == []


class TestStatus:
    def test_what_is_loaded_and_what_each_model_is_for(self, client, ollama, monkeypatch):
        monkeypatch.setenv("OLLAMA_MODEL", "qwen3-coder:30b")
        monkeypatch.setenv("WEB_PLANNER_MODEL", "qwen3:4b")
        monkeypatch.setenv("WEB_DISTILLER_MODEL", "qwen3:4b")
        out = client.get("/api/models/status", environ_base=LAN).get_json()
        assert out["loaded"][0]["name"] == "qwen3-coder:30b"
        assert out["loaded"][0]["size_vram"] == 19_000
        assert out["roles"]["qwen3-coder:30b"] == ["default answer"]
        assert out["roles"]["qwen3:4b"] == ["search planner", "page distiller"]

    def test_ollama_asleep_is_an_answer_not_a_500(self, client, monkeypatch):
        def asleep():
            raise ValueError("Could not reach Ollama")
        monkeypatch.setattr(ollama_client, "loaded_models", asleep)
        resp = client.get("/api/models/status", environ_base=LAN)
        assert resp.status_code == 200
        assert resp.get_json()["error"] == "Could not reach Ollama"


class TestThePanel:
    def page(self):
        return chat_ui.render_page("t")

    def test_it_has_a_tab_and_a_shortcut_by_the_model_picker(self):
        page = self.page()
        assert 'id="tabModels"' in page and 'id="modelPane"' in page
        assert 'getElementById("manageModels").addEventListener("click", () => openDrawer("models"))' in page

    def test_deleting_asks_first_and_says_what_it_breaks(self):
        page = self.page()
        body = page[page.index("async function deleteModel"):]
        body = body[:body.index("\n      }\n")]
        assert "window.confirm(" in body
        assert "This app is set to use it as: " in body
        assert body.index("window.confirm(") < body.index('fetch("api/models/delete"')

    def test_names_go_in_as_text(self):
        page = self.page()
        body = page[page.index("function modelRow"):page.index("async function refreshModelPane")]
        assert "title.textContent = name" in body and "innerHTML" not in body

    def test_a_finished_download_refreshes_the_model_picker(self):
        page = self.page()
        body = page[page.index("async function pullModel"):]
        done = body[body.index("if (msg.done)"):]
        assert done.index("await loadModels();") < done.index("pulling[name].remove()")


class TestTheOllamaCalls:
    """The client itself, against a stand-in Ollama speaking its real shapes."""

    @pytest.fixture
    def server(self, monkeypatch):
        from http.server import BaseHTTPRequestHandler, HTTPServer
        seen = []

        class H(BaseHTTPRequestHandler):
            def reply(self, code, body, ctype="application/json"):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def payload(self):
                return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))

            def do_GET(self):
                self.reply(200, json.dumps({"models": [{"name": "a:1b", "size_vram": 5}]}).encode())

            def do_DELETE(self):
                seen.append(("delete", self.payload()))
                self.reply(404, b'{"error":"model not found"}')

            def do_POST(self):
                body = self.payload()
                seen.append((self.path, body))
                if self.path == "/api/pull":
                    self.reply(200, b'{"status":"pulling manifest"}\n'
                                    b'{"status":"pulling x","total":10,"completed":5}\n'
                                    b'{"error":"disk full"}\n', "application/x-ndjson")
                else:
                    self.reply(200, b'{"done":true}')

            def log_message(self, *a):
                pass

        httpd = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        monkeypatch.setenv("OLLAMA_HOST", f"http://127.0.0.1:{httpd.server_port}")
        yield seen
        httpd.shutdown()
        httpd.server_close()

    def test_loaded(self, server):
        assert ollama_client.loaded_models() == [{"name": "a:1b", "size_vram": 5}]

    def test_a_pull_reports_progress_then_its_error(self, server):
        got = []
        with pytest.raises(ValueError, match="disk full"):
            for line in ollama_client.pull_model("a:1b"):
                got.append(line)
        assert got[-1]["completed"] == 5
        assert server[0] == ("/api/pull", {"model": "a:1b", "stream": True})

    def test_unload_asks_for_no_keep_alive(self, server):
        ollama_client.unload_model("a:1b")
        assert server[-1] == ("/api/generate", {"model": "a:1b", "keep_alive": 0})

    def test_deleting_what_is_not_there_says_so(self, server):
        with pytest.raises(ValueError, match="not installed"):
            ollama_client.delete_model("a:1b")
