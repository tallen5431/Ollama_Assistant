"""Research recipes: a routine that also says how to search."""

from __future__ import annotations

import importlib
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import app as app_module
import chat_ui
import recipes
import store
import web
from conftest import allow_loopback


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAT_DB", str(tmp_path / "chat.db"))
    yield


@pytest.fixture
def client(monkeypatch):
    for key in ("CHAT_AUTH", "CHAT_AUTH_USER", "CHAT_AUTH_PASSWORD"):
        monkeypatch.delenv(key, raising=False)
    return importlib.reload(app_module).app.test_client()


class TestCleaning:
    def test_sites_however_they_are_typed(self):
        got = recipes.clean({"sites": "https://www.Docs.Python.org/3/library, github.com,"
                                      " not a site, http://localhost:8080"})
        assert got == {"sites": ["docs.python.org", "github.com"]}

    def test_at_most_five_sites_and_no_repeats(self):
        sites = [f"s{i}.example.com" for i in range(8)] + ["s0.example.com"]
        assert recipes.clean({"sites": sites})["sites"] == sites[:5]

    def test_numbers_are_clamped(self):
        assert recipes.clean({"rounds": 9, "pages": 0}) == {"rounds": 3, "pages": 1}
        assert recipes.clean({"rounds": "2", "pages": "x"}) == {"rounds": 2}

    def test_zero_rounds_is_a_setting_not_an_absence(self):
        assert recipes.clean({"rounds": 0}) == {"rounds": 0}

    def test_only_known_values(self):
        assert recipes.clean({"fresh": "decade", "format": "poem", "extra": 1}) == {}
        assert recipes.clean({"fresh": "Week", "format": "TABLE"}) == {"fresh": "week",
                                                                         "format": "table"}

    @pytest.mark.parametrize("raw", [None, "", [], "table", 5])
    def test_anything_else_is_no_recipe(self, raw):
        assert recipes.clean(raw) == {}


class TestWhatARecipeDoes:
    def test_one_site(self):
        assert recipes.query_for("pathlib glob", {"sites": ["docs.python.org"]}) == \
            "pathlib glob site:docs.python.org"

    def test_several_sites(self):
        assert recipes.query_for("x", {"sites": ["a.com", "b.org"]}) == "x (site:a.com OR site:b.org)"

    def test_no_sites_leaves_the_query_alone(self):
        assert recipes.query_for("x", {"fresh": "day"}) == "x"

    def test_a_subdomain_counts_and_a_lookalike_does_not(self):
        recipe = {"sites": ["python.org"]}
        assert recipes.allowed("https://docs.python.org/3/", recipe)
        assert not recipes.allowed("https://notpython.org/", recipe)
        assert not recipes.allowed("https://python.org.evil.example/", recipe)

    def test_the_planner_is_told(self):
        note = recipes.planner_note({"sites": ["a.com"], "fresh": "week"})
        assert "limited to a.com" in note and "past week" in note
        assert recipes.planner_note({}) == ""

    def test_the_answer_layout(self):
        assert "Markdown table" in recipes.answer_note({"format": "table"})
        assert recipes.answer_note({}) == ""

    def test_describe(self):
        assert recipes.describe({"sites": ["a.com"], "fresh": "week", "rounds": 0,
                                 "pages": 4, "format": "table"}) == \
            "only a.com · past week · 0 extra round(s) · 4 page(s) · table"


class TestStorage:
    def test_a_recipe_is_kept_with_its_routine(self):
        made = store.create_routine("R", "b", recipe={"sites": "a.com", "format": "table"})
        assert made["recipe"] == {"sites": ["a.com"], "format": "table"}
        assert store.get_routine(made["id"])["recipe"] == made["recipe"]

    def test_a_routine_without_one_has_an_empty_one(self):
        made = store.create_routine("R", "b")
        assert store.list_routines()[0]["recipe"] == {} and made["recipe"] == {}

    def test_changing_and_clearing_it(self):
        rid = store.create_routine("R", "b", recipe={"fresh": "day"})["id"]
        assert store.update_routine(rid, {"recipe": {"pages": 4}})["recipe"] == {"pages": 4}
        assert store.update_routine(rid, {"recipe": {}})["recipe"] == {}

    def test_the_starter_recipes(self):
        made = store.create_recipe_starters()
        names = [r["name"] for r in made]
        assert names == [name for name, _, _ in recipes.STARTERS]
        assert all(r["web"] is True and r["recipe"] for r in made)
        assert store.create_recipe_starters() == [], "installed twice"

    def test_the_old_starters_still_install_as_they_were(self):
        made = store.create_starters()
        assert made and all(r["recipe"] == {} for r in made)


class TestRoutes:
    def test_saving_one_cleans_it(self, client):
        out = client.post("/api/routines", json={
            "name": "R", "body": "b", "recipe": {"sites": "www.a.com", "pages": 99,
                                                 "format": "nonsense"}}).get_json()
        assert out["recipe"] == {"sites": ["a.com"], "pages": 6}

    def test_editing_one(self, client):
        rid = store.create_routine("R", "b")["id"]
        out = client.patch(f"/api/routines/{rid}", json={"recipe": {"fresh": "week"}}).get_json()
        assert out["routine"]["recipe"] == {"fresh": "week"}

    def test_installing_the_recipes(self, client):
        out = client.post("/api/routines/recipes", json={}).get_json()
        assert len(out["routines"]) == len(recipes.STARTERS)

    def test_installing_needs_json(self, client):
        """It reads no body, so a form on another site could otherwise POST it."""
        assert client.post("/api/routines/recipes", data="").status_code == 415


class TestTheEngineFilter:
    def test_searxng_is_given_the_time_range(self, monkeypatch):
        monkeypatch.setenv("WEB_ENABLED", "1")
        monkeypatch.setattr(web, "effective_search_url", lambda: "http://searx.local")
        seen = {}
        monkeypatch.setattr(web, "_search_searxng",
                            lambda base, q, limit, time_range="": seen.update(t=time_range) or
                            [{"url": "https://a.com/"}])
        web.search("x", time_range="week")
        assert seen["t"] == "week"

    def test_duckduckgo_is_given_its_own_code(self, monkeypatch):
        monkeypatch.setenv("WEB_ENABLED", "1")
        monkeypatch.setattr(web, "effective_search_url", lambda: "")
        asked = []

        def fake_get(url, **kwargs):
            asked.append(url)
            raise web.WebError("stop here")
        monkeypatch.setattr(web, "_get", fake_get)
        with pytest.raises(web.WebError):
            web.search("x", time_range="month")
        assert asked and all(u.endswith("&df=m") for u in asked)

    def test_the_url_built_for_searxng(self, monkeypatch):
        asked = []

        def fake_get(url, **kwargs):
            asked.append(url)
            raise web.WebError("stop")
        monkeypatch.setattr(web, "_get", fake_get)
        with pytest.raises(web.WebError):
            web._search_searxng("http://s", "x", 3, time_range="day")
        assert asked[0].endswith("&time_range=day")

    def test_a_recent_search_is_cached_apart(self, monkeypatch):
        monkeypatch.setenv("WEB_ENABLED", "1")
        monkeypatch.setattr(web, "effective_search_url", lambda: "")
        calls = []
        monkeypatch.setattr(web, "_search_duckduckgo",
                            lambda q, limit, time_range="": calls.append(time_range) or
                            [{"url": "https://a.com/"}])
        web.search("x")
        web.search("x", time_range="day")
        assert calls == ["", "day"]


# --------------------------------------------------------------------------
# A whole turn
# --------------------------------------------------------------------------

class _Ollama(BaseHTTPRequestHandler):
    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        self.server.requests.append(payload)
        system = payload["messages"][0]["content"] if payload.get("messages") else ""
        if payload.get("stream") is False:
            reply = "DONE" if "You check whether pages" in system else "Q: widget price"
            body = json.dumps({"message": {"content": reply}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        self.wfile.write(json.dumps({"message": {"content": "ok"}, "done": True,
                                     "eval_count": 1}).encode() + b"\n")

    def log_message(self, *args):
        pass


class _Site(BaseHTTPRequestHandler):
    def do_GET(self):
        body = f"<html><body><p>Page {self.path}: widget costs $42.</p></body></html>".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def turn(client, monkeypatch):
    servers = []
    for handler in (_Ollama, _Site):
        server = HTTPServer(("127.0.0.1", 0), handler)
        server.requests = []
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
    ollama, site = servers
    monkeypatch.setenv("OLLAMA_HOST", f"http://127.0.0.1:{ollama.server_port}")
    monkeypatch.setenv("WEB_ENABLED", "1")
    monkeypatch.setenv("WEB_FOLLOW_ON_SEARCH", "0")
    allow_loopback(monkeypatch)
    base = f"http://127.0.0.1:{site.server_port}"
    asked = []

    def search(query, limit=3, time_range=""):
        asked.append((query, time_range))
        # One result on the recipe's site (by its path), one elsewhere.
        return [{"url": f"{base}/shop.example.com/widget", "title": "on site"},
                {"url": f"{base}/elsewhere/widget", "title": "off site"}]
    monkeypatch.setattr(web, "search", search)
    # The stand-in site is one host, so "on the site" is decided by path here.
    monkeypatch.setattr(recipes, "allowed",
                        lambda url, recipe: not recipe.get("sites") or
                        any(s in url for s in recipe["sites"]))

    def ask(routine_id=None):
        resp = client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "widget price?"}], "web": True,
            **({"routine_id": routine_id} if routine_id else {})})
        return [json.loads(l) for l in resp.get_data(as_text=True).splitlines() if l.strip()]

    yield {"ask": ask, "asked": asked, "ollama": ollama}
    for server in servers:
        server.shutdown()
        server.server_close()


def _steps(out):
    return {o["debug"]["step"]: o["debug"]["detail"] for o in out if "debug" in o}


def _systems(ollama):
    answer = [r for r in ollama.requests if r.get("stream")][-1]
    return [m["content"] for m in answer["messages"] if m["role"] == "system"]


class TestATurnWithARecipe:
    def make(self, recipe):
        return store.create_routine("R", "b", web=True, recipe=recipe)["id"]

    def test_sites_and_recency_reach_the_search(self, turn):
        rid = self.make({"sites": ["shop.example.com"], "fresh": "week", "rounds": 0})
        out = turn["ask"](rid)
        assert turn["asked"] == [("widget price site:shop.example.com", "week")]
        sources = [o["sources"] for o in out if "sources" in o][0]
        assert len(sources) == 1 and "/shop.example.com/" in sources[0]["url"], \
            "an off-site result was read"
        assert "only shop.example.com · past week" in _steps(out)["Research recipe"]

    def test_the_planner_is_told_about_the_limits(self, turn):
        rid = self.make({"sites": ["shop.example.com"], "rounds": 0})
        turn["ask"](rid)
        planner = [r for r in turn["ollama"].requests if r.get("stream") is False][0]
        assert "limited to shop.example.com" in planner["messages"][0]["content"]

    def test_the_answer_layout_reaches_the_model(self, turn):
        rid = self.make({"format": "table", "rounds": 0})
        turn["ask"](rid)
        assert any("Markdown table" in s for s in _systems(turn["ollama"]))

    def test_rounds_override_the_server_setting(self, turn, monkeypatch):
        monkeypatch.setenv("WEB_SEARCH_ROUNDS", "2")
        out = turn["ask"](self.make({"rounds": 0}))
        assert "Search round 2" not in _steps(out)

    def test_pages_override_the_server_setting(self, turn, monkeypatch):
        monkeypatch.setenv("WEB_MAX_DOCS", "3")
        out = turn["ask"](self.make({"pages": 1, "rounds": 0}))
        sources = [o["sources"] for o in out if "sources" in o][0]
        assert len(sources) == 1

    def test_a_message_from_no_routine_is_unchanged(self, turn):
        out = turn["ask"]()
        assert turn["asked"][0] == ("widget price", "")
        assert "Research recipe" not in _steps(out)
        assert not any("Answer format" in s for s in _systems(turn["ollama"]))

    def test_an_unknown_routine_is_no_recipe(self, turn):
        out = turn["ask"]("not-a-routine")
        assert "Research recipe" not in _steps(out)


class TestThePage:
    def page(self):
        return chat_ui.render_page("t")

    def test_the_message_says_which_routine_it_came_from(self):
        assert "routine_id: (pendingRoutine && pendingRoutine.routine.id) || undefined" \
            in self.page()

    def test_the_editor_saves_the_recipe(self):
        page = self.page()
        assert "recipe: readRecipe() };" in page
        for field in ("rSites", "rFresh", "rRounds", "rPages", "rFormat"):
            assert f'id="{field}"' in page

    def test_the_recipe_section_opens_only_when_there_is_one(self):
        assert 'document.getElementById("rRecipe").open = Object.keys(recipe).length > 0;' \
            in self.page()

    def test_the_recipes_can_be_installed_from_the_list(self):
        page = self.page()
        assert '"api/routines/recipes"' in page
        assert "add.addEventListener(\"click\", addRecipes);" in page
