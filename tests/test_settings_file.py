"""The settings file, and the two processes it exists to keep in agreement.

Reported from the NucBox: `tools/check_web.py` printed a healthy SearXNG at
127.0.0.1:8888, three real results, "Both halves work" — and the very next
search in the UI came back with a DuckDuckGo captcha. Nothing was broken.
SEARXNG_URL had been exported in a terminal, which configures the next program
*that terminal* starts; the app is started by the server manager and never saw
it. Two environments, one of them invisible, and a green light from the tool
whose whole job was to say whether search worked.

So: a file both of them read, and a check that says which one a value came from.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

import config
import web

ROOT = Path(config.__file__).resolve().parent


class TestReadingTheFile:
    def test_a_setting_in_the_file_is_applied(self, tmp_path):
        env = {}
        path = tmp_path / ".env"
        path.write_text("SEARXNG_URL=http://127.0.0.1:8888\n")
        assert config.load_env_file(path, env) == ["SEARXNG_URL"]
        assert env["SEARXNG_URL"] == "http://127.0.0.1:8888"

    def test_the_real_environment_wins(self, tmp_path):
        """Where the server manager puts HOST and PORT, and where an operator
        overriding one setting for one run expects to put it."""
        env = {"SEARXNG_URL": "http://from-the-manager:8888"}
        path = tmp_path / ".env"
        path.write_text("SEARXNG_URL=http://from-the-file:8888\n")
        assert config.load_env_file(path, env) == []
        assert env["SEARXNG_URL"] == "http://from-the-manager:8888"

    def test_no_file_is_the_normal_case(self, tmp_path):
        assert config.load_env_file(tmp_path / "absent", {}) == []

    def test_a_directory_where_the_file_should_be_is_not_a_crash(self, tmp_path):
        (tmp_path / ".env").mkdir()
        assert config.load_env_file(tmp_path / ".env", {}) == []

    @pytest.mark.parametrize("line, expected", [
        ("KEY=value", "value"),
        ("  KEY = value  ", "value"),
        ("export KEY=value", "value"),
        ('KEY="value"', "value"),
        ("KEY='value'", "value"),
        ("KEY=", ""),
        ("KEY=http://host:8888/path?a=b", "http://host:8888/path?a=b"),
        ('KEY="p@ss word#1"', "p@ss word#1"),
    ])
    def test_the_shapes_a_line_comes_in(self, line, expected):
        assert config.parse_env_file(line) == {"KEY": expected}

    @pytest.mark.parametrize("line", [
        "# KEY=value",
        "   # KEY=value",
        "",
        "   ",
        "KEY without an equals sign",
        "9KEY=value",
        "KEY-WITH-DASHES=value",
        "=value",
    ])
    def test_what_is_skipped(self, line):
        assert config.parse_env_file(line) == {}

    def test_a_value_is_never_executed(self):
        """Read line by line rather than sourced, so a settings file cannot run
        commands — it is edited over SSH by whoever is fixing something."""
        parsed = config.parse_env_file("KEY=$(rm -rf /)\nOTHER=`whoami`\n")
        assert parsed == {"KEY": "$(rm -rf /)", "OTHER": "`whoami`"}

    def test_a_realistic_file(self, tmp_path):
        path = tmp_path / ".env"
        path.write_text(
            "# Settings for Ollama Chat\n"
            "\n"
            "OLLAMA_HOST=http://192.168.1.50:11434\n"
            "  # search\n"
            "SEARXNG_URL=http://127.0.0.1:8888\n"
            "export WEB_MAX_DOCS=5\n"
        )
        env = {"WEB_MAX_DOCS": "3"}
        applied = config.load_env_file(path, env)
        assert sorted(applied) == ["OLLAMA_HOST", "SEARXNG_URL"]
        assert env["OLLAMA_HOST"] == "http://192.168.1.50:11434"
        assert env["WEB_MAX_DOCS"] == "3", "already set, so the file does not touch it"


class TestWhereTheFileIsLookedFor:
    def test_beside_the_code_by_default(self, monkeypatch):
        monkeypatch.delenv("CHAT_ENV_FILE", raising=False)
        path = config.env_file_path()
        assert path.name == ".env"
        assert path.parent == Path(config.__file__).resolve().parent

    def test_chat_env_file_moves_it(self, monkeypatch):
        monkeypatch.setenv("CHAT_ENV_FILE", "/etc/ollama-chat.env")
        assert config.env_file_path() == Path("/etc/ollama-chat.env")


class TestSayingWhereAValueCameFrom:
    """The distinction the check tool needs: a value in the file reaches the
    app, a value in this shell does not."""

    def test_from_the_file(self, monkeypatch):
        monkeypatch.setenv("SEARXNG_URL", "http://127.0.0.1:8888")
        monkeypatch.setattr(config, "_FROM_ENV_FILE", ("SEARXNG_URL",))
        assert config.setting_origin("SEARXNG_URL") == "the .env file"

    def test_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("SEARXNG_URL", "http://127.0.0.1:8888")
        monkeypatch.setattr(config, "_FROM_ENV_FILE", ())
        assert config.setting_origin("SEARXNG_URL") == "this process's environment"

    def test_unset_has_no_origin(self, monkeypatch):
        monkeypatch.delenv("SEARXNG_URL", raising=False)
        monkeypatch.setattr(config, "_FROM_ENV_FILE", ())
        assert config.setting_origin("SEARXNG_URL") == ""

    def test_set_to_nothing_counts_as_unset(self, monkeypatch):
        monkeypatch.setenv("SEARXNG_URL", "   ")
        monkeypatch.setattr(config, "_FROM_ENV_FILE", ("SEARXNG_URL",))
        assert config.setting_origin("SEARXNG_URL") == ""


class TestTheBackendIsNamedOutLoud:
    """Which of the two this process is using was not answerable from anywhere,
    and it is the first thing worth knowing when a search returns a captcha."""

    def test_searxng_is_named_with_its_address(self, monkeypatch):
        monkeypatch.setattr(web, "get_search_url", lambda: "http://127.0.0.1:8888")
        assert web.search_backend() == "SearXNG at http://127.0.0.1:8888"

    def test_the_fallback_says_it_is_scraping(self, monkeypatch):
        monkeypatch.setattr(web, "get_search_url", lambda: "")
        said = web.search_backend()
        assert "DuckDuckGo" in said
        assert "SEARXNG_URL" in said

    def test_the_startup_banner_prints_it(self):
        """Because the banner is where the disagreement would have shown."""
        source = Path(config.__file__).resolve().parent / "app.py"
        text = source.read_text(encoding="utf-8")
        assert "search_backend()" in text, "the banner must name the backend"


class TestTheAdviceSaysWhereToPutIt:
    """Naming only the variable read as already done to someone who had set it
    — in a terminal, where the app could not see it."""

    def test_the_duckduckgo_failure_names_the_file(self, monkeypatch):
        monkeypatch.setattr(web, "get_search_url", lambda: "")
        monkeypatch.setattr(web, "web_enabled", lambda: True)

        class Blocked:
            ok = True
            status_code = 200
            encoding = "utf-8"
            headers = {"Content-Type": "text/html"}

            def iter_content(self, size):
                yield (b"<html><body>Unfortunately, bots use DuckDuckGo too. "
                       b"Please complete the following challenge.</body></html>")

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(web, "_get", lambda url, **kw: Blocked())
        with pytest.raises(web.WebError) as caught:
            web.search("anything", 3)
        said = str(caught.value)
        assert "bot challenge" in said, "the existing diagnosis is unchanged"
        assert "SEARXNG_URL" in said
        assert ".env" in said, "and now says where it has to go"
        assert "shell export does not reach" in said


class TestTheCheckToolNoLongerLies:
    """`Both halves work` was true of the shell it ran in and false of the app,
    which is the failure this whole change is about."""

    def check_web(self):
        import importlib.util
        root = Path(config.__file__).resolve().parent
        spec = importlib.util.spec_from_file_location(
            "check_web_under_test", root / "tools" / "check_web.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def run(self, monkeypatch, origin):
        monkeypatch.setattr(config, "setting_origin", lambda name: origin)
        tool = self.check_web()
        out = io.StringIO()
        with redirect_stdout(out):
            shared = tool.shared_with_the_app()
        return shared, out.getvalue()

    def test_a_value_from_the_file_is_shared(self, monkeypatch):
        shared, said = self.run(monkeypatch, "the .env file")
        assert shared is True
        assert said == ""

    def test_nothing_set_at_all_is_not_this_warning(self, monkeypatch):
        """The DuckDuckGo fallback is reported by the search check itself; this
        one is only about a value the app cannot see."""
        shared, said = self.run(monkeypatch, "")
        assert shared is True
        assert said == ""

    def test_a_value_only_this_shell_has_is_called_out(self, monkeypatch):
        shared, said = self.run(monkeypatch, "this process's environment")
        assert shared is False
        assert "SEARXNG_URL" in said
        assert ".env" in said
        assert "does not" in said, "it has to say the app will not see it"

    def test_the_closing_summary_stops_claiming_everything_works(self):
        source = Path(config.__file__).resolve().parent / "tools" / "check_web.py"
        text = source.read_text(encoding="utf-8")
        assert "Both halves work here." in text, "scoped to the shell it ran in"
        assert "shared_with_the_app()" in text


class TestTheLaunchersReadItToo:
    """config.py reading the file is not enough on its own: Start.sh exports
    OLLAMA_HOST whether or not anyone asked, so a value in the file would reach
    Python already overridden by a fallback nobody chose."""

    def source(self, name):
        return (Path(config.__file__).resolve().parent / name).read_text(encoding="utf-8")

    @pytest.mark.parametrize("name, fallback", [
        ("Start.sh", 'export OLLAMA_HOST="${OLLAMA_HOST:-'),
        ("Start.bat", "if not defined OLLAMA_HOST"),
    ])
    def test_the_file_is_read_before_the_defaults(self, name, fallback):
        text = self.source(name)
        assert "ENV_FILE" in text, f"{name} must read the settings file"
        assert fallback in text, "this test is pinned to the launcher's fallback"
        assert text.index("ENV_FILE") < text.index(fallback), \
            "otherwise the launcher's own fallback wins over the file"

    def test_the_shell_launcher_does_not_source_it(self):
        text = self.source("Start.sh")
        assert ". $ENV_FILE" not in text
        assert "source " not in text, "a settings file must not be able to run commands"


@pytest.mark.parametrize("key", ["SEARXNG_URL", "OLLAMA_HOST", "CHAT_AUTH_PASSWORD"])
def test_the_example_file_documents_the_settings_that_matter(key):
    text = (Path(config.__file__).resolve().parent / ".env.example").read_text("utf-8")
    assert key in text


def test_the_real_file_is_not_committed():
    """It holds CHAT_AUTH_PASSWORD."""
    text = (Path(config.__file__).resolve().parent / ".gitignore").read_text("utf-8")
    assert any(line.strip() == ".env" for line in text.splitlines())


class TestTheSuiteIsInsulatedFromADevelopersOwnFile:
    """Clearing it per-test is too late, and the failure is remote-controlled by
    a file that is not in the repo.

    config.py reads .env into the real environment on first import, and the
    modules under test capture some of it at *their* import — app.py reads
    AUTH_ENABLED once and registers a before_request hook from it. Collection
    imports those modules before any fixture runs, so a .env holding the
    CHAT_AUTH_* pair the README documents put the whole suite in front of an app
    with auth on, answering 401 to tests that never asked for it. Popping the
    variables in a fixture cannot unregister the hook.
    """

    def as_pytest_would(self, tmp_path, dotenv):
        """conftest, then app — the order collection imports them in.

        Asserted on that sequence rather than by running a nested suite: a
        nested run self-heals, because no_auth_left_behind reloads app at the
        first teardown, so whether it fails depends on which file you point it
        at and in what order. The invariant does not. This is the moment the
        damage is done, and it is the same moment either way.
        """
        env_file = tmp_path / "dotenv"
        env_file.write_text(dotenv, encoding="utf-8")
        code = (
            "import sys; sys.path.insert(0, %r)\n"
            "import conftest\n"
            "import app, os\n"
            "print(app.AUTH_ENABLED, os.getenv('CHAT_AUTH_USER'), os.getenv('SEARXNG_URL'))\n"
            % str(ROOT / "tests")
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                capture_output=True, text=True,
                                env={**os.environ, "CHAT_ENV_FILE": str(env_file)})
        assert result.returncode == 0, result.stderr[-2000:]
        return result.stdout.split()

    def test_auth_settings_in_the_file_never_reach_the_app_under_test(self, tmp_path):
        """The exact pair the README tells you to put there."""
        enabled, user, _ = self.as_pytest_would(
            tmp_path, "CHAT_AUTH_USER=someone\nCHAT_AUTH_PASSWORD=hunter2\n")
        assert enabled == "False", \
            "app captured AUTH_ENABLED at import, with a hook no fixture can remove"
        assert user == "None"

    def test_nor_do_settings_the_tests_assert_defaults_for(self, tmp_path):
        _, _, searx = self.as_pytest_would(
            tmp_path, "SEARXNG_URL=http://127.0.0.1:8888\nWEB_MAX_DOCS=9\n")
        assert searx == "None"

    def test_a_real_suite_run_is_clean_with_one_sitting_there(self, tmp_path):
        """The end-to-end version of the two above, on the file whose tests
        actually went red when this was found."""
        env_file = tmp_path / "dotenv"
        env_file.write_text("CHAT_AUTH_USER=someone\nCHAT_AUTH_PASSWORD=hunter2\n",
                            encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_turn_survival.py", "-q"],
            cwd=ROOT, capture_output=True, text=True,
            env={**os.environ, "CHAT_ENV_FILE": str(env_file)})
        assert result.returncode == 0, result.stdout[-3000:]
        assert " 401 " not in result.stdout, "the app under test had auth switched on"

    def test_the_clearing_happens_before_anything_is_collected(self):
        """Pinned to where it is, because a fixture is the obvious place to put
        it and the obvious place is the one that does not work."""
        text = (ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
        cleared = text.index("config._FROM_ENV_FILE = ()")
        first_fixture = text.index("@pytest.fixture")
        assert cleared < first_fixture, \
            "clearing it in a fixture runs after collection has already imported app"
