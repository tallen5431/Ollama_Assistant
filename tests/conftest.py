"""Make the app modules importable from the tests directory."""

from __future__ import annotations

import functools
import importlib
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Modules whose functions get stubbed a lot. "app" is the one that actually
# matters: it binds its dependencies by value at import time, so a stub that
# was live when app was reloaded becomes app's permanent "original" and
# monkeypatch dutifully restores it forever.
_WATCHED = ("app", "ollama_client", "web", "store", "voice")


def _is_test_stub(value: object) -> bool:
    """Whether a callable came from a test rather than from the app.

    __module__ alone is not enough: a unittest.mock object reports "unittest.mock"
    and a functools.partial has no __module__ at all, so both slipped past the
    first version of this check.

    Anything that objects to being inspected is not a stub. app.py's namespace
    holds Flask's ``request``, a proxy that raises "working outside of request
    context" the moment an attribute is read from it.
    """
    try:
        return _classify(value)
    except Exception:  # noqa: BLE001 - an object that resists inspection is not ours
        return False


def _classify(value: object) -> bool:
    module = getattr(value, "__module__", None) or ""
    if module.startswith("test_") or module == "conftest":
        return True
    if module.startswith("unittest.mock") or type(value).__module__.startswith("unittest.mock"):
        return True
    if isinstance(value, functools.partial):
        return _classify(value.func)
    # A closure defined in a test module: the function object itself may report
    # the app's module after a reload, but its code object cannot lie about the
    # file it was compiled from.
    code = getattr(value, "__code__", None)
    filename = getattr(code, "co_filename", "") if code else ""
    return "/tests/" in filename.replace("\\", "/")


@pytest.fixture(autouse=True)
def no_dotenv_in_tests():
    """A developer's own .env must not decide what the tests see.

    config.py reads it at import, into the real environment, so without this a
    box with SEARXNG_URL in its .env fails every test that asserts the unset
    default — and does so only on that box, which is the worst kind.

    Saved and restored by hand rather than with monkeypatch: an autouse fixture
    that *asks* for monkeypatch drags it to the front of the setup order, which
    puts its teardown after no_stub_left_behind's and makes the guard below
    report every test that patched a module as a leak.
    """
    import os
    import config
    saved = {key: os.environ.pop(key)
             for key in config.env_file_keys() if key in os.environ}
    from_file = config._FROM_ENV_FILE
    config._FROM_ENV_FILE = ()
    try:
        yield
    finally:
        os.environ.update(saved)
        config._FROM_ENV_FILE = from_file


@pytest.fixture(autouse=True)
def clear_process_caches():
    """Memoised state lives for the process, so tests would inherit it.

    The installed-model list is cached for a few seconds; without this, a test
    that points at one fake Ollama sees the previous test's model list.
    """
    import ollama_client
    ollama_client.invalidate_models_cache()
    yield
    ollama_client.invalidate_models_cache()


@pytest.fixture(autouse=True)
def no_auth_left_behind():
    """Put app.py back to open after a test that reloaded it with auth on.

    ``importlib.reload`` with CHAT_AUTH_* set registers a before_request handler
    and leaves AUTH_ENABLED True on the module for the rest of the session.
    monkeypatch takes the environment variables back but cannot take the
    handler back, so every later test that touches the app gets a 401 from a
    module it never configured — /api/health returns HTML, get_json() returns
    None, and the failure lands nowhere near its cause.

    Repaired rather than reported, unlike the stub guard above: this one is
    fixed by exactly the reload that caused it, with the environment now clean,
    and the check is two env reads on tests that did nothing wrong.
    """
    yield
    module = sys.modules.get("app")
    if module is None or not getattr(module, "AUTH_ENABLED", False):
        return
    import authz
    if authz.auth_enabled():
        return          # the environment still says auth is on; that is correct
    importlib.reload(module)


@pytest.fixture(autouse=True)
def no_stub_left_behind():
    """Fail the test that leaks a stub, not the twenty tests downstream.

    Patching a module and *then* reloading app.py binds the stub into app's own
    global, so monkeypatch records the stub as the original and restores it
    forever. That happened: it cost 14 failures in test_web_chat.py that had
    nothing to do with test_web_chat.py, and the full suite hid it because an
    unrelated module reload happened to sit in between.
    """
    yield
    leaked = []
    for name in _WATCHED:
        module = sys.modules.get(name)
        if module is None:
            continue
        for attr, value in list(vars(module).items()):
            # Dunders only. Private names are exactly the ones these tests patch
            # (_is_public, _get_model, _download), so skipping them missed the
            # leak wherever it was most likely.
            if attr.startswith("__"):
                continue
            try:
                if not callable(value):
                    continue
            except Exception:  # noqa: BLE001 - a proxy that needs a request context
                continue
            if _is_test_stub(value):
                leaked.append(f"{name}.{attr}")
    if leaked:
        # Reported, not repaired: reloading a module during teardown to undo the
        # leak takes the rest of the session with it, which is worse than the
        # problem. Downstream tests may fail too — the *first* failure is the
        # one to read.
        pytest.fail(
            f"{', '.join(leaked)} still held a test stub after this test. Patch the "
            "reloaded app module, not the dependency it imported from — otherwise "
            "every later test in the session inherits it."
        )


def page_script(page: str) -> str:
    """The page's own script, out of the rendered HTML.

    The longest block, not the first: the page carries a second, tiny script in
    <head> that applies a saved theme before the stylesheet paints, and every
    helper that reached for ``<script>`` with a non-greedy match started
    getting that one instead.
    """
    blocks = re.findall(r"<script>(.*?)</script>", page, re.S)
    if not blocks:
        raise AssertionError("the page has no script block")
    return max(blocks, key=len)
