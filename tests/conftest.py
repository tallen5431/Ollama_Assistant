"""Make the app modules importable from the tests directory."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Modules whose functions get stubbed a lot. "app" is the one that actually
# matters: it binds its dependencies by value at import time, so a stub that
# was live when app was reloaded becomes app's permanent "original" and
# monkeypatch dutifully restores it forever.
_WATCHED = ("app", "ollama_client", "web", "store", "voice")


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
    for name in _WATCHED:
        module = sys.modules.get(name)
        if module is None:
            continue
        for attr, value in list(vars(module).items()):
            if attr.startswith("_") or not callable(value):
                continue
            origin = getattr(value, "__module__", None) or ""
            if origin.startswith("test_") or origin == "conftest":
                pytest.fail(
                    f"{name}.{attr} is still a stub from {origin} after this test. "
                    "Patch the reloaded app module, not the dependency it imported "
                    "from — otherwise every later test in the session inherits it."
                )
