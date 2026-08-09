"""The setup instructions have to describe the files they sit next to.

Nothing here imports the app. These are the steps someone follows on a box
where the search has just stopped working, usually in a hurry, and a command
that is subtly wrong costs an evening — the README's fallback `docker run`
spells out the same container as searxng/docker-compose.yml, and two copies of
a container spec are exactly the thing that drifts apart in silence.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text()
COMPOSE = (ROOT / "searxng" / "docker-compose.yml").read_text()


def docker_run_block() -> str:
    """The README's compose-free fallback, as one string."""
    at = README.index("docker run -d --name searxng")
    return README[at:README.index("```", at)]


class TestTheComposeFreeFallbackIsTheSameContainer:
    """`docker compose` is the v2 plugin, and Ubuntu's docker.io package ships
    without it — reported from the NucBox as "unknown shorthand flag: 'd' in
    -d", because the -d never reaches a compose that would understand it. The
    README gives a plain `docker run` for that case, and it has to start the
    same thing: a fallback that quietly differs on the port sends someone to a
    URL nothing is listening on, with the app still saying search is broken.
    """

    @pytest.mark.parametrize("what, pattern", [
        ("image", r"image:\s*(\S+)"),
        ("port mapping", r"ports:\s*\n\s*-\s*\"?([\d.]+:\d+:\d+)\"?"),
        ("base URL", r"(SEARXNG_BASE_URL=\S+)"),
        ("mount point", r"-\s*\./:(\S+)"),
    ])
    def test_the_settings_that_matter_agree(self, what, pattern):
        found = re.search(pattern, COMPOSE)
        assert found, f"{what} is no longer where this test looks in the compose file"
        assert found.group(1) in docker_run_block(), \
            f"the README's docker run has a different {what} than the compose file"

    def test_it_keeps_the_capabilities_the_compose_file_drops(self):
        """The container runs with almost nothing. A fallback that skips this
        is a more privileged container than the documented one, which is not a
        difference anyone would notice from the outside."""
        block = docker_run_block()
        assert "--cap-drop ALL" in block
        for cap in re.findall(r"-\s*(CHOWN|SETGID|SETUID)", COMPOSE):
            assert f"--cap-add {cap}" in block, cap

    def test_and_the_log_cap_that_stops_it_filling_the_disk(self):
        block = docker_run_block()
        assert "max-size=1m" in block and "max-file=1" in block


class TestTheStepsCanBeFollowedInOrder:
    """Reported: `.venv/bin/python: No such file or directory`. The block above
    says `cd searxng`, and the check below is run from the project root where
    the virtualenv is — with nothing in between saying to come back."""

    def test_the_searxng_steps_return_to_the_project_root(self):
        start = README.index("cd searxng")
        end = README.index("tools/check_web.py", start)
        assert "\ncd ..\n" in README[start:end], \
            "the setup walks into searxng/ and never comes back, but the next " \
            "command it gives needs the project root"

    def test_the_compose_variants_are_named_where_the_command_is(self):
        start = README.index("docker compose up -d")
        window = README[start:start + 1200]
        assert "docker-compose" in window, "the v1 binary is not mentioned"
        assert "unknown shorthand flag" in window, \
            "the error someone actually sees is what they will search for"
