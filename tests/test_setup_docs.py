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

    def test_the_key_generator_does_not_assume_openssl(self):
        """`$(openssl …)` on a box without openssl expands to nothing and the
        sed silently writes an empty key; the container then restart-loops with
        the reason only in docker logs. python3 is already required to run this
        app at all, so it is the safer generator."""
        start = README.index("cd searxng")
        window = README[start:start + 700]
        assert "openssl rand" not in window, "openssl is not on every box"
        assert "secrets.token_hex" in window
        assert "grep secret_key settings.yml" in window, \
            "the result has to be checked, not the command trusted"

    def test_the_tracked_config_trap_is_written_down(self):
        """settings.yml is tracked, so an update can put the default key back
        and stop a working instance — with the reason only in docker logs."""
        start = README.index("cd searxng")
        window = README[start:start + 1400]
        assert "tracked in this repository" in window
        assert "git pull" in window

    @pytest.mark.parametrize("error", [
        # No compose v2 plugin: the -d never reaches a compose that would
        # understand it, so Docker's root command rejects it and prints its own
        # usage — which reads like a typo rather than a missing plugin.
        "unknown shorthand flag: 'd' in -d",
        # And the trap on the other side. The hyphenated docker-compose is v1,
        # end-of-life since 2023, and against a current Docker Engine it dies
        # here on the *second* run — the recreate path — so it looks like the
        # compose file broke rather than the tool.
        "KeyError: 'ContainerConfig'",
    ])
    def test_the_errors_someone_will_paste_into_a_search_box_are_here(self, error):
        start = README.index("docker compose up -d")
        window = README[start:start + 2500]
        assert error in window, f"{error!r} is not named where the command is"

    def test_and_v1_is_named_as_the_trap_it_is_rather_than_an_option(self):
        """It was offered here as a fallback, which sent someone straight into
        the KeyError above. Two wrong turns on the same paragraph."""
        start = README.index("docker compose up -d")
        window = README[start:start + 2500]
        assert "Do not reach for the hyphenated" in window
        assert "use the v1 binary if that is what is there" not in README
