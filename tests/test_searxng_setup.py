"""The SearXNG container's config must not live where git works.

Reported while pulling a branch that touched nothing under searxng/:

    error: unable to unlink old 'searxng/docker-compose.yml': Permission denied
    error: unable to unlink old 'searxng/settings.yml': Permission denied
    fatal: Could not reset index file to revision 'FETCH_HEAD'.

The compose file mounted `./` — the repo's own directory — as the container's
config directory, and the image takes ownership of that on startup. Unlinking a
file needs write permission on the *directory*, so once the container had it,
git could not replace anything in there and refused the whole checkout.

The live config now lives in searxng/instance/, which is gitignored. The
container owns its config, git owns the worktree, and they are no longer the
same place.
"""

from __future__ import annotations

import re
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SEARXNG = ROOT / "searxng"


def compose() -> str:
    return (SEARXNG / "docker-compose.yml").read_text(encoding="utf-8")


def mounts() -> list:
    """The volume lines, without hand-rolling a YAML parser or adding one."""
    text = compose()
    block = re.search(r"^    volumes:\n((?:^      - .*\n)+)", text, re.M)
    assert block, "the compose file must declare volumes"
    return [line.strip()[2:] for line in block.group(1).splitlines()]


class TestTheContainerDoesNotOwnTheRepo:
    def test_the_repo_directory_is_not_mounted(self):
        for mount in mounts():
            host = mount.split(":")[0]
            assert host not in ("./", "."), (
                "mounting the repo's own directory hands it to the container, "
                "and git can no longer check anything out into it"
            )

    def test_the_config_mount_is_the_ignored_directory(self):
        assert any(m.startswith("./instance:") for m in mounts()), \
            "the container's config belongs in searxng/instance/"

    def test_that_directory_is_gitignored(self):
        ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        assert any(line.strip() == "searxng/instance/" for line in ignored)

    def test_git_agrees_it_is_ignored(self):
        """The line above is only worth having if git reads it the same way."""
        result = subprocess.run(
            ["git", "check-ignore", "-q", "searxng/instance/settings.yml"],
            cwd=ROOT, capture_output=True)
        assert result.returncode == 0, "git does not consider the live config ignored"

    def test_the_compose_file_is_not_handed_to_the_container(self):
        """It was, under `./`, and it had no business reading it."""
        for mount in mounts():
            assert not mount.startswith("./docker-compose.yml")


class TestTheLiveConfigIsNotTracked:
    def test_the_template_is_what_ships(self):
        assert (SEARXNG / "settings.yml.example").is_file()

    def test_the_live_config_is_not_committed(self):
        """It holds the instance's secret key, and `sed -i` into a tracked file
        is one `git add -A` away from committing it."""
        tracked = subprocess.run(
            ["git", "ls-files", "searxng/"],
            cwd=ROOT, capture_output=True, text=True).stdout.split()
        assert "searxng/settings.yml" not in tracked
        assert "searxng/instance/settings.yml" not in tracked

    @pytest.mark.parametrize("setting", ["limiter: false", "- json", "safe_search: 0"])
    def test_the_template_still_carries_the_settings_that_matter(self, setting):
        """A fresh SearXNG fails on the first two; they are the whole reason
        this file is shipped rather than left to the image's defaults."""
        assert setting in (SEARXNG / "settings.yml.example").read_text(encoding="utf-8")

    def test_the_placeholder_key_is_still_the_one_setup_replaces(self):
        template = (SEARXNG / "settings.yml.example").read_text(encoding="utf-8")
        script = (SEARXNG / "setup.sh").read_text(encoding="utf-8")
        assert "ultrasecretkey" in template
        assert "ultrasecretkey" in script, "otherwise setup.sh seeds an unstartable instance"


class TestSetupScript:
    def test_it_is_executable(self):
        mode = (SEARXNG / "setup.sh").stat().st_mode
        assert mode & stat.S_IXUSR, "documented as ./setup.sh"

    def run(self, cwd):
        return subprocess.run(["bash", str(SEARXNG / "setup.sh")],
                              cwd=cwd, capture_output=True, text=True)

    @pytest.fixture
    def instance(self, tmp_path, monkeypatch):
        """A throwaway checkout, so the test never writes into the real one."""
        work = tmp_path / "searxng"
        work.mkdir()
        for name in ("setup.sh", "settings.yml.example"):
            (work / name).write_bytes((SEARXNG / name).read_bytes())
        (work / "setup.sh").chmod(0o755)
        return work

    def test_it_writes_a_live_config_with_a_real_key(self, instance):
        result = subprocess.run(["bash", "./setup.sh"], cwd=instance,
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        live = (instance / "instance" / "settings.yml").read_text(encoding="utf-8")
        assert "ultrasecretkey" not in live, "SearXNG refuses to start on the placeholder"
        key = re.search(r'secret_key:\s*"([0-9a-f]+)"', live)
        assert key and len(key.group(1)) == 64

    def test_two_runs_do_not_cost_you_your_key(self, instance):
        subprocess.run(["bash", "./setup.sh"], cwd=instance, capture_output=True)
        first = (instance / "instance" / "settings.yml").read_text(encoding="utf-8")
        result = subprocess.run(["bash", "./setup.sh"], cwd=instance,
                                capture_output=True, text=True)
        assert result.returncode == 0
        assert (instance / "instance" / "settings.yml").read_text(encoding="utf-8") == first
        assert "SKIP" in result.stdout

    def test_the_key_is_not_world_readable(self, instance):
        subprocess.run(["bash", "./setup.sh"], cwd=instance, capture_output=True)
        mode = (instance / "instance" / "settings.yml").stat().st_mode
        assert not mode & stat.S_IROTH

    def test_it_says_so_rather_than_half_working_without_a_template(self, tmp_path):
        work = tmp_path / "searxng"
        work.mkdir()
        (work / "setup.sh").write_bytes((SEARXNG / "setup.sh").read_bytes())
        result = subprocess.run(["bash", "./setup.sh"], cwd=work,
                                capture_output=True, text=True)
        assert result.returncode == 1
        assert "missing" in result.stdout + result.stderr


def test_the_documented_steps_are_the_ones_that_exist():
    """The compose header is where someone setting this up actually looks."""
    header = compose()
    assert "./setup.sh" in header
    assert "sed -i" not in header, "that step moved into setup.sh"
