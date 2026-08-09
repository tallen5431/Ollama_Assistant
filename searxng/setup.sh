#!/usr/bin/env bash
# Prepare instance/settings.yml for the container, once.
#
# The live config deliberately does not live in the repo. The image takes
# ownership of its config directory on startup, and when that directory was the
# repo's own, git could no longer replace files in it — a pull came back with
# "unable to unlink old 'searxng/docker-compose.yml': Permission denied" and
# checked nothing out. instance/ is gitignored, so the container can own it and
# git never has an opinion.
#
# Safe to re-run: an existing instance/settings.yml is left exactly as it is,
# secret key and all.
#
#   ./setup.sh
#   docker compose up -d

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TEMPLATE="settings.yml.example"
LIVE="instance/settings.yml"

if [[ ! -f "$TEMPLATE" ]]; then
  echo "[ERROR] $TEMPLATE is missing. Run this from a checkout of the repo."
  exit 1
fi

mkdir -p instance

if [[ -f "$LIVE" ]]; then
  echo "[SKIP] $LIVE already exists — leaving it and its secret key alone."
  echo "       Delete it first if you want a fresh one from $TEMPLATE."
  exit 0
fi

# openssl is not on every box, and this is the one step that must not be
# skipped: SearXNG refuses to start on the placeholder key.
if command -v openssl >/dev/null 2>&1; then
  SECRET="$(openssl rand -hex 32)"
elif command -v python3 >/dev/null 2>&1; then
  SECRET="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
else
  echo "[ERROR] Need openssl or python3 to generate a secret key."
  exit 1
fi

# Written through a temporary file rather than edited in place: `set -e` then
# stops a half-substituted config from ever being called settings.yml, and the
# umask means the key is never briefly world-readable on the way. Plain `sed`
# rather than `sed -i`, which takes an argument on BSD and none on GNU.
(
  umask 077
  sed "s|ultrasecretkey|$SECRET|" "$TEMPLATE" > "$LIVE.tmp"
)
mv "$LIVE.tmp" "$LIVE"

echo "[OK] Wrote $LIVE with a fresh secret key."
echo "     Edit that file, not $TEMPLATE — the template only seeds new installs."
echo "     Next: docker compose up -d"
