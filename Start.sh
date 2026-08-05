#!/usr/bin/env bash
# Ollama Chat launcher (portable, venv-safe).
# The HTTP Server Manager passes HOST/PORT (and any overrides) in the environment.

set -euo pipefail

# --- App root (this folder) ---
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

# --- Prefer a per-project venv ---
VENV_DIR="$APP_DIR/.venv"
VENV_PY="$VENV_DIR/bin/python"
PY_CMD=""

if [[ -x "$VENV_PY" ]]; then
  PY_CMD="$VENV_PY"
fi

if [[ -z "$PY_CMD" ]]; then
  if [[ -n "${PY_EXE:-}" && -x "$PY_EXE" ]]; then
    SYS_PY="$PY_EXE"
  else
    SYS_PY="$(command -v python3 || true)"
  fi
  if [[ -z "${SYS_PY:-}" ]]; then
    echo "[ERROR] No python3 found. Install it or set PY_EXE, then retry."
    exit 1
  fi
  echo "[SETUP] Creating virtual environment at $VENV_DIR..."
  "$SYS_PY" -m venv "$VENV_DIR" || { echo "[ERROR] venv creation failed"; exit 1; }
  PY_CMD="$VENV_PY"
fi

# --- Install dependencies ---
# A network hiccup here must not stop the app. The box reboots after a power
# cut, this runs before DNS is up, pip exits 1, and under `set -e` the whole
# launch dies — discovered days later from the phone. If the dependencies are
# already installed (the steady state on every reboot but the first), a failed
# install is nothing to stop for. The same pattern is used for vosk below.
#
# pip/setuptools/wheel are deliberately not upgraded on every start: it is the
# one unpinned step in an otherwise pinned install, so it is the only thing
# that can change under the app without anyone asking it to.
if [[ -f "$APP_DIR/requirements.txt" ]]; then
  echo "[SETUP] Installing dependencies..."
  if ! "$PY_CMD" -m pip install -r "$APP_DIR/requirements.txt"; then
    echo "[WARN] Could not install dependencies (no network yet?)."
    if "$PY_CMD" -c 'import flask, requests, waitress' 2>/dev/null; then
      echo "[WARN] They are already present — starting anyway."
    else
      echo "[ERROR] And they are not already installed. Cannot start."
      exit 1
    fi
  fi
fi

# --- Optional voice support (vosk has no wheel on some platforms) ---
# A failure here must not stop the app: voice input is optional, and without it
# the mic button just stays hidden.
if [[ -f "$APP_DIR/requirements-voice.txt" ]]; then
  echo "[SETUP] Installing optional voice support..."
  if ! "$PY_CMD" -m pip install -r "$APP_DIR/requirements-voice.txt"; then
    echo "[WARN] Voice support (vosk) could not be installed on this platform."
    echo "[WARN] Continuing without it — the mic button will be hidden."
  fi
fi

# --- Stream output to the manager's log pane ---
# Python block-buffers stdout when it isn't a terminal, so the startup banner
# and any print() would otherwise show up late or not at all.
export PYTHONUNBUFFERED=1

# --- Network (Server Manager passes HOST/PORT in env) ---
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8070}"

# --- Ollama server (where the model runs). Point this at your desktop's LAN or
# Tailscale address (e.g. http://192.168.1.50:11434) so the app on the server
# manager can reach it — set it here or in the program's env. ---
export OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.1:8b}"

# Keep the bind address and OLLAMA_HOST on separate lines, bind address first.
# The server manager infers a program's port by scanning its log output, and a
# full http://host:port URL outranks a bare 0.0.0.0:port bind address — so with
# both on one line it would show Ollama's port (11434) instead of ours.
echo "[RUN] Starting Ollama Chat on ${HOST}:${PORT}"
echo "[RUN] OLLAMA_HOST=${OLLAMA_HOST}  OLLAMA_MODEL=${OLLAMA_MODEL}"
exec "$PY_CMD" "$APP_DIR/app.py"
