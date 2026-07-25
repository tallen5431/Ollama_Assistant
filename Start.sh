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
if [[ -f "$APP_DIR/requirements.txt" ]]; then
  echo "[SETUP] Installing dependencies..."
  "$PY_CMD" -m pip install --upgrade pip setuptools wheel >/dev/null
  "$PY_CMD" -m pip install -r "$APP_DIR/requirements.txt"
fi

# --- Network (Server Manager passes HOST/PORT in env) ---
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8070}"

# --- Ollama server (where the model runs). Point this at your desktop's LAN or
# Tailscale address so the app on the server manager can reach it. ---
export OLLAMA_HOST="${OLLAMA_HOST:-http://100.98.112.1:11434}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.1:8b}"

echo "[RUN] Starting Ollama Chat on ${HOST}:${PORT}  (OLLAMA_HOST=${OLLAMA_HOST})"
exec "$PY_CMD" "$APP_DIR/app.py"
