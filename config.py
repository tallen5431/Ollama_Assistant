#!/usr/bin/env python3
"""Config + logging helpers for the Ollama Chat app.

Everything is driven by environment variables so the HTTP Server Manager can
override host/port/model without touching the code. Kept small and dependency
free so the rest of the app can stay focused on routing and UI.
"""

from __future__ import annotations

import logging
import os
from typing import Tuple

LOGGER_NAME = "ollama_chat"


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure basic logging and return the shared logger."""
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
    )
    return logging.getLogger(LOGGER_NAME)


# Shared logger used across the app.
logger = configure_logging()


def get_ollama_base() -> str:
    """Return the Ollama base URL (no trailing slash).

    Points at wherever Ollama is running. When the app runs on the server
    manager and the model runs on your desktop, set ``OLLAMA_HOST`` to the
    desktop's LAN or Tailscale address, e.g. ``http://192.168.1.50:11434``.
    A trailing ``/v1`` (the OpenAI-compatible form other cards use) is accepted
    and stripped so the native Ollama API is used.
    """
    base = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").strip()
    base = base.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base


def get_default_model() -> str:
    """Return the default model name for chat."""
    return os.getenv("OLLAMA_MODEL", "llama3.1:8b")


def get_request_timeout() -> float:
    """Return the per-request timeout (seconds) for talking to Ollama."""
    try:
        return float(os.getenv("OLLAMA_TIMEOUT", "300"))
    except (TypeError, ValueError):
        return 300.0


def get_max_body_bytes() -> int:
    """Return the maximum accepted request body size, in bytes.

    Guards the in-memory WAV upload on ``/api/transcribe``. ``CHAT_MAX_BODY_MB``
    overrides the 25 MB default (roughly 13 minutes of 16 kHz mono audio).
    """
    try:
        mb = float(os.getenv("CHAT_MAX_BODY_MB", "25"))
    except (TypeError, ValueError):
        mb = 25.0
    if mb <= 0:
        mb = 25.0
    return int(mb * 1024 * 1024)


def get_app_title() -> str:
    """Human-friendly title shown in the browser tab and header."""
    return os.getenv("CHAT_TITLE", "Ollama Chat")


def get_host_port(default_port: int = 8070) -> Tuple[str, int]:
    """Return (host, port) for the server.

    ``HOST`` and ``PORT`` are read from the environment (the server manager
    injects these), falling back to ``0.0.0.0:8070`` when unset or invalid.
    """
    host = os.getenv("HOST", "0.0.0.0").strip() or "0.0.0.0"
    port_str = os.getenv("PORT", str(default_port))
    try:
        port = int(port_str)
    except (TypeError, ValueError):
        port = default_port
    return host, port
