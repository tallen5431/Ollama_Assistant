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


def _flag(name: str, default: str = "1") -> bool:
    """Read a boolean-ish environment variable."""
    return os.getenv(name, default).strip().lower() not in ("0", "false", "no", "off", "")


def _number(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def web_enabled() -> bool:
    """Whether the app may reach the public web at all (operator kill switch)."""
    return _flag("WEB_ENABLED")


def get_search_url() -> str:
    """Base URL of a SearXNG instance, e.g. ``http://127.0.0.1:8888``.

    When unset, search falls back to DuckDuckGo's HTML endpoint, which needs no
    key but is best-effort — self-hosting SearXNG is the sturdier option.
    """
    return os.getenv("SEARXNG_URL", "").strip().rstrip("/")


def get_planner_model() -> str:
    """Model used to turn a message into search queries.

    Empty means "use whichever model is answering". Pointing this at a small
    model (``qwen2.5-coder:0.5b``, ``qwen3.5:4b``) makes planning quick and
    keeps a big answering model from being invoked twice per turn — but only
    if both fit in VRAM at once, otherwise Ollama swaps them and it is slower.
    """
    return os.getenv("WEB_PLANNER_MODEL", "").strip()


def get_web_max_docs() -> int:
    """How many fetched pages to put in front of the model at once."""
    return max(1, int(_number("WEB_MAX_DOCS", 3)))


def get_web_timeout() -> float:
    """Per-request timeout when fetching a page or running a search."""
    return _number("WEB_TIMEOUT", 15.0)


def get_web_max_chars() -> int:
    """How much extracted text to keep from a single page."""
    return int(_number("WEB_MAX_CHARS", 6000))


def get_web_max_bytes() -> int:
    """Hard cap on a downloaded document, before extraction."""
    return int(_number("WEB_MAX_BYTES", 2 * 1024 * 1024))


def get_num_ctx() -> int:
    """Context window to request when web context is attached.

    Ollama defaults to a modest window (commonly 4096) whatever the model
    supports, and a fetched page will silently push the conversation out of it.
    """
    return int(_number("OLLAMA_NUM_CTX", 8192))


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
