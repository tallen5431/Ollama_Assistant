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
    """Return the per-request read timeout (seconds) for talking to Ollama."""
    try:
        return float(os.getenv("OLLAMA_TIMEOUT", "300"))
    except (TypeError, ValueError):
        return 300.0


def get_connect_timeout() -> float:
    """How long to wait for the TCP connection to Ollama, separately.

    A generous read timeout is right — a 30b model can take minutes to think —
    but passing it as a scalar makes it the *connect* timeout too. When the
    desktop holding the models is asleep it drops the SYN rather than refusing,
    so the connection attempt hung for the full budget: over two minutes of
    silence before the user was told anything, and a web turn paid it on every
    call it made.
    """
    try:
        return max(1.0, float(os.getenv("OLLAMA_CONNECT_TIMEOUT", "5")))
    except (TypeError, ValueError):
        return 5.0


def get_timeouts() -> tuple:
    """(connect, read), the shape requests wants."""
    return (get_connect_timeout(), get_request_timeout())


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


def get_vision_model() -> str:
    """Model used to read an attached image when planning a web search.

    Empty means "pick an installed model that can see". Set WEB_VISION_MODEL to
    pin a specific one (e.g. ``minicpm-v``).
    """
    return os.getenv("WEB_VISION_MODEL", "").strip()


def get_photo_keep_days() -> float:
    """How long a stored photo stays in the history, in days. 0 keeps forever.

    Photos are the bulk of what history costs — the browser caps a re-encode at
    roughly 900KB, which is 1.2MB of base64 text in the row — and what is worth
    having a year later is the reading that came off the picture, not the
    picture. That reading is already in the reply and, for a routine, in the
    records. So the pixels expire on a timer and the words never do.
    """
    return max(0.0, _number("PHOTO_KEEP_DAYS", 30.0))


def get_web_max_docs() -> int:
    """How many fetched pages to put in front of the model at once."""
    return max(1, int(_number("WEB_MAX_DOCS", 3)))


def get_distiller_model() -> str:
    """Small model that cuts a fetched page down to what bears on the question.

    Unset means off, and off is the default: this costs one model call per page,
    and on a single-GPU desktop a model that has to be swapped in can cost more
    in latency than the context it saves is worth. Set it to something small
    (``qwen3:4b``, ``llama3.2:3b``) that fits alongside the answering model and
    the pages arrive as a few hundred characters each instead of six thousand.
    """
    return os.getenv("WEB_DISTILLER_MODEL", "").strip()


def get_web_timeout() -> float:
    """Per-request timeout when fetching a page or running a search."""
    return _number("WEB_TIMEOUT", 15.0)


def get_web_max_chars() -> int:
    """How much extracted text to keep from a single page."""
    return int(_number("WEB_MAX_CHARS", 6000))


def get_web_max_bytes() -> int:
    """Hard cap on a downloaded document, before extraction."""
    return int(_number("WEB_MAX_BYTES", 2 * 1024 * 1024))


def get_web_follow_links() -> int:
    """How many pages linked from a page you pasted may also be read.

    A wiki article often answers half the question and points at the page with
    the other half. One hop, same site only, and a small model picks which
    links are worth opening. 0 turns it off.
    """
    return max(0, min(4, int(_number("WEB_FOLLOW_LINKS", 2))))


def get_photo_meta_default() -> bool:
    """Whether a browser that has never chosen starts with photo details on.

    The date, camera and GPS position a photo carries are read in the browser
    and sent with it. On a box only you can reach — over Tailscale, say — that
    is plainly useful and on is the right default. Set ``PHOTO_META=0`` where
    that is not true; a browser that has used the toggle keeps its own answer
    either way.
    """
    return _flag("PHOTO_META", "1")


def get_share_photo_location() -> bool:
    """Whether a photo's coordinates may inform a web search.

    The planner runs on your own hardware, but what it writes is sent to a
    search engine — so this is the one setting on which a photo's position can
    leave the house. ``WEB_SHARE_LOCATION=0`` keeps the date and camera in the
    planner's view and drops the position.
    """
    return _flag("WEB_SHARE_LOCATION", "1")


def get_image_turns() -> int:
    """How many recent image-bearing turns keep their attachments.

    A vision model re-reads every image in the thread on every turn, which is
    slow and rarely intended. Raise CHAT_IMAGE_TURNS if you compare images
    across turns ("here is before… here is after"); 0 sends none at all.
    """
    return max(0, int(_number("CHAT_IMAGE_TURNS", 1)))


def get_keep_alive() -> str:
    """How long Ollama should hold the answering model in VRAM after a turn.

    Empty means "whatever Ollama is configured to do" (five minutes by
    default). Set OLLAMA_KEEP_ALIVE to something like ``30m`` to keep a 30b
    resident between turns and skip its load time — worth it only if the VRAM
    is not wanted by anything else. Helper models are unloaded immediately
    regardless; a one-shot planner call has no reason to squat.
    """
    return os.getenv("OLLAMA_KEEP_ALIVE", "").strip()


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
