#!/usr/bin/env python3
"""Config + logging helpers for CodeSmith Ollama Helper.

This module centralizes environment access and logging setup so
`app.py` can stay focused on HTTP routing logic.
"""

from __future__ import annotations

import logging
import os
from typing import Tuple, Optional

LOGGER_NAME = "codesmith_ollama_helper"


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure basic logging and return the shared logger.

    Called on import so the helper always has a usable logger, but you
    can customize this further in the future if needed.
    """
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
    )
    return logging.getLogger(LOGGER_NAME)


# Shared logger instance used by the rest of the modules
logger = configure_logging()


def get_ollama_base() -> str:
    """Return the Ollama base URL (no trailing slash)."""
    base = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").strip()
    if base.endswith("/"):
        base = base[:-1]
    return base


def get_default_model() -> str:
    """Return the default model name for Ollama."""
    return os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")


def get_heavy_model(default: Optional[str] = None) -> str:
    """Return the "heavy" model name for Ollama.

    Typically configured via OLLAMA_MODEL_HEAVY; if unset, falls back
    to ``default`` (if provided) or the primary default model.
    """
    base = default or get_default_model()
    return os.getenv("OLLAMA_MODEL_HEAVY", base)


def get_host_port(default_port: int = 8070) -> Tuple[str, int]:
    """Return (host, port) for the Flask app.

    HOST and PORT are read from the environment, falling back to
    0.0.0.0:8070 when unset or invalid.
    """
    host = os.getenv("HOST", "0.0.0.0")
    port_str = os.getenv("PORT", str(default_port))
    try:
        port = int(port_str)
    except ValueError:
        port = default_port
    return host, port
