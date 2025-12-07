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
        logger.warning("Invalid PORT value '%s', using default %d", port_str, default_port)
        port = default_port

    if not (1 <= port <= 65535):
        logger.warning("Port %d out of valid range, using default %d", port, default_port)
        port = default_port

    return host, port


def get_max_upload_size() -> int:
    """Return max upload size in bytes (default: 50MB)."""
    default_mb = 50
    try:
        mb = int(os.getenv("MAX_UPLOAD_SIZE_MB", str(default_mb)))
        if mb <= 0:
            raise ValueError("Size must be positive")
        return mb * 1024 * 1024
    except (ValueError, TypeError):
        logger.warning("Invalid MAX_UPLOAD_SIZE_MB, using default %dMB", default_mb)
        return default_mb * 1024 * 1024


def get_request_timeout() -> float:
    """Return default request timeout in seconds (default: 120)."""
    try:
        timeout = float(os.getenv("OLLAMA_TIMEOUT", "120"))
        if timeout <= 0:
            raise ValueError("Timeout must be positive")
        return timeout
    except (ValueError, TypeError):
        logger.warning("Invalid OLLAMA_TIMEOUT, using default 120s")
        return 120.0


def get_max_snapshots() -> int:
    """Return maximum number of snapshots to keep (default: 100, 0=unlimited)."""
    try:
        max_count = int(os.getenv("MAX_SNAPSHOTS", "100"))
        if max_count < 0:
            raise ValueError("Max snapshots cannot be negative")
        return max_count
    except (ValueError, TypeError):
        logger.warning("Invalid MAX_SNAPSHOTS, using default 100")
        return 100


def validate_config() -> None:
    """Validate configuration on startup and log warnings for issues."""
    logger.info("Validating configuration...")

    ollama_base = get_ollama_base()
    if not ollama_base.startswith(("http://", "https://")):
        logger.warning("OLLAMA_HOST should start with http:// or https://: %s", ollama_base)

    host, port = get_host_port()
    logger.info("Flask will bind to %s:%d", host, port)

    logger.info("Max upload size: %d MB", get_max_upload_size() // (1024 * 1024))
    logger.info("Ollama timeout: %.1f seconds", get_request_timeout())
    logger.info("Max snapshots: %d (0=unlimited)", get_max_snapshots())
