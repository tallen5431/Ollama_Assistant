#!/usr/bin/env python3
"""Thin HTTP client wrapper around a local Ollama server.

This keeps all direct HTTP calls to Ollama in one place so that
`app.py` is mostly about request/response shaping.
"""

from __future__ import annotations

from typing import Any, Dict

import requests

from config_ollama_helper import get_ollama_base, logger


def post_ollama(path: str, payload: Dict[str, Any], timeout: float = 120.0) -> Dict[str, Any]:
    """POST JSON to Ollama and return parsed JSON or raise ValueError.

    Parameters
    ----------
    path:
        API path such as "/api/chat" or "/api/tags".
    payload:
        JSON-serializable body to send.
    timeout:
        Request timeout in seconds.
    """
    url = get_ollama_base() + path

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        logger.error("Error calling Ollama at %s: %s", url, exc)
        raise ValueError(f"Error calling Ollama: {exc}") from exc

    if not resp.ok:
        logger.error("Ollama error (%s): %s", resp.status_code, resp.text[:512])
        raise ValueError(
            f"Ollama returned HTTP {resp.status_code}: {resp.text[:256]}"
        )

    try:
        return resp.json()
    except ValueError as exc:
        logger.error("Invalid JSON from Ollama at %s: %s", url, resp.text[:512])
        raise ValueError("Invalid JSON from Ollama") from exc
