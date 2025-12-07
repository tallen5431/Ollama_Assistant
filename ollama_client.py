#!/usr/bin/env python3
"""Thin HTTP client wrapper around a local Ollama server.

This keeps all direct HTTP calls to Ollama in one place so that
`app.py` is mostly about request/response shaping.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Generator
import json

import requests

from config_ollama_helper import get_ollama_base, get_request_timeout, logger


def post_ollama(
    path: str,
    payload: Dict[str, Any],
    timeout: Optional[float] = None,
    stream: bool = False
) -> Dict[str, Any]:
    """POST JSON to Ollama and return parsed JSON or raise ValueError.

    Parameters
    ----------
    path:
        API path such as "/api/chat" or "/api/tags".
    payload:
        JSON-serializable body to send.
    timeout:
        Request timeout in seconds. If None, uses default from config.
    stream:
        If True, enable streaming mode (caller should use post_ollama_stream instead).
    """
    if timeout is None:
        timeout = get_request_timeout()

    url = get_ollama_base() + path

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except requests.Timeout as exc:
        logger.error("Timeout calling Ollama at %s after %.1fs", url, timeout)
        raise ValueError(f"Request to Ollama timed out after {timeout}s") from exc
    except requests.ConnectionError as exc:
        logger.error("Connection error calling Ollama at %s: %s", url, exc)
        raise ValueError(f"Could not connect to Ollama at {url}. Is Ollama running?") from exc
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


def post_ollama_stream(
    path: str,
    payload: Dict[str, Any],
    timeout: Optional[float] = None
) -> Generator[Dict[str, Any], None, None]:
    """POST JSON to Ollama with streaming enabled.

    Yields parsed JSON chunks as they arrive from the streaming response.

    Parameters
    ----------
    path:
        API path such as "/api/chat" or "/api/generate".
    payload:
        JSON-serializable body to send. Will automatically set stream=True.
    timeout:
        Request timeout in seconds. If None, uses default from config.

    Yields
    ------
    dict
        Parsed JSON objects from the stream.
    """
    if timeout is None:
        timeout = get_request_timeout()

    url = get_ollama_base() + path
    payload_copy = dict(payload)
    payload_copy["stream"] = True

    try:
        with requests.post(url, json=payload_copy, timeout=timeout, stream=True) as resp:
            if not resp.ok:
                logger.error("Ollama error (%s): %s", resp.status_code, resp.text[:512])
                raise ValueError(
                    f"Ollama returned HTTP {resp.status_code}: {resp.text[:256]}"
                )

            for line in resp.iter_lines():
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        logger.warning("Invalid JSON in stream: %s", line[:256])
                        continue
    except requests.Timeout as exc:
        logger.error("Timeout calling Ollama at %s after %.1fs", url, timeout)
        raise ValueError(f"Request to Ollama timed out after {timeout}s") from exc
    except requests.ConnectionError as exc:
        logger.error("Connection error calling Ollama at %s: %s", url, exc)
        raise ValueError(f"Could not connect to Ollama at {url}. Is Ollama running?") from exc
    except requests.RequestException as exc:
        logger.error("Error calling Ollama at %s: %s", url, exc)
        raise ValueError(f"Error calling Ollama: {exc}") from exc


def check_ollama_health() -> bool:
    """Check if Ollama server is reachable.

    Returns
    -------
    bool
        True if Ollama is healthy, False otherwise.
    """
    try:
        url = get_ollama_base() + "/api/tags"
        resp = requests.get(url, timeout=5.0)
        return resp.ok
    except requests.RequestException:
        return False
