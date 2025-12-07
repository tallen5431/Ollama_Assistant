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


def wait_for_ollama(
    max_retries: int = 5,
    initial_delay: float = 2.0,
    timeout: float = 5.0
) -> bool:
    """Wait for Ollama server to become available with exponential backoff.

    Parameters
    ----------
    max_retries:
        Maximum number of connection attempts.
    initial_delay:
        Initial delay between retries in seconds.
    timeout:
        Timeout for each health check request.

    Returns
    -------
    bool
        True if Ollama became available, False if max retries exceeded.
    """
    import time

    delay = initial_delay
    url = get_ollama_base()

    logger.info("Checking Ollama connectivity at %s...", url)

    for attempt in range(max_retries):
        try:
            endpoint = url + "/api/tags"
            resp = requests.get(endpoint, timeout=timeout)
            if resp.ok:
                logger.info("✓ Successfully connected to Ollama")
                return True
        except requests.RequestException as exc:
            if attempt < max_retries - 1:
                logger.warning(
                    "Attempt %d/%d: Cannot connect to Ollama at %s (%s). Retrying in %.1fs...",
                    attempt + 1,
                    max_retries,
                    url,
                    type(exc).__name__,
                    delay
                )
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                logger.error(
                    "Attempt %d/%d: Cannot connect to Ollama at %s (%s)",
                    attempt + 1,
                    max_retries,
                    url,
                    type(exc).__name__
                )

    return False


def get_ollama_startup_instructions() -> str:
    """Return helpful instructions for starting Ollama.

    Returns
    -------
    str
        Multi-line string with platform-specific instructions.
    """
    instructions = """
╔══════════════════════════════════════════════════════════════════════╗
║                    Ollama Server Not Running                         ║
╚══════════════════════════════════════════════════════════════════════╝

The CodeSmith Ollama Helper requires Ollama to be running.

To start Ollama:

  1. Check if Ollama is installed:
     $ ollama --version

  2. If not installed, download from: https://ollama.ai/

  3. Start Ollama service:
     • macOS/Linux:   $ ollama serve
     • Windows:       Ollama runs automatically, check system tray

  4. Verify Ollama is running:
     $ ollama list

  5. Pull a model if needed:
     $ ollama pull qwen2.5-coder:7b

  6. Check the configured Ollama host:
     Current: {ollama_host}

If Ollama is running on a different host/port, set OLLAMA_HOST:
  $ export OLLAMA_HOST=http://your-host:11434

To disable this check and start anyway, set:
  $ export REQUIRE_OLLAMA_ON_STARTUP=false

For more information: https://github.com/ollama/ollama
"""
    from config_ollama_helper import get_ollama_base
    return instructions.format(ollama_host=get_ollama_base())

