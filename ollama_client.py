#!/usr/bin/env python3
"""Thin HTTP client wrapper around an Ollama server.

Keeps all direct HTTP calls to Ollama in one place so ``app.py`` stays focused
on request/response shaping. Talks to the *native* Ollama API (``/api/chat``,
``/api/tags``).

Requests deliberately bypass any ambient HTTP proxy (``proxies=...``): the model
usually runs on your desktop reached over LAN/Tailscale, and a system proxy set
for internet traffic must not swallow that direct call.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List

import requests

from config import get_ollama_base, get_request_timeout, logger

# Never route Ollama calls through a proxy — the server is on the local
# network / tailnet and must be reached directly.
_NO_PROXY = {"http": None, "https": None}


def get_ollama(path: str, timeout: float | None = None) -> Dict[str, Any]:
    """GET JSON from Ollama and return parsed JSON, or raise ValueError."""
    url = get_ollama_base() + path
    timeout = get_request_timeout() if timeout is None else timeout

    try:
        resp = requests.get(url, timeout=timeout, proxies=_NO_PROXY)
    except requests.RequestException as exc:
        logger.error("Error calling Ollama at %s: %s", url, exc)
        raise ValueError(f"Could not reach Ollama at {get_ollama_base()}: {exc}") from exc

    return _parse(resp, url)


def post_ollama(path: str, payload: Dict[str, Any], timeout: float | None = None) -> Dict[str, Any]:
    """POST JSON to Ollama and return parsed JSON, or raise ValueError."""
    url = get_ollama_base() + path
    timeout = get_request_timeout() if timeout is None else timeout

    try:
        resp = requests.post(url, json=payload, timeout=timeout, proxies=_NO_PROXY)
    except requests.RequestException as exc:
        logger.error("Error calling Ollama at %s: %s", url, exc)
        raise ValueError(f"Could not reach Ollama at {get_ollama_base()}: {exc}") from exc

    return _parse(resp, url)


def _parse(resp: requests.Response, url: str) -> Dict[str, Any]:
    """Validate an Ollama HTTP response and return its JSON body."""
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


def list_models() -> List[Dict[str, Any]]:
    """Return the list of installed models from Ollama's ``/api/tags``."""
    data = get_ollama("/api/tags")
    models = data.get("models") or data.get("tags") or []
    return models if isinstance(models, list) else []


def chat(model: str, messages: List[Dict[str, str]]) -> str:
    """Send a chat completion (non-streaming) and return the reply text."""
    payload = {"model": model, "messages": messages, "stream": False}
    data = post_ollama("/api/chat", payload)
    message = data.get("message") or {}
    return message.get("content") or data.get("response") or ""


def chat_stream(model: str, messages: List[Dict[str, str]]) -> Iterator[str]:
    """Stream a chat completion, yielding raw NDJSON lines from Ollama.

    Each line is a JSON object: incremental ``{"message": {"content": "..."}}``
    chunks (and ``"thinking"`` for reasoning models), ending with a
    ``{"done": true, "eval_count": ..., "eval_duration": ...}`` summary that
    carries the token-usage stats.
    """
    url = get_ollama_base() + "/api/chat"
    payload = {"model": model, "messages": messages, "stream": True}

    try:
        resp = requests.post(
            url, json=payload, stream=True, timeout=get_request_timeout(), proxies=_NO_PROXY
        )
    except requests.RequestException as exc:
        logger.error("Error calling Ollama at %s: %s", url, exc)
        raise ValueError(f"Could not reach Ollama at {get_ollama_base()}: {exc}") from exc

    if not resp.ok:
        text = resp.text[:256]
        resp.close()
        logger.error("Ollama error (%s): %s", resp.status_code, text)
        raise ValueError(f"Ollama returned HTTP {resp.status_code}: {text}")

    # Decode explicitly rather than via iter_lines(decode_unicode=True): that
    # flag is a silent no-op when the response has no inferable encoding, and
    # Ollama streams "application/x-ndjson" with no charset, so requests would
    # hand back raw bytes. Always yield str so callers can treat it as text.
    with resp:
        for raw in resp.iter_lines():
            if raw:
                yield raw.decode("utf-8", errors="replace")
