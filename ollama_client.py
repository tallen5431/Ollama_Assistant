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
        data = resp.json()
    except ValueError as exc:
        logger.error("Invalid JSON from Ollama at %s: %s", url, resp.text[:512])
        raise ValueError("Invalid JSON from Ollama") from exc
    # Callers index this like an object; a bare array or string would raise
    # AttributeError past the ValueError handlers and become a bare 500.
    if not isinstance(data, dict):
        logger.error("Unexpected JSON shape from Ollama at %s: %s", url, type(data).__name__)
        raise ValueError("Unexpected JSON from Ollama (expected an object)")
    return data


def list_models() -> List[Dict[str, Any]]:
    """Return the list of installed models from Ollama's ``/api/tags``."""
    data = get_ollama("/api/tags")
    models = data.get("models") or data.get("tags") or []
    return models if isinstance(models, list) else []


def chat(
    model: str,
    messages: List[Dict[str, str]],
    options: Dict[str, Any] | None = None,
) -> str:
    """Send a chat completion (non-streaming) and return the reply text."""
    payload: Dict[str, Any] = {"model": model, "messages": messages, "stream": False}
    if options:
        payload["options"] = options
    data = post_ollama("/api/chat", payload)
    message = data.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    return content or data.get("response") or ""


def chat_stream(
    model: str,
    messages: List[Dict[str, str]],
    options: Dict[str, Any] | None = None,
) -> Iterator[str]:
    """Stream a chat completion, yielding raw NDJSON lines from Ollama.

    Each line is a JSON object: incremental ``{"message": {"content": "..."}}``
    chunks (and ``"thinking"`` for reasoning models), ending with a
    ``{"done": true, "eval_count": ..., "eval_duration": ...}`` summary that
    carries the token-usage stats.
    """
    url = get_ollama_base() + "/api/chat"
    payload: Dict[str, Any] = {"model": model, "messages": messages, "stream": True}
    if options:
        # e.g. {"num_ctx": 8192} — Ollama's default window is small, and an
        # attached web page will silently push the conversation out of it.
        payload["options"] = options

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


# Markers Ollama reports for models that accept images. "clip" covers the llava
# family, "mllama" covers llama3.2-vision; the name check catches builds whose
# details block is sparse, which varies by Ollama version.
# Deliberately no "gemma3": Ollama reports that family for the text-only
# gemma3:270m and :1b as well as the vision-capable sizes, so the family
# alone cannot tell them apart. The capability list below does.
_VISION_FAMILIES = {"clip", "mllama", "qwen2vl", "qwen2_5_vl", "siglip"}
_VISION_NAMES = ("llava", "vision", "minicpm-v", "moondream", "vl", "bakllava", "ocr")

# Models built to transcribe an image rather than reason about one. They are the
# best choice for pulling exact text out of a screenshot and a poor choice for
# answering a question about it, so the two roles are kept apart.
_OCR_NAMES = ("ocr", "got-ocr", "trocr", "donut")


def model_name(model: Dict[str, Any]) -> str:
    """The name Ollama would accept for this /api/tags entry."""
    if not isinstance(model, dict):
        return ""
    return str(model.get("name") or model.get("model") or "")


def is_ocr(model: Dict[str, Any]) -> bool:
    """Whether the model is a text-extraction specialist rather than a VLM."""
    return any(marker in model_name(model).lower() for marker in _OCR_NAMES)


def has_vision(model: Dict[str, Any]) -> bool:
    """Whether an /api/tags entry describes a model that can read images."""
    if not isinstance(model, dict):
        return False
    # Ollama's explicit capability list is authoritative wherever it is present,
    # so it is checked FIRST. Deciding on the family instead declares
    # gemma3:270m vision-capable, and being smallest it would then win both the
    # auto-switch and the reader role — sending a screenshot to a blind model.
    caps = model.get("capabilities")
    if isinstance(caps, list) and caps:
        return any(str(c).lower() == "vision" for c in caps)

    details = model.get("details")
    if isinstance(details, dict):
        families = details.get("families") or []
        if isinstance(families, list):
            if {str(f).lower() for f in families} & _VISION_FAMILIES:
                return True
        if str(details.get("family") or "").lower() in _VISION_FAMILIES:
            return True
    return any(marker in model_name(model).lower() for marker in _VISION_NAMES)


def vision_models(include_ocr: bool = True) -> List[str]:
    """Names of installed models that can read images, smallest first.

    Smallest first because both callers want a cheap pass rather than the best
    possible one, and because silently loading a 19 GB model because someone
    pasted a screenshot is a poor surprise.

    ``include_ocr=False`` drops the transcription specialists, which is what you
    want when choosing a model to *answer* a question about an image.
    """
    try:
        models = list_models()
    except ValueError:
        return []
    picked = [
        (m.get("size") or 0, model_name(m))
        for m in models
        if has_vision(m) and (include_ocr or not is_ocr(m))
    ]
    return [name for _, name in sorted(picked) if name]


def ocr_models() -> List[str]:
    """Installed text-extraction models, smallest first."""
    try:
        models = list_models()
    except ValueError:
        return []
    picked = [(m.get("size") or 0, model_name(m)) for m in models if is_ocr(m)]
    return [name for _, name in sorted(picked) if name]
