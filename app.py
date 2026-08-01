#!/usr/bin/env python3
"""Ollama Chat — a simple general chatbot you can plug into the HTTP Server Manager.

A small Flask API in front of a local Ollama server, plus a single-page chat UI
on the root page. The model runs on your desktop (Ollama); this app runs on the
server manager and talks to it over LAN/Tailscale via ``OLLAMA_HOST``.

Endpoints:
  GET  /              chat UI
  GET  /healthz       plain "ok" health probe (open even when auth is on)
  GET  /api/health    JSON status (host, default model)
  GET  /api/models    installed models (proxy to Ollama /api/tags)
  POST /api/chat      chat completion (proxy to Ollama /api/chat)

Configuration is entirely via environment variables — see config.py / authz.py.
"""

from __future__ import annotations

import json
import os
import socket
from typing import Any, Dict, List, Optional

from flask import Flask, Response, jsonify, request, stream_with_context
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.middleware.proxy_fix import ProxyFix

import authz
import voice
from chat_ui import render_page
from config import (
    get_app_title,
    get_default_model,
    get_host_port,
    get_max_body_bytes,
    get_ollama_base,
    logger,
)
from ollama_client import chat as ollama_chat
from ollama_client import chat_stream, list_models

# -------------------------------------------------------------------
# Flask app
# -------------------------------------------------------------------

app = Flask(__name__)

# No CORS: the UI ships with this app and only calls relative paths, so nothing
# legitimate is cross-origin. Allowing any origin would let a page the user
# happens to visit drive their local model over the LAN and read the reply.

# Cap request bodies so a single oversized POST (notably the raw WAV upload to
# /api/transcribe, which is buffered in memory) can't exhaust RAM.
app.config["MAX_CONTENT_LENGTH"] = get_max_body_bytes()

# Respect X-Forwarded-* headers so the app works behind the manager's reverse
# proxy (Caddy) as well as when accessed directly.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1, x_prefix=1)


@app.errorhandler(RequestEntityTooLarge)
def _too_large(_exc: RequestEntityTooLarge) -> Any:
    """Return the body-size rejection as JSON — the UI only parses JSON."""
    limit = app.config["MAX_CONTENT_LENGTH"]
    return jsonify({"error": f"Request body too large (limit {limit // (1024 * 1024)} MB)"}), 413


# -------------------------------------------------------------------
# Optional HTTP Basic Auth (makes internet exposure safe)
# -------------------------------------------------------------------
AUTH_ENABLED = authz.auth_enabled()
if AUTH_ENABLED:
    _AUTH_REALM = os.environ.get("CHAT_AUTH_REALM", "Ollama Chat")

    @app.before_request
    def _enforce_basic_auth() -> Optional[Response]:
        # /healthz stays open so a tunnel or uptime monitor can probe it.
        if request.path.rstrip("/") == "/healthz":
            return None
        auth = request.authorization
        if (
            auth
            and (auth.type or "").lower() == "basic"
            and authz.credentials_match(auth.username, auth.password)
        ):
            return None
        return Response(
            "Authentication required.",
            401,
            {"WWW-Authenticate": f'Basic realm="{_AUTH_REALM}"'},
        )


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------


@app.route("/", methods=["GET"])
def index() -> Any:
    """Serve the chat page."""
    return render_page(get_app_title())


@app.route("/healthz", methods=["GET"], strict_slashes=False)
def healthz() -> Any:
    """Plain-text health probe for the server manager / tunnels."""
    return "ok", 200


@app.route("/api/health", methods=["GET"])
def health() -> Any:
    """JSON status: which Ollama host and default model are configured."""
    return jsonify(
        {
            "status": "ok",
            "ollama_host": get_ollama_base(),
            "default_model": get_default_model(),
            "auth": AUTH_ENABLED,
            "voice": voice.voice_available(),
        }
    )


@app.route("/api/models", methods=["GET"])
def api_models() -> Any:
    """List installed models from Ollama, plus the configured default."""
    try:
        models = list_models()
    except ValueError as exc:
        return jsonify({"error": str(exc), "models": [], "default": get_default_model()}), 502
    return jsonify({"models": models, "default": get_default_model()})


@app.route("/api/chat", methods=["POST"])
def api_chat() -> Any:
    """Chat completion. Accepts a ``messages`` array or a single ``prompt``.

    Streams token-by-token NDJSON by default (what the UI consumes). Pass
    ``{"stream": false}`` to get a single JSON ``{"model", "reply"}`` object.
    """
    body = request.get_json(silent=True) or {}
    model = body.get("model") or get_default_model()
    messages: Optional[List[Dict[str, str]]] = body.get("messages")

    if messages is None:
        prompt = body.get("prompt")
        if not prompt:
            return jsonify({"error": "Missing 'messages' or 'prompt' in request"}), 400
        messages = [{"role": "user", "content": str(prompt)}]

    if not isinstance(messages, list) or not messages:
        return jsonify({"error": "'messages' must be a non-empty list"}), 400

    # Non-streaming path — single JSON object.
    if body.get("stream") is False:
        try:
            reply = ollama_chat(model, messages)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 502
        return jsonify({"model": model, "reply": reply})

    # Streaming path — pass Ollama's NDJSON lines straight through. Any failure
    # (including one raised mid-stream while iterating) is turned into a final
    # JSON error line rather than a bare HTTP 500, so the UI can show why.
    @stream_with_context
    def generate() -> Any:
        try:
            for line in chat_stream(model, messages):
                yield line + "\n"
        except Exception as exc:  # noqa: BLE001 - surface any error to the client
            logger.exception("Chat stream failed")
            message = str(exc) or exc.__class__.__name__
            yield json.dumps({"error": message}) + "\n"

    return Response(generate(), mimetype="application/x-ndjson")


@app.route("/api/voice/models", methods=["GET"])
def api_voice_models() -> Any:
    """List downloadable + already-downloaded Vosk speech models."""
    if not voice.voice_available():
        return jsonify({"error": "Voice input is not available (vosk not installed)."}), 501
    return jsonify(voice.list_models())


@app.route("/api/voice/download", methods=["POST"])
def api_voice_download() -> Any:
    """Download a catalog Vosk model so it's ready before recording."""
    if not voice.voice_available():
        return jsonify({"error": "Voice input is not available (vosk not installed)."}), 501
    body = request.get_json(silent=True) or {}
    model_id = body.get("id")
    if not model_id:
        return jsonify({"error": "Missing 'id'"}), 400
    try:
        info = voice.download_model(str(model_id))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Voice model download failed")
        return jsonify({"error": f"Download failed: {exc}"}), 500
    return jsonify(info)


@app.route("/api/transcribe", methods=["POST"])
def api_transcribe() -> Any:
    """Transcribe posted WAV audio to text with Vosk (offline).

    The Vosk model is chosen with a ``?model=<id>`` query parameter (defaults to
    the configured default language).
    """
    if not voice.voice_available():
        return jsonify({"error": "Voice input is not available (vosk not installed)."}), 501

    audio = request.get_data()
    if not audio:
        return jsonify({"error": "No audio received"}), 400

    model_id = request.args.get("model") or None
    try:
        text = voice.transcribe(audio, model_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - defensive (model load/download)
        logger.exception("Transcription failed")
        return jsonify({"error": f"Transcription failed: {exc}"}), 500

    return jsonify({"text": text, "model": model_id or voice.default_model_id()})


# -------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------


def _resolve_bind_host(host: str, default: str = "0.0.0.0") -> str:
    """Return a host safe to hand to waitress.

    Some launchers inject a bogus value (e.g. an unsubstituted ``HOST=PORT``
    placeholder). Waitress runs ``getaddrinfo`` on the host and aborts the whole
    process when it can't resolve — so warn and fall back to all interfaces.
    """
    host = (host or "").strip()
    if not host:
        return default
    if host in ("0.0.0.0", "::", "*"):
        return host
    try:
        socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError, ValueError):
        logger.warning("HOST=%r is not a resolvable bind address; using %s", host, default)
        return default
    return host


def main() -> None:
    host, port = get_host_port(default_port=8070)
    host = _resolve_bind_host(host)

    print("=" * 60)
    print(f"💬 {get_app_title()} started")
    print("=" * 60)
    print(f"  Listening on : http://{host}:{port}")
    print(f"  Ollama host  : {get_ollama_base()}")
    print(f"  Default model: {get_default_model()}")
    print(f"  Auth         : {'ON — Basic Auth required' if AUTH_ENABLED else 'OFF (LAN only)'}")
    print("=" * 60)

    try:
        from waitress import serve

        # Backstop the app-level cap at the socket layer so a hugely oversized
        # upload is refused before waitress spools it. Deliberately set higher
        # than MAX_CONTENT_LENGTH: waitress rejects with a plain-text body, so
        # anything merely over the limit should reach Flask and get the JSON
        # error the UI can display.
        serve(
            app,
            host=host,
            port=port,
            max_request_body_size=app.config["MAX_CONTENT_LENGTH"] * 2,
        )
    except ImportError:
        # Fall back to the Flask dev server if waitress isn't installed.
        logger.warning("waitress not installed; using the Flask dev server")
        app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
