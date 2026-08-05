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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from flask import Flask, Response, jsonify, request, stream_with_context
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.middleware.proxy_fix import ProxyFix

import authz
import store
import voice
import web
from chat_ui import render_page
from config import (
    get_app_title,
    get_default_model,
    get_host_port,
    get_max_body_bytes,
    get_num_ctx,
    get_ollama_base,
    get_vision_model,
    get_web_max_docs,
    logger,
    web_enabled,
)
from ollama_client import chat as ollama_chat
from ollama_client import chat_stream, has_vision, is_ocr, list_models, ocr_models, vision_models

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
    # :.3g not // — a sub-megabyte cap otherwise reports itself as "limit 0 MB".
    return jsonify({"error": f"Request body too large (limit {limit / (1024 * 1024):.3g} MB)"}), 413


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
            "web": web_enabled(),
            "history": store.available(),
        }
    )


@app.route("/api/models", methods=["GET"])
def api_models() -> Any:
    """List installed models from Ollama, plus the configured default."""
    try:
        models = list_models()
    except ValueError as exc:
        return jsonify({"error": str(exc), "models": [], "default": get_default_model()}), 502
    # Tag each model so the UI can route an attached image to something that can
    # actually see it — and away from the OCR specialists, which transcribe well
    # and reason poorly.
    for model in models:
        if isinstance(model, dict):
            model["vision"] = has_vision(model)
            model["ocr"] = is_ocr(model)
    return jsonify({
        "models": models,
        "default": get_default_model(),
        # Smallest general vision model: loading 19 GB because someone pasted a
        # screenshot is a poor surprise, and the user can still pick another.
        "vision_default": next(iter(vision_models(include_ocr=False)), None),
        # A transcriber, if installed: lets a text-only model handle an image
        # instead of the UI taking the conversation away from it.
        "ocr_default": next(iter(ocr_models()), None),
    })


# -------------------------------------------------------------------
# Conversation history
# -------------------------------------------------------------------


@app.route("/api/conversations", methods=["GET"])
def api_conversations() -> Any:
    """Every stored conversation, most recently updated first."""
    return jsonify({"conversations": store.list_conversations()})


@app.route("/api/conversations", methods=["POST"])
def api_conversation_create() -> Any:
    """Start a conversation and return its record."""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        body = {}
    model = body.get("model")
    return jsonify(store.create(str(body.get("title") or ""),
                                str(model) if model is not None else None))


@app.route("/api/conversations/<convo_id>", methods=["GET"])
def api_conversation_get(convo_id: str) -> Any:
    """One conversation with its full message list."""
    convo = store.get(convo_id)
    if convo is None:
        return jsonify({"error": "No such conversation"}), 404
    return jsonify(convo)


@app.route("/api/conversations/<convo_id>", methods=["PATCH"])
def api_conversation_rename(convo_id: str) -> Any:
    """Rename a conversation."""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        body = {}
    title = str(body.get("title") or "").strip()
    if not title:
        return jsonify({"error": "Missing 'title'"}), 400
    if not store.rename(convo_id, title):
        return jsonify({"error": "No such conversation"}), 404
    return jsonify({"ok": True, "title": title})


@app.route("/api/conversations/<convo_id>", methods=["DELETE"])
def api_conversation_delete(convo_id: str) -> Any:
    """Delete a conversation and its messages."""
    if not store.delete(convo_id):
        return jsonify({"error": "No such conversation"}), 404
    return jsonify({"ok": True})


@app.route("/api/conversations/<convo_id>/messages", methods=["POST"])
def api_conversation_add_message(convo_id: str) -> Any:
    """Append one message to a conversation."""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        body = {}
    role = str(body.get("role") or "")
    if role not in ("user", "assistant"):
        return jsonify({"error": "'role' must be 'user' or 'assistant'"}), 400
    images = body.get("images") if isinstance(body.get("images"), list) else None
    sources = body.get("sources") if isinstance(body.get("sources"), list) else None
    if not store.add_message(convo_id, role, str(body.get("content") or ""), images, sources):
        return jsonify({"error": "No such conversation"}), 404
    return jsonify({"ok": True})


@app.route("/api/chat", methods=["POST"])
def api_chat() -> Any:
    """Chat completion. Accepts a ``messages`` array or a single ``prompt``.

    Streams token-by-token NDJSON by default (what the UI consumes). Pass
    ``{"stream": false}`` to get a single JSON ``{"model", "reply"}`` object.
    """
    # A top-level array/string/number is valid JSON but has no .get, and
    # `or {}` only guards against falsy — so normalise before touching it.
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        body = {}
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

    use_web = bool(body.get("web")) and web_enabled()

    # Streaming path — pass Ollama's NDJSON lines straight through, preceded by
    # any web-grounding progress. Any failure (including one raised mid-stream
    # while iterating) is turned into a final JSON error line rather than a bare
    # HTTP 500, so the UI can show why.
    @stream_with_context
    def generate() -> Any:
        try:
            # Always send num_ctx, not only on web turns: changing a model's
            # load options mid-conversation makes Ollama reload the runner and
            # throw away the KV cache.
            convo, options = messages, {"num_ctx": get_num_ctx()}

            # An image attached to a model that cannot see used to force a
            # switch, which meant giving up your best model exactly when you
            # were debugging with a screenshot. Transcribe it instead and hand
            # the text over, so a coder model keeps the question.
            transcript = None
            # Any image still in the thread, not only a newly attached one, so a
            # follow-up question about the same screenshot keeps its context.
            images = web.conversation_images(messages)
            if images and not _model_has_vision(model):
                reader, is_ocr_reader = _image_reader(model)
                if reader:
                    yield _line({"status": f"Reading the image with {reader}…"})
                    transcript = web.read_images(images, reader, ocr=is_ocr_reader)
                    # Drop the base64 once it is transcribed: this model will
                    # never read it, and it costs the body limit every turn.
                    convo = web.with_context(
                        web.strip_images(messages),
                        web.image_context(transcript, ocr=is_ocr_reader),
                    )

            if use_web:
                documents: List[Dict[str, str]] = []
                for line in _gather_web(model, messages, documents, transcript):
                    yield line
                if documents:
                    # Layer the page context on whatever the image already added.
                    convo = web.with_context(convo, web.build_context(documents))
                    yield _line(
                        {"sources": [{"url": d["url"], "title": d["title"]} for d in documents]}
                    )
            for line in chat_stream(model, convo, options=options):
                yield line + "\n"
        except Exception as exc:  # noqa: BLE001 - surface any error to the client
            logger.exception("Chat stream failed")
            message = str(exc) or exc.__class__.__name__
            yield json.dumps({"error": message}) + "\n"

    return Response(generate(), mimetype="application/x-ndjson")


def _line(obj: Dict[str, Any]) -> str:
    """One NDJSON line for the client."""
    return json.dumps(obj) + "\n"


def _host_of(url: str) -> str:
    try:
        return urlparse(url).hostname or url
    except ValueError:
        return url


def _gather_web(
    model: str,
    messages: List[Dict[str, str]],
    documents: List[Dict[str, str]],
    transcript: Optional[str] = None,
) -> Any:
    """Collect web documents for this turn, yielding progress lines as it goes.

    Appends what it retrieves to ``documents``. A URL in the message is taken as
    an explicit instruction to read that page; otherwise the model is asked
    whether a search is warranted. Retrieval problems are reported and skipped —
    never fatal, since answering without the web beats not answering.
    """
    urls = web.find_urls(web.last_user_text(messages))
    if urls:
        for url in urls[:2]:
            yield _line({"status": f"Reading {_host_of(url)}…"})
            try:
                documents.append(web.fetch(url))
            except web.WebError as exc:
                yield _line({"status": str(exc)})
        return

    # An attached image usually carries the specifics worth searching for — the
    # exact error text, a version number — while the typed message is often just
    # "what's this?". Read it first so the planner has something to work with.
    # Reuse the transcription the chat path already made, if any — one OCR pass
    # per turn, serving both the answer and the search.
    image_note = transcript
    images = web.conversation_images(messages)
    if images and image_note is None:
        reader, is_reader_ocr = _image_reader(model)
        if reader:
            yield _line({"status": "Reading the image…" if is_reader_ocr else "Looking at the image…"})
            image_note = web.read_images(images, reader, ocr=is_reader_ocr)

    yield _line({"status": "Working out what to search for…"})
    queries = web.plan_searches(messages, model, image_note=image_note)
    if queries is None:
        yield _line({"status": "Could not plan a search; answering without one."})
        return
    if not queries:
        yield _line({"status": ""})
        return

    yield _line({"status": "Searching: " + " · ".join(queries)})
    # Concurrently: three searches then several fetches, run one after another,
    # each with its own timeout, is the sum of every round trip before the user
    # sees a single token.
    groups, failures = _run_all(web.search, queries)

    max_docs = get_web_max_docs()
    # Interleave so each query contributes, then fetch more candidates than
    # needed since some will be paywalled, JS-only, or plain unreachable.
    results = web.merge_results(groups, limit=max_docs * 2)
    if not results:
        yield _line({"status": failures[0] if failures else "No search results found."})
        return

    urls = [r["url"] for r in results]
    yield _line({"status": "Reading " + ", ".join(_host_of(u) for u in urls[:max_docs]) + "…"})
    fetched, _ = _run_all(web.fetch, urls)
    documents.extend(fetched[:max_docs])

    if not documents:
        yield _line({"status": "Found results but couldn't read any of them."})


def _model_has_vision(name: str) -> bool:
    """Whether the answering model can read an image itself.

    OCR models are excluded: they can technically take an image, but they
    transcribe rather than answer, so a conversation should not be handed to one.
    """
    try:
        return name in vision_models(include_ocr=False)
    except Exception:  # noqa: BLE001 - Ollama unreachable; assume it cannot
        return False


def _image_reader(answering_model: str) -> Any:
    """Pick the model that reads an attached image, and say whether it is OCR.

    An OCR model is preferred: what makes a good search query out of a
    screenshot is the exact error text, which transcription gets right and prose
    description paraphrases away. Falls back to the answering model when it can
    already see, then to the smallest general vision model.
    """
    pinned = get_vision_model()
    if pinned:
        # Only honour a pin that is actually installed — otherwise every image
        # read 404s and the user is told "no readable text was found", which is
        # simply false.
        try:
            installed = set(vision_models())
        except Exception:  # noqa: BLE001
            installed = set()
        if not installed or pinned in installed:
            return pinned, "ocr" in pinned.lower()
        logger.warning("WEB_VISION_MODEL=%r is not installed; ignoring it", pinned)
    try:
        ocr = next(iter(ocr_models()), "")
        if ocr:
            return ocr, True
        vision = vision_models()
        if answering_model in vision:
            return answering_model, False
        return next(iter(vision), ""), False
    except Exception:  # noqa: BLE001 - Ollama unreachable; skip the pass
        return "", False


def _run_all(func: Any, items: List[Any]) -> Any:
    """Run ``func`` over ``items`` concurrently, returning (results, errors).

    Order is preserved so interleaving and ranking stay deterministic; a WebError
    for one item is collected rather than raised, since one dead link must not
    cost the others.
    """
    if not items:
        return [], []
    results: List[Any] = [None] * len(items)
    errors: List[str] = []
    with ThreadPoolExecutor(max_workers=min(4, len(items))) as pool:
        futures = {pool.submit(func, item): i for i, item in enumerate(items)}
        for future in as_completed(futures):
            index = futures[future]
            try:
                results[index] = future.result()
            except web.WebError as exc:
                errors.append(str(exc))
            except Exception as exc:  # noqa: BLE001 - one bad item, not the turn
                logger.info("Retrieval failed for %r: %s", items[index], exc)
                errors.append(str(exc))
    return [r for r in results if r], errors


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
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        body = {}
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
