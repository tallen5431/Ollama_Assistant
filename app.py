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

import csv
import io
import json
import os
import queue
import socket
import sqlite3
import threading
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import urlparse

from flask import Flask, Response, jsonify, request, stream_with_context
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.middleware.proxy_fix import ProxyFix

import authz
import records
import store
import voice
import web
from chat_ui import render_page
from config import (
    get_app_title,
    get_default_model,
    get_host_port,
    get_image_turns,
    get_keep_alive,
    get_max_body_bytes,
    get_num_ctx,
    get_ollama_base,
    get_photo_keep_days,
    get_photo_meta_default,
    get_planner_model,
    get_share_photo_location,
    get_vision_model,
    get_web_follow_links,
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

# Keys go out in the order they were built, not alphabetically. A record's
# fields are the order the routine declared — start odometer, end odometer,
# distance — and sorting them turned a trip into "average speed" first. Nothing
# here reads JSON by key order, so this only affects how it looks.
app.json.sort_keys = False

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


# Records kept before extraction learned to strip LaTeX still hold it, and they
# are read back as table cells and as CSV — neither of which renders TeX. Once,
# at startup, and a no-op on a log that never had any.
try:
    _tidied = records.tidy_stored()
    if _tidied:
        logger.info("Tidied LaTeX out of %d stored record(s)", _tidied)
except Exception:  # noqa: BLE001 - a tidy-up must never stop the app starting
    logger.exception("Could not tidy stored records")


@app.route("/manifest.webmanifest", methods=["GET"])
def manifest() -> Any:
    """Make a home-screen shortcut open as an app rather than a browser tab.

    Worth about 100px of a 844px phone — the address bar and the tab strip —
    on the device this is mostly used from, and it gives the shortcut a name
    and an icon instead of a screenshot of the page.
    """
    title = get_app_title()
    resp = jsonify({
        "name": title,
        "short_name": title[:12],
        "start_url": ".",
        "scope": ".",
        "display": "standalone",
        "orientation": "any",
        "background_color": "#0a0e17",
        "theme_color": "#0a0e17",
        "icons": [{"src": _ICON, "sizes": "any", "type": "image/svg+xml",
                   "purpose": "any maskable"}],
    })
    resp.headers["Content-Type"] = "application/manifest+json"
    return resp


# The same mark the page uses for its favicon, as a data URL so there is no
# second file to serve and no request to 404 behind the server manager.
_ICON = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'"
    "%3E%3Crect width='32' height='32' rx='7' fill='%232563eb'/%3E%3Cpath d='M8 11h16M8 "
    "16h16M8 21h10' stroke='white' stroke-width='2.6' stroke-linecap='round'/%3E%3C/svg%3E"
)


@app.route("/healthz", methods=["GET"], strict_slashes=False)
def healthz() -> Any:
    """Plain-text health probe for the server manager / tunnels."""
    return "ok", 200


def _names_match(wanted: str, installed: List[str]) -> bool:
    """Whether ``wanted`` names one of ``installed``, Ollama's way.

    Ollama treats ``X`` and ``X:latest`` as the same model — that is the form
    ``ollama pull`` accepts and the form several of these models were installed
    under. Exact string equality flagged a perfectly working default as missing
    and told the owner to go pick a different one.
    """
    def canonical(name: str) -> str:
        name = (name or "").strip()
        return name[:-7] if name.endswith(":latest") else name

    target = canonical(wanted)
    return any(canonical(name) == target for name in installed if name)


@app.route("/api/health", methods=["GET"])
def health() -> Any:
    """JSON status, including whether Ollama actually answers.

    This used to echo the configured host without ever calling it, so nothing
    outside the browser could tell "app up, Ollama dead" from "all well" — and
    that is exactly the state the server-manager card shows. The probe uses a
    short timeout of its own: this endpoint is on the page-load path, and a
    sleeping desktop must not stall it.

    /healthz is deliberately left alone; it is the manager's liveness check and
    must keep measuring only whether this process is up.
    """
    reachable, detail, installed = True, "", []
    try:
        installed = [
            m.get("name") or m.get("model")
            for m in list_models(timeout=(3, 5))
            if isinstance(m, dict)
        ]
    except Exception as exc:  # noqa: BLE001 - the whole point is to report it
        reachable, detail = False, str(exc)

    default = get_default_model()
    return jsonify(
        {
            "status": "ok",
            "ollama_host": get_ollama_base(),
            "ollama_reachable": reachable,
            "ollama_error": detail,
            "model_count": len(installed),
            "default_model": default,
            # A default naming a model that has been deleted or renamed since
            # is a silent failure on every turn until someone notices.
            "default_installed": bool(installed) and _names_match(default, installed),
            "auth": AUTH_ENABLED,
            "voice": voice.voice_available(),
            "web": web_enabled(),
            "history": store.available(),
            # The browser trims image payloads before sending, so it needs
            # the same number the server would use.
            "image_turns": get_image_turns(),
            # Whether a browser that has never touched the toggle starts with
            # photo details on, so the deployment decides rather than the page.
            "photo_meta": get_photo_meta_default(),
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
    #
    # Copies, not in-place: list_models memoises, so mutating what it returns
    # would write these keys into the shared cache that every other caller reads.
    tagged = [
        {**m, "vision": has_vision(m), "ocr": is_ocr(m)} if isinstance(m, dict) else m
        for m in models
    ]
    return jsonify({
        "models": tagged,
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


@app.errorhandler(sqlite3.Error)
def _store_unavailable(exc: sqlite3.Error) -> Any:
    """Report a database failure as JSON, not as an HTML 500.

    None of the history routes caught sqlite3.Error, so a corrupt or full
    database produced Flask's HTML error page — which the client parsed as a
    failed fetch and then ignored, because saveMessage checked neither resp.ok
    nor the body. History silently stopped accumulating while the chat carried
    on working perfectly.

    ProgrammingError and InterfaceError are ours, not the disk's: a bad query
    or a wrong binding count would otherwise be reported to the user as a
    corrupt database and logged as one line with no stack, which is exactly the
    wrong place to start looking. Those get a traceback.
    """
    if isinstance(exc, (sqlite3.ProgrammingError, sqlite3.InterfaceError)):
        logger.exception("Bug in a history query, not a database problem")
        return jsonify({"error": "Internal error in the history layer."}), 500
    logger.error("History database error: %s", exc)
    return jsonify({
        # "Saved data", not "conversation history": routines live in the same
        # file behind the same write probe, and this is what they report too.
        "error": "Saved data is unavailable — the database could not be "
                 "written. Chatting still works; nothing is being saved.",
        "history": False,
    }), 503


@app.route("/api/conversations/stats", methods=["GET"])
def api_conversation_stats() -> Any:
    """What history is costing on disk, for the drawer to show."""
    out = store.stats()
    out["photo_keep_days"] = get_photo_keep_days()
    return jsonify(out)


@app.route("/api/conversations", methods=["GET"])
def api_conversations() -> Any:
    """Every stored conversation, most recently updated first."""
    # Hung off the one request every client makes on every visit, and rate
    # limited to once an hour inside the store. A timer thread would have to be
    # started, stopped and kept out of the tests to do work that can wait.
    store.maybe_forget_images(get_photo_keep_days())
    return jsonify({"conversations": store.list_conversations()})


@app.route("/api/search", methods=["GET"])
def api_search() -> Any:
    """Find a phrase across the saved conversations and the kept records."""
    return jsonify(store.search(request.args.get("q", "")))


@app.route("/api/photos/forget", methods=["POST"])
def api_forget_photos() -> Any:
    """Apply the photo retention now, rather than waiting for the sweep."""
    days = request.get_json(silent=True) or {}
    if not isinstance(days, dict):
        days = {}
    # The body may name its own cutoff, so "free up space now" can offer a
    # shorter one than the standing policy without changing the policy.
    try:
        cutoff = float(days.get("days", get_photo_keep_days()))
    except (TypeError, ValueError):
        cutoff = get_photo_keep_days()
    return jsonify(store.forget_images(max(0.0, cutoff)))


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
    meta = body.get("image_meta") if isinstance(body.get("image_meta"), list) else None
    if not store.add_message(convo_id, role, str(body.get("content") or ""), images,
                             sources, meta):
        return jsonify({"error": "No such conversation"}), 404
    return jsonify({"ok": True})


# -------------------------------------------------------------------
# Routines — saved prompts
# -------------------------------------------------------------------

# What a PATCH is allowed to change. Membership is tested against the request
# body rather than read with .get(), because null is a real value here: absent
# means "leave it alone", explicit null means "stop forcing this toggle".
_ROUTINE_FIELDS = ("name", "body", "photos", "web", "photo_meta", "record")


def _tri(value: Any) -> Optional[bool]:
    """Three-state: absent or null means leave the toggle where the user has it."""
    return None if value is None else bool(value)


def _count(value: Any) -> int:
    """A photo count from a request body. The store clamps the range.

    OverflowError as well as the obvious two: Python's json accepts
    ``Infinity``, and ``int(inf)`` raises rather than returning anything.
    """
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return 0


@app.route("/api/routines", methods=["GET"])
def api_routines() -> Any:
    """Every saved routine, in strip order."""
    return jsonify({"routines": store.list_routines()})


@app.route("/api/routines", methods=["POST"])
def api_routine_create() -> Any:
    """Save a routine and return its record."""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        body = {}
    name = str(body.get("name") or "").strip()
    text = str(body.get("body") or "").strip()
    if not name:
        return jsonify({"error": "Missing 'name'"}), 400
    if not text:
        return jsonify({"error": "Missing 'body'"}), 400
    record = body.get("record")
    return jsonify(store.create_routine(
        name, text, _count(body.get("photos")),
        _tri(body.get("web")), _tri(body.get("photo_meta")),
        record if isinstance(record, list) else None,
    ))


@app.route("/api/routines/starters", methods=["POST"])
def api_routine_starters() -> Any:
    """Install the shipped routines, skipping any name already taken.

    The JSON requirement is the point of the check, not a formality: it is the
    only routines route that reads no body, so without it a plain HTML form on
    any page could POST here cross-origin — a form post is a "simple" request
    and is never preflighted. Every other route here is already unreachable
    that way, by needing JSON or a method a form cannot send.
    """
    if not request.is_json:
        return jsonify({"error": "Expected a JSON request"}), 415
    return jsonify({"routines": store.create_starters()})


@app.route("/api/routines/<routine_id>", methods=["PATCH"])
def api_routine_update(routine_id: str) -> Any:
    """Change some of a routine, returning the whole normalised record."""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        body = {}
    fields: Dict[str, Any] = {}
    for field in _ROUTINE_FIELDS:
        if field not in body:
            continue
        if field in ("name", "body"):
            fields[field] = str(body[field] or "").strip()
        elif field == "photos":
            fields[field] = _count(body[field])
        elif field == "record":
            fields[field] = body[field] if isinstance(body[field], list) else []
        else:
            fields[field] = _tri(body[field])
    if not fields:
        return jsonify({"error": "Nothing to update"}), 400
    if not fields.get("name", "x") or not fields.get("body", "x"):
        return jsonify({"error": "A routine needs a name and a prompt"}), 400
    routine = store.update_routine(routine_id, fields)
    if routine is None:
        return jsonify({"error": "No such routine"}), 404
    return jsonify({"ok": True, "routine": routine})


@app.route("/api/routines/<routine_id>", methods=["DELETE"])
def api_routine_delete(routine_id: str) -> Any:
    """Remove a routine."""
    if not store.delete_routine(routine_id):
        return jsonify({"error": "No such routine"}), 404
    return jsonify({"ok": True})


# -------------------------------------------------------------------
# Records — what a routine run wrote down
# -------------------------------------------------------------------


@app.route("/api/records", methods=["GET"])
def api_records() -> Any:
    """Kept records, newest first. ``?routine=`` narrows to one."""
    rows = store.list_records(str(request.args.get("routine") or ""))
    return jsonify({"records": rows, "columns": store.record_columns(rows)})


@app.route("/api/records", methods=["POST"])
def api_record_create() -> Any:
    """Restate a routine's answer as its declared fields, and keep it.

    Called by the browser once the reply has finished, rather than from inside
    the streaming generator: that path is delicate, and a record is worth less
    than the answer already on screen. A failure here is a 200 with no record.
    """
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        body = {}
    answer = str(body.get("answer") or "").strip()
    fields = body.get("fields")
    if not answer:
        return jsonify({"error": "Missing 'answer'"}), 400
    if not isinstance(fields, list) or not fields:
        return jsonify({"error": "Missing 'fields'"}), 400

    model = str(body.get("model") or "") or get_default_model()
    extracted = records.extract(answer, [str(f) for f in fields], model)
    kept = store.add_record(
        str(body.get("routine_name") or "Routine"), extracted,
        str(body.get("routine_id") or "") or None,
        str(body.get("conversation_id") or "") or None,
    )
    # 200 either way. "Nothing could be pulled out of that answer" is an
    # outcome, not an error, and the reply it came from is still on screen.
    return jsonify({"record": kept})


@app.route("/api/records/<record_id>", methods=["PATCH"])
def api_record_update(record_id: str) -> Any:
    """Correct a record. The extraction is good, not beyond question."""
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or not isinstance(body.get("fields"), dict):
        return jsonify({"error": "Missing 'fields'"}), 400
    record = store.update_record(record_id, body["fields"])
    if record is None:
        return jsonify({"error": "No such record"}), 404
    return jsonify({"ok": True, "record": record})


@app.route("/api/records/<record_id>", methods=["DELETE"])
def api_record_delete(record_id: str) -> Any:
    """Remove one record."""
    if not store.delete_record(record_id):
        return jsonify({"error": "No such record"}), 404
    return jsonify({"ok": True})


# Excel and Sheets read a leading =, + or @ as a formula, so a value that
# arrived here as text becomes something that runs when the file is opened.
# Not "-": a negative number is an ordinary reading — a temperature, a
# correction — and quoting it would corrupt the log to prevent nothing, since
# a spreadsheet only treats "-" as a formula when what follows is not a number.
_CSV_FORMULA = ("=", "+", "@", "\t", "\r")


def _csv_safe(value: str) -> str:
    """One cell, with a leading formula character defused."""
    text = str(value or "")
    return "'" + text if text.startswith(_CSV_FORMULA) else text


@app.route("/api/records.csv", methods=["GET"])
def api_records_csv() -> Any:
    """The whole log as CSV, for a spreadsheet or another machine's importer.

    A real endpoint rather than a download built in the browser: pulling this
    with curl from wherever the records are meant to end up is the point.
    """
    rows = store.list_records(str(request.args.get("routine") or ""))
    columns = store.record_columns(rows)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["taken_at", "routine"] + columns)
    for row in rows:
        stamp = datetime.fromtimestamp(row["created_at"]).isoformat(timespec="seconds")
        writer.writerow([stamp, _csv_safe(row["routine_name"])] +
                        [_csv_safe(row["fields"].get(name, "")) for name in columns])
    return Response(
        buffer.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": 'attachment; filename="records.csv"'},
    )


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
            reply = ollama_chat(model, web.strip_image_meta(messages))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 502
        return jsonify({"model": model, "reply": reply})

    use_web = bool(body.get("web")) and web_enabled()
    # Which stored thread this turn belongs to, if any. The browser used to
    # write the reply down itself once the stream finished, which is fine right
    # up until the tab is not there when it finishes: a phone that locks past
    # its wake lock, an app switch, a handoff from wifi to cellular. Two
    # minutes of a 30b model's work then existed nowhere at all. Given the id,
    # this end writes the turn down as soon as it has one — including when the
    # client is the thing that went away.
    convo_id = body.get("conversation_id")
    convo_id = convo_id if isinstance(convo_id, str) and convo_id else None
    last_user = messages[-1] if isinstance(messages[-1], dict) else {}
    answer: List[str] = []
    thinking: List[str] = []
    kept_steps: List[Dict[str, Any]] = []
    kept_sources: List[Dict[str, str]] = []

    # Streaming path — pass Ollama's NDJSON lines straight through, preceded by
    # any web-grounding progress. Any failure (including one raised mid-stream
    # while iterating) is turned into a final JSON error line rather than a bare
    # HTTP 500, so the UI can show why.
    def run_turn() -> Iterator[str]:
        """The whole turn, as NDJSON lines. Any failure becomes a final line."""
        try:
            # Always send num_ctx, not only on web turns: changing a model's
            # load options mid-conversation makes Ollama reload the runner and
            # throw away the KV cache.
            options = {"num_ctx": get_num_ctx()}

            # A photo's own record of when and where it was taken. Even a
            # vision model needs telling: the browser re-encodes every upload
            # through a canvas, so the pixels that arrive here carry no EXIF at
            # all and no model, however good its eyes, can read it off them.
            #
            # Read it once and then take it off the messages, so that after this
            # point the facts travel as prose that was deliberately put
            # somewhere, rather than as a JSON field riding along everywhere.
            photo_meta = web.conversation_image_meta(messages)
            meta_context = web.image_metadata(photo_meta)
            carried = sum(1 for m in photo_meta if m)
            if not web.conversation_images(messages):
                pass                      # nothing attached; no photo step at all
            elif meta_context:
                yield _step("Photo details",
                            f"{carried} photo(s) carried their own record; "
                            f"{len(meta_context)} characters given to the model",
                            text=meta_context)
            else:
                yield _step("Photo details",
                            "none — the toggle is off, or the file carried no "
                            "EXIF (a screenshot, or a copy that stripped it)")
            # The planner gets a shorter version. It runs on your own hardware,
            # but what it writes is sent to a search engine — so the position is
            # the one part of this that can leave the house, and it has its own
            # switch (WEB_SHARE_LOCATION).
            photo_note = web.metadata_note(photo_meta,
                                           with_location=get_share_photo_location())
            turns = web.strip_image_meta(messages)

            # Only the most recent image-bearing turn keeps its payload. A
            # vision model otherwise re-reads every screenshot in the thread on
            # every turn, and turn six is almost never about turn one's image.
            # Earlier turns keep their text, so the conversation still reads.
            convo = web.keep_recent_images(turns, keep_turns=get_image_turns())

            # An image attached to a model that cannot see used to force a
            # switch, which meant giving up your best model exactly when you
            # were debugging with a screenshot. Transcribe it instead and hand
            # the text over, so a coder model keeps the question.
            transcript = None
            # Any image still in the thread, not only a newly attached one, so a
            # follow-up question about the same screenshot keeps its context.
            images = web.conversation_images(turns)
            if images:
                yield _step(
                    "Images", f"{len(images)} in the thread; {model} "
                    + ("reads them itself" if _model_has_vision(model)
                       else "cannot see, so they are transcribed for it"))
            if images and not _model_has_vision(model):
                reader, is_ocr_reader = _image_reader(model)
                if reader:
                    yield _line({"status": f"Reading the image with {reader}…"})
                    describing = not is_ocr_reader
                    try:
                        transcript = web.read_images(images, reader, ocr=is_ocr_reader,
                                                     answering_model=model)
                        # An OCR model that found nothing is not the end of it:
                        # a photo of a plant has no text, and telling the user
                        # to go change a dropdown while a vision model sits idle
                        # is a poor answer to "what is this?".
                        if transcript is None and is_ocr_reader:
                            describer = _describing_model(model)
                            if describer:
                                yield _line({"status": f"No text in it — looking at it with {describer}…"})
                                transcript = web.read_images(images, describer, ocr=False,
                                                             answering_model=model)
                                describing = True
                        context = web.image_context(transcript, ocr=not describing)
                    except web.ReadFailed as exc:
                        # A reader-model OOM is routine when a 30b model holds
                        # the GPU. Reporting it as "no readable text was found"
                        # is a confident, wrong claim about the user's image.
                        logger.warning("Image read via %s failed: %s", reader, exc)
                        yield _line({"status": f"{reader} could not read the image."})
                        context = web.image_failed_context()
                    # Drop the base64 once it is transcribed: this model will
                    # never read it, and it costs the body limit every turn.
                    convo = web.with_context(web.strip_images(turns), context)

            if meta_context:
                convo = web.with_context(convo, meta_context)

            if use_web:
                documents: List[Dict[str, str]] = []
                outcome: Dict[str, bool] = {}
                for line in _gather_web(model, turns, documents, transcript, outcome,
                                        photo_note=photo_note):
                    yield line
                if not documents and outcome.get("attempted"):
                    # Retrieval ran and produced nothing. Say so, or the answer
                    # comes back from stale memory sounding exactly like a
                    # sourced one — the worst possible failure mode for a
                    # feature whose whole point is not guessing.
                    convo = web.with_context(convo, web.no_results_context())
                if documents:
                    # Layer the page context on whatever the image already added,
                    # bounded so the pages cannot crowd the conversation out of
                    # the window — Ollama drops the oldest turns silently.
                    convo = web.with_context(
                        convo,
                        web.build_context(documents,
                                          char_budget=web.context_budget(get_num_ctx())),
                    )
                    kept_sources.extend(
                        {"url": d["url"], "title": d["title"]} for d in documents)
                    yield _line({"sources": list(kept_sources)})
            yield _step(
                "Sent to the model",
                f"{len(convo)} turns, {sum(len(str(t.get('content') or '')) for t in convo)} "
                f"characters of text, num_ctx {options.get('num_ctx')}",
                system=[str(t.get("content") or "")[:2000]
                        for t in convo if t.get("role") == "system"])
            for line in chat_stream(model, convo, options=options,
                                    keep_alive=get_keep_alive() or None):
                answer.append(_content_of(line))
                thinking.append(_thinking_of(line))
                yield line + "\n"
        except Exception as exc:  # noqa: BLE001 - surface any error to the client
            logger.exception("Chat stream failed")
            message = str(exc) or exc.__class__.__name__
            yield json.dumps({"error": message}) + "\n"

    if not convo_id:
        # Nothing to write down, so nothing to protect: stream straight through
        # and let a departed client stop the work, as it always has. This is
        # the path an API client and a history-less browser take.
        return Response(stream_with_context(run_turn()),
                        mimetype="application/x-ndjson")

    # With somewhere to save it, the work outlives the connection. A WSGI
    # response is pulled by the client: when the phone locks or the app is
    # switched away, the server stops being asked for the next chunk and so
    # stops generating — which is why a saved turn on its own only kept the
    # first sentence. Producing on a thread and relaying through a queue means
    # the model finishes its answer whether or not anyone is still listening.
    lines: "queue.Queue[Optional[str]]" = queue.Queue()
    cancel = _start_turn(convo_id)

    def produce() -> None:
        try:
            for line in run_turn():
                if cancel.is_set():
                    break     # closes chat_stream, which lets Ollama go
                entry = _debug_of(line)
                if entry is not None:
                    kept_steps.append(entry)
                lines.put(line)
        finally:
            _end_turn(convo_id, cancel)
            # Written before the sentinel, not after: the browser refreshes its
            # list the moment the stream ends, and putting the save afterwards
            # made that a race the list usually lost — a finished thread with
            # no title until something else happened to refresh it.
            # Saved even when cancelled: Stop usually means "I have read
            # enough", and the half you read is worth keeping.
            _keep_turn(convo_id, last_user, "".join(answer), kept_sources,
                       "".join(thinking), kept_steps)
            lines.put(None)

    threading.Thread(target=produce, name="chat-turn", daemon=True).start()

    def relay() -> Iterator[str]:
        while True:
            line = lines.get()
            if line is None:
                return
            yield line

    return Response(stream_with_context(relay()), mimetype="application/x-ndjson")


# One in-flight turn per conversation, so Stop only needs to name the thread.
# Without this the detached producer would keep a 30b model busy for a minute
# after the button that exists to stop it was pressed.
_TURNS: Dict[str, threading.Event] = {}
_TURNS_LOCK = threading.Lock()


def _start_turn(convo_id: str) -> threading.Event:
    flag = threading.Event()
    with _TURNS_LOCK:
        earlier = _TURNS.get(convo_id)
        if earlier is not None:
            earlier.set()      # a second turn in one thread supersedes the first
        _TURNS[convo_id] = flag
    return flag


def _end_turn(convo_id: str, flag: threading.Event) -> None:
    with _TURNS_LOCK:
        if _TURNS.get(convo_id) is flag:
            _TURNS.pop(convo_id, None)


@app.route("/api/chat/cancel", methods=["POST"])
def api_chat_cancel() -> Any:
    """Stop the turn running for a conversation, if there is one."""
    body = request.get_json(silent=True)
    convo_id = body.get("conversation_id") if isinstance(body, dict) else None
    with _TURNS_LOCK:
        flag = _TURNS.get(convo_id) if isinstance(convo_id, str) else None
    if flag is not None:
        flag.set()
    return jsonify({"cancelled": flag is not None})


def _thinking_of(line: str) -> str:
    """The reasoning in one of Ollama's NDJSON lines, if it carries any."""
    try:
        obj = json.loads(line)
    except (ValueError, TypeError):
        return ""
    message = obj.get("message") if isinstance(obj, dict) else None
    if not isinstance(message, dict):
        return ""
    return str(message.get("thinking") or "")


def _content_of(line: str) -> str:
    """The reply text in one of Ollama's NDJSON lines, if it carries any."""
    try:
        obj = json.loads(line)
    except (ValueError, TypeError):
        return ""
    message = obj.get("message") if isinstance(obj, dict) else None
    if not isinstance(message, dict):
        return ""
    return str(message.get("content") or "")


def _keep_turn(convo_id: Optional[str], user: Dict[str, Any], reply: str,
               sources: List[Dict[str, str]], thinking: str = "",
               steps: Optional[List[Dict[str, Any]]] = None) -> None:
    """Write the question and its answer down. Never raises into the stream."""
    if not convo_id:
        return
    # The stored reply is the answer, not the model talking to itself: a
    # reasoning block is scratch work, and a thread reopened tomorrow should
    # read the way it read today.
    text = web.strip_thinking(reply).strip()
    if not text:
        return
    try:
        store.save_turn(convo_id, user, text, sources or None,
                        thinking=thinking, steps=steps or None)
    except Exception:  # noqa: BLE001 - history is the extra, never the turn
        logger.exception("Could not save the turn to %s", convo_id)


def _line(obj: Dict[str, Any]) -> str:
    """One NDJSON line for the client."""
    return json.dumps(obj) + "\n"


def _step(name: str, detail: str = "", **extra: Any) -> str:
    """One line for the panel that says what this turn actually did.

    The status line says what is happening now and is replaced by the next one;
    this accumulates, so afterwards you can see why an answer came out the way
    it did — which photo details were read, what was searched for, what came
    back, what was put in front of the model. Kept out of the way behind a
    collapsed panel, because it is only ever wanted when something looks wrong.
    """
    entry: Dict[str, Any] = {"step": name}
    if detail:
        entry["detail"] = detail
    entry.update(extra)
    return _line({"debug": entry})


def _debug_of(line: str) -> Optional[Dict[str, Any]]:
    """The panel entry in a line, if it is one. Recorded where the lines are
    relayed rather than at each yield, so a step added later is kept without
    anyone having to remember to keep it."""
    try:
        obj = json.loads(line)
    except (ValueError, TypeError):
        return None
    entry = obj.get("debug") if isinstance(obj, dict) else None
    return entry if isinstance(entry, dict) else None


def _host_of(url: str) -> str:
    try:
        return urlparse(url).hostname or url
    except ValueError:
        return url


def _follow_links(
    model: str,
    question: str,
    source: Dict[str, Any],
    documents: List[Dict[str, str]],
    budget: int,
) -> Any:
    """Open a couple of the pages ``source`` links to, if any look relevant.

    Same-site only, one hop, and every URL still goes through the address guard
    in fetch() — a link is chosen by a model from content written by a stranger,
    so it gets no more trust than a pasted URL does.
    """
    candidates = [
        link for link in (source.get("links") or [])
        if web.same_site(source.get("url", ""), link["url"])
        and link["url"] not in {d.get("url") for d in documents}
        and link["url"] not in {d.get("requested") for d in documents}
    ]
    if not candidates:
        return

    picker = get_planner_model() or model
    try:
        chosen = web.choose_links(question, candidates, picker, max_links=budget,
                                  answering_model=model)
    except Exception as exc:  # noqa: BLE001 - an enhancement, never a requirement
        logger.warning("Link picking failed: %s", exc)
        return
    if not chosen:
        return

    yield _line({"status": "Following: " + " · ".join(c["text"][:40] for c in chosen)})
    fetched, failures = _run_all(web.fetch, [c["url"] for c in chosen])
    documents.extend(fetched)
    if failures and not fetched:
        yield _line({"status": "Could not read the linked pages."})


def _gather_web(
    model: str,
    messages: List[Dict[str, str]],
    documents: List[Dict[str, str]],
    transcript: Optional[str] = None,
    outcome: Optional[Dict[str, bool]] = None,
    photo_note: str = "",
) -> Any:
    """Collect web documents for this turn, yielding progress lines as it goes.

    Appends what it retrieves to ``documents``. A URL in the message is taken as
    an explicit instruction to read that page; otherwise the model is asked
    whether a search is warranted. Retrieval problems are reported and skipped —
    never fatal, since answering without the web beats not answering.

    ``outcome["attempted"]`` records whether retrieval was actually tried, which
    the caller needs to tell "searched and found nothing" apart from "decided no
    search was needed" — identical from the outside, opposite in meaning.
    """
    if outcome is None:
        outcome = {}
    question = web.last_user_text(messages)
    urls = web.find_urls(question)
    if urls:
        outcome["attempted"] = True
        for url in urls[:2]:
            yield _line({"status": f"Reading {_host_of(url)}…"})
            try:
                documents.append(web.fetch(url))
            except web.WebError as exc:
                yield _line({"status": str(exc)})

        # A linked page is rarely self-contained: a wiki article answers half
        # the question and points at the page with the other half. Ask a small
        # model which of its links are worth opening, and follow a couple.
        # Only from a page the *user* chose, and only within the same site —
        # following a model's pick of an arbitrary outbound link is a much
        # larger surface for very little gain.
        budget = get_web_follow_links()
        for source in list(documents):
            if budget < 1:
                break
            yield from _follow_links(model, question, source, documents, budget)
            budget = 0     # one hop, from the first page only
        return

    # An attached image usually carries the specifics worth searching for — the
    # exact error text, a version number — while the typed message is often just
    # "what's this?". Read it first so the planner has something to work with.
    # Reuse the transcription the chat path already made, if any — one OCR pass
    # per turn, serving both the answer and the search.
    # last_user_images, not conversation_images: only an image attached to *this*
    # turn should shape the search. Scanning the whole thread meant a screenshot
    # from turn 1 was still steering queries at turn 11 ("what's the weather in
    # Paris" planned against a stale traceback). That gate has to cover the
    # reused transcript too, which is read from the whole thread by design —
    # otherwise the stale image walks straight back in through the reuse path.
    image_note = None
    images = web.last_user_images(messages)
    if images:
        image_note = transcript
        if image_note is None:
            reader, is_reader_ocr = _image_reader(model)
            if reader:
                yield _line({"status": "Reading the image…" if is_reader_ocr else "Looking at the image…"})
                try:
                    image_note = web.read_images(images, reader, ocr=is_reader_ocr,
                                                 answering_model=model)
                except web.ReadFailed as exc:
                    # Planning without the image beats not answering.
                    logger.warning("Image read via %s failed: %s", reader, exc)
                    yield _line({"status": "Could not read the image; searching without it."})
                    yield _step("Read the image for search", f"{reader} failed: {exc}")
                else:
                    yield _step("Read the image for search",
                                f"{reader} ({'OCR' if is_reader_ocr else 'description'})",
                                text=image_note or "(nothing came back)")
            else:
                yield _step("Read the image for search",
                            "no reader available — the search is planned from "
                            "your words alone, which for a photo with no "
                            "question in it is very little to go on")

    yield _line({"status": "Working out what to search for…"})
    # Only alongside an image attached to *this* turn, for the same reason the
    # transcription is gated that way: last week's photo should not still be
    # steering today's queries.
    queries = web.plan_searches(messages, model, image_note=image_note,
                                photo_note=photo_note if images else "")
    if queries is None:
        yield _line({"status": "Could not plan a search; answering without one."})
        yield _step("Planned searches", "the planner could not be reached")
        return
    if not queries:
        yield _line({"status": ""})
        yield _step("Planned searches",
                    "none — the planner judged that a search would not help")
        return

    outcome["attempted"] = True
    yield _step("Planned searches", " · ".join(queries))
    yield _line({"status": "Searching: " + " · ".join(queries)})
    # Concurrently: three searches then several fetches, run one after another,
    # each with its own timeout, is the sum of every round trip before the user
    # sees a single token.
    groups, failures = _run_all(web.search, queries)

    max_docs = get_web_max_docs()
    # Interleave so each query contributes, then fetch more candidates than
    # needed since some will be paywalled, JS-only, or plain unreachable.
    results = web.merge_results(groups, limit=max_docs * 2)
    # Deduped: three queries hitting the same broken backend produced the same
    # sentence three times, which reads as three different problems.
    unique: List[str] = []
    for note in failures:
        if note not in unique:
            unique.append(note)
    yield _step("Search results",
                f"{len(results)} from {len(groups)} query group(s)"
                + (f"; {' | '.join(unique)}" if unique else ""),
                urls=[r["url"] for r in results[:12]])
    if not results:
        yield _line({"status": failures[0] if failures else "No search results found."})
        return

    urls = [r["url"] for r in results]
    yield _line({"status": "Reading " + ", ".join(_host_of(u) for u in urls[:max_docs]) + "…"})
    fetched, fetch_failures = _run_all(web.fetch, urls, enough=max_docs)
    documents.extend(fetched[:max_docs])
    yield _step("Pages read", f"{len(fetched)} of {len(urls)} tried"
                + (f"; failures: {'; '.join(fetch_failures[:4])}" if fetch_failures else ""),
                urls=[d.get("url", "") for d in fetched[:max_docs]])

    # Paywalled, JS-only and dead pages are routine. Their search snippets are
    # already paid for, so use them to fill out the budget rather than throwing
    # away everything that query found. Labelled as summaries in the context.
    if len(documents) < max_docs:
        documents.extend(
            web.snippet_documents(results, documents, limit=max_docs - len(documents))
        )

    if not documents:
        yield _line({"status": "Found results but couldn't read any of them."})
    yield _step("Context from the web",
                f"{len(documents)} document(s), "
                f"{sum(len(d.get('text') or '') for d in documents)} characters")


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


def _describing_model(answering_model: str) -> str:
    """A general vision model to describe an image, or "" if none is installed.

    Used only when the OCR pass found no text at all. Transcription is the
    right first choice — it keeps the exact error string a screenshot carries,
    which prose paraphrases away — but a photo of a plant has no text in it,
    and the reply then told the user to change a dropdown while a perfectly
    good vision model sat idle.

    Smallest first, deliberately: silently loading 19 GB because someone
    photographed their dinner is a poor surprise.
    """
    try:
        vision = vision_models(include_ocr=False)
    except Exception:  # noqa: BLE001 - Ollama unreachable
        return ""
    if answering_model in vision:
        return answering_model
    return next(iter(vision), "")


# How long a better-ranked page may keep the turn waiting once enough others
# have landed. Long enough to matter on a normal site, short enough that a slow
# one cannot reintroduce the dead air the early return removed.
_STRAGGLER_GRACE = 1.5


def _run_all(func: Any, items: List[Any], enough: Optional[int] = None) -> Any:
    """Run ``func`` over ``items`` concurrently, returning (results, errors).

    Order is preserved so interleaving and ranking stay deterministic; a WebError
    for one item is collected rather than raised, since one dead link must not
    cost the others.

    ``enough`` stops waiting once that many have succeeded. Fetching six
    candidates to keep three meant the turn waited for the slowest of six even
    when three had long since landed — worst case two full timeouts of dead air
    before the first token, since a pool of four also ran them in two waves.
    """
    if not items:
        return [], []
    results: List[Any] = [None] * len(items)
    errors: List[str] = []
    done = 0
    # These are pure I/O waits, so the pool can be as wide as the work — one
    # wave rather than two, and a stuck fetch no longer delays a queued one.
    #
    # Not a `with` block: the context manager joins every worker on exit, which
    # would undo the early return entirely — the turn would still wait for the
    # slowest of six after the third had landed.
    # thread_name_prefix marks these in a stack dump; more importantly, the
    # abandoned ones are joined by concurrent.futures' interpreter-exit hook, so
    # restarting the card mid-fetch would stall for the remaining budget. Each
    # fetch is bounded by its own wall clock, so that wait is finite — but the
    # deadline is what keeps it short, not the pool.
    pool = ThreadPoolExecutor(max_workers=min(8, len(items)),
                              thread_name_prefix="retrieval")
    futures: Dict[Any, int] = {}
    try:
        futures = {pool.submit(func, item): i for i, item in enumerate(items)}
        def record(future: Any) -> bool:
            index = futures[future]
            try:
                results[index] = future.result()
                return True
            except web.WebError as exc:
                errors.append(str(exc))
            except Exception as exc:  # noqa: BLE001 - one bad item, not the turn
                logger.info("Retrieval failed for %r: %s", items[index], exc)
                errors.append(str(exc))
            return False

        for future in as_completed(futures):
            if record(future):
                done += 1
            if enough is not None and done >= enough:
                break

        # Enough have landed. Stopping dead here would select by completion
        # order rather than by the ranking that chose these candidates — a fast
        # low-ranked page beating a slow top-ranked one. Give any *better*
        # ranked stragglers a short, bounded wait; the caller keeps the best few
        # by rank, so whatever arrives in time simply improves the selection.
        #
        # A bounded wait(), not another pass over as_completed(): that blocks
        # until the next completion whenever it is, which is not a grace period
        # at all — it is the full wait the early return exists to avoid.
        if enough is not None and done >= enough:
            best_kept = max((i for i, r in enumerate(results) if r), default=-1)
            better = [f for f, i in futures.items() if not f.done() and i < best_kept]
            if better:
                done_futures, _ = wait(better, timeout=_STRAGGLER_GRACE)
                for future in done_futures:
                    record(future)
    finally:
        # Drop what has not started; abandon what has. Each in-flight fetch is
        # bounded by its own wall-clock deadline, so nothing runs forever.
        # cancel() rather than shutdown(cancel_futures=...), which needs 3.9.
        for future in futures:
            future.cancel()
        pool.shutdown(wait=False)
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
    # Checked before a byte is streamed: once the response has started there is
    # no status code left to send, and "unknown model" is knowable now.
    if str(model_id) not in voice.CATALOG:
        return jsonify({"error": f"Unknown voice model: {str(model_id)!r}"}), 400
    # Streamed, in the same NDJSON the chat uses. The large English model is
    # 1.8 GB — ten minutes on a domestic line — and ten minutes of a silent
    # request is indistinguishable from a broken one, which is what this
    # looked like. The work runs on a thread so a slow line cannot stall the
    # relay, and the client sees a percentage the whole way down.
    lines: "queue.Queue[Optional[str]]" = queue.Queue()
    sent = [0.0]

    def progress(done: int, total: int) -> None:
        # Rate limited to whole percents: a 1.8 GB file in 256KB chunks is
        # 7000 callbacks, and 7000 lines of JSON is its own small download.
        share = (done / total) if total else 0.0
        if total and share - sent[0] < 0.01 and done < total:
            return
        sent[0] = share
        lines.put(_line({"downloaded": done, "total": total,
                         "percent": round(share * 100)}))

    def fetch() -> None:
        try:
            lines.put(_line(voice.download_model(str(model_id), progress)))
        except ValueError as exc:
            lines.put(_line({"error": str(exc)}))
        except Exception as exc:  # noqa: BLE001 - report, never 500 mid-stream
            logger.exception("Voice model download failed")
            lines.put(_line({"error": f"Download failed: {exc}"}))
        finally:
            lines.put(None)

    threading.Thread(target=fetch, name="voice-download", daemon=True).start()

    def relay() -> Iterator[str]:
        while True:
            line = lines.get()
            if line is None:
                return
            yield line

    return Response(stream_with_context(relay()), mimetype="application/x-ndjson")


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
