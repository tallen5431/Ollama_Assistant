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
import re
import socket
import sqlite3
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import urlparse

from flask import Flask, Response, jsonify, request, stream_with_context
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from werkzeug.middleware.proxy_fix import ProxyFix

import authz
import records
import store
import fields
import values
import voice
import web
from chat_ui import render_page
from config import (
    get_app_title,
    get_default_model,
    get_distiller_model,
    get_host_port,
    get_image_turns,
    get_keep_alive,
    get_max_body_bytes,
    get_num_ctx,
    get_ollama_base,
    get_photo_keep_days,
    get_photo_meta_default,
    get_photo_read_each,
    get_planner_model,
    get_search_url,
    get_server_threads,
    get_share_photo_location,
    get_vision_model,
    get_web_fetch_hops,
    get_web_follow_links,
    get_web_follow_on_search,
    get_web_max_hops,
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


@app.errorhandler(HTTPException)
def _http_error(exc: HTTPException) -> Any:
    """Answer an /api/ request as JSON whatever went wrong.

    Every handler here returns JSON and the page parses it as JSON — but a 404
    from an unmatched path, or a 405 from the wrong method, came from Werkzeug
    as an HTML page. `await resp.json()` throws on that, so the reason arrived
    at the browser as a parse error with the actual status lost. This is the
    same fix _too_large already applies to 413, applied to the rest.

    Pages keep the HTML: a person who mistypes a URL should see the friendly
    page, not a JSON blob.
    """
    if not request.path.startswith("/api/"):
        return exc
    return jsonify({"error": exc.description or exc.name}), exc.code or 500


# -------------------------------------------------------------------
# Optional HTTP Basic Auth (makes internet exposure safe)
# -------------------------------------------------------------------
AUTH_ENABLED = authz.auth_enabled()
# Auth off is the default and a fine choice behind Tailscale. Auth off because
# a setting did not take is a different thing that looked identical from the
# outside, so it is said out loud — once, at import, where the startup banner
# will be read.
_AUTH_PROBLEM = authz.misconfigured()
if _AUTH_PROBLEM:
    logger.warning("%s", _AUTH_PROBLEM)
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

# And the same for the shape of the values themselves: "102,072", "102,072 mi"
# and "100,409 miles" are one column written three ways, none of which sorts
# against the others. Records already kept are the ones that need this most,
# since they are the whole log. Lossless — a value that changes has the model's
# own wording written to `raw` first — and idempotent, so a tidy log is a no-op.
try:
    store.normalise_stored()
except Exception:  # noqa: BLE001 - same rule: never stop the app starting
    logger.exception("Could not standardise stored records")


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
            # Which search backend this process actually has. SEARXNG_URL is
            # set wherever the environment for this app comes from, which is
            # not the shell anyone is standing in — so an export that never
            # reached the app looks exactly like an app ignoring it, and the
            # only way to tell them apart was to run a search and read the
            # panel. Named "searxng" rather than echoed as a URL when unset, so
            # the answer to "is my instance live?" is one field.
            "search_backend": web.effective_search_url() or "duckduckgo",
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
    #
    # Started, not awaited: the sweep ends in a VACUUM, which rewrites the whole
    # database and blocks every writer — measured at 2.8 s on 210 MB of stored
    # photos. That is the drawer hanging on open, once an hour, for work nobody
    # asked for. /api/photos/forget is still synchronous, because there someone
    # did ask and is waiting for the number.
    store.sweep_in_background(get_photo_keep_days())
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
    saved = store.create_routine(
        name, text, _count(body.get("photos")),
        _tri(body.get("web")), _tri(body.get("photo_meta")),
        record if isinstance(record, list) else None,
    )
    # Saved either way, and told what is wrong with it. A formula naming a
    # field that does not exist is not an error anywhere downstream — it
    # computes to nothing, every run, and an empty column looks exactly like a
    # run with no data. Saying so here is the difference between a typo fixed
    # in seconds and one found in a month of records.
    return jsonify({**saved, "problems": _record_problems(saved.get("record"))})


def _record_problems(record: Any) -> List[str]:
    """What is wrong with a routine's field declarations, in words."""
    return fields.problems(fields.parse(record), record) if record else []


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
    # Not named `fields`: that is the module which reads the declarations, and
    # shadowing it here would turn every call on it into a method on a dict.
    changes: Dict[str, Any] = {}
    for field in _ROUTINE_FIELDS:
        if field not in body:
            continue
        if field in ("name", "body"):
            changes[field] = str(body[field] or "").strip()
        elif field == "photos":
            changes[field] = _count(body[field])
        elif field == "record":
            changes[field] = body[field] if isinstance(body[field], list) else []
        else:
            changes[field] = _tri(body[field])
    if not changes:
        return jsonify({"error": "Nothing to update"}), 400
    if not changes.get("name", "x") or not changes.get("body", "x"):
        return jsonify({"error": "A routine needs a name and a prompt"}), 400
    routine = store.update_routine(routine_id, changes)
    if routine is None:
        return jsonify({"error": "No such routine"}), 404
    return jsonify({"ok": True, "routine": routine,
                    "problems": _record_problems(routine.get("record"))})


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
    # Not named `fields`: that is the module that reads them, and shadowing it
    # here made every call below a method on a list.
    wanted = body.get("fields")
    if not answer:
        return jsonify({"error": "Missing 'answer'"}), 400
    if not isinstance(wanted, list) or not wanted:
        return jsonify({"error": "Missing 'fields'"}), 400

    model = str(body.get("model") or "") or get_default_model()
    declared = fields.parse(wanted)
    # Only the fields nothing can derive are asked for. A rate is arithmetic
    # over two other fields, and asking a language model to do arithmetic in
    # prose is how one trip came to be logged at $26.23 an hour and, five
    # minutes later, at $23.19 — the second time from a row with no elapsed
    # time in it at all. What can be computed is computed, below, in Python.
    asked = fields.to_ask(declared)
    extracted = records.extract(
        answer, [f.name for f in asked], model,
        kinds={f.name: f.kind for f in asked if f.kind})
    # Straight from the photos' own files, before anything is computed, so a
    # field declared "= earliest photo taken" is exact and the elapsed time
    # built on it is exact too. The model is not asked for these at all: it has
    # no labels on the pictures to match a time against, which is precisely how
    # it came to report somebody else's.
    photos = body.get("photos")
    taken = fields.from_photos(declared, photos if isinstance(photos, list) else None)
    read = {**extracted, **{k: v for k, v in taken.items() if v}}
    # The kinds the columns actually hold, not only the ones written down.
    # Without this, arithmetic worked *only* on a routine whose every input
    # carried an explicit "name: kind" — write the obvious
    #
    #     Start odometer
    #     End odometer
    #     Miles = End odometer - Start odometer
    #
    # and Miles came out blank on every run, reporting "nothing recorded for
    # End odometer or Start odometer" while both sat in the row reading
    # "102,072 mi". Asking storage means the sum reads a value exactly as the
    # column that files it does, so the two can never disagree.
    if read:
        known = store.column_kinds(str(body.get("routine_name") or "Routine"), read,
                                   fields.kinds(declared))
        computed, gaps = fields.compute(declared, read, known)
    else:
        computed, gaps = {}, []
    # Declared order, so the table reads the way the routine was written.
    row = {f.name: computed.get(f.name, read.get(f.name, ""))
           for f in declared}
    wrong = fields.mismatches(declared, extracted)
    if gaps or wrong:
        logger.info("Record for %s: %s", body.get("routine_name"),
                    "; ".join(gaps + [f"{k}: {v}" for k, v in wrong.items()]))
    kept = store.add_record(
        str(body.get("routine_name") or "Routine"), row,
        str(body.get("routine_id") or "") or None,
        str(body.get("conversation_id") or "") or None,
        declared=fields.kinds(declared),
    )
    # 200 either way. "Nothing could be pulled out of that answer" is an
    # outcome, not an error, and the reply it came from is still on screen.
    # The gaps travel with it: a blank hourly rate is only reassuring once you
    # know it is blank because no elapsed time was recorded, rather than
    # because something broke.
    return jsonify({"record": kept, "gaps": gaps,
                    "mismatched": [f"{k}: {v}" for k, v in wrong.items()]})


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
_CSV_FORMULA = ("=", "+", "@", "\t", "\r")

# "-" is the awkward one. A negative number is an ordinary reading — a
# temperature, a correction — and quoting every one of them would corrupt the
# log to prevent nothing. But a spreadsheet does evaluate a leading "-" when
# what follows is not simply a number: "-2+3" opens as 1, and "-1+cmd|'/c
# calc'!A0" opens as rather more than that. So: leave the readings alone and
# defuse the rest, which is what the comment here used to claim was happening.
_FORMULA_CHARS = re.compile(r"[+*/()!|=&^\[\]{}]")


def _csv_safe(value: str) -> str:
    """One cell, with a leading formula character defused."""
    text = str(value or "")
    if text.startswith(_CSV_FORMULA):
        return "'" + text
    # A leading "-" only matters when an operator or a reference follows it:
    # "-5 °C" and "-1,250.75" are readings, "-2+3" opens as 1, and
    # "-1+cmd|'/c calc'!A0" opens as rather more than that.
    if text.startswith("-") and _FORMULA_CHARS.search(text):
        return "'" + text
    return text


@app.route("/api/records.csv", methods=["GET"])
def api_records_csv() -> Any:
    """The whole log as CSV, for a spreadsheet or another machine's importer.

    A real endpoint rather than a download built in the browser: pulling this
    with curl from wherever the records are meant to end up is the point.
    """
    rows = store.list_records(str(request.args.get("routine") or ""))
    columns = store.record_columns(rows)
    # The unit moves to the header and the cell keeps the number alone. A cell
    # reading "$115.94" or "93 mi" is text to a spreadsheet: it will not sum, it
    # will not chart, and it sorts "100 mi" before "93 mi". Under a header
    # saying "Total earnings (USD)" the bare 115.94 says exactly as much and is
    # a number. Timestamps and prose are written out as they stand.
    # Declared kinds first, inference only where nothing was declared — the
    # same order the table and the record writer use. Without this a column
    # declared "text" and stored as text was exported as a number.
    declared: Dict[str, str] = {}
    for routine in {r["routine_name"] for r in rows}:
        for name, kind in store.declared_kinds(routine).items():
            declared.setdefault(name, kind)
    kinds, heads = {}, []
    for name in columns:
        seen = [r["fields"].get(name, "") for r in rows]
        kinds[name] = declared.get(name) or values.column_kind(seen)
        unit = values.unit_label(kinds[name], seen)
        heads.append(f"{name} ({unit})" if unit else name)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["taken_at", "routine"] + heads)
    for row in rows:
        stamp = datetime.fromtimestamp(row["created_at"]).isoformat(timespec="seconds")
        writer.writerow(
            [stamp, _csv_safe(row["routine_name"])] +
            [_csv_safe(values.number_of(row["fields"].get(name, ""), kinds[name]))
             for name in columns])
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
            # Read off the messages *as they will be sent*, not off the whole
            # thread. The block numbers its lines "Image 1", "Image 2", and the
            # images carry no numbering of their own — so the numbering is only
            # true if it counts the same photos the model is about to see.
            # Trimming happens below for the real conversation; this runs it
            # early, on a copy that still has the EXIF attached to it.
            photo_meta = web.sent_image_meta(
                web.keep_recent_images(messages, keep_turns=get_image_turns()))
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
                    transcript, context = yield from _read_images_for(
                        model, images, reader, is_ocr_reader)
                    # Drop the base64 once it is transcribed: this model will
                    # never read it, and it costs the body limit every turn.
                    convo = web.with_context(web.strip_images(turns), context)
            elif images and len(images) > 1 and get_photo_read_each():
                # A model that *can* see, given each photo read on its own so
                # that the labels are reliable. The pictures carry no labels,
                # and the photo details beside them say "Image 1", "Image 2" —
                # so using a capture time means aligning two lists across two
                # messages by position, with nothing in the input to anchor it.
                # That is where a two-odometer routine goes wrong, and it goes
                # wrong on large models too. Read separately, the readings
                # arrive already numbered and the join is text to text.
                #
                # The images stay: this is an anchor for them, not a substitute,
                # and the preamble tells the model to trust its own eyes over a
                # reading that disagrees.
                yield _line({"status": f"Reading {len(images)} photos one at a time…"})
                try:
                    transcript = web.read_images(images, model, ocr=False,
                                                 answering_model=model)
                except web.ReadFailed as exc:
                    # An enhancement. Losing it costs the labels, not the turn.
                    logger.warning("Per-photo read via %s failed: %s", model, exc)
                    yield _step("Read each photo", f"{model} failed: {exc}")
                else:
                    context = web.per_photo_context(transcript)
                    if context:
                        convo = web.with_context(convo, context)
                    yield _step("Read each photo",
                                f"{model}, {len(images)} photo(s) read separately "
                                "so the numbering is reliable",
                                text=transcript or "(nothing came back)")

            if meta_context:
                convo = web.with_context(convo, meta_context)

            # Where the numbered links point, and how many times the model may
            # still ask for one. Both empty unless a web turn fills them in.
            documents: List[Dict[str, str]] = []
            link_ids: Dict[str, Dict[str, str]] = {}
            hops = 0
            # The conversation *without* the page context, kept so that a hop
            # can rebuild rather than stack: inserting a second block would
            # leave the model reading two overlapping copies of the same pages,
            # numbered differently, which is worse than not hopping at all.
            grounded = convo

            if use_web:
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
                    grounded = convo
                    hops = get_web_fetch_hops()
                    convo = web.with_context(
                        convo, _web_context(documents, turns, link_ids, hops))
                    kept_sources.extend(
                        {"url": d["url"], "title": d["title"]} for d in documents)
                    yield _line({"sources": list(kept_sources)})
            yield _step(
                "Sent to the model",
                f"{len(convo)} turns, {sum(len(str(t.get('content') or '')) for t in convo)} "
                f"characters of text, num_ctx {options.get('num_ctx')}",
                system=[str(t.get("content") or "")[:2000]
                        for t in convo if t.get("role") == "system"])

            # The model may answer, or — while a hop is left and it was told it
            # could — ask for one of the numbered links to be read first. A
            # request is swallowed rather than shown: it is the machinery
            # talking, not the reply, and the user gets the answer that comes
            # back with the page in front of it.
            for _ in range(hops + 1):
                # The table only goes in while an offer actually stands. With
                # the feature off the model was never told it could ask, so a
                # reply that happens to look like a request is just a reply —
                # and holding it back would swallow somebody's answer about
                # this very feature.
                wanted = yield from _stream_answer(
                    model, convo, options, link_ids if hops else {},
                    answer, thinking)
                if not wanted:
                    break
                link = link_ids.get(wanted)
                if not link:
                    # A number that was never on the list. Nothing to fetch, so
                    # ask again with the offer withdrawn rather than showing the
                    # user a marker where their answer should be.
                    logger.info("Model asked for link [%s], which is not on the "
                                "list it was shown", wanted)
                    yield _step("Asked to read a link",
                                f"[{wanted}] is not one of the links offered; "
                                "answering from what was already retrieved")
                # The map *shows* more than may be opened: WEB_LINK_SCOPE is
                # about what the model can see and WEB_FOLLOW_SCOPE about what
                # it can talk us into visiting, and the second is the stricter
                # of the two on purpose. Asking by number must not be the way
                # round it — without this check the default settings list every
                # outbound link and then open any of them on request, which is
                # exactly the surface following was kept narrow to avoid.
                elif not web.followable({"url": link.get("source", "")}, link):
                    logger.info("Model asked for link [%s] (%s), which leaves the "
                                "site it was found on", wanted, link.get("url"))
                    yield _step("Asked to read a link",
                                f"[{wanted}] leaves {_host_of(link.get('source', ''))} "
                                "— only pages of a site already retrieved may be "
                                "opened (WEB_FOLLOW_SCOPE)")
                # Already read. A link stays on the map after it is followed, so
                # with more than one hop the model can ask for the same page
                # twice — which fetched it twice, listed it twice as a source,
                # and spent the last hop learning nothing.
                elif any(link["url"] in (d.get("url"), d.get("requested"))
                         for d in documents):
                    yield _step("Asked to read a link",
                                f"[{wanted}] has already been read")
                else:
                    yield _line({"status": f"Opening {_host_of(link['url'])}…"})
                    try:
                        documents.append(web.fetch(link["url"]))
                    # Not just WebError. Following a link is an enhancement, and
                    # an enhancement that can take the whole turn down with it is
                    # a worse bug than the one it fixes — the reply is already
                    # paid for and the pages to answer from are already here.
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Fetching requested link %s failed: %s",
                                       link["url"], exc)
                        yield _line({"status": f"Could not read {_host_of(link['url'])}."})
                        yield _step("Asked to read a link",
                                    f"[{wanted}] {link['url']} — {exc}")
                    else:
                        yield _step("Asked to read a link",
                                    f"the model asked for [{wanted}]",
                                    urls=[documents[-1].get("url", "")])
                        kept_sources.append({"url": documents[-1]["url"],
                                             "title": documents[-1]["title"]})
                        yield _line({"sources": list(kept_sources)})
                # One offer per hop, and none once they are spent, so the model
                # is never invited to ask for something it cannot be given.
                hops = max(0, hops - 1)
                convo = web.with_context(
                    grounded, _web_context(documents, turns, link_ids, hops))
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
    cancel = _start_turn(convo_id)

    def produce(emit: Any) -> None:
        try:
            for line in run_turn():
                if cancel.is_set():
                    break     # closes chat_stream, which lets Ollama go
                entry = _debug_of(line)
                if entry is not None:
                    kept_steps.append(entry)
                emit(line)
        finally:
            try:
                # In produce's own finally, so it completes before the stream
                # closes: the browser refreshes its list the moment the reply
                # ends, and saving afterwards made that a race the list usually
                # lost — a finished thread with no title until something else
                # refreshed it. Saved even when cancelled: Stop usually means
                # "I have read enough", and the half you read is worth keeping.
                #
                # Before _end_turn, not after. /api/chat/status is nothing but
                # membership in _TURNS, so between deregistering and committing
                # there is a window where a reconnecting page is told the turn
                # is not running *and* finds nothing written — which reads to it
                # as "the reply did not survive", about a reply sitting in the
                # database milliseconds later.
                _keep_turn(convo_id, last_user, "".join(answer), kept_sources,
                           "".join(thinking), kept_steps)
            finally:
                # Nested, because a turn left registered forever would break
                # Stop and /api/chat/status for that conversation for the life
                # of the process. _keep_turn swallows store failures, but the
                # reasoning-strip ahead of them is outside its try.
                _end_turn(convo_id, cancel)

    return _detached_stream(produce, "chat-turn")


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


@app.route("/api/chat/status")
def api_chat_status() -> Any:
    """Is a turn still being generated for this conversation?

    Generation is detached from the connection that asked for it, so a phone
    that goes into a pocket mid-reply leaves a turn running that it can no
    longer see. Without this the page cannot tell that from a server that has
    died: both look like a stream that stopped arriving, and the safe reading
    of the second one — discard the turn — throws away the first one.

    With it, a dropped connection becomes something to wait out.
    """
    convo_id = request.args.get("conversation_id") or ""
    with _TURNS_LOCK:
        running = convo_id in _TURNS
    return jsonify({"running": running})


def _message_field(line: str, field: str) -> str:
    """One field of the ``message`` object in an Ollama NDJSON line.

    Tolerant on purpose: this runs on every line of every reply, and a line
    that is not what was expected is not worth ending a turn over.
    """
    try:
        obj = json.loads(line)
    except (ValueError, TypeError):
        return ""
    message = obj.get("message") if isinstance(obj, dict) else None
    if not isinstance(message, dict):
        return ""
    return str(message.get(field) or "")


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


# How long the relay waits for the producer before writing to find out whether
# anyone is still listening. Long enough that an ordinary turn — where tokens
# arrive continuously — never sends one, short enough that a phone in a pocket
# does not hold a worker for the whole Ollama timeout.
_HEARTBEAT_SECONDS = 20.0


def _detached_stream(produce: Any, name: str) -> Response:
    """Run ``produce`` on a thread and relay the lines it emits to the client.

    A WSGI response is *pulled*: when the client stops asking for the next
    chunk, the generator stops being advanced and the work stops with it. That
    is right for work whose only purpose is the response, and wrong for work
    worth finishing — a reply the server can write down, a download that is
    most of the way through a gigabyte.

    ``produce`` is called with an ``emit`` function and may block for as long
    as it likes. Anything it does in its own ``finally`` happens before the
    stream closes, which is what lets a turn be saved before the browser is
    told the reply has ended.

    The relay must not simply block until the producer speaks. Measured: six
    turns against an Ollama that accepts and never answers took every one of
    waitress's four worker threads, and closing all six clients did not give a
    single one back — /healthz stopped answering for the length of the Ollama
    timeout, five minutes, and the server manager's card reads that as the app
    being down rather than busy. A departed client cannot be detected under
    waitress except by writing to it, so the relay writes something harmless
    when the producer has been quiet, and lets the write do the detecting.
    """
    lines: "queue.Queue[Optional[str]]" = queue.Queue()

    def run() -> None:
        try:
            produce(lines.put)
        except Exception:  # noqa: BLE001 - a thread has nowhere to raise to
            # Both callers report their own failures, so this is the backstop
            # for the one that forgets: without it the exception goes to the
            # thread's default handler and the client sees a stream that simply
            # stops, which reads as a finished reply that was cut short.
            logger.exception("The %s stream failed", name)
            lines.put(_line({"error": "The server stopped part-way through."}))
        finally:
            lines.put(None)

    threading.Thread(target=run, name=name, daemon=True).start()

    def relay() -> Iterator[str]:
        while True:
            try:
                line = lines.get(timeout=_HEARTBEAT_SECONDS)
            except queue.Empty:
                # Nothing to relay, so nothing has been written, so nothing has
                # yet noticed that the client left. Under waitress there is no
                # way to ask whether the socket is still there — the only thing
                # that finds out is a write. So write: a blank line is not a
                # record, both readers in the page skip it explicitly, and
                # NDJSON allows it. Against a departed client it raises, Flask
                # closes this generator, and the worker goes back to the pool.
                yield "\n"
                continue
            if line is None:
                return
            yield line

    return Response(stream_with_context(relay()), mimetype="application/x-ndjson")


def _web_context(
    documents: List[Dict[str, str]],
    turns: List[Dict[str, str]],
    link_ids: Dict[str, Dict[str, str]],
    hops: int,
) -> str:
    """The fenced page block, with the link numbering it hands back recorded.

    ``link_ids`` is emptied and refilled on every call, because the numbering
    is only valid for the block it was rendered with: a hop adds a document,
    every list below it shifts, and a table left over from the previous round
    would resolve [3.2] against a page that is now [4.2].
    """
    return web.build_context(
        documents,
        char_budget=web.context_budget(get_num_ctx()),
        question=web.last_user_text(turns),
        link_ids=link_ids,
        may_fetch=hops > 0,
    )


def _stream_answer(
    model: str,
    convo: List[Dict[str, str]],
    options: Dict[str, Any],
    link_ids: Dict[str, Dict[str, str]],
    answer: List[str],
    thinking: List[str],
) -> Any:
    """Stream one reply, unless it turns out to be a request to read a link.

    Returns the link id asked for, or "" when this was an ordinary answer — in
    which case every line has already been yielded and nothing was held back.

    The holding is the awkward part and cannot be avoided: a request has to be
    recognised before it is shown, or the user watches "FETCH: [2.3]" arrive,
    sit there, and then be followed by a second answer with no explanation of
    what the first line was. So content is buffered while it could still become
    a request — which for ordinary prose is one token, since "The" cannot — and
    released the moment it cannot. Reasoning is never held: it is not the reply,
    the user is already watching it scroll, and a model that thinks for thirty
    seconds before asking for a link would otherwise look frozen.
    """
    held: List[str] = []
    holding = bool(link_ids)

    def flush() -> Any:
        for line in held:
            answer.append(_message_field(line, "content"))
            yield line + "\n"
        held.clear()

    for line in chat_stream(model, convo, options=options,
                            keep_alive=get_keep_alive() or None):
        thinking.append(_message_field(line, "thinking"))
        if not holding:
            answer.append(_message_field(line, "content"))
            yield line + "\n"
            continue
        # Thinking-only lines go straight out; everything else waits, including
        # the final empty one, or the reply would arrive after its own end.
        if not _message_field(line, "content") and _message_field(line, "thinking"):
            yield line + "\n"
            continue
        held.append(line)
        if web.fetch_pending("".join(_message_field(l, "content") for l in held)):
            continue
        holding = False
        yield from flush()

    if held:
        wanted = web.fetch_request("".join(_message_field(l, "content") for l in held))
        if wanted:
            return wanted
        # Held to the end and not a request after all — an empty reply, or one
        # short enough to still look like one. It is the answer; show it.
        yield from flush()
    return ""


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


# However the settings multiply out, retrieval never puts more than this many
# documents in front of the model. WEB_MAX_DOCS × WEB_FOLLOW_LINKS × WEB_MAX_HOPS
# reaches fifteen at the permitted maximums, and fifteen documents do not fit in
# any window this app runs at — build_context would hand each one its 800-character
# floor and overrun the budget by several times. A ceiling here is the one place
# that cannot be got wrong by a combination of settings that each look reasonable.
_DOC_CEILING = 8


def _link_candidates(
    question: str,
    sources: List[Dict[str, Any]],
    documents: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Links worth offering the picker, best first, across every page given.

    Pooled and ranked as one list rather than per page. A search turn retrieves
    three pages from three sites; asking the picker about each in turn is three
    model calls to answer one question, and it cannot see that the best link on
    the whole turn was on page two.
    """
    seen = {d.get("url") for d in documents} | {d.get("requested") for d in documents}
    pool: List[Dict[str, str]] = []
    for source in sources:
        for link in (source.get("links") or []):
            url = link.get("url")
            # followable(), not same_site(): what may be *opened* is a stricter
            # question than what may be listed, and it has its own setting.
            if not url or url in seen or not web.followable(source, link):
                continue
            seen.add(url)
            pool.append(link)
    # No `here`: these come from several pages at once, so a same-site bonus
    # would mean "same site as whichever page this link happened to be on",
    # which is every one of them and therefore nothing.
    return web.rank_links(pool, question)


def _follow_links(
    model: str,
    question: str,
    sources: List[Dict[str, Any]],
    documents: List[Dict[str, str]],
    budget: int,
) -> Any:
    """Open a couple of the pages ``sources`` link to, if any look relevant.

    Every URL still goes through the address guard in fetch(), and by default
    only same-site links are candidates at all — a link is chosen by a model
    from content written by a stranger, so it gets no more trust than a pasted
    URL does. Returns the documents it added, so the caller can hop from them.
    """
    candidates = _link_candidates(question, sources, documents)
    if not candidates:
        return []

    picker = get_planner_model() or model
    try:
        chosen = web.choose_links(question, candidates, picker, max_links=budget,
                                  answering_model=model)
    except Exception as exc:  # noqa: BLE001 - an enhancement, never a requirement
        logger.warning("Link picking failed: %s", exc)
        return []
    if not chosen:
        return []

    yield _line({"status": "Following: " + " · ".join(c["text"][:40] for c in chosen)})
    fetched, failures = _run_all(web.fetch, [c["url"] for c in chosen])
    documents.extend(fetched)
    if failures and not fetched:
        yield _line({"status": "Could not read the linked pages."})
    yield _step("Followed links",
                f"{len(chosen)} chosen, {len(fetched)} read"
                + (f"; failures: {'; '.join(failures[:3])}" if failures else ""),
                urls=[d.get("url", "") for d in fetched])
    return fetched


def _deepen(
    model: str,
    question: str,
    documents: List[Dict[str, str]],
) -> Any:
    """Follow links outward from what has been retrieved, up to the hop limit.

    Hop one is the case this started as: the page answers half the question and
    points at the page with the other half. Hop two is the one that needed a
    setting of its own — the specification linked from the release note linked
    from the search result — and it is off by default because each hop is
    another picker call and another round of fetches on hardware that is
    usually running the answering model at the same time.
    """
    budget = get_web_follow_links()
    hops = get_web_max_hops()
    if budget < 1 or hops < 1:
        return
    frontier = list(documents)
    for _ in range(hops):
        room = _DOC_CEILING - len(documents)
        if not frontier or room < 1:
            return
        fetched = yield from _follow_links(model, question, frontier, documents,
                                           min(budget, room))
        if not fetched:
            return          # nothing chosen or nothing readable; no deeper to go
        frontier = fetched


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
        # Only within the same site by default — following a model's pick of an
        # arbitrary outbound link is a much larger surface for very little gain.
        yield from _deepen(model, question, documents)
        # Deliberately not distilled. A page reached by searching is one this
        # app chose, and cutting it down to the question it was chosen to
        # answer loses nothing. A page the *user* pasted is a deliberate act,
        # and "summarise this" is a question no extractive pass can cut to —
        # it would come back with the sentences containing the word summarise,
        # or with nothing. Their link, their whole page.
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
                read_failed = False
                try:
                    image_note = web.read_images(images, reader, ocr=is_reader_ocr,
                                                 answering_model=model)
                except web.ReadFailed as exc:
                    # Planning without the image beats not answering.
                    read_failed = True
                    logger.warning("Image read via %s failed: %s", reader, exc)
                    yield _line({"status": "Could not read the image; searching without it."})
                    yield _step("Read the image for search", f"{reader} failed: {exc}")
                else:
                    yield _step("Read the image for search",
                                f"{reader} ({'OCR' if is_reader_ocr else 'description'})",
                                text=image_note or "(nothing came back)")
                # Look at it before searching for it.
                #
                # An OCR model is the right reader for a screenshot and the
                # wrong one for a photograph, and it is preferred unconditionally
                # — so a photo of an animal was transcribed, found to contain no
                # text, and the search was planned from the user's words alone.
                # "what creature is this" searched verbatim returns pages of
                # advertising for animal-identifier apps and nothing about the
                # animal. Ask something that can see to say what it is, and
                # search for that instead.
                if is_reader_ocr and not _reads_as_text(image_note):
                    describer = _describing_model(model)
                    if describer:
                        yield _line({"status": "No text in it — looking at what it shows…"})
                        try:
                            image_note = web.read_images(images, describer, ocr=False,
                                                         answering_model=model)
                        except web.ReadFailed as exc:
                            logger.warning("Image description via %s failed: %s",
                                           describer, exc)
                            yield _step("Looked at the image instead",
                                        f"{describer} failed: {exc}")
                        else:
                            # Say which of the two happened. This block also
                            # runs after the read *failed*, where "no text to
                            # transcribe" contradicts the line above it — the
                            # panel said glm-ocr had fallen over and then that
                            # there was nothing in the image to transcribe.
                            why = (f"{reader} could not be asked" if read_failed
                                   else "no text to transcribe")
                            yield _step(
                                "Looked at the image instead",
                                f"{why}, so {describer} described it",
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

    # The same half-an-answer problem the pasted-URL path has, and for a long
    # time the search path did not do this at all: a search lands on the
    # overview and the specifics are one click away, exactly as they are on a
    # page pasted by hand. Before the snippet documents are added, so the
    # picker is only ever offered pages that were actually read — a snippet has
    # no links, and its own URL is already a document.
    if get_web_follow_on_search():
        yield from _deepen(model, question, documents)

    # Paywalled, JS-only and dead pages are routine. Their search snippets are
    # already paid for, so use them to fill out the budget rather than throwing
    # away everything that query found. Labelled as summaries in the context.
    if len(documents) < max_docs:
        documents.extend(
            web.snippet_documents(results, documents, limit=max_docs - len(documents))
        )

    if not documents:
        yield _line({"status": "Found results but couldn't read any of them."})
    yield from _distil_documents(question, documents, model)
    yield _step("Context from the web",
                f"{len(documents)} document(s), "
                f"{sum(len(d.get('text') or '') for d in documents)} characters")


def _distil_documents(question: str, documents: List[Dict[str, str]],
                      model: str) -> Any:
    """Replace each page's text with the part of it that answers the question.

    Six thousand characters a page is most of an 8192-token window before the
    conversation is even added, and nearly all of it is navigation prose,
    boilerplate and the parts of the article about something else. A small model
    reading each page and copying out the relevant sentences turns that into a
    few hundred — which both leaves room for the conversation and means more
    pages can be read, not fewer.

    Off unless WEB_DISTILLER_MODEL is set. A page whose distillation fails keeps
    its full text: a summariser that can lose information is a worse bug than a
    long context, and this one is allowed to fail as often as it likes.
    """
    distiller = get_distiller_model()
    # A search snippet is already a couple of hundred characters and already
    # labelled as a lead rather than a source. There is nothing to cut out of
    # it, and asking would spend a model call to risk it coming back as
    # "nothing relevant" — losing the only thing that page contributed.
    pages = [d for d in documents if not d.get("snippet_only")]
    if not distiller or not pages:
        return
    yield _line({"status": f"Reading {len(pages)} page(s) down to what you asked…"})
    before = [len(d.get("text") or "") for d in pages]

    # Each call carries its own index back, because _run_all drops falsy results
    # — it is built for fetches, where a failure means "no document" and the
    # survivors are all that matter. Here a failure means "this page keeps its
    # full text", and zipping the shortened list back against the pages would
    # hand page two the distillation of page one. Wrong text under a citation is
    # far worse than a long one.
    #
    # It also catches for itself: _run_all logs the item it was given, which
    # here would be a whole page in the log for every distiller hiccup.
    def one(pair: Any) -> Any:
        index, doc = pair
        try:
            return index, web.distil(question, doc, distiller, answering_model=model)
        except Exception:  # noqa: BLE001 - distil reports its own; this is the belt
            logger.exception("Distilling %s failed", doc.get("url"))
            return index, None

    # Concurrently, like the fetches: one slow page must not serialise the rest.
    done, _ = _run_all(one, list(enumerate(pages)))
    shortened = {index: text for index, text in done if text}
    kept = 0
    for index, doc in enumerate(pages):
        short = shortened.get(index)
        if short:
            doc["text"] = short
            doc["distilled"] = True
        else:
            kept += 1
    after = [len(d.get("text") or "") for d in pages]
    yield _step(
        "Distilled",
        f"{distiller}: {sum(before)} → {sum(after)} characters"
        + (f"; {kept} page(s) kept in full because it could not be asked" if kept else ""),
        # Bounded, like every other step that carries bulk text ("Sent to the
        # model" slices 2000, the image reads 600). This one showed each page's
        # text whole — and on the path where the distiller could not be asked,
        # "whole" is the untouched page. Three of those is ~18 KB streamed to
        # the browser and written into chat.db on every web turn, on exactly
        # the turns where distilling achieved nothing.
        text="\n\n".join(
            f"{d.get('url', '')}\n{b} → {a} chars\n{_clipped(d.get('text') or '')}"
            for d, b, a in zip(pages, before, after)))


# Enough of a distillation to judge it by, which is the panel's whole job. A
# page kept in full is far longer than this and does not need showing twice.
_PANEL_TEXT_MAX = 1500


def _clipped(text: str) -> str:
    if len(text) <= _PANEL_TEXT_MAX:
        return text
    return text[:_PANEL_TEXT_MAX].rsplit(" ", 1)[0] + " …[shown in part]"


def _read_images_for(model: str, images: List[str], reader: str,
                     is_ocr_reader: bool) -> Any:
    """Read this turn's images for a model that cannot see them.

    An image attached to a model without eyes used to force a switch, which
    meant giving up your best model exactly when you were debugging with a
    screenshot. Transcribing it and handing the text over lets a coder model
    keep the question.

    Yields status lines and returns ``(transcript, context)``: the transcript
    so the search planner can reuse it rather than paying for a second pass,
    and the context to put in front of the answering model.
    """
    yield _line({"status": f"Reading the image with {reader}…"})
    describing = not is_ocr_reader
    try:
        transcript = web.read_images(images, reader, ocr=is_ocr_reader,
                                     answering_model=model)
        # An OCR model that found nothing is not the end of it: a photo of a
        # plant has no text, and telling the user to go change a dropdown while
        # a vision model sits idle is a poor answer to "what is this?".
        #
        # `is None` was too narrow. An OCR model handed a photograph rarely
        # returns nothing — it returns a sentence saying there is nothing, which
        # is text by every measure except the one that matters, so this fallback
        # never fired for the case it was written for.
        if is_ocr_reader and not _reads_as_text(transcript):
            describer = _describing_model(model)
            if describer:
                yield _line({"status": f"No text in it — looking at it with {describer}…"})
                # Its own guard. Inside the outer one, a describer that OOMs —
                # routine when a 30b holds the GPU — landed in the handler
                # below, which blames `reader` and returns image_failed_context:
                # the wrong model named, and a transcription that worked thrown
                # away. Failing here just means keeping what OCR gave.
                try:
                    transcript = web.read_images(images, describer, ocr=False,
                                                 answering_model=model)
                    describing = True
                except web.ReadFailed as exc:
                    logger.warning("Image description via %s failed: %s", describer, exc)
        yield _step("Read the image",
                    f"{reader} ({'description' if describing else 'OCR'})",
                    text=transcript or "(nothing came back)")
        return transcript, web.image_context(transcript, ocr=not describing)
    except web.ReadFailed as exc:
        # A reader-model OOM is routine when a 30b model holds the GPU.
        # Reporting it as "no readable text was found" is a confident, wrong
        # claim about the user's image.
        logger.warning("Image read via %s failed: %s", reader, exc)
        yield _line({"status": f"{reader} could not read the image."})
        yield _step("Read the image", f"{reader} failed: {exc}")
        return None, web.image_failed_context()


def _model_has_vision(name: str) -> bool:
    """Whether the answering model can read an image itself.

    OCR models are excluded: they can technically take an image, but they
    transcribe rather than answer, so a conversation should not be handed to one.
    """
    try:
        return name in vision_models(include_ocr=False)
    except Exception:  # noqa: BLE001 - Ollama unreachable; assume it cannot
        return False


# What an OCR model says when there was nothing to transcribe. It answers in
# prose rather than with an empty string, so length alone reads that as text.
# describe_images joins several readings as "[image 1] … [image 2] …".
_IMAGE_SPLIT = re.compile(r"\[image \d+\]")

_NO_TEXT_SAID = (
    "no text", "no visible text", "no readable text", "no legible text",
    "does not contain any text", "doesn't contain any text",
    "there is no text", "no words", "not contain text", "no discernible text",
)


def _reads_as_text(note: Optional[str]) -> bool:
    """Did the OCR pass actually find text worth planning a search from?

    A transcription of a photograph of an animal is empty, or a sentence
    explaining that it is empty. Either way there is nothing in it to search
    for, and the planner is left with only the user's words — which for "what
    creature is this" is how you end up searching that phrase and getting three
    pages of advertising for animal-identifier apps.
    """
    said = " ".join((note or "").split())
    if not said:
        return False
    # Several images come back joined as "[image 1] … [image 2] …", and one
    # blank photo next to a screenshot must not discard the screenshot's text.
    # Any image that yielded something is enough to plan a search from.
    parts = [p for p in _IMAGE_SPLIT.split(said) if p.strip()] or [said]
    return any(_one_reading_is_text(part) for part in parts)


def _one_reading_is_text(said: str) -> bool:
    # Take the denial out and judge what is left. Matching "no text" anywhere
    # discarded a screenshot of a form error reading "Validation failed: no
    # text entered in the Name field. Please complete every required box" —
    # a perfectly good transcription of a page that happens to be *about*
    # missing text — and replaced that error string with prose describing a
    # screenshot. It is a denial only when removing it leaves nothing behind.
    left = " ".join(part for part in _SENTENCE_SPLIT.split(said)
                    if not any(p in part.lower() for p in _NO_TEXT_SAID))
    # Punctuation and stray marks off a collar tag are not a search query.
    return sum(ch.isalnum() for ch in left) >= 12


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


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
        recorded = set()

        def record(future: Any) -> bool:
            recorded.add(future)
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
            best_kept = max((i for i, r in enumerate(results) if r is not None), default=-1)
            # "not yet recorded", not "not yet done". A future that finished in
            # the moment between the break and this line is done, so it was
            # excluded here — and nothing had called record() on it, so its
            # result was simply thrown away. That is the top search result
            # arriving on time and being dropped in favour of a lower-ranked
            # page, which is the exact selection this grace period exists to
            # prevent. Reproduced 10 times out of 10 with the window widened;
            # in the wild the window is microseconds, which is why it read as
            # working. wait() returns the already-done ones on its first check,
            # so nothing waits for them.
            better = [f for f, i in futures.items() if f not in recorded and i < best_kept]
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
    # `is not None`, not truthiness: a search that legitimately matched nothing
    # returns [], which record() counts as a success — so `enough` was reached
    # while the caller got fewer than that back. No caller can produce a falsy
    # success today; the disagreement is the kind that only shows up later.
    return [r for r in results if r is not None], errors


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
    def produce(emit: Any) -> None:
        sent = [0.0]

        def progress(done: int, total: int) -> None:
            # Rate limited to whole percents: a 1.8 GB file in 256KB chunks is
            # 7000 callbacks, and 7000 lines of JSON is its own small download.
            share = (done / total) if total else 0.0
            if total and share - sent[0] < 0.01 and done < total:
                return
            sent[0] = share
            emit(_line({"downloaded": done, "total": total,
                        "percent": round(share * 100)}))

        try:
            emit(_line(voice.download_model(str(model_id), progress)))
        except ValueError as exc:
            emit(_line({"error": str(exc)}))
        except Exception as exc:  # noqa: BLE001 - report, never 500 mid-stream
            logger.exception("Voice model download failed")
            emit(_line({"error": f"Download failed: {exc}"}))

    return _detached_stream(produce, "voice-download")


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
    if _AUTH_PROBLEM:
        print(f"  ⚠️  {_AUTH_PROBLEM}")
    # Which search backend, because "did my SEARXNG_URL take?" is otherwise
    # unanswerable without running a search and reading the panel. SEARXNG_URL
    # is set where the server manager passes environment, not in the shell that
    # started the container, and an export that never reached the app looks
    # exactly like an app that ignored it.
    if not web_enabled():
        print("  Web          : OFF (WEB_ENABLED=0)")
    elif get_search_url():
        print(f"  Web search   : {get_search_url()}")
    elif web.local_searxng():
        print(f"  Web search   : {web.local_searxng()} (found running; SEARXNG_URL is unset)")
    else:
        print("  Web search   : DuckDuckGo (no SEARXNG_URL — see the README)")
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
            # waitress's own default is 4, and a chat turn holds its worker for
            # as long as the model takes. Margin, not a fix — the relay's
            # heartbeat is what stops a departed client keeping one at all.
            threads=get_server_threads(),
            max_request_body_size=app.config["MAX_CONTENT_LENGTH"] * 2,
        )
    except ImportError:
        # Fall back to the Flask dev server if waitress isn't installed.
        logger.warning("waitress not installed; using the Flask dev server")
        app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
