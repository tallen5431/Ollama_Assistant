#!/usr/bin/env python3
"""CodeSmith Ollama Helper - modular + patch-aware version

Small Flask API in front of a local Ollama server, plus a tiny
in-browser chat UI on the root page.

This version is split into:
- config_ollama_helper.py: env + logging helpers
- ollama_client.py: thin HTTP wrapper around Ollama
- codesmith_patch_utils.py: CodeSmith patch helpers + normalization
- ui_root.py: HTML UI for the root page
- app.py (this file): Flask routes, uploads, and entrypoint.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

from config_ollama_helper import (
    get_default_model,
    get_heavy_model,
    get_host_port,
    get_max_snapshots,
    get_max_upload_size,
    get_ollama_base,
    get_ollama_startup_retries,
    get_ollama_startup_retry_delay,
    logger,
    require_ollama_on_startup,
    validate_config,
)
from ollama_client import (
    check_ollama_health,
    get_ollama_startup_instructions,
    post_ollama,
    post_ollama_stream,
    wait_for_ollama,
)
from codesmith_patch_utils import (
    build_code_system_prompt,
    normalize_patch_text,
)
from ui_root import ROOT_HTML


# -------------------------------------------------------------------
# Paths / upload directory
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).parent.resolve()
UPLOAD_DIR = Path(
    os.getenv("SNAPSHOT_UPLOAD_DIR", BASE_DIR / "uploads")
).resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Allowed MIME types for snapshot uploads
ALLOWED_MIME_TYPES = {"application/json", "text/plain"}
ALLOWED_EXTENSIONS = {".json"}


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------


def cleanup_old_snapshots() -> None:
    """Remove oldest snapshots if count exceeds MAX_SNAPSHOTS."""
    max_count = get_max_snapshots()
    if max_count == 0:
        return  # Unlimited

    try:
        snapshots = sorted(
            UPLOAD_DIR.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if len(snapshots) > max_count:
            to_delete = snapshots[max_count:]
            for snapshot in to_delete:
                try:
                    snapshot.unlink()
                    logger.info("Deleted old snapshot: %s", snapshot.name)
                except OSError as exc:
                    logger.warning("Failed to delete snapshot %s: %s", snapshot.name, exc)
    except Exception as exc:
        logger.error("Error during snapshot cleanup: %s", exc)


def validate_snapshot_file(filename: str, content: bytes) -> Optional[str]:
    """Validate uploaded snapshot file.

    Returns None if valid, or an error message string if invalid.
    """
    # Check file extension
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return f"Invalid file type. Only {', '.join(ALLOWED_EXTENSIONS)} files are allowed"

    # Check if content is valid JSON
    try:
        obj = json.loads(content)
    except json.JSONDecodeError as exc:
        return f"Invalid JSON file: {exc}"

    # Basic schema validation
    if not isinstance(obj, dict):
        return "Snapshot must be a JSON object"

    # Check for required fields
    if "files" not in obj:
        return "Snapshot missing 'files' field"

    if not isinstance(obj.get("files"), list):
        return "'files' field must be a list"

    return None


# -------------------------------------------------------------------
# Flask app + routes
# -------------------------------------------------------------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = get_max_upload_size()
CORS(app)


# -------------------------------------------------------------------
# Error handlers
# -------------------------------------------------------------------


@app.errorhandler(413)
def request_entity_too_large(error: Any) -> Any:
    """Handle file upload size exceeded."""
    max_mb = get_max_upload_size() // (1024 * 1024)
    return jsonify(
        {"error": f"File too large. Maximum upload size is {max_mb}MB"}
    ), 413


@app.errorhandler(500)
def internal_server_error(error: Any) -> Any:
    """Handle internal server errors."""
    logger.error("Internal server error: %s", error)
    return jsonify({"error": "Internal server error"}), 500


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------


@app.route("/", methods=["GET"])
def index() -> Any:
    """Root page: chat box + endpoint list."""
    return ROOT_HTML


@app.route("/api/health", methods=["GET"])
def health() -> Any:
    """Health check with Ollama connectivity status."""
    ollama_healthy = check_ollama_health()
    status = "ok" if ollama_healthy else "degraded"

    return jsonify(
        {
            "status": status,
            "ollama_host": get_ollama_base(),
            "ollama_connected": ollama_healthy,
            "default_model": get_default_model(),
            "heavy_model": get_heavy_model(get_default_model()),
            "max_upload_mb": get_max_upload_size() // (1024 * 1024),
        }
    )


@app.route("/api/models", methods=["GET"])
def list_models() -> Any:
    """Proxy to Ollama's /api/tags to list installed models."""
    try:
        data = post_ollama("/api/tags", {})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 502

    models = data.get("models") or data.get("tags") or data
    return jsonify({"models": models})


@app.route("/api/chat", methods=["POST"])
def chat() -> Any:
    """Generic chat endpoint that passes through to Ollama's /api/chat."""
    body = request.get_json(silent=True) or {}
    model = body.get("model") or get_default_model()
    messages: Optional[List[Dict[str, str]]] = body.get("messages")
    stream = body.get("stream", False)

    if messages is None:
        prompt = body.get("prompt")
        if not prompt:
            return (
                jsonify({"error": "Missing 'messages' or 'prompt' in request"}),
                400,
            )
        messages = [{"role": "user", "content": str(prompt)}]

    # Validate messages structure
    if not isinstance(messages, list):
        return jsonify({"error": "'messages' must be a list"}), 400

    for msg in messages:
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            return jsonify({"error": "Each message must have 'role' and 'content'"}), 400

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }

    try:
        data = post_ollama("/api/chat", payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 502

    message = data.get("message") or {}
    content = message.get("content") or data.get("response") or ""

    return jsonify(
        {
            "model": model,
            "messages": messages,
            "reply": content,
            "raw": data,
        }
    )


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream() -> Any:
    """Streaming chat endpoint for long-running model responses."""
    from flask import Response

    body = request.get_json(silent=True) or {}
    model = body.get("model") or get_default_model()
    messages: Optional[List[Dict[str, str]]] = body.get("messages")

    if messages is None:
        prompt = body.get("prompt")
        if not prompt:
            return (
                jsonify({"error": "Missing 'messages' or 'prompt' in request"}),
                400,
            )
        messages = [{"role": "user", "content": str(prompt)}]

    # Validate messages structure
    if not isinstance(messages, list):
        return jsonify({"error": "'messages' must be a list"}), 400

    for msg in messages:
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            return jsonify({"error": "Each message must have 'role' and 'content'"}), 400

    payload = {
        "model": model,
        "messages": messages,
    }

    def generate() -> Generator[str, None, None]:
        """Generator function for streaming responses."""
        try:
            for chunk in post_ollama_stream("/api/chat", payload):
                yield json.dumps(chunk) + "\n"
        except ValueError as exc:
            error_chunk = {"error": str(exc)}
            yield json.dumps(error_chunk) + "\n"

    return Response(generate(), mimetype="application/x-ndjson")



@app.route("/api/code-assist", methods=["POST"])
def code_assist() -> Any:
    """CodeSmith-friendly endpoint for prompt+code assistance."""
    body = request.get_json(silent=True) or {}
    prompt = body.get("prompt")
    code = body.get("code")
    filename = body.get("filename")
    extra_system = body.get("system")
    model = body.get("model") or get_default_model()

    if not prompt and not code:
        return jsonify({"error": "At least 'prompt' or 'code' must be provided"}), 400

    system_prompt = build_code_system_prompt(extra_system)

    parts: List[str] = []
    if prompt:
        parts.append(prompt.strip())
    if code:
        header = "\n\nCode snippet"
        if filename:
            header += f" from {filename}"
        header += ":\n\n"
        parts.append(header + code)

    user_content = "\n".join(parts).strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }

    try:
        data = post_ollama("/api/chat", payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 502

    message = data.get("message") or {}
    content = message.get("content") or data.get("response") or ""

    return jsonify(
        {
            "model": model,
            "reply": content,
            "messages": messages,
            "raw": data,
        }
    )


@app.route("/api/patch/normalize", methods=["POST"])
def patch_normalize() -> Any:
    """Normalize a model's patch output into a valid CodeSmith patch JSON.

    Request body:
        {
          "raw": "<model reply text>",
          "root_path": "...",           # optional
          "file_hashes": { ... }        # optional
        }

    Response:
        200: { "patch": { ...normalized patch object... } }
        400: { "error": "..." } on parse/validation failure.
    """
    body = request.get_json(silent=True) or {}
    raw = body.get("raw")
    root_path = body.get("root_path")
    file_hashes = body.get("file_hashes")

    if raw is None:
        return jsonify({"error": "Missing 'raw' in request"}), 400

    try:
        patch_obj = normalize_patch_text(
            str(raw),
            root_path=str(root_path) if root_path is not None else None,
            file_hashes=file_hashes if isinstance(file_hashes, Dict) else None,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify({"patch": patch_obj})


@app.route("/api/upload-snapshot", methods=["POST"])
def upload_snapshot() -> Any:
    """Upload a CodeSmith snapshot JSON and save it under the uploads folder.

    Expects multipart/form-data with a "file" field containing the
    snapshot JSON. Returns basic metadata about the saved file and,
    when possible, some snapshot details (schema, root_path, file_count).

    Enhanced with validation and automatic cleanup of old snapshots.
    """
    if "file" not in request.files:
        return jsonify({"error": "Missing 'file' in form-data"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    if not filename:
        return jsonify({"error": "Invalid filename"}), 400

    # Read file content for validation
    try:
        content = file.read()
    except Exception as exc:
        logger.error("Failed to read uploaded file: %s", exc)
        return jsonify({"error": "Failed to read uploaded file"}), 500

    # Validate file
    validation_error = validate_snapshot_file(filename, content)
    if validation_error:
        return jsonify({"error": validation_error}), 400

    # Save file
    dest = UPLOAD_DIR / filename
    try:
        with dest.open("wb") as fh:
            fh.write(content)
        logger.info("Uploaded snapshot: %s (%d bytes)", filename, len(content))
    except OSError as exc:
        logger.error("Failed to save uploaded snapshot %s: %s", dest, exc)
        return jsonify({"error": f"Failed to save file: {exc}"}), 500

    # Parse metadata
    size = dest.stat().st_size
    meta: Optional[Dict[str, Any]] = None

    try:
        obj = json.loads(content)
        schema = obj.get("schema")
        root_path = obj.get("root_path")
        files = obj.get("files") or []
        file_count = len(files) if isinstance(files, list) else None
        meta = {
            "schema": schema,
            "root_path": root_path,
            "file_count": file_count,
        }
    except Exception as exc:
        logger.warning("Failed to inspect uploaded snapshot %s: %s", dest, exc)

    # Cleanup old snapshots
    cleanup_old_snapshots()

    return jsonify(
        {
            "filename": filename,
            "path": str(dest),
            "size": size,
            "meta": meta,
        }
    )


@app.route("/api/snapshots", methods=["GET"])
def list_snapshots() -> Any:
    """List uploaded snapshot JSON files plus basic metadata.

    This scans the uploads directory for ``*.json`` files, tries to
    parse them as CodeSmith snapshots, and returns lightweight metadata
    that you can use to pick which snapshot to work with in Ollama.
    """
    snapshots: List[Dict[str, Any]] = []

    try:
        candidates = sorted(
            UPLOAD_DIR.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to scan uploads dir %s: %s", UPLOAD_DIR, exc)
        return jsonify({"error": f"Failed to scan uploads dir: {exc}"}), 500

    for p in candidates:
        meta: Optional[Dict[str, Any]] = None
        try:
            with p.open("r", encoding="utf-8") as fh:
                obj = json.load(fh)
            files = obj.get("files") or []
            meta = {
                "schema": obj.get("schema"),
                "root_path": obj.get("root_path"),
                "file_count": len(files) if isinstance(files, list) else None,
                "total_size": obj.get("total_size"),
            }
        except Exception as exc:  # pragma: no cover - best-effort introspection
            logger.warning("Failed to load snapshot %s: %s", p, exc)

        stat = p.stat()
        snapshots.append(
            {
                "filename": p.name,
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "meta": meta,
            }
        )

    return jsonify({"snapshots": snapshots})


@app.route("/api/snapshot/files", methods=["POST"])
def snapshot_files() -> Any:
    """Decode a snapshot into per-file text snippets for model consumption.

    Request JSON body:
        {
          "filename": "snapshot-....json",       # required
          "paths": ["app.py", ...],             # optional filter list
          "max_total_chars": 16000,              # optional, global char budget
          "max_file_chars": 4000,                # optional, per-file cap
          "include_source": true                 # optional, default true
        }

    Response JSON:
        {
          "snapshot": { ...meta... },
          "files": [
            {
              "path": "app.py",
              "size": 20340,
              "tags": ["python"],
              "is_binary": false,
              "language": "python",
              "snippet": "...text...",
              "snippet_truncated": true
            },
            ...
          ],
          "truncated": true|false,              # true if global budget hit
          "max_total_chars": 16000,
          "max_file_chars": 4000
        }
    """
    body = request.get_json(silent=True) or {}
    filename = body.get("filename")
    if not filename:
        return jsonify({"error": "Missing 'filename' in request"}), 400

    paths_filter = body.get("paths")
    if paths_filter is not None and not isinstance(paths_filter, list):
        return jsonify({"error": "'paths' must be a list of relative paths"}), 400

    try:
        max_total_chars = int(body.get("max_total_chars", 16000))
    except (TypeError, ValueError):
        max_total_chars = 16000

    try:
        max_file_chars = int(body.get("max_file_chars", 4000))
    except (TypeError, ValueError):
        max_file_chars = 4000

    include_source = bool(body.get("include_source", True))

    snapshot_path = UPLOAD_DIR / filename
    if not snapshot_path.is_file():
        return jsonify({"error": f"Snapshot not found: {filename}"}), 404

    try:
        with snapshot_path.open("r", encoding="utf-8") as fh:
            snap = json.load(fh)
    except Exception as exc:
        logger.error("Failed to load snapshot %s: %s", snapshot_path, exc)
        return jsonify({"error": f"Failed to read snapshot: {exc}"}), 500

    files = snap.get("files") or []
    if not isinstance(files, list):
        return jsonify({"error": "Snapshot missing 'files' list"}), 400

    def guess_language(path: str, tags: Optional[List[str]]) -> Optional[str]:
        """Very small helper to guess language for display/hints."""
        if tags:
            for t in tags:
                lower = str(t).lower()
                if lower in ("python", "javascript", "typescript", "text", "config"):
                    return lower
        lower = path.lower()
        if lower.endswith(".py"):
            return "python"
        if lower.endswith(".js"):
            return "javascript"
        if lower.endswith(".ts"):
            return "typescript"
        if lower.endswith(".json"):
            return "json"
        if lower.endswith(".md"):
            return "markdown"
        if lower.endswith(".txt"):
            return "text"
        return None

    selected_files: List[Dict[str, Any]] = []
    remaining = max(0, int(max_total_chars))
    truncated_global = False

    for entry in files:
        path = entry.get("path")
        if not isinstance(path, str):
            continue
        if paths_filter is not None and path not in paths_filter:
            continue

        size = entry.get("size")
        tags = entry.get("tags") or []
        is_binary = bool(entry.get("is_binary"))
        text = entry.get("source") if include_source else None

        snippet: Optional[str] = None
        truncated_file = False

        if include_source and text is not None and not is_binary and remaining > 0:
            text = str(text)
            per_file_limit = max(0, int(max_file_chars)) if max_file_chars is not None else len(text)
            allowed = min(len(text), per_file_limit, remaining)
            snippet = text[:allowed]
            if allowed < len(text):
                truncated_file = True
            remaining -= allowed
            if remaining <= 0:
                truncated_global = True

        selected_files.append(
            {
                "path": path,
                "size": size,
                "tags": tags,
                "is_binary": is_binary,
                "language": guess_language(path, tags),
                "snippet": snippet,
                "snippet_truncated": truncated_file,
            }
        )

        if truncated_global:
            break

    snapshot_meta = {
        "schema": snap.get("schema"),
        "root_path": snap.get("root_path"),
        "file_count": len(files),
        "total_size": snap.get("total_size"),
    }

    return jsonify(
        {
            "snapshot": snapshot_meta,
            "files": selected_files,
            "truncated": truncated_global,
            "max_total_chars": max_total_chars,
            "max_file_chars": max_file_chars,
        }
    )


# -------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------


def main() -> None:
    import sys

    # Validate configuration on startup
    validate_config()

    # Check Ollama connectivity with retry logic
    max_retries = get_ollama_startup_retries()
    retry_delay = get_ollama_startup_retry_delay()
    require_ollama = require_ollama_on_startup()

    ollama_available = wait_for_ollama(
        max_retries=max_retries,
        initial_delay=retry_delay,
        timeout=5.0
    )

    if not ollama_available:
        if require_ollama:
            # Print helpful instructions and exit
            print(get_ollama_startup_instructions(), file=sys.stderr)
            logger.error("Ollama is not available and REQUIRE_OLLAMA_ON_STARTUP is enabled. Exiting.")
            sys.exit(1)
        else:
            logger.warning(
                "Cannot connect to Ollama at %s. The server will start but API calls will fail.",
                get_ollama_base()
            )
            logger.warning("Set REQUIRE_OLLAMA_ON_STARTUP=true to enforce Ollama availability at startup.")

    host, port = get_host_port(default_port=8070)
    logger.info(
        "Starting CodeSmith Ollama Helper on %s:%s (OLLAMA_HOST=%s, MODEL=%s, HEAVY_MODEL=%s)",
        host,
        port,
        get_ollama_base(),
        get_default_model(),
        get_heavy_model(get_default_model()),
    )
    logger.info("Upload directory: %s", UPLOAD_DIR)
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
