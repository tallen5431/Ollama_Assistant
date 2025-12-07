#!/usr/bin/env python3
"""CodeSmith patch helpers for the Ollama Helper.

This module contains:
- A compact description string for the CodeSmith patch schema
- Helpers to build a system prompt that references the schema
- Utilities to normalize a model's patch-like response into a
  well-formed CodeSmith patch JSON object.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


# -------------------------------------------------------------------
# CodeSmith patch schema helper (for prompts)
# -------------------------------------------------------------------

CODESMITH_PATCH_SCHEMA_HELP: str = """
codesmith.patch.v1:
{
  "schema": "codesmith.patch.v1",
  "root_path": "<project root on disk>",
  "transactional": true,
  "file_hashes": { "path": "sha256", ... },
  "edits": [
    {
      "operation": "modify" | "create" | "delete",
      "path": "relative/path.py",

      # modify (partial edit)
      "search": "old text",
      "replace": "new text",

      # OR modify (regex)
      "pattern": "regex",
      "replacement": "text",
      "is_regex": true,

      # OR modify/create (full content)
      "content": "... full file text ...",

      # Optional safety
      "expected_hash": "<sha256>",
      "must_match": true
    },
    ...
  ]
}
""".strip()


# -------------------------------------------------------------------
# System / prompt helpers
# -------------------------------------------------------------------


def build_code_system_prompt(user_system: Optional[str] = None) -> str:
    """System prompt tailored for CodeSmith-style code assistance.

    Includes a compact summary of the CodeSmith patch schema so
    individual requests don't need to paste it again.
    """
    base = (
        "You are a code assistant helping the user work on projects managed "
        "by CodeSmith. Prefer small, well-scoped changes. When possible, "
        "suggest changes in terms of patches or minimal edits, and keep "
        "your answers concise and focused on code.\n\n"
        "CodeSmith patch schema (summary):\n"
        f"{CODESMITH_PATCH_SCHEMA_HELP}"
    )
    if user_system:
        return user_system.strip() + "\n\n" + base
    return base


# -------------------------------------------------------------------
# Patch JSON normalization helpers
# -------------------------------------------------------------------


def _strip_markdown_fences(text: str) -> str:
    """Remove common ```json ... ``` style fences from model output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop first line (``` or ```json)
        lines = lines[1:]
        # Drop trailing ``` if present
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def _extract_json_block(text: str) -> str:
    """Heuristically extract the main {...} block from text.

    Finds the first '{' and the last '}' and returns that slice.
    If that fails, the original text is returned unchanged.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _ensure_patch_wrapper(
    obj: Any,
    *,
    root_path: Optional[str] = None,
    file_hashes: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Normalize parsed JSON into a full CodeSmith patch dict.

    Accepts a variety of shapes:
    - A complete patch object (schema == codesmith.patch.v1).
    - An object with an 'edits' list.
    - A bare list of edits.

    In all cases we fill in:
    - schema = codesmith.patch.v1
    - transactional = true (if missing)
    - root_path / file_hashes if provided and not already present.
    """
    if isinstance(obj, dict) and obj.get("schema") == "codesmith.patch.v1":
        patch = obj
    elif isinstance(obj, dict) and "edits" in obj:
        patch = {
            "schema": "codesmith.patch.v1",
            "transactional": True,
            "edits": obj.get("edits", []),
        }
    elif isinstance(obj, list):
        patch = {
            "schema": "codesmith.patch.v1",
            "transactional": True,
            "edits": obj,
        }
    else:
        raise ValueError(
            "Cannot normalize patch: JSON must be an object with 'edits' or a list of edits"
        )

    if root_path is not None and not patch.get("root_path"):
        patch["root_path"] = root_path
    if file_hashes is not None and not patch.get("file_hashes"):
        patch["file_hashes"] = file_hashes
    if "transactional" not in patch:
        patch["transactional"] = True

    return patch


def normalize_patch_text(
    raw_text: str,
    *,
    root_path: Optional[str] = None,
    file_hashes: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Best-effort normalization of a model's patch response.

    Responsibilities:
    - Strip Markdown fences (```json ... ```).
    - Extract the main JSON-looking block from surrounding prose.
    - Parse it as JSON.
    - Ensure it is a well-formed CodeSmith patch object with schema,
      transactional flag, and (optionally) root_path/file_hashes.

    Raises ValueError with a readable message if parsing/normalization
    fails so the caller can surface it nicely to the UI.
    """
    cleaned = _strip_markdown_fences(raw_text)
    cleaned = _extract_json_block(cleaned)

    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse model output as JSON: {exc}") from exc

    return _ensure_patch_wrapper(obj, root_path=root_path, file_hashes=file_hashes)
