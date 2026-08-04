#!/usr/bin/env python3
"""Conversation storage, so history survives a reload and follows you between devices.

SQLite rather than the browser: the whole point is starting a thread on a
desktop and finishing it on a phone, which per-browser storage cannot do.

Connections are opened per call rather than shared. Waitress serves requests on
a thread pool, and a single sqlite3 connection is not safe across threads;
per-call connections in WAL mode keep concurrent readers off each other's backs
while writes stay short. The cost is a file open per request, which is noise
next to talking to a language model.

Anything stored here is readable by anyone who can reach the app, so an install
that keeps history should also set CHAT_AUTH — see the README.

Environment: CHAT_DB — path to the database file (default ./chat.db).
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from config import logger

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    model       TEXT,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    seq             INTEGER NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    images          TEXT,
    sources         TEXT,
    created_at      REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages (conversation_id, seq);
CREATE INDEX IF NOT EXISTS idx_conversations_updated
    ON conversations (updated_at DESC);
"""

_TITLE_MAX = 60


def db_path() -> Path:
    """Where the database lives. Read per call so tests can redirect it."""
    return Path(os.getenv("CHAT_DB", "chat.db")).expanduser()


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    """Yield a connection with the schema applied and foreign keys on."""
    path = db_path()
    if path.parent and str(path.parent) not in ("", "."):
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10)
    try:
        conn.row_factory = sqlite3.Row
        # WAL lets a reader run while a write is in flight, which matters as
        # soon as two devices have the app open.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def available() -> bool:
    """Whether history can be stored (the file is creatable and writable)."""
    try:
        with _connect():
            return True
    # OSError too, not just sqlite3.Error: _connect() creates the parent
    # directory first, so an unwritable or nonsensical CHAT_DB raises
    # PermissionError/FileExistsError/EROFS before sqlite is ever reached — and
    # this is called from /api/health, which the whole UI bootstraps from.
    except (sqlite3.Error, OSError) as exc:
        logger.warning("Conversation history unavailable (%s): %s", db_path(), exc)
        return False


def _title_from(text: str) -> str:
    """A conversation title taken from its opening message."""
    clean = " ".join((text or "").split())
    if not clean:
        return "New chat"
    return clean[:_TITLE_MAX] + ("…" if len(clean) > _TITLE_MAX else "")


def create(title: str = "", model: Optional[str] = None) -> Dict[str, Any]:
    """Start a conversation and return its record."""
    now = time.time()
    convo_id = uuid.uuid4().hex
    with _connect() as conn:
        conn.execute(
            "INSERT INTO conversations (id, title, model, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (convo_id, _title_from(title), model, now, now),
        )
    return {"id": convo_id, "title": _title_from(title), "model": model,
            "created_at": now, "updated_at": now, "message_count": 0}


def list_conversations(limit: int = 200) -> List[Dict[str, Any]]:
    """Most recently updated conversations first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT c.id, c.title, c.model, c.created_at, c.updated_at,"
            "       (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count"
            "  FROM conversations c"
            " ORDER BY c.updated_at DESC"
            " LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def get(convo_id: str) -> Optional[Dict[str, Any]]:
    """A conversation with its messages, or None when it doesn't exist."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, title, model, created_at, updated_at FROM conversations WHERE id = ?",
            (convo_id,),
        ).fetchone()
        if row is None:
            return None
        messages = conn.execute(
            "SELECT role, content, images, sources FROM messages"
            " WHERE conversation_id = ? ORDER BY seq",
            (convo_id,),
        ).fetchall()

    convo = dict(row)
    convo["messages"] = [
        {
            "role": m["role"],
            "content": m["content"],
            "images": json.loads(m["images"]) if m["images"] else None,
            "sources": json.loads(m["sources"]) if m["sources"] else None,
        }
        for m in messages
    ]
    return convo


def add_message(
    convo_id: str,
    role: str,
    content: str,
    images: Optional[List[str]] = None,
    sources: Optional[List[Dict[str, str]]] = None,
) -> bool:
    """Append a message. Returns False when the conversation is gone.

    The first user message also names an untitled conversation, so a thread
    identifies itself in the list without anyone having to rename it.
    """
    now = time.time()
    with _connect() as conn:
        exists = conn.execute(
            "SELECT title FROM conversations WHERE id = ?", (convo_id,)
        ).fetchone()
        if exists is None:
            return False

        seq_row = conn.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 AS next FROM messages WHERE conversation_id = ?",
            (convo_id,),
        ).fetchone()
        try:
            conn.execute(
                "INSERT INTO messages (conversation_id, seq, role, content, images, sources, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    convo_id,
                    seq_row["next"],
                    role,
                    content or "",
                    json.dumps(images) if images else None,
                    json.dumps(sources) if sources else None,
                    now,
                ),
            )
        except sqlite3.IntegrityError:
            # The conversation was deleted between the check above and here —
            # another device, most likely. Same answer as "never existed".
            return False
        if role == "user" and exists["title"] in ("", "New chat"):
            conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (_title_from(content), now, convo_id),
            )
        else:
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, convo_id)
            )
    return True


def rename(convo_id: str, title: str) -> bool:
    """Set a conversation's title. Returns False when it doesn't exist."""
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (_title_from(title) or "New chat", time.time(), convo_id),
        )
        return cur.rowcount > 0


def delete(convo_id: str) -> bool:
    """Remove a conversation and its messages. False when it didn't exist."""
    with _connect() as conn:
        # Explicit, rather than relying on the cascade: PRAGMA foreign_keys is
        # per-connection and easy to lose in a future refactor.
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (convo_id,))
        cur = conn.execute("DELETE FROM conversations WHERE id = ?", (convo_id,))
        return cur.rowcount > 0


def stats() -> Dict[str, int]:
    """Row counts and on-disk size, for the UI to show what history costs."""
    with _connect() as conn:
        convos = conn.execute("SELECT COUNT(*) AS n FROM conversations").fetchone()["n"]
        msgs = conn.execute("SELECT COUNT(*) AS n FROM messages").fetchone()["n"]
    try:
        size = db_path().stat().st_size
    except OSError:
        size = 0
    return {"conversations": convos, "messages": msgs, "bytes": size}
