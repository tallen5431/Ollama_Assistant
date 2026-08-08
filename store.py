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
    image_meta      TEXT,
    created_at      REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS routines (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    body        TEXT NOT NULL,
    photos      INTEGER NOT NULL DEFAULT 0,
    web         INTEGER,
    photo_meta  INTEGER,
    position    REAL NOT NULL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages (conversation_id, seq);
CREATE INDEX IF NOT EXISTS idx_conversations_updated
    ON conversations (updated_at DESC);
-- What a routine run produced, as fields rather than prose. Deliberately not
-- foreign-keyed to routines: deleting the routine you used last year must not
-- delete the year of records you kept with it, which is why the name is copied
-- in rather than looked up.
CREATE TABLE IF NOT EXISTS records (
    id              TEXT PRIMARY KEY,
    routine_id      TEXT,
    routine_name    TEXT NOT NULL,
    conversation_id TEXT,
    fields          TEXT NOT NULL,
    created_at      REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_routines_position
    ON routines (position, created_at);
CREATE INDEX IF NOT EXISTS idx_records_created
    ON records (created_at DESC);
"""

# Columns added after the first release. CREATE TABLE IF NOT EXISTS leaves an
# existing table at whatever shape it already had, so without this an in-place
# upgrade keeps the old columns and every read of a new one raises.
#
# A whole new *table* needs no entry here — executescript runs the schema on
# every connection, so an existing database grows it on the next request. Only
# a new column in an existing table does.
_ADDED_COLUMNS = (
    ("messages", "image_meta", "TEXT"),
    # The reasoning panel and the "what it did" panel were live stream state
    # only, so a thread opened on the other device had the reply and neither of
    # them — and those are exactly what you go looking for when an answer
    # reads oddly, which is usually later and elsewhere.
    ("messages", "thinking", "TEXT"),
    ("messages", "steps", "TEXT"),
    # What a routine records after each run: a JSON list of field names.
    ("routines", "record", "TEXT"),
)

_TITLE_MAX = 60

_ROUTINE_NAME_MAX = 40        # a chip that still fits across a phone
# The body becomes an ordinary user turn and is re-sent with every later turn of
# that thread, so a stray paste must not quietly own the context window.
_ROUTINE_BODY_MAX = 4000
_ROUTINE_PHOTOS_MAX = 4       # what the composer accepts and web._MAX_IMAGES_READ reads


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
        _migrate(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _migrate(conn: sqlite3.Connection) -> None:
    """Add any column a database written by an earlier version is missing.

    ALTER TABLE ADD COLUMN is cheap — sqlite rewrites no rows, it only edits the
    stored schema — so running the check on every connection costs a schema
    lookup, not a table scan.
    """
    for table, column, kind in _ADDED_COLUMNS:
        names = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        # An empty set means the table does not exist, which _SCHEMA has just
        # ruled out; adding a column to nothing would raise.
        if names and column not in names:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")


def available() -> bool:
    """Whether history can be stored — checked by writing, not by opening.

    Opening proves almost nothing: on an existing database sqlite allocates no
    page, so a full disk and a corrupt file both opened cleanly and this
    returned True forever while every write failed. The realistic trigger on a
    small always-on box is not disk-full but SQLITE_CORRUPT after a power cut,
    and no reload recovered from it because nothing ever noticed.
    """
    try:
        with _connect() as conn:
            # BEGIN IMMEDIATE takes a write lock and forces the journal to be
            # created, which is what actually fails on a read-only mount or a
            # full disk — then rolls back, so nothing is left in the file. On a
            # corrupt database _connect's schema step has already raised.
            # The short busy timeout keeps this off the critical path: it runs
            # on every page load, and the default 10 s would stall the UI
            # behind an ordinary concurrent write.
            conn.execute("PRAGMA busy_timeout = 300")
            conn.execute("BEGIN IMMEDIATE")
            conn.rollback()
            return True
    except sqlite3.OperationalError as exc:
        # Being locked means the database exists and something is writing to
        # it — a sign of health, not of failure. Reporting "history
        # unavailable" because another device saved a message at the same
        # moment would be worse than not checking at all.
        if "locked" in str(exc).lower() or "busy" in str(exc).lower():
            return True
        logger.warning("Conversation history unavailable (%s): %s", db_path(), exc)
        return False
    # OSError too, not just sqlite3.Error: _connect() creates the parent
    # directory first, so an unwritable or nonsensical CHAT_DB raises
    # PermissionError/FileExistsError/EROFS before sqlite is ever reached — and
    # this is called from /api/health, which the whole UI bootstraps from.
    except (sqlite3.Error, OSError) as exc:
        logger.warning("Conversation history unavailable (%s): %s", db_path(), exc)
        return False


def _loads(raw: Any) -> Any:
    """A JSON column, or None if it will not parse.

    A row that cannot be decoded used to raise out of get(), which 500s the
    conversation route — so one bad write made that thread permanently
    unopenable, with no way to reach the messages either side of it. A missing
    attachment is a far smaller loss than a missing conversation.
    """
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        logger.warning("Ignoring an unreadable JSON column in %s", db_path().name)
        return None


def _as_list(value: Any) -> Optional[list]:
    """A list, or nothing. Anything else stored here comes back out and is
    iterated by the browser, where a string turns into a page error on every
    later open of that thread."""
    return value if isinstance(value, list) and value else None


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
            "SELECT role, content, images, sources, image_meta, thinking, steps"
            " FROM messages"
            " WHERE conversation_id = ? ORDER BY seq",
            (convo_id,),
        ).fetchall()

    convo = dict(row)
    convo["messages"] = [
        {
            "role": m["role"],
            "content": m["content"],
            "images": _loads(m["images"]),
            "sources": _loads(m["sources"]),
            # Reopening a thread and asking "where was that taken?" should get
            # the same answer as asking it the first time round.
            "image_meta": _loads(m["image_meta"]),
            # So does opening it to work out why an answer came out the way it
            # did — which is the whole point of these two panels.
            "thinking": m["thinking"] or None,
            "steps": _loads(m["steps"]),
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
    image_meta: Optional[List[Optional[Dict[str, Any]]]] = None,
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
                "INSERT INTO messages"
                " (conversation_id, seq, role, content, images, sources, image_meta, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    convo_id,
                    seq_row["next"],
                    role,
                    content or "",
                    json.dumps(_as_list(images)) if _as_list(images) else None,
                    json.dumps(_as_list(sources)) if _as_list(sources) else None,
                    # `any` rather than truthiness: [None, {...}] is a real
                    # entry for a second photo, [None, None] is nothing at all.
                    json.dumps(image_meta) if _as_list(image_meta)
                    and any(image_meta) else None,
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


# Long enough for a reasoning model on a hard question, short enough that a
# thread of them does not become the database. Truncated rather than dropped:
# the beginning of a scratchpad is the part that says what it decided to do.
_THINKING_MAX = 20000


def save_turn(
    convo_id: str,
    user: Dict[str, Any],
    reply: str,
    sources: Optional[List[Dict[str, str]]] = None,
    thinking: str = "",
    steps: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """Write a question and its answer together, or write neither.

    The browser used to save both, after the stream finished. That is fine
    until the tab is not there when it finishes — a phone that locks past its
    wake lock, an app switch, a Tailscale handoff from wifi to cellular — and
    then a reply the server spent two minutes generating exists nowhere at all.
    This is the server writing down what it produced, so a reload finds it.

    Both halves in one transaction because half a turn is worse than none: a
    question with no answer reads as an unanswered question rather than as a
    dropped connection.
    """
    if not convo_id or not (reply or "").strip():
        return False
    now = time.time()
    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT title FROM conversations WHERE id = ?", (convo_id,)
        ).fetchone()
        if row is None:
            return False
        seq = conn.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 AS next FROM messages"
            " WHERE conversation_id = ?", (convo_id,),
        ).fetchone()["next"]
        content = str(user.get("content") or "")
        images = _as_list(user.get("images"))
        meta = _as_list(user.get("image_meta"))
        conn.execute(
            "INSERT INTO messages"
            " (conversation_id, seq, role, content, images, sources, image_meta, created_at)"
            " VALUES (?, ?, 'user', ?, ?, NULL, ?, ?)",
            (convo_id, seq, content,
             json.dumps(images) if images else None,
             json.dumps(meta) if meta and any(meta) else None, now),
        )
        kept_thinking = (thinking or "")[:_THINKING_MAX] or None
        conn.execute(
            "INSERT INTO messages"
            " (conversation_id, seq, role, content, images, sources, image_meta,"
            "  thinking, steps, created_at)"
            " VALUES (?, ?, 'assistant', ?, NULL, ?, NULL, ?, ?, ?)",
            (convo_id, seq + 1, reply,
             json.dumps(sources) if sources else None,
             kept_thinking, json.dumps(steps) if steps else None, now),
        )
        if row["title"] in ("", "New chat"):
            conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (_title_from(content), now, convo_id),
            )
        else:
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, convo_id)
            )
    return True


def is_empty(convo_id: str) -> bool:
    """Whether a conversation has no messages at all.

    The browser now creates the thread before the reply starts, so that the
    server has somewhere to write it. A turn that produces nothing therefore
    leaves an empty thread behind, and this is how the browser knows to take it
    away again rather than leaving a row named after a question nobody answered.
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM messages WHERE conversation_id = ? LIMIT 1", (convo_id,)
        ).fetchone()
    return row is None


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
        deleted = cur.rowcount > 0
    if deleted:
        # Deleting an image-heavy thread frees a lot of pages, and the README
        # says deleting reclaims space — so make that true. No-ops unless there
        # is enough dead space to be worth the rewrite.
        compact()
    return deleted


# -------------------------------------------------------------------
# Routines — saved prompts you tap instead of typing
# -------------------------------------------------------------------
#
# A routine is a name, a body of prompt text, and three things it declares
# about the turn it is used on: how many photos it expects, and whether web
# access and photo details should be forced on or off. Those two are
# three-state on purpose — NULL means "leave the toggle where the user has it",
# which is a real third answer. Making every routine assert a position on the
# web is how you end up with routines that are landmines.


def _routine_name(text: str) -> str:
    """A chip label: one line, trimmed to something that fits."""
    clean = " ".join((text or "").split())
    return clean[:_ROUTINE_NAME_MAX]


def _tri_column(value: Any) -> Optional[bool]:
    """A three-state column back into a bool or None.

    ``bool(row["web"])`` on its own is wrong here: it collapses NULL and 0 into
    False, which turns "no opinion" into "force it off" on the first read.
    """
    return None if value is None else bool(value)


# What one run of a routine may write down. Names only — the values come from
# the answer. Kept small: a record with twenty columns is a form, and nobody
# fills in a form at the roadside.
_RECORD_FIELDS_MAX = 12
_RECORD_NAME_MAX = 32


def _record_fields(raw: Any) -> List[str]:
    """The field-name list off a routine row, however it was stored."""
    if not raw:
        return []
    try:
        names = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return []
    if not isinstance(names, list):
        return []
    out = []
    for name in names[:_RECORD_FIELDS_MAX]:
        clean = " ".join(str(name or "").split())[:_RECORD_NAME_MAX].strip()
        if clean and clean not in out:
            out.append(clean)
    return out


def _routine_row(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "body": row["body"],
        "photos": int(row["photos"] or 0),
        "web": _tri_column(row["web"]),
        "photo_meta": _tri_column(row["photo_meta"]),
        "position": row["position"],
        "record": _record_fields(row["record"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def create_routine(
    name: str,
    body: str,
    photos: int = 0,
    web: Optional[bool] = None,
    photo_meta: Optional[bool] = None,
    record: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Save a routine and return its record."""
    now = time.time()
    rid = uuid.uuid4().hex
    name = _routine_name(name)
    body = (body or "")[:_ROUTINE_BODY_MAX]
    photos = max(0, min(_ROUTINE_PHOTOS_MAX, int(photos or 0)))
    record = _record_fields(record)
    with _connect() as conn:
        # New ones go on the end of the strip. Ordering by "recently used" would
        # move a tap target that a thumb has learned where to find.
        #
        # Under the write lock, so two routines saved at once cannot read the
        # same MAX and land on the same position — which would leave their order
        # decided by the created_at tiebreak rather than by the strip.
        conn.execute("BEGIN IMMEDIATE")
        position = conn.execute(
            "SELECT COALESCE(MAX(position), 0) + 1 AS next FROM routines"
        ).fetchone()["next"]
        conn.execute(
            "INSERT INTO routines"
            " (id, name, body, photos, web, photo_meta, position, record,"
            "  created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (rid, name, body, photos,
             None if web is None else int(web),
             None if photo_meta is None else int(photo_meta),
             position, json.dumps(record) if record else None, now, now),
        )
    return {"id": rid, "name": name, "body": body, "photos": photos,
            "web": web, "photo_meta": photo_meta, "position": position,
            "record": record, "created_at": now, "updated_at": now}


def list_routines(limit: int = 100) -> List[Dict[str, Any]]:
    """Every routine, in the order they appear on the strip."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM routines ORDER BY position, created_at LIMIT ?", (limit,)
        ).fetchall()
    return [_routine_row(row) for row in rows]


# Only these may be written, and the SET clause is built from this tuple rather
# than from anything the request said — the column names are never interpolated
# from input.
_ROUTINE_COLUMNS = ("name", "body", "photos", "web", "photo_meta",
                    "position", "record")


def update_routine(rid: str, fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Change some of a routine. None when there is no such routine.

    Takes a dict rather than a row of optional parameters because ``None`` is a
    meaningful *value* for ``web`` and ``photo_meta``, so there is no sentinel
    that would tell "not supplied" from "set to nothing" apart.

    Returns the whole updated record, not a bool: the caller re-renders the chip
    from it, and the store is where the name gets truncated and the photo count
    clamped, so the client has to be told what was actually kept.
    """
    updates: Dict[str, Any] = {}
    for column in _ROUTINE_COLUMNS:
        if column not in fields:
            continue
        value = fields[column]
        if column == "name":
            updates[column] = _routine_name(value)
        elif column == "body":
            updates[column] = (value or "")[:_ROUTINE_BODY_MAX]
        elif column == "photos":
            updates[column] = max(0, min(_ROUTINE_PHOTOS_MAX, int(value or 0)))
        elif column in ("web", "photo_meta"):
            updates[column] = None if value is None else int(bool(value))
        elif column == "record":
            names = _record_fields(value)
            updates[column] = json.dumps(names) if names else None
        else:
            updates[column] = float(value)
    if not updates:
        return None

    clause = ", ".join(f"{column} = ?" for column in updates)
    with _connect() as conn:
        cur = conn.execute(
            f"UPDATE routines SET {clause}, updated_at = ? WHERE id = ?",
            (*updates.values(), time.time(), rid),
        )
        if cur.rowcount < 1:
            return None
        row = conn.execute("SELECT * FROM routines WHERE id = ?", (rid,)).fetchone()
    return _routine_row(row) if row else None


def delete_routine(rid: str) -> bool:
    """Remove a routine. False when it didn't exist."""
    with _connect() as conn:
        cur = conn.execute("DELETE FROM routines WHERE id = ?", (rid,))
        return cur.rowcount > 0


# Four to start with, installed only when asked for — never seeded on connect.
# A write inside _connect() would leave an open transaction, and available()'s
# BEGIN IMMEDIATE then raises "cannot start a transaction within a transaction",
# which is neither "locked" nor "busy" — so the whole history UI would vanish.
#
# Each is (name, body, photos, web, photo_meta).
_STARTERS = (
    (
        "🚗 Trip",
        "Two photos of the same car's odometer are attached: one from the start "
        "of a trip and one from the end.\n\n"
        "Read the main odometer number in each photo, digit by digit, and say "
        "which reading came from which photo. If a photo also shows a smaller "
        "trip meter, report that separately and never mix the two. If the last "
        "digit sits in its own box or a different colour it is tenths — say "
        "whether you counted it. Say whether the dial reads miles or "
        "kilometres, and if it does not say, say which you assumed.\n\n"
        "The photo details above label the two photos \"Image 1\" and \"Image 2\" "
        "in the order they were attached, and those are the same two photos as "
        "\"[image 1]\" and \"[image 2]\" in any transcription above. Quote both "
        "capture times back to me — I am asking for them, so the usual rule "
        "about not reciting photo details does not apply here.\n\n"
        "Then give me:\n"
        "- Distance: the larger reading minus the smaller. Do not assume the "
        "first photo is the start; the later capture time is the end of the trip.\n"
        "- Elapsed time: the gap between the two capture times.\n"
        "- Average speed over that elapsed time.\n\n"
        "If either photo has no readable number, or has no capture time in the "
        "photo details above, say exactly which photo is missing which thing and "
        "stop there. Do not estimate either one, and do not answer from one photo "
        "alone. If the photo with the later time shows the smaller reading, say "
        "so plainly rather than reporting a negative distance.\n\n"
        "Capture times carry no time zone. If this trip could have crossed one, "
        "say the elapsed time may be off by whole hours.",
        2, False, True,
    ),
    (
        "📊 Before / after",
        "Two photos of the same thing are attached, taken at different times.\n\n"
        "Say what is different between them, most significant first — not a "
        "general description of each. If either shows a number, a gauge or a "
        "reading, give both values and the difference. Ignore differences that "
        "are only camera angle, distance or lighting.\n\n"
        "The photo details above label them \"Image 1\" and \"Image 2\" in the "
        "order they were attached. Quote both capture times and the gap between "
        "them; I am asking for them, so report them. If a capture time is "
        "missing for either photo, say which one and do not estimate it.\n\n"
        "If the two photos are not of the same subject, say so and stop.",
        2, None, True,
    ),
    (
        "📄 Read this",
        "Transcribe every readable word and number in this photo, exactly as it "
        "appears, keeping the line breaks.\n\n"
        "Copy any serial number, part number, code, price or total character for "
        "character. Do not tidy the spelling, convert units, or correct "
        "arithmetic that does not add up — if a printed total disagrees with the "
        "items, give both numbers and say which is which. Where something is "
        "genuinely unreadable, write [unreadable] in its place rather than "
        "guessing at it.\n\n"
        "Do not summarise, and do not explain what the thing is unless I ask.",
        # Photo details forced off: a label or a receipt has no interesting time
        # or place, and this is the routine most likely to be pointed at
        # something in someone else's house.
        1, None, False,
    ),
    (
        "✍️ Plain words",
        "Explain the following in plain language, for someone who is not in the "
        "field.\n\n"
        "Keep it under 200 words, define any term you have to use, and end with "
        "the one sentence that matters most. If it is ambiguous, say which part "
        "and why instead of picking one reading and running with it.",
        0, None, None,
    ),
)


def create_starters() -> List[Dict[str, Any]]:
    """Install the shipped routines, skipping any name already taken.

    Idempotent, and idempotent *under a double tap*: the read of what is
    already there and the inserts share one connection and one transaction, so
    two requests arriving together cannot both find the table empty and both
    install a full set. Reading on one connection and writing on another —
    which is what calling create_routine in a loop did — left exactly that gap.
    """
    now = time.time()
    made: List[Dict[str, Any]] = []
    with _connect() as conn:
        # BEGIN IMMEDIATE takes the write lock up front, so the second request
        # waits here rather than racing us to the same conclusion.
        conn.execute("BEGIN IMMEDIATE")
        taken = {row["name"] for row in conn.execute("SELECT name FROM routines")}
        position = conn.execute(
            "SELECT COALESCE(MAX(position), 0) AS top FROM routines"
        ).fetchone()["top"]
        for name, body, photos, web, photo_meta in _STARTERS:
            if name in taken:
                continue
            position += 1
            rid = uuid.uuid4().hex
            conn.execute(
                "INSERT INTO routines"
                " (id, name, body, photos, web, photo_meta, position, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (rid, name, body, photos,
                 None if web is None else int(web),
                 None if photo_meta is None else int(photo_meta),
                 position, now, now),
            )
            made.append({"id": rid, "name": name, "body": body, "photos": photos,
                         "web": web, "photo_meta": photo_meta, "position": position,
                         "created_at": now, "updated_at": now})
    return made


# -------------------------------------------------------------------
# Records — what a routine run wrote down
# -------------------------------------------------------------------
#
# A row per run: which routine, when, and the fields it produced. This is the
# thing you keep — a year of trips outlives the conversations that made them,
# and outlives the routine too, which is why the name is copied in rather than
# joined and why deleting a routine leaves its records alone.

_RECORD_VALUE_MAX = 300       # a field, not an essay


def _record_row(row: sqlite3.Row) -> Dict[str, Any]:
    try:
        fields = json.loads(row["fields"])
    except (TypeError, ValueError):
        fields = {}
    return {
        "id": row["id"],
        "routine_id": row["routine_id"],
        "routine_name": row["routine_name"],
        "conversation_id": row["conversation_id"],
        "fields": fields if isinstance(fields, dict) else {},
        "created_at": row["created_at"],
    }


def _clean_fields(fields: Any) -> Dict[str, str]:
    """Whatever came back from the extraction, as a flat dict of short strings."""
    if not isinstance(fields, dict):
        return {}
    out: Dict[str, str] = {}
    for key, value in list(fields.items())[:_RECORD_FIELDS_MAX]:
        name = " ".join(str(key or "").split())[:_RECORD_NAME_MAX].strip()
        if not name:
            continue
        if value is None or value is False:
            text = ""
        elif isinstance(value, (list, tuple)):
            text = ", ".join(str(v) for v in value)
        else:
            text = str(value)
        out[name] = " ".join(text.split())[:_RECORD_VALUE_MAX]
    return out


def add_record(
    routine_name: str,
    fields: Dict[str, Any],
    routine_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Keep one run. None when there was nothing worth keeping.

    A record whose every field came back empty is not a record — it is the
    extraction having failed, and storing it would put a blank row in a log the
    owner is meant to be able to trust.
    """
    clean = _clean_fields(fields)
    if not any(v for v in clean.values()):
        return None
    now = time.time()
    rid = uuid.uuid4().hex
    name = _routine_name(routine_name) or "Routine"
    with _connect() as conn:
        conn.execute(
            "INSERT INTO records"
            " (id, routine_id, routine_name, conversation_id, fields, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (rid, routine_id or None, name, conversation_id or None,
             json.dumps(clean), now),
        )
    return {"id": rid, "routine_id": routine_id or None, "routine_name": name,
            "conversation_id": conversation_id or None, "fields": clean,
            "created_at": now}


def list_records(routine_name: str = "", limit: int = 500) -> List[Dict[str, Any]]:
    """Records, newest first, optionally for one routine."""
    with _connect() as conn:
        if routine_name:
            rows = conn.execute(
                "SELECT * FROM records WHERE routine_name = ?"
                " ORDER BY created_at DESC LIMIT ?", (routine_name, limit)).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM records ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [_record_row(row) for row in rows]


def update_record(rid: str, fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Correct a record. None when there is no such record.

    Editable on purpose: the fields are extracted by a model from its own prose,
    which is good enough to be worth keeping and not good enough to be beyond
    question. A log you cannot correct is one you stop trusting.
    """
    clean = _clean_fields(fields)
    if not clean:
        return None
    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM records WHERE id = ?", (rid,)).fetchone()
        if row is None:
            return None
        # Merged, not replaced. Correcting one cell from the table would
        # otherwise silently drop every column the edit did not mention, which
        # is a poor property for the thing you are keeping records in.
        merged = dict(_record_row(row)["fields"])
        merged.update(clean)
        conn.execute("UPDATE records SET fields = ? WHERE id = ?",
                     (json.dumps(_clean_fields(merged)), rid))
        row = conn.execute("SELECT * FROM records WHERE id = ?", (rid,)).fetchone()
    return _record_row(row) if row else None


def delete_record(rid: str) -> bool:
    """Remove one record. False when it didn't exist."""
    with _connect() as conn:
        return conn.execute("DELETE FROM records WHERE id = ?", (rid,)).rowcount > 0


def record_columns(records: List[Dict[str, Any]]) -> List[str]:
    """Every field name across these records, in the order first seen.

    Routines change, so two runs of the same one can disagree about their
    columns. A table has to show both rather than the intersection.
    """
    columns: List[str] = []
    for record in records:
        for name in record.get("fields", {}):
            if name not in columns:
                columns.append(name)
    return columns


_SEARCH_LIMIT = 60
_SNIPPET_PAD = 60


def _snippet(text: str, needle: str) -> str:
    """The part of a message the match is in, with a little either side."""
    at = text.lower().find(needle.lower())
    if at < 0:
        return text[:2 * _SNIPPET_PAD].strip()
    start = max(0, at - _SNIPPET_PAD)
    end = min(len(text), at + len(needle) + _SNIPPET_PAD)
    return ("…" if start else "") + text[start:end].strip() + ("…" if end < len(text) else "")


def search(query: str, limit: int = _SEARCH_LIMIT) -> Dict[str, List[Dict[str, Any]]]:
    """Find a phrase in the conversations and in the records.

    LIKE rather than FTS5: this database holds one household's chat history —
    thousands of rows, not millions — where a scan costs single-digit
    milliseconds and needs no second copy of every message kept in step by
    triggers. The failure mode of a stale index is a search that quietly misses
    what you are looking for, which is the one thing a search must not do.
    """
    needle = str(query or "").strip()
    if len(needle) < 2:
        return {"conversations": [], "records": []}
    like = "%" + needle.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
    limit = max(1, min(int(limit or _SEARCH_LIMIT), 200))
    with _connect() as conn:
        # One row per conversation, carrying the newest message that matched —
        # a thread where a word appears nine times is one result, not nine.
        rows = conn.execute(
            "SELECT c.id, c.title, c.updated_at,"
            "       (SELECT m.content FROM messages m"
            "         WHERE m.conversation_id = c.id AND m.content LIKE ? ESCAPE '\\'"
            "         ORDER BY m.seq DESC LIMIT 1) AS hit"
            "  FROM conversations c"
            " WHERE c.title LIKE ? ESCAPE '\\'"
            "    OR EXISTS (SELECT 1 FROM messages m"
            "                WHERE m.conversation_id = c.id"
            "                  AND m.content LIKE ? ESCAPE '\\')"
            " ORDER BY c.updated_at DESC LIMIT ?",
            (like, like, like, limit),
        ).fetchall()
        found = [
            {"id": r["id"], "title": r["title"], "updated_at": r["updated_at"],
             "snippet": _snippet(r["hit"], needle) if r["hit"] else ""}
            for r in rows
        ]
        # Records are searched on their values, which is where "Uber" or a
        # date actually lives; the routine name is searched too.
        kept = [
            _record_row(r) for r in conn.execute(
                "SELECT * FROM records"
                " WHERE routine_name LIKE ? ESCAPE '\\' OR fields LIKE ? ESCAPE '\\'"
                " ORDER BY created_at DESC LIMIT ?", (like, like, limit)).fetchall()
        ]
    return {"conversations": found, "records": kept}


# Photos are the bulk of what history costs: the browser caps a re-encode at
# about 900KB, which is 1.2MB of base64 in the row, and a two-photo routine run
# every day is most of a gigabyte a year. What is worth keeping a year later is
# almost never the picture of the odometer — it is the number that was read off
# it, which is already in the text and in the records. So the pixels expire and
# the words do not.
_SWEEP_EVERY = 3600.0
_last_sweep = 0.0


def forget_images(older_than_days: float) -> Dict[str, int]:
    """Drop stored photos older than the cutoff, keeping every word.

    Returns what it did. ``older_than_days`` of 0 or less keeps them forever.
    """
    days = float(older_than_days or 0)
    if days <= 0:
        return {"messages": 0, "bytes": 0}
    cutoff = time.time() - days * 86400.0
    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(LENGTH(images)), 0) AS b"
            "  FROM messages WHERE images IS NOT NULL AND created_at < ?", (cutoff,)
        ).fetchone()
        if not row["n"]:
            return {"messages": 0, "bytes": 0}
        conn.execute(
            "UPDATE messages SET images = NULL"
            " WHERE images IS NOT NULL AND created_at < ?", (cutoff,)
        )
        done = {"messages": row["n"], "bytes": row["b"]}
    logger.info("Forgot %d stored photo(s), %d KB of text", done["messages"],
                done["bytes"] // 1024)
    # The rows are now dead space; hand it back if there is enough to bother.
    compact()
    return done


def maybe_forget_images(older_than_days: float) -> Dict[str, int]:
    """forget_images, at most once an hour.

    Hung off ordinary traffic rather than a timer thread: this app is one
    process serving one household, and a background thread that has to be
    started, stopped and kept out of the tests earns its keep only if the work
    cannot wait for the next request. This can.
    """
    global _last_sweep
    now = time.time()
    if now - _last_sweep < _SWEEP_EVERY:
        return {"messages": 0, "bytes": 0}
    _last_sweep = now
    try:
        return forget_images(older_than_days)
    except sqlite3.Error as exc:      # a sweep must never cost a request
        logger.warning("Could not sweep old photos: %s", exc)
        return {"messages": 0, "bytes": 0}


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


# Only bother when a meaningful share of the file is dead space; a VACUUM
# rewrites the whole database, so doing it after every delete would be silly.
_COMPACT_MIN_FREE_FRACTION = 0.25
_COMPACT_MIN_BYTES = 4 * 1024 * 1024


def compact() -> int:
    """Give freed pages back to the filesystem. Returns the bytes reclaimed.

    Gated on there being enough dead space to be worth it: VACUUM rewrites the
    whole database, so running it after every delete would be silly. Deleting
    one small conversation does nothing; deleting an image-heavy one, or
    several, reclaims the lot.
    """
    try:
        before = db_path().stat().st_size
    except OSError:
        return 0
    try:
        with _connect() as conn:
            page_size = conn.execute("PRAGMA page_size").fetchone()[0]
            free = conn.execute("PRAGMA freelist_count").fetchone()[0]
            total = conn.execute("PRAGMA page_count").fetchone()[0]
            if not total or not page_size:
                return 0
            free_bytes = free * page_size
            if free_bytes < _COMPACT_MIN_BYTES and \
                    (free / total) < _COMPACT_MIN_FREE_FRACTION:
                return 0
            conn.isolation_level = None        # VACUUM cannot run in a transaction
            conn.execute("VACUUM")
            # In WAL mode a VACUUM writes the compacted database into the WAL;
            # the main file is not truncated until a checkpoint. Without this,
            # VACUUM appeared to succeed while the file stayed exactly the same
            # size — 4.6 MB with 99% of its pages on the freelist.
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except (sqlite3.Error, OSError) as exc:
        logger.warning("Could not compact %s: %s", db_path(), exc)
        return 0
    try:
        reclaimed = before - db_path().stat().st_size
    except OSError:
        return 0
    if reclaimed > 0:
        logger.info("Reclaimed %.1f MB from %s", reclaimed / (1024 * 1024), db_path())
    return max(0, reclaimed)
