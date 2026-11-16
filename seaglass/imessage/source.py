"""`imessage/source.py` — the ONLY module aware of Apple's `chat.db` schema.

Everything downstream (chunker, embedder, search) consumes the clean
`Message` dataclass defined here. See development-plans/PLAN.md §6 Phase 1.

Key gotchas encoded here, each documented in PLAN.md / DESIGN-NOTES.md:

* `attributedBody` is a legacy **typedstream** blob, not a keyed archive
  (see `attributedbody.py`).
* Apple epoch units are **not uniform across history** -- pre-High-Sierra
  rows store seconds since 2001-01-01, newer rows store nanoseconds.
  Detected per-row by magnitude.
* `message.ROWID` is global and chronological **across all chats** -- never
  identify a chunk's messages by ROWID range (DESIGN-NOTES.md §9, "A bare
  message.ROWID range as chunk membership"). Chunk membership lives in the
  `chunk_message` junction table built later by the chunker, not here.
* Per ADDENDUM.md §4: on a machine mid-iCloud-backfill, `message.ROWID` is
  **not** a proxy for arrival/chronological order -- a backfilled old
  message can get a fresh, high ROWID. That matters for the sync design
  (index/sync.py), not for this module, which just reads what's there.
"""

from __future__ import annotations

import dataclasses
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

from seaglass.imessage.attributedbody import decode_attributed_body

APPLE_EPOCH_UNIX = 978307200  # 2001-01-01T00:00:00Z, as unix seconds
NS_VS_S_THRESHOLD = 1e11  # PLAN.md §6 Phase 1: verify against your own oldest messages

# Tables/columns this module depends on. Checked on open; a macOS update
# that changes chat.db's shape should fail loudly here, not silently
# downstream. Kept in sync with seaglass/probe.py's check_schema_shape.
EXPECTED_SCHEMA = {
    "message": {
        "ROWID", "text", "attributedBody", "date", "date_edited",
        "date_retracted", "is_from_me", "handle_id", "associated_message_type",
    },
    "chat": {"ROWID", "style"},
    "chat_message_join": {"chat_id", "message_id"},
    "chat_handle_join": {"chat_id", "handle_id"},
    "handle": {"ROWID", "id"},
    "attachment": {"ROWID", "filename"},
    "message_attachment_join": {"message_id", "attachment_id"},
}


class SchemaDriftError(RuntimeError):
    """Raised when chat.db's shape doesn't match EXPECTED_SCHEMA."""


def apple_to_unix(value: int) -> float:
    """Convert an Apple-epoch `date` value (seconds OR nanoseconds,
    depending on macOS version at write time) to unix seconds.

    ⚠️ Verify the threshold against your own oldest messages (PLAN.md §6
    Phase 0) -- this is a magnitude heuristic, not a documented API contract.
    """
    secs = value / 1e9 if value > NS_VS_S_THRESHOLD else value
    return secs + APPLE_EPOCH_UNIX


def assert_schema(con: sqlite3.Connection, attached_as: str = "im") -> None:
    """Fail loudly if chat.db's shape doesn't match what this module expects.

    Call this immediately after ATTACHing, before any extraction runs.
    """
    missing = {}
    for table, columns in EXPECTED_SCHEMA.items():
        try:
            rows = con.execute(f"pragma {attached_as}.table_info({table})").fetchall()
        except sqlite3.OperationalError as error:
            missing[table] = f"table query failed: {error}"
            continue
        present = {row[1] for row in rows}
        gap = columns - present
        if gap:
            missing[table] = f"missing columns: {sorted(gap)}"
    if missing:
        raise SchemaDriftError(
            f"chat.db schema drift detected: {missing}. A macOS update likely "
            "changed the schema; update EXPECTED_SCHEMA in seaglass/imessage/source.py."
        )


@dataclasses.dataclass(frozen=True)
class Message:
    """A single, cleaned message. The only representation the rest of the
    system should ever see -- no raw chat.db rows leak past this module.
    """

    rowid: int
    chat_id: int
    ts: float  # unix seconds
    is_from_me: bool
    handle: Optional[str]  # raw handle id (phone/email); resolve via contacts.py
    text: Optional[str]  # resolved: message.text, else decoded attributedBody
    date_edited: Optional[float]
    date_retracted: Optional[float]
    has_attachment: bool


def connect_readonly(chat_db: Path) -> sqlite3.Connection:
    """Read-only, live ATTACH against Apple's chat.db.

    ⚠️ `mode=ro`, never `immutable=1` -- immutable asserts the file cannot
    change and yields silently corrupt reads on a live database
    (PLAN.md §6 Phase 1). Requires Full Disk Access for the caller.
    """
    con = sqlite3.connect(":memory:", uri=True, check_same_thread=False)
    try:
        uri = chat_db.expanduser().absolute().as_uri() + "?mode=ro"
        con.execute("ATTACH DATABASE ? AS im", (uri,))
        assert_schema(con)
        con.execute("PRAGMA query_only = ON")
    except Exception:
        con.close()
        raise
    return con


def apple_timestamp_sql(column: str = "m.date") -> str:
    """Internal SQL equivalent of apple_to_unix, including mixed-unit history."""
    return (
        f"(CASE WHEN {column} > {NS_VS_S_THRESHOLD} "
        f"THEN {column} / 1e9 ELSE {column} END + {APPLE_EPOCH_UNIX})"
    )


def newest_message_ts(con: sqlite3.Connection) -> Optional[float]:
    value = con.execute(f"SELECT MAX({apple_timestamp_sql()}) FROM im.message m").fetchone()[0]
    return float(value) if value is not None else None


def count_messages_newer_than(con: sqlite3.Connection, ts: float) -> int:
    return con.execute(
        f"SELECT COUNT(*) FROM im.message m WHERE {apple_timestamp_sql()} > ?",
        (ts,),
    ).fetchone()[0]


def _recent_selection(
    *,
    since_ts: float,
    before_ts: float,
    chat_id: Optional[int] = None,
    chat_ids: Optional[Sequence[int]] = None,
    sender_handles: Sequence[str] = (),
    from_me: Optional[bool] = None,
    has_media: bool = False,
) -> tuple[str, list]:
    has_attachment = (
        "EXISTS (SELECT 1 FROM im.message_attachment_join maj WHERE maj.message_id = m.ROWID)"
    )
    conditions = [
        "m.associated_message_type = 0",
        f"{apple_timestamp_sql()} >= ?",
        f"{apple_timestamp_sql()} <= ?",
        f"(LENGTH(TRIM(m.text)) > 0 OR LENGTH(m.attributedBody) > 0 OR {has_attachment})",
    ]
    params: list = [since_ts, before_ts]
    if chat_id is not None:
        conditions.append("cmj.chat_id = ?")
        params.append(chat_id)
    if chat_ids is not None:
        conditions.append(f"cmj.chat_id IN ({','.join('?' for _ in chat_ids)})")
        params.extend(chat_ids)
    if sender_handles:
        conditions.append(f"h.id IN ({','.join('?' for _ in sender_handles)}) AND m.is_from_me = 0")
        params.extend(sender_handles)
    elif from_me is not None:
        conditions.append("m.is_from_me = ?")
        params.append(int(from_me))
    if has_media:
        conditions.append(has_attachment)
    selection = f"""
        FROM im.message m
        JOIN im.chat_message_join cmj ON cmj.message_id = m.ROWID
        LEFT JOIN im.handle h ON h.ROWID = m.handle_id
        WHERE {' AND '.join(conditions)}
    """
    return selection, params


def read_recent_messages(
    con: sqlite3.Connection,
    *,
    since_ts: float,
    before_ts: float,
    limit: int,
    offset: int = 0,
    include_text: bool = True,
    session_page: Optional[tuple[int, int]] = None,
    **filters,
) -> tuple[List[Message], Optional[int]]:
    """Bounded, date-ordered live rows, without any index or model dependency.

    An optional (offset, limit) day/chat session page supports the legacy
    search result contract. Both selection and paging happen before bodies
    are decoded; high-ROWID iCloud backfill cannot hide recent rows.
    """
    selection, params = _recent_selection(
        since_ts=since_ts, before_ts=before_ts, **filters
    )
    total_sessions = None
    if session_page is not None:
        day = f"date({apple_timestamp_sql()}, 'unixepoch', 'localtime')"
        grouped = f"SELECT cmj.chat_id, {day} AS day {selection} GROUP BY cmj.chat_id, day"
        total_sessions = con.execute(f"SELECT COUNT(*) FROM ({grouped})", params).fetchone()[0]
        session_offset, session_limit = session_page
        groups = con.execute(
            grouped + f" ORDER BY MAX({apple_timestamp_sql()}) DESC, cmj.chat_id DESC LIMIT ? OFFSET ?",
            params + [session_limit, session_offset],
        ).fetchall()
        if not groups:
            return [], total_sessions
        selection += " AND (" + " OR ".join(
            f"(cmj.chat_id = ? AND {day} = ?)" for _ in groups
        ) + ")"
        params += [value for group in groups for value in group]
    bodies = "m.text, m.attributedBody" if include_text else "NULL, NULL"
    rows = con.execute(
        f"""
        SELECT DISTINCT m.ROWID, cmj.chat_id, m.date, m.is_from_me, h.id, {bodies},
               EXISTS (SELECT 1 FROM im.message_attachment_join maj WHERE maj.message_id = m.ROWID)
        {selection}
        ORDER BY {apple_timestamp_sql()} DESC, m.ROWID DESC, cmj.chat_id DESC
        LIMIT ? OFFSET ?
        """,
        params + [limit, offset],
    ).fetchall()
    messages = []
    for rowid, chat, date, is_from_me, handle, text, attributed_body, has_attachment in rows:
        if include_text and not text and attributed_body:
            text = decode_attributed_body(attributed_body)
            if text is None:
                # The historical decoder is deliberately best effort. A live
                # lookup must not silently claim an unreadable body is absent.
                raise RuntimeError("Live chat.db attributedBody decoding failed; update the typedstream decoder.")
        messages.append(Message(
            rowid=rowid, chat_id=chat, ts=apple_to_unix(date),
            is_from_me=bool(is_from_me), handle=handle, text=text,
            date_edited=None, date_retracted=None, has_attachment=bool(has_attachment),
        ))
    return messages, total_sessions


_MESSAGE_QUERY = """
    SELECT
        m.ROWID,
        cmj.chat_id,
        m.date,
        m.is_from_me,
        h.id AS handle,
        m.text,
        m.attributedBody,
        m.date_edited,
        m.date_retracted,
        EXISTS (
            SELECT 1 FROM im.message_attachment_join maj WHERE maj.message_id = m.ROWID
        ) AS has_attachment
    FROM im.message m
    JOIN im.chat_message_join cmj ON cmj.message_id = m.ROWID
    LEFT JOIN im.handle h ON h.ROWID = m.handle_id
    WHERE m.associated_message_type = 0
    {chat_filter}
    ORDER BY m.date ASC
"""


def iter_messages(con: sqlite3.Connection, chat_id: Optional[int] = None) -> Iterator[Message]:
    """Yield cleaned `Message` rows, keeping read transactions short.

    Filters out tapbacks/reactions and other associated-message types
    (`associated_message_type != 0` -- PLAN.md §6 Phase 1: "pin this filter
    set; changing it later alters both body renderings and forces a full
    re-embed"). Text falls back from `message.text` to a decoded
    `attributedBody`, matching the documented extraction order.
    """
    chat_filter = "AND cmj.chat_id = :chat_id" if chat_id is not None else ""
    query = _MESSAGE_QUERY.format(chat_filter=chat_filter)
    params = {"chat_id": chat_id} if chat_id is not None else {}
    cursor = con.execute(query, params)
    for row in cursor:
        (
            rowid, row_chat_id, date, is_from_me, handle, text, attributed_body,
            date_edited, date_retracted, has_attachment,
        ) = row
        if not text and attributed_body:
            text = decode_attributed_body(attributed_body)
        yield Message(
            rowid=rowid,
            chat_id=row_chat_id,
            ts=apple_to_unix(date),
            is_from_me=bool(is_from_me),
            handle=handle,
            text=text,
            date_edited=apple_to_unix(date_edited) if date_edited else None,
            date_retracted=apple_to_unix(date_retracted) if date_retracted else None,
            has_attachment=bool(has_attachment),
        )


@dataclasses.dataclass(frozen=True)
class AttachmentRow:
    attachment_id: int
    message_id: int
    filename: Optional[str]


def fetch_attachments_for_messages(
    con: sqlite3.Connection, msg_ids: Iterable[int]
) -> Dict[int, List[AttachmentRow]]:
    """Batch-fetch attachment rows for a set of message ROWIDs, keyed by
    `message_id`. Used by index/render.py's `format_lexical` to build the
    inline media placeholder. Batches to stay well under SQLite's default
    999-variable limit for `IN (...)`.
    """
    ids = list(msg_ids)
    result: Dict[int, List[AttachmentRow]] = {}
    batch_size = 500
    for start in range(0, len(ids), batch_size):
        batch = ids[start : start + batch_size]
        placeholders = ",".join("?" for _ in batch)
        query = f"""
            SELECT maj.message_id, a.ROWID, a.filename
            FROM im.message_attachment_join maj
            JOIN im.attachment a ON a.ROWID = maj.attachment_id
            WHERE maj.message_id IN ({placeholders})
        """
        for message_id, attachment_id, filename in con.execute(query, batch):
            result.setdefault(message_id, []).append(
                AttachmentRow(attachment_id=attachment_id, message_id=message_id, filename=filename)
            )
    return result
