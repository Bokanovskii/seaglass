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
from typing import Dict, Iterable, Iterator, List, Optional

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
    con = sqlite3.connect(":memory:")
    con.execute(f"ATTACH DATABASE 'file:{chat_db}?mode=ro' AS im")
    assert_schema(con)
    return con


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
