"""`search/hydrate.py` — PLAN.md §6 Phase 4, step 7: hydrate the final
~8 surviving sessions (and only those) with raw messages for display and
citation. Everything upstream of this module works with chunk ids and
compressed `body_semantic` blobs; this is the one place that goes back to
`chat.db` for the original per-message rows.

⚠️ Always join through `chunk_message`, never a bare `message.ROWID`
range -- `message.ROWID` is global and chronological *across all chats*,
so a naive range spans other conversations (PLAN.md §6 step 7).
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence

from seaglass.imessage.attributedbody import decode_attributed_body
from seaglass.imessage.contacts import ContactIndex
from seaglass.imessage.source import apple_to_unix
from seaglass.search.rank import Session

# Stay well under SQLite's default 999-variable limit for `IN (...)`.
_SQLITE_VARIABLE_BATCH = 800



@dataclasses.dataclass(frozen=True)
class HydratedMessage:
    message_id: int
    ts: float
    is_from_me: bool
    sender: Optional[str]  # resolved contact display name, else raw handle, else None (is_from_me)
    text: Optional[str]
    has_attachment: bool
    # Human-readable kind for attachment-only messages ("Photo", "Video",
    # ...). Attachment-only messages have no text at all, so without this
    # the client has nothing to render but an empty bubble.
    attachment_kind: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class HydratedSession:
    chat_id: int
    day: str
    score: float
    hit_messages: List[HydratedMessage]  # from chunks that actually scored
    context_messages: List[HydratedMessage]  # ±2 expansion, zero ranking weight


def _resolve_sender(is_from_me: int, handle: Optional[str], contact_index: Optional[ContactIndex]) -> Optional[str]:
    if is_from_me:
        return None  # the client already knows who "me" is; no name needed
    if handle is None:
        return None
    if contact_index is not None:
        name = contact_index.resolve_handle(handle)
        if name is not None:
            return name
    return handle


def _attachment_label(mime_type: Optional[str], uti: Optional[str], is_sticker: int) -> str:
    """Best-effort human name for one attachment. chat.db is inconsistent
    here -- plenty of rows carry a NULL mime_type and only a UTI (and some
    have neither) -- so fall back through both before giving up.
    """
    if is_sticker:
        return 'Sticker'
    mime = (mime_type or '').lower()
    uti_value = (uti or '').lower()
    if mime == 'image/gif' or 'gif' in uti_value:
        return 'GIF'
    if mime.startswith('image/') or uti_value.startswith('public.hei') or uti_value in {'public.jpeg', 'public.png'}:
        return 'Photo'
    if mime.startswith('video/') or 'movie' in uti_value or 'mpeg-4' in uti_value:
        return 'Video'
    if mime.startswith('audio/') or 'audio' in uti_value:
        return 'Audio message'
    if mime == 'application/pdf' or uti_value == 'com.adobe.pdf':
        return 'PDF'
    if mime == 'text/vcard' or 'vcard' in uti_value:
        return 'Contact card'
    return 'Attachment'


def _fetch_attachment_kinds(chat_con, msg_ids: Sequence[int]) -> Dict[int, str]:
    """Map message ROWID -> a short label describing its attachments, e.g.
    "Photo" or "3 photos". Attachment-only messages carry no text, so this
    is the only thing the UI can show for them besides an empty bubble.

    The `attachment` table's columns vary between macOS releases, so probe
    for the ones we want and substitute literals for any that are missing
    rather than failing the whole hydration with "no such column".
    """
    if not msg_ids:
        return {}
    available = {row[1] for row in chat_con.execute("PRAGMA im.table_info(attachment)")}
    if not available:
        return {}
    mime_expr = "a.mime_type" if "mime_type" in available else "NULL"
    uti_expr = "a.uti" if "uti" in available else "NULL"
    sticker_expr = "COALESCE(a.is_sticker, 0)" if "is_sticker" in available else "0"

    kinds: Dict[int, str] = {}
    for start in range(0, len(msg_ids), _SQLITE_VARIABLE_BATCH):
        batch = msg_ids[start:start + _SQLITE_VARIABLE_BATCH]
        placeholders = ",".join("?" for _ in batch)
        by_message: Dict[int, List[str]] = {}
        for message_id, mime_type, uti, is_sticker in chat_con.execute(
            f"""
            SELECT maj.message_id, {mime_expr}, {uti_expr}, {sticker_expr}
            FROM im.message_attachment_join maj
            JOIN im.attachment a ON a.ROWID = maj.attachment_id
            WHERE maj.message_id IN ({placeholders})
            """,
            list(batch),
        ):
            by_message.setdefault(message_id, []).append(
                _attachment_label(mime_type, uti, is_sticker)
            )
        for message_id, labels in by_message.items():
            if len(labels) == 1:
                kinds[message_id] = labels[0]
            elif len(set(labels)) == 1:
                kinds[message_id] = f'{len(labels)} {labels[0].lower()}s'
            else:
                kinds[message_id] = f'{len(labels)} attachments'
    return kinds


def hydrate_sessions(
    index_con,
    chat_con,
    sessions: Sequence[Session],
    contact_index: Optional[ContactIndex] = None,
) -> List[HydratedSession]:
    """Step 7: for each session's hit and context chunks, pull raw
    messages via `chunk_message`, resolve sender display names, and
    return them split into `hit_messages` (chunks that scored) vs
    `context_messages` (expansion only) -- both are needed for display,
    only the first group should ever be described as "why this matched".

    `index_con` provides the `chunk_message` join table; `chat_con` (a
    live/snapshot `chat.db` connection via `imessage.source.connect_readonly`)
    provides the actual message text/attachments. Both are required --
    hydration is the one step that needs both databases at once.
    """
    hydrated: List[HydratedSession] = []
    for session in sessions:
        hit_messages = _hydrate_chunks(index_con, chat_con, session.hit_chunk_ids, contact_index)
        context_messages = _hydrate_chunks(index_con, chat_con, session.context_chunk_ids, contact_index)
        hydrated.append(
            HydratedSession(
                chat_id=session.chat_id,
                day=session.day,
                score=session.score,
                hit_messages=hit_messages,
                context_messages=context_messages,
            )
        )
    return hydrated


def _hydrate_chunks(
    index_con, chat_con, chunk_ids: Sequence[int], contact_index: Optional[ContactIndex]
) -> List[HydratedMessage]:
    if not chunk_ids:
        return []
    placeholders = ",".join("?" for _ in chunk_ids)
    msg_ids = [
        row[0]
        for row in index_con.execute(
            f"SELECT msg_id FROM chunk_message WHERE chunk_id IN ({placeholders}) ORDER BY msg_id",
            list(chunk_ids),
        ).fetchall()
    ]
    if not msg_ids:
        return []
    msg_placeholders = ",".join("?" for _ in msg_ids)
    attachment_kinds = _fetch_attachment_kinds(chat_con, msg_ids)
    query = f"""
        SELECT m.ROWID, m.text, m.attributedBody, m.date, m.is_from_me, h.id,
               EXISTS (SELECT 1 FROM im.message_attachment_join maj WHERE maj.message_id = m.ROWID)
        FROM im.message m
        LEFT JOIN im.handle h ON h.ROWID = m.handle_id
        WHERE m.ROWID IN ({msg_placeholders})
        ORDER BY m.date ASC
    """
    messages: List[HydratedMessage] = []
    for rowid, text, attributed_body, date, is_from_me, handle, has_attachment in chat_con.execute(
        query, msg_ids
    ):
        if not text and attributed_body:
            text = decode_attributed_body(attributed_body)
        kind = attachment_kinds.get(rowid) if has_attachment else None
        if not (text or "").strip() and not kind:
            # System rows (chat.db `item_type != 0`: group renames, shared
            # location start/stop, participant changes) carry neither text
            # nor an attachment and would render as an empty bubble. They
            # are filtered here rather than in the snapshot query because
            # that filter set is pinned -- changing it forces a full
            # re-embed (see imessage/source.py).
            continue
        sender = _resolve_sender(is_from_me, handle, contact_index)
        messages.append(
            HydratedMessage(
                message_id=rowid,
                ts=apple_to_unix(date),
                is_from_me=bool(is_from_me),
                sender=sender,
                text=text,
                has_attachment=bool(has_attachment),
                attachment_kind=kind,
            )
        )
    return messages
