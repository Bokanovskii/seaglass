from __future__ import annotations

from typing import Optional

from seaglass.imessage.attributedbody import decode_attributed_body
from seaglass.imessage.contacts import ContactIndex
from seaglass.imessage.source import apple_to_unix
from seaglass.search.hydrate import HydratedMessage, _fetch_attachment_kinds, _resolve_sender


def fetch_conversation(
    chat_con,
    chat_id: int,
    around_ts: Optional[float] = None,
    limit: int = 50,
    contact_index: Optional[ContactIndex] = None,
) -> dict:
    if around_ts is not None:
        id_date_rows = chat_con.execute(
            """
            SELECT m.ROWID, m.date
            FROM im.message m
            JOIN im.chat_message_join cmj ON cmj.message_id = m.ROWID
            WHERE cmj.chat_id = ?
            ORDER BY m.date ASC
            """,
            (chat_id,),
        ).fetchall()
        if not id_date_rows:
            target_ids = []
        else:
            converted = [(rowid, apple_to_unix(date)) for rowid, date in id_date_rows]
            closest_idx = min(range(len(converted)), key=lambda i: abs(converted[i][1] - around_ts))
            half = limit // 2
            lo = max(0, closest_idx - half)
            hi = min(len(converted), lo + limit)
            lo = max(0, hi - limit)
            target_ids = [rowid for rowid, _ in converted[lo:hi]]
        if not target_ids:
            return {"chat_id": chat_id, "n_messages": 0, "messages": []}
        placeholders = ",".join("?" for _ in target_ids)
        query = f"""
            SELECT m.ROWID, m.text, m.attributedBody, m.date, m.is_from_me, h.id
            FROM im.message m
            LEFT JOIN im.handle h ON h.ROWID = m.handle_id
            WHERE m.ROWID IN ({placeholders})
        """
        rows = chat_con.execute(query, target_ids).fetchall()
    else:
        rows = chat_con.execute(
            """
            SELECT m.ROWID, m.text, m.attributedBody, m.date, m.is_from_me, h.id
            FROM im.message m
            JOIN im.chat_message_join cmj ON cmj.message_id = m.ROWID
            LEFT JOIN im.handle h ON h.ROWID = m.handle_id
            WHERE cmj.chat_id = ?
            ORDER BY m.date DESC
            LIMIT ?
            """,
            (chat_id, limit),
        ).fetchall()

    messages = []
    attachment_kinds = _fetch_attachment_kinds(chat_con, [row[0] for row in rows])
    for rowid, text, attributed_body, date, is_from_me, handle in rows:
        if not text and attributed_body:
            text = decode_attributed_body(attributed_body)
        sender = _resolve_sender(is_from_me, handle, contact_index)
        kind = attachment_kinds.get(rowid)
        messages.append(
            HydratedMessage(
                message_id=rowid,
                ts=apple_to_unix(date),
                is_from_me=bool(is_from_me),
                sender=sender,
                text=text,
                has_attachment=kind is not None,
                attachment_kind=kind,
            )
        )

    messages.sort(key=lambda m: m.ts)
    return {
        "chat_id": chat_id,
        "n_messages": len(messages),
        "messages": [
            {
                "message_id": m.message_id,
                "ts": m.ts,
                "is_from_me": m.is_from_me,
                "sender": m.sender,
                "text": m.text,
                "has_attachment": m.has_attachment,
                "attachment_kind": m.attachment_kind,
            }
            for m in messages
        ],
    }
