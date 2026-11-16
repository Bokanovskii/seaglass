"""Live chat membership filters, independent of semantic retrieval."""

from __future__ import annotations

from typing import Optional, Sequence, Set


def resolve_participant_chat_ids(chat_con, handle_ids: Sequence[str]) -> Set[int]:
    """Include both current roster membership and past message authors."""
    if not handle_ids:
        return set()
    placeholders = ",".join("?" for _ in handle_ids)
    roster = f"""
        SELECT DISTINCT chj.chat_id
        FROM im.chat_handle_join chj
        JOIN im.handle h ON h.ROWID = chj.handle_id
        WHERE h.id IN ({placeholders})
    """
    authored = f"""
        SELECT DISTINCT cmj.chat_id
        FROM im.message m
        JOIN im.handle h ON h.ROWID = m.handle_id
        JOIN im.chat_message_join cmj ON cmj.message_id = m.ROWID
        WHERE h.id IN ({placeholders}) AND m.is_from_me = 0
    """
    params = list(handle_ids)
    return (
        {row[0] for row in chat_con.execute(roster, params)}
        | {row[0] for row in chat_con.execute(authored, params)}
    )


def resolve_group_chat_ids(chat_con, is_group: bool) -> Set[int]:
    rows = chat_con.execute(
        """
        SELECT c.ROWID, c.style, COUNT(DISTINCT chj.handle_id)
        FROM im.chat c
        LEFT JOIN im.chat_handle_join chj ON chj.chat_id = c.ROWID
        GROUP BY c.ROWID, c.style
        """
    ).fetchall()
    matched = set()
    for chat_id, style, participant_count in rows:
        chat_is_group = int((style or 0) not in (45,)) if style is not None else participant_count > 1
        if bool(chat_is_group) is bool(is_group):
            matched.add(chat_id)
    return matched


def resolve_chat_id_filter(chat_con, parsed_query) -> Optional[Set[int]]:
    """None means unfiltered; an empty set must fail closed."""
    if chat_con is None:
        return None
    filters: list[Set[int]] = []
    if parsed_query.people_participant:
        filters.append(resolve_participant_chat_ids(chat_con, parsed_query.people_participant))
    if getattr(parsed_query, "is_group", None) is not None:
        filters.append(resolve_group_chat_ids(chat_con, parsed_query.is_group))
    if getattr(parsed_query, "chat_ids", None):
        filters.append(set(parsed_query.chat_ids))
    if not filters:
        return None
    chat_id_filter = filters[0]
    for subset in filters[1:]:
        chat_id_filter &= subset
    return chat_id_filter
