from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional

STYLE_ONE_TO_ONE = 45


@dataclasses.dataclass(frozen=True)
class ChatMetadata:
    chat_id: int
    is_group: bool
    title: str
    participants: tuple[str, ...]
    participant_count: int
    chat_identifier: Optional[str] = None


class ChatMetadataCache:
    def __init__(self, by_chat_id: Dict[int, ChatMetadata]):
        self._by_chat_id = by_chat_id

    @classmethod
    def build(cls, chat_con, present_chat_ids: Optional[set[int]] = None) -> "ChatMetadataCache":
        chats = chat_con.execute(
            """
            SELECT c.ROWID, c.style, h.id
            FROM im.chat c
            LEFT JOIN im.handle h ON h.ROWID = c.ROWID
            ORDER BY c.ROWID
            """
        ).fetchall()
        participants_by_chat: dict[int, List[str]] = {}
        for chat_id, handle in chat_con.execute(
            """
            SELECT chj.chat_id, h.id
            FROM im.chat_handle_join chj
            JOIN im.handle h ON h.ROWID = chj.handle_id
            ORDER BY chj.chat_id, h.id
            """
        ):
            participants_by_chat.setdefault(chat_id, []).append(handle)

        by_chat_id: Dict[int, ChatMetadata] = {}
        for chat_id, style, chat_identifier in chats:
            if present_chat_ids is not None and chat_id not in present_chat_ids:
                continue
            participants = tuple(participants_by_chat.get(chat_id, []))
            participant_count = len(participants)
            is_group = classify_chat(style, participant_count)
            title = format_chat_title(participants, chat_identifier)
            by_chat_id[chat_id] = ChatMetadata(
                chat_id=chat_id,
                is_group=is_group,
                title=title,
                participants=participants,
                participant_count=participant_count,
                chat_identifier=chat_identifier,
            )
        return cls(by_chat_id)

    def get(self, chat_id: int) -> Optional[ChatMetadata]:
        return self._by_chat_id.get(chat_id)

    def all(self) -> List[ChatMetadata]:
        return list(self._by_chat_id.values())


def classify_chat(style: Optional[int], participant_count: int) -> bool:
    if style is not None:
        return int(style or 0) not in (STYLE_ONE_TO_ONE,)
    return participant_count > 1


def format_chat_title(participants: tuple[str, ...], chat_identifier: Optional[str]) -> str:
    if participants:
        return ", ".join(participants)
    if chat_identifier:
        return chat_identifier
    return "Unknown conversation"
