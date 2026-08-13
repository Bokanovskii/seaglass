from __future__ import annotations

import dataclasses
from dataclasses import replace

from seaglass.search.parse import ParsedQuery


@dataclasses.dataclass
class SearchFilters:
    people_handles: list[str] = dataclasses.field(default_factory=list)
    people_names: list[str] = dataclasses.field(default_factory=list)
    is_group: bool | None = None
    chat_ids: list[int] | None = None
    date_from: float | None = None
    date_to: float | None = None
    has_media: bool | None = None


def apply_filters(parsed: ParsedQuery, filters: SearchFilters, contact_index=None) -> ParsedQuery:
    people = list(parsed.people_participant)
    if filters.people_handles:
        people = list(filters.people_handles)
    elif filters.people_names and contact_index is not None:
        resolved: list[str] = []
        for name in filters.people_names:
            resolved.extend(contact_index.handle_ids_for_names(name))
        if resolved:
            people = resolved

    updated = replace(
        parsed,
        people_participant=people,
        date_from=filters.date_from if filters.date_from is not None else parsed.date_from,
        date_to=filters.date_to if filters.date_to is not None else parsed.date_to,
        has_media=filters.has_media if filters.has_media is not None else parsed.has_media,
    )
    setattr(updated, 'is_group', filters.is_group)
    setattr(updated, 'chat_ids', filters.chat_ids)
    return updated
