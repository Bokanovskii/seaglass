from __future__ import annotations

from seaglass.app.filters import SearchFilters, apply_filters
from seaglass.imessage.contacts import Contact, ContactIndex
from seaglass.search.parse import ParsedQuery


def _contacts():
    return ContactIndex([
        Contact(identifier='1', display_name='Alice Chen', handles=('+15551234567',)),
        Contact(identifier='2', display_name='Bob Smith', handles=('+15557654321',)),
    ])


def test_explicit_handles_override_inferred_people():
    parsed = ParsedQuery(raw='query', semantic='query', people_participant=['old'])
    updated = apply_filters(parsed, SearchFilters(people_handles=['+15551234567']))
    assert updated.people_participant == ['+15551234567']


def test_free_typed_names_resolve_with_contact_index():
    parsed = ParsedQuery(raw='query', semantic='query')
    updated = apply_filters(parsed, SearchFilters(people_names=['Alice Chen']), contact_index=_contacts())
    assert updated.people_participant == ['+15551234567']


def test_explicit_dates_override_inferred_dates():
    parsed = ParsedQuery(raw='query', semantic='query', date_from=1.0, date_to=2.0)
    updated = apply_filters(parsed, SearchFilters(date_from=10.0, date_to=20.0))
    assert updated.date_from == 10.0
    assert updated.date_to == 20.0


def test_group_and_chat_filters_attached():
    parsed = ParsedQuery(raw='query', semantic='query')
    updated = apply_filters(parsed, SearchFilters(is_group=True, chat_ids=[1, 2]))
    assert updated.is_group is True
    assert updated.chat_ids == [1, 2]
