from __future__ import annotations

from seaglass.app.assist import AssistCircuitBreaker, build_prompt, cache_key, ensure_cache, get_cached_parse, merge_ghcp_parse, put_cached_parse, should_assist
from seaglass.imessage.contacts import Contact, ContactIndex
from seaglass.search.parse import ParsedQuery


def _contacts():
    return ContactIndex([
        Contact(identifier='1', display_name='Alice Chen', handles=('+15551234567',)),
        Contact(identifier='2', display_name='Bob Smith', handles=('+15557654321',)),
    ])


def test_build_prompt_mentions_structured_contract():
    prompt = build_prompt('photos from Alice', today='2026-08-12', weekday='Wednesday', tz_name='America/Los_Angeles')
    assert 'return ONLY a JSON object' in prompt
    assert 'photos from Alice' in prompt


def test_should_assist_auto_only_for_weak_parse():
    parsed = ParsedQuery(raw='find that lease discussion', semantic='find that lease discussion')
    assert should_assist('auto', parsed) is True
    parsed.people_participant = ['+15551234567']
    assert should_assist('auto', parsed) is False


def test_merge_ghcp_parse_drops_unknown_people_and_bad_dates():
    parsed = ParsedQuery(raw='lease with Alice', semantic='lease')
    merged, changes, expansions = merge_ghcp_parse(parsed, {'people': ['Unknown Person'], 'date_from': '2099-01-01', 'date_to': '2099-01-02', 'semantic': 'lease', 'expansions': ['paperwork']}, _contacts(), (0, 2000000000))
    assert merged.people_participant == []
    assert merged.date_from is None
    assert expansions == ['paperwork']


def test_merge_ghcp_parse_accepts_valid_changes():
    parsed = ParsedQuery(raw='photos with Alice', semantic='photos')
    merged, changes, expansions = merge_ghcp_parse(parsed, {'people': ['Alice Chen'], 'date_from': '2024-01-01', 'date_to': '2024-01-03', 'has_media': True, 'is_group': False, 'semantic': 'photos lease', 'expansions': ['image', 'photo']}, _contacts(), (0, 2000000000))
    assert '+15551234567' in merged.people_participant
    assert merged.has_media is True
    assert merged.is_group is False
    assert 'people' in changes
    assert 'image' in expansions


def test_cache_round_trip(tmp_path):
    con = ensure_cache(tmp_path / 'app.db')
    key = cache_key('photos', prompt_version='v1', today='2026-08-12', aliases_mtime=0)
    put_cached_parse(con, key, 'photos', {'semantic': 'photos'})
    assert get_cached_parse(con, key) == {'semantic': 'photos'}


def test_circuit_breaker_opens_after_three_failures():
    breaker = AssistCircuitBreaker()
    breaker.record_failure(); breaker.record_failure(); breaker.record_failure()
    assert breaker.open is True
