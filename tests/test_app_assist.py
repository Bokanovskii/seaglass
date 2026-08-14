from __future__ import annotations

from seaglass.app.assist import (
    AssistCircuitBreaker,
    assisted_search_args,
    build_prompt,
    cache_key,
    describe_parse,
    ensure_cache,
    get_cached_parse,
    merge_ghcp_parse,
    put_cached_parse,
    should_assist,
)
from seaglass.app.filters import SearchFilters
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


def test_should_assist_auto_fires_when_a_name_went_unresolved():
    # The query that exposed this: "recent messages from kaya" resolved a
    # date but no person, and the old rule read *any* extracted filter as
    # "no help needed" -- so the one query that needed assist never asked.
    parsed = ParsedQuery(raw='recent messages from kaya', semantic='')
    parsed.date_from = 1.0
    assert should_assist('auto', parsed) is True


def test_should_assist_auto_skips_a_confident_parse():
    parsed = ParsedQuery(raw='messages from Alice yesterday', semantic='')
    parsed.people_participant = ['+15551234567']
    parsed.date_from = 1.0
    assert should_assist('auto', parsed) is False


def test_should_assist_auto_skips_short_queries():
    assert should_assist('auto', ParsedQuery(raw='lease', semantic='lease')) is False


def test_should_assist_force_always_asks():
    parsed = ParsedQuery(raw='messages from Alice yesterday', semantic='')
    parsed.people_participant = ['+15551234567']
    parsed.date_from = 1.0
    assert should_assist('force', parsed) is True
    assert should_assist('off', parsed) is False


def test_merge_reports_people_it_could_not_resolve():
    parsed = ParsedQuery(raw='messages from nobody', semantic='messages')
    merged = merge_ghcp_parse(parsed, {'people': ['Nobody'], 'semantic': 'messages'}, _contacts(), (0, 2000000000))
    assert merged.unresolved_people == ['Nobody']
    assert merged.parse.people_participant == []


def test_merge_resolves_a_lower_case_name():
    # Copilot copies the name span verbatim from the query, so it hands back
    # "alice" for "messages from alice" -- which used to resolve to nothing,
    # leaving a banner that promised a filter that could never be applied.
    parsed = ParsedQuery(raw='messages from alice', semantic='messages')
    merged = merge_ghcp_parse(parsed, {'people': ['alice chen'], 'semantic': 'messages'}, _contacts(), (0, 2000000000))
    assert merged.parse.people_participant == ['+15551234567']


def test_assisted_search_args_move_the_parse_into_filters():
    parsed = ParsedQuery(raw='recent messages from alice', semantic='')
    parsed.people_participant = ['+15551234567']
    parsed.people_sender = ['+15551234567']
    parsed.date_from, parsed.date_to = 100.0, 200.0
    text, filters = assisted_search_args(parsed, SearchFilters())
    # The engine re-parses whatever text it gets, so anything not expressed
    # as a filter is dropped.
    assert text == ''
    assert filters.people_handles == ['+15551234567']
    assert filters.people_sender == ['+15551234567']
    assert (filters.date_from, filters.date_to) == (100.0, 200.0)


def test_assisted_search_args_defer_to_filters_the_user_set():
    parsed = ParsedQuery(raw='lease with alice', semantic='lease')
    parsed.people_participant = ['+15551234567']
    parsed.date_from, parsed.date_to = 100.0, 200.0
    explicit = SearchFilters(people_handles=['+15557654321'], date_from=1.0, date_to=2.0)
    _text, filters = assisted_search_args(parsed, explicit)
    assert filters.people_handles == ['+15557654321']
    assert (filters.date_from, filters.date_to) == (1.0, 2.0)


def test_describe_parse_names_resolved_contacts_and_flags_the_rest():
    parsed = ParsedQuery(raw='photos from alice', semantic='photos')
    parsed.people_participant = ['+15551234567']
    parsed.has_media = True
    described = describe_parse(parsed, _contacts(), ['zzz'])
    assert 'Alice Chen' in described
    assert 'with media' in described
    assert 'no contact matched zzz' in described
    assert '+15551234567' not in described
