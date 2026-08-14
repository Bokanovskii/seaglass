"""A filters-only query must not be wrong just because a sync has not run.

"latest from Vamski" needs no models and no ranking -- it is a lookup, and
chat.db can answer it completely at any moment. Routing it through the
index anyway meant the query most sensitive to staleness was the one query
guaranteed to miss the answer. These tests pin the tail path that fixes
that: what it reads, how it merges with indexed sessions, and that it
still respects every filter the query asked for.
"""

from __future__ import annotations

import dataclasses

from seaglass.app.engine import SearchOptions, _merge_sessions
from seaglass.app.filters import SearchFilters
from seaglass.imessage.source import connect_readonly
from seaglass.search.hydrate import (
    HydratedMessage,
    HydratedSession,
    hydrate_recent_messages,
    sessions_from_messages,
)

from conftest import build_fixture_chat_db
from test_app_engine import APPLE_EPOCH_START, _snapshot_and_live


def _message(message_id, ts, text="hi", sender="+1555", is_from_me=False):
    return HydratedMessage(
        message_id=message_id,
        ts=ts,
        is_from_me=is_from_me,
        sender=sender,
        text=text,
        has_attachment=False,
        attachment_kind=None,
        handle=sender,
    )


def _session(chat_id, day, messages, context=()):
    return HydratedSession(
        chat_id=chat_id,
        day=day,
        score=0.0,
        hit_messages=list(messages),
        context_messages=list(context),
    )


class TestHydrateRecentMessages:
    def test_reads_only_messages_newer_than_the_horizon(self, tmp_path):
        chat_db = build_fixture_chat_db(
            tmp_path,
            [{
                'chat_id': 1,
                'handles': ['+15551234567'],
                'messages': [
                    ('already indexed', APPLE_EPOCH_START, False, 0),
                    ('brand new', APPLE_EPOCH_START + 5000, False, 0),
                ],
            }],
        )
        con = connect_readonly(chat_db)
        from seaglass.imessage.source import apple_to_unix

        horizon = apple_to_unix(APPLE_EPOCH_START + 10)
        pairs = hydrate_recent_messages(con, since_ts=horizon)

        assert [m.text for _, m in pairs] == ['brand new']
        assert all(chat_id == 1 for chat_id, _ in pairs)

    def test_returns_newest_first(self, tmp_path):
        chat_db = build_fixture_chat_db(
            tmp_path,
            [{
                'chat_id': 1,
                'handles': ['+15551234567'],
                'messages': [
                    ('older', APPLE_EPOCH_START + 100, False, 0),
                    ('newer', APPLE_EPOCH_START + 200, False, 0),
                ],
            }],
        )
        con = connect_readonly(chat_db)
        pairs = hydrate_recent_messages(con, since_ts=0)
        assert [m.text for _, m in pairs] == ['newer', 'older']

    def test_a_zero_limit_reads_nothing(self, tmp_path):
        chat_db = build_fixture_chat_db(
            tmp_path,
            [{'chat_id': 1, 'handles': ['+1'], 'messages': [('x', APPLE_EPOCH_START, False, 0)]}],
        )
        con = connect_readonly(chat_db)
        assert hydrate_recent_messages(con, since_ts=0, limit=0) == []

    def test_backfilled_messages_do_not_hide_the_new_ones(self, tmp_path):
        """A message synced from another device is appended to chat.db now
        but dated whenever it happened, so a high ROWID can hold an old
        message. Scanning ROWID-descending and stopping at the first old
        row would drop every genuinely new message behind it -- which is
        precisely what this function exists to find. Measured against a
        real chat.db, 3000 consecutive rows held 1234 such inversions.
        """
        from seaglass.imessage.source import apple_to_unix

        chat_db = build_fixture_chat_db(
            tmp_path,
            [{
                'chat_id': 1,
                'handles': ['+15551234567'],
                'messages': [
                    ('already indexed', APPLE_EPOCH_START, False, 0),
                    ('genuinely new', APPLE_EPOCH_START + 5000, False, 0),
                    # Inserted last (highest ROWID), dated oldest: backfill.
                    ('backfilled from icloud', APPLE_EPOCH_START - 90000, False, 0),
                ],
            }],
        )
        con = connect_readonly(chat_db)
        horizon = apple_to_unix(APPLE_EPOCH_START + 10)
        texts = [m.text for _, m in hydrate_recent_messages(con, since_ts=horizon)]

        assert 'genuinely new' in texts
        assert 'backfilled from icloud' not in texts

    def test_results_are_ordered_by_date_not_by_rowid(self, tmp_path):
        chat_db = build_fixture_chat_db(
            tmp_path,
            [{
                'chat_id': 1,
                'handles': ['+15551234567'],
                'messages': [
                    ('newest but lowest rowid', APPLE_EPOCH_START + 900, False, 0),
                    ('oldest but highest rowid', APPLE_EPOCH_START + 100, False, 0),
                ],
            }],
        )
        con = connect_readonly(chat_db)
        texts = [m.text for _, m in hydrate_recent_messages(con, since_ts=0)]
        assert texts == ['newest but lowest rowid', 'oldest but highest rowid']


class TestSessionsFromMessages:
    def test_groups_by_chat_and_day_newest_session_first(self):
        day = 86400
        pairs = [
            (1, _message(1, day * 10)),
            (1, _message(2, day * 10 + 60)),
            (2, _message(3, day * 20)),
        ]
        sessions = sessions_from_messages(pairs)
        assert len(sessions) == 2
        # chat 2's day is later, so it leads.
        assert sessions[0].chat_id == 2
        assert [m.message_id for m in sessions[1].hit_messages] == [2, 1]

    def test_produces_sessions_with_no_context(self):
        sessions = sessions_from_messages([(1, _message(1, 86400 * 10))])
        assert sessions[0].context_messages == []


class TestMergeSessions:
    def test_a_new_day_is_appended(self):
        indexed = [_session(1, '2024-01-01', [_message(1, 100)])]
        tail = [_session(1, '2024-01-02', [_message(2, 200)])]
        merged = _merge_sessions(tail, indexed)
        assert [(s.chat_id, s.day) for s in merged] == [(1, '2024-01-01'), (1, '2024-01-02')]

    def test_the_same_day_becomes_one_session_not_two(self):
        indexed = [_session(1, '2024-01-01', [_message(1, 100)])]
        tail = [_session(1, '2024-01-01', [_message(2, 200)])]
        merged = _merge_sessions(tail, indexed)
        assert len(merged) == 1
        assert [m.message_id for m in merged[0].hit_messages] == [2, 1]

    def test_a_message_on_both_sides_appears_once(self):
        """A sync landing mid-query puts the same message in the index and
        in the tail. Showing it twice would look like a duplicate send."""
        indexed = [_session(1, '2024-01-01', [_message(1, 100)])]
        tail = [_session(1, '2024-01-01', [_message(1, 100), _message(2, 200)])]
        merged = _merge_sessions(tail, indexed)
        assert [m.message_id for m in merged[0].hit_messages] == [2, 1]

    def test_context_messages_on_the_indexed_session_survive(self):
        indexed = [_session(1, '2024-01-01', [_message(1, 100)], context=[_message(9, 50)])]
        tail = [_session(1, '2024-01-01', [_message(2, 200)])]
        merged = _merge_sessions(tail, indexed)
        assert [m.message_id for m in merged[0].context_messages] == [9]

    def test_an_empty_tail_changes_nothing(self):
        indexed = [_session(1, '2024-01-01', [_message(1, 100)])]
        assert _merge_sessions([], indexed) == indexed


class TestEngineServesTheTail:
    def test_a_filters_only_query_returns_an_unindexed_message(self, tmp_path, monkeypatch):
        engine = _snapshot_and_live(
            tmp_path, monkeypatch,
            [('brand new unindexed message', APPLE_EPOCH_START + 5000)],
        )
        assert engine.status()['n_messages_since_index'] == 1

        payload = engine.search('', SearchFilters(), SearchOptions())
        texts = [
            m['text']
            for session in payload['sessions']
            for m in session['messages'] + session.get('context_messages', [])
        ]
        assert 'brand new unindexed message' in texts
        assert payload['unindexed_included'] >= 1
        assert payload['served_from'] == 'chat_db+index'

    def test_the_answer_is_still_flagged_stale_but_declares_its_coverage(self, tmp_path, monkeypatch):
        engine = _snapshot_and_live(
            tmp_path, monkeypatch, [('brand new unindexed message', APPLE_EPOCH_START + 5000)],
        )
        payload = engine.search('', SearchFilters(), SearchOptions())
        # The index really is behind -- we do not lie about that. What is
        # new is that the caller can see the gap did not cost it anything.
        assert payload['index_stale'] is True
        assert payload['unindexed_included'] >= 1

    def test_a_current_index_does_no_tail_work(self, tmp_path, monkeypatch):
        engine = _snapshot_and_live(tmp_path, monkeypatch, [])
        payload = engine.search('', SearchFilters(), SearchOptions())
        assert payload['unindexed_included'] == 0
        assert payload['served_from'] == 'index'

    def test_the_tail_is_narrowed_by_sender_like_any_other_message(self, tmp_path, monkeypatch):
        """A too-new message must not dodge the filters the query asked
        for. It is merged before the sender filter, not appended after."""
        engine = _snapshot_and_live(
            tmp_path, monkeypatch, [('brand new unindexed message', APPLE_EPOCH_START + 5000)],
        )
        payload = engine.search('', SearchFilters(from_me=True), SearchOptions())
        texts = [
            m['text']
            for session in payload['sessions']
            for m in session['messages']
        ]
        # The tail message is incoming, so a "from me" query must not show it.
        assert 'brand new unindexed message' not in texts


def _two_chat_snapshot_and_live(tmp_path, monkeypatch, extra_in_chat_2):
    """A stale index over two chats, with the unindexed message landing in
    chat 2 -- so a query scoped to chat 1 has something to wrongly leak."""
    from seaglass.app.engine import SearchEngine
    from seaglass.imessage.contacts import ContactIndex
    from seaglass.index.build import build_index

    from test_app_engine import FakeReranker
    from conftest import FakeEmbeddingModel

    base = [
        {'chat_id': 1, 'handles': ['+15551111111'], 'messages': [('chat one baseline', APPLE_EPOCH_START, False, 0)]},
        {'chat_id': 2, 'handles': ['+15552222222'], 'messages': [('chat two baseline', APPLE_EPOCH_START + 10, False, 0)]},
    ]
    snap_dir = tmp_path / 'snap'
    snap_dir.mkdir()
    snapshot_db = build_fixture_chat_db(snap_dir, base)
    index_db = tmp_path / 'index.db'
    build_index(snapshot_db, index_db, embedding_model=FakeEmbeddingModel(), batch_size=10)

    live_dir = tmp_path / 'live'
    live_dir.mkdir()
    live = [dict(chat, messages=list(chat['messages'])) for chat in base]
    live[1]['messages'] += [(text, date, False, 0) for text, date in extra_in_chat_2]
    live_db = build_fixture_chat_db(live_dir, live)

    monkeypatch.setattr('seaglass.app.engine.EmbeddingModel', FakeEmbeddingModel)
    monkeypatch.setattr('seaglass.app.engine.CrossEncoderReranker', FakeReranker)
    monkeypatch.setattr('seaglass.app.engine.ContactIndex.load', lambda: ContactIndex([]))
    engine = SearchEngine(str(index_db), str(snapshot_db), chat_db_source=str(live_db))
    engine.warmup(progress=lambda name: __import__('contextlib').nullcontext())
    return engine


class TestTheTailRespectsChatScoping:
    """`people_participant` / `is_group` / `chat_ids` narrow the *candidate
    chunks* upstream, so the tail never passes through them. It has to
    apply them itself or a query scoped to one person answers with
    somebody else's messages."""

    def test_a_participant_scoped_query_does_not_leak_another_chats_tail(self, tmp_path, monkeypatch):
        engine = _two_chat_snapshot_and_live(
            tmp_path, monkeypatch, [('unindexed message in chat two', APPLE_EPOCH_START + 5000)],
        )
        payload = engine.search(
            '', SearchFilters(people_handles=['+15551111111']), SearchOptions(),
        )
        texts = [
            m['text']
            for session in payload['sessions']
            for m in session['messages'] + session.get('context_messages', [])
        ]
        assert 'unindexed message in chat two' not in texts
        assert all(session['chat_id'] == 1 for session in payload['sessions'])

    def test_the_tail_still_arrives_when_the_query_scopes_to_its_own_chat(self, tmp_path, monkeypatch):
        engine = _two_chat_snapshot_and_live(
            tmp_path, monkeypatch, [('unindexed message in chat two', APPLE_EPOCH_START + 5000)],
        )
        payload = engine.search(
            '', SearchFilters(people_handles=['+15552222222']), SearchOptions(),
        )
        texts = [
            m['text']
            for session in payload['sessions']
            for m in session['messages'] + session.get('context_messages', [])
        ]
        assert 'unindexed message in chat two' in texts
