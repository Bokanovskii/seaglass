from __future__ import annotations

from seaglass.app.engine import SearchEngine, SearchOptions
from seaglass.app.filters import SearchFilters
from seaglass.imessage.contacts import Contact, ContactIndex
from seaglass.index.build import build_index
from seaglass.index.embed import EmbeddingModel
from seaglass.search.rerank import CrossEncoderReranker

from conftest import FakeEmbeddingModel, build_fixture_chat_db

APPLE_EPOCH_START = 700000000


class FakeReranker:
    def score(self, pairs):
        scores = []
        for query, text in pairs:
            scores.append(float(len(set(query.lower().split()) & set(text.lower().split()))))
        return scores


def _fixture(tmp_path):
    chat_db = build_fixture_chat_db(
        tmp_path,
        [
            {'chat_id': 1, 'handles': ['+15551234567'], 'messages': [('dinner plans with alice', APPLE_EPOCH_START, False, 0), ('photo attached', APPLE_EPOCH_START + 10, True, 0)]},
            {'chat_id': 2, 'handles': ['+15550000001', '+15550000002'], 'messages': [('group tax discussion', APPLE_EPOCH_START + 1000, False, 0), ('lease paperwork', APPLE_EPOCH_START + 1010, False, 1)]},
        ],
    )
    index_db = tmp_path / 'index.db'
    build_index(chat_db, index_db, embedding_model=FakeEmbeddingModel(), batch_size=10)
    return chat_db, index_db


def _engine(tmp_path, monkeypatch):
    chat_db, index_db = _fixture(tmp_path)
    monkeypatch.setattr('seaglass.app.engine.EmbeddingModel', FakeEmbeddingModel)
    monkeypatch.setattr('seaglass.app.engine.CrossEncoderReranker', FakeReranker)
    monkeypatch.setattr('seaglass.app.engine.ContactIndex.load', lambda: ContactIndex([Contact(identifier='1', display_name='Alice Chen', handles=('+15551234567',))]))
    engine = SearchEngine(str(index_db), str(chat_db))
    engine.warmup(progress=lambda name: __import__('contextlib').nullcontext())
    return engine


def test_engine_search_returns_formatted_sessions(tmp_path, monkeypatch):
    engine = _engine(tmp_path, monkeypatch)
    payload = engine.search('dinner plans', SearchFilters(), SearchOptions())
    assert payload['n_sessions'] >= 1
    assert payload['sessions'][0]['title']


def test_engine_applies_group_filter(tmp_path, monkeypatch):
    engine = _engine(tmp_path, monkeypatch)
    payload = engine.search('discussion', SearchFilters(is_group=True), SearchOptions())
    assert all(session['is_group'] for session in payload['sessions'])


def test_engine_applies_people_filter(tmp_path, monkeypatch):
    engine = _engine(tmp_path, monkeypatch)
    payload = engine.search('dinner', SearchFilters(people_handles=['+15551234567']), SearchOptions())
    assert payload['n_sessions'] >= 1
    assert all('+15551234567' in session['participants'] for session in payload['sessions'])


def test_status_before_warmup_reports_not_ready():
    from seaglass.app.engine import SearchEngine

    engine = SearchEngine('/tmp/does-not-exist-index.db', None)
    status = engine.status()
    assert status['index_ready'] is False
    assert status['n_chunks'] == 0
    assert status['hydration_available'] is False


def _snapshot_and_live(tmp_path, monkeypatch, live_messages):
    """Build a stale snapshot chat.db + index, plus a *live* chat.db that
    additionally contains `live_messages` (list of (text, apple_date))."""
    snap_dir = tmp_path / 'snap'
    snap_dir.mkdir()
    base = [('dinner plans with alice', APPLE_EPOCH_START, False, 0), ('photo attached', APPLE_EPOCH_START + 10, True, 0)]
    snapshot_db = build_fixture_chat_db(snap_dir, [{'chat_id': 1, 'handles': ['+15551234567'], 'messages': base}])
    index_db = tmp_path / 'index.db'
    build_index(snapshot_db, index_db, embedding_model=FakeEmbeddingModel(), batch_size=10)

    live_dir = tmp_path / 'live'
    live_dir.mkdir()
    extra = [(text, date, False, 0) for text, date in live_messages]
    live_db = build_fixture_chat_db(live_dir, [{'chat_id': 1, 'handles': ['+15551234567'], 'messages': base + extra}])

    monkeypatch.setattr('seaglass.app.engine.EmbeddingModel', FakeEmbeddingModel)
    monkeypatch.setattr('seaglass.app.engine.CrossEncoderReranker', FakeReranker)
    monkeypatch.setattr('seaglass.app.engine.ContactIndex.load', lambda: ContactIndex([]))
    engine = SearchEngine(str(index_db), str(snapshot_db), chat_db_source=str(live_db))
    engine.warmup(progress=lambda name: __import__('contextlib').nullcontext())
    return engine


def test_status_counts_messages_from_live_chat_db_not_snapshot(tmp_path, monkeypatch):
    engine = _snapshot_and_live(tmp_path, monkeypatch, [('brand new message', APPLE_EPOCH_START + 5000), ('another new one', APPLE_EPOCH_START + 6000)])
    status = engine.status()
    assert status['n_messages_since_index'] == 2
    assert status['chat_db_max_ts'] > status['most_recent_chunk_ts']


def test_status_reports_zero_when_live_db_matches_index(tmp_path, monkeypatch):
    engine = _snapshot_and_live(tmp_path, monkeypatch, [])
    assert engine.status()['n_messages_since_index'] == 0


def test_status_normalizes_nanosecond_dates_like_source(tmp_path, monkeypatch):
    # Big Sur+ writes nanoseconds-since-2001; the live query must apply the
    # same magnitude heuristic as imessage/source.apple_to_unix.
    ns_date = int((APPLE_EPOCH_START + 7000) * 1e9)
    engine = _snapshot_and_live(tmp_path, monkeypatch, [('nanosecond era message', ns_date)])
    status = engine.status()
    assert status['n_messages_since_index'] == 1
    assert abs(status['chat_db_max_ts'] - (APPLE_EPOCH_START + 7000 + 978307200)) < 1


def test_status_degrades_gracefully_when_live_db_missing(tmp_path, monkeypatch):
    engine = _snapshot_and_live(tmp_path, monkeypatch, [('new', APPLE_EPOCH_START + 5000)])
    engine._close_live_chat_connection()
    engine.chat_db_source = str(tmp_path / 'gone' / 'chat.db')
    engine.chat_db = None
    status = engine.status()
    assert status['n_messages_since_index'] == 0
    assert status['chat_db_max_ts'] is None
    assert status['index_ready'] is True


def test_status_reuses_cached_live_connection(tmp_path, monkeypatch):
    engine = _snapshot_and_live(tmp_path, monkeypatch, [('new', APPLE_EPOCH_START + 5000)])
    engine.status()
    first = engine._live_chat_con
    engine.status()
    assert engine._live_chat_con is first is not None


def test_blank_query_browses_newest_instead_of_scoring_noise(tmp_path, monkeypatch):
    """An empty query has no words to match on: embedding the empty string
    gives an arbitrary vector-space direction, so the old behaviour was
    dozens of confidently-scored but meaningless sessions.
    """
    engine = _engine(tmp_path, monkeypatch)
    payload = engine.search('   ', SearchFilters(), SearchOptions())

    assert payload['sessions'], 'browse should still show something to look at'
    # newest first, and the models are skipped entirely
    starts = [s['messages'][0]['ts'] for s in payload['sessions']]
    assert starts == sorted(starts, reverse=True)
    assert 'browse' in payload['timings']
    assert 'rerank' not in payload['timings']


def test_blank_query_still_honours_filters(tmp_path, monkeypatch):
    engine = _engine(tmp_path, monkeypatch)
    payload = engine.search('', SearchFilters(chat_ids=[2]), SearchOptions())
    assert payload['sessions']
    assert {s['chat_id'] for s in payload['sessions']} == {2}


def test_blank_query_paginates(tmp_path, monkeypatch):
    engine = _engine(tmp_path, monkeypatch)
    first = engine.search('', SearchFilters(), SearchOptions(max_sessions=1))
    assert first['has_more'] is (first['total_sessions'] > 1)
    if first['has_more']:
        second = engine.search('', SearchFilters(), SearchOptions(max_sessions=1, offset=first['next_offset']))
        assert second['sessions'][0]['chat_id'] != first['sessions'][0]['chat_id'] or \
            second['sessions'][0]['messages'][0]['ts'] != first['sessions'][0]['messages'][0]['ts']


def test_page_cache_is_not_shared_across_different_filters(tmp_path, monkeypatch):
    """The cache holds a reranked *candidate pool*, and filters are what
    narrow that pool -- so a filter missing from the key made switching
    Any/1:1/Group replay the previous query's results verbatim.
    """
    engine = _engine(tmp_path, monkeypatch)
    everything = engine.search('discussion', SearchFilters(), SearchOptions())
    one_chat = engine.search('discussion', SearchFilters(chat_ids=[1]), SearchOptions())
    assert {s['chat_id'] for s in one_chat['sessions']} <= {1}
    assert one_chat['sessions'] != everything['sessions'] or everything['n_sessions'] <= 1

    grouped = engine.search('discussion', SearchFilters(is_group=True), SearchOptions())
    solo = engine.search('discussion', SearchFilters(is_group=False), SearchOptions())
    assert not ({s['chat_id'] for s in grouped['sessions']} & {s['chat_id'] for s in solo['sessions']})


def test_status_reports_live_chat_readability(tmp_path, monkeypatch):
    """A dead live-chat.db connection (the usual cause: Full Disk Access not
    granted to *this* app identity) made the sync banner report "up to
    date" forever. The UI needs to be able to tell the difference.
    """
    engine = _engine(tmp_path, monkeypatch)
    assert engine.status()['live_chat_readable'] is True

    monkeypatch.setattr(engine, '_live_chat_connection', lambda: None)
    status = engine.status()
    assert status['live_chat_readable'] is False
    assert status['n_messages_since_index'] == 0


def test_freshness_is_cached_between_calls(tmp_path, monkeypatch):
    # Reading the live chat.db costs ~0.25s -- roughly a whole query -- so
    # it must not be paid on every search.
    engine = _snapshot_and_live(tmp_path, monkeypatch, [('new', APPLE_EPOCH_START + 5000)])
    engine.invalidate_freshness()  # the helper may already have primed it
    calls = []
    original = engine._live_chat_freshness
    engine._live_chat_freshness = lambda ts: (calls.append(ts), original(ts))[1]

    engine.status()
    engine.status()
    assert len(calls) == 1


def test_freshness_cache_is_dropped_after_a_build(tmp_path, monkeypatch):
    # Otherwise the app keeps reporting "N messages behind" for a whole
    # TTL after the user has just synced -- the one moment they are looking.
    engine = _snapshot_and_live(tmp_path, monkeypatch, [('new', APPLE_EPOCH_START + 5000)])
    engine.status()
    engine.invalidate_freshness()
    calls = []
    original = engine._live_chat_freshness
    engine._live_chat_freshness = lambda ts: (calls.append(ts), original(ts))[1]

    engine.status()
    assert len(calls) == 1
