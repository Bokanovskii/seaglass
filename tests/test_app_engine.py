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
