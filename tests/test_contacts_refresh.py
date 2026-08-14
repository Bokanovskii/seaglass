"""Contacts change under a long-running app; the index does not.

Chunk bodies say "Me"/"Them" and store no names, so an added, renamed or
deleted contact never needs a reindex -- but the app read the Contacts
store exactly once, during warmup, and kept that snapshot forever. The
failure mode is not "shows a raw handle": an unknown name still fuzzy
matches, so a newly added contact silently resolves to somebody else.
"""

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from seaglass.app.engine import (
    CONTACTS_TTL_SECONDS,
    SearchEngine,
    _contacts_fingerprint,
)
from seaglass.imessage.contacts import Contact, ContactIndex, ContactsUnavailableError

from conftest import build_fixture_chat_db

APPLE_EPOCH_START = 600000000


def _contact(name, handle, ident=None):
    return Contact(identifier=ident or name, display_name=name, handles=(handle,))


def _engine(contacts):
    """An engine with a contact set but no databases, for the cache logic."""
    engine = SearchEngine('/nonexistent/index.db')
    engine.contact_index = ContactIndex(list(contacts))
    engine._contacts_fingerprint = _contacts_fingerprint(engine.contact_index)
    engine._contacts_loaded_at = 1000.0
    return engine


def _load_returns(monkeypatch, contacts):
    calls = []

    def fake_load(*args, **kwargs):
        calls.append(1)
        return ContactIndex(list(contacts))

    monkeypatch.setattr('seaglass.app.engine.ContactIndex.load', fake_load)
    return calls


class TestTheTtlDecidesWhenToReRead:
    def test_a_fresh_cache_is_not_re_read(self, monkeypatch):
        engine = _engine([_contact('Ben Galindo-Navarro', '+1425')])
        calls = _load_returns(monkeypatch, [])

        refreshed = engine.maybe_refresh_contacts(now=1000.0 + CONTACTS_TTL_SECONDS - 1)

        assert refreshed is False
        assert calls == [], 'a ~300ms read must not run on every query'

    def test_an_aged_cache_is_re_read(self, monkeypatch):
        engine = _engine([_contact('Ben Galindo-Navarro', '+1425')])
        calls = _load_returns(monkeypatch, [_contact('Ben Gibson', '+1303')])

        refreshed = engine.maybe_refresh_contacts(now=1000.0 + CONTACTS_TTL_SECONDS + 1)

        assert refreshed is True
        assert len(calls) == 1

    def test_a_failed_read_does_not_retry_on_every_query(self, monkeypatch):
        """A denied grant must not turn every subsequent query into another
        blocking attempt, so the failure has to reset the TTL too."""
        engine = _engine([_contact('Kaya', '+1206')])
        attempts = []

        def boom(*args, **kwargs):
            attempts.append(1)
            raise ContactsUnavailableError('denied')

        monkeypatch.setattr('seaglass.app.engine.ContactIndex.load', boom)

        engine.maybe_refresh_contacts(now=1e9)
        engine.maybe_refresh_contacts(now=1e9 + 1)

        assert len(attempts) == 1

    def test_a_failed_read_keeps_the_contacts_it_already_had(self, monkeypatch):
        engine = _engine([_contact('Kaya', '+1206')])

        def boom(*args, **kwargs):
            raise ContactsUnavailableError('denied')

        monkeypatch.setattr('seaglass.app.engine.ContactIndex.load', boom)
        engine.maybe_refresh_contacts(now=1e9)

        assert engine.contact_index.resolve_handle('+1206') == 'Kaya'


class TestTheContactSetActuallyUpdates:
    def test_an_added_contact_becomes_resolvable(self, monkeypatch):
        engine = _engine([_contact('Ben Galindo-Navarro', '+1425')])
        _load_returns(monkeypatch, [
            _contact('Ben Galindo-Navarro', '+1425'),
            _contact('Ben Gibson', '+1303'),
        ])

        assert engine.contact_index.resolve_handle('+1303') is None
        engine.refresh_contacts()
        assert engine.contact_index.resolve_handle('+1303') == 'Ben Gibson'

    def test_a_deleted_contact_stops_resolving(self, monkeypatch):
        engine = _engine([_contact('Old Name', '+1555')])
        _load_returns(monkeypatch, [])

        engine.refresh_contacts()

        assert engine.contact_index.resolve_handle('+1555') is None

    def test_a_renamed_contact_resolves_to_the_new_name(self, monkeypatch):
        engine = _engine([_contact('Maiden Name', '+1555', ident='x')])
        _load_returns(monkeypatch, [_contact('Married Name', '+1555', ident='x')])

        engine.refresh_contacts()

        assert engine.contact_index.resolve_handle('+1555') == 'Married Name'


class TestAnUnchangedReadCostsNothingExtra:
    """A reload every 5 minutes must not keep throwing away the ranked-page
    cache, or "load more" pays a full embed + rerank for no reason."""

    def test_an_identical_contact_set_reports_no_change(self, monkeypatch):
        contacts = [_contact('Kaya', '+1206')]
        engine = _engine(contacts)
        _load_returns(monkeypatch, contacts)

        assert engine.refresh_contacts() is False

    def test_an_identical_contact_set_keeps_the_page_cache(self, monkeypatch):
        contacts = [_contact('Kaya', '+1206')]
        engine = _engine(contacts)
        _load_returns(monkeypatch, contacts)
        engine._page_cache['key'] = ['ranked']

        engine.refresh_contacts()

        assert 'key' in engine._page_cache

    def test_a_changed_contact_set_drops_the_page_cache(self, monkeypatch):
        engine = _engine([_contact('Kaya', '+1206')])
        _load_returns(monkeypatch, [_contact('Kaya', '+1206'), _contact('New', '+1999')])
        engine._page_cache['key'] = ['ranked']

        engine.refresh_contacts()

        assert 'key' not in engine._page_cache, 'cached rankings carry resolved names'

    def test_a_rename_alone_counts_as_a_change(self, monkeypatch):
        engine = _engine([_contact('Before', '+1206', ident='x')])
        _load_returns(monkeypatch, [_contact('After', '+1206', ident='x')])

        assert engine.refresh_contacts() is True


def _warmed_engine(tmp_path, monkeypatch, contacts):
    from seaglass.index.build import build_index
    from test_app_engine import FakeReranker
    from conftest import FakeEmbeddingModel

    chats = [{
        'chat_id': 1,
        'handles': ['+15551111111'],
        'messages': [('lets get dinner tomorrow', APPLE_EPOCH_START, False, 0)],
    }]
    chat_db = build_fixture_chat_db(tmp_path, chats)
    index_db = tmp_path / 'index.db'
    build_index(chat_db, index_db, embedding_model=FakeEmbeddingModel(), batch_size=10)

    monkeypatch.setattr('seaglass.app.engine.EmbeddingModel', FakeEmbeddingModel)
    monkeypatch.setattr('seaglass.app.engine.CrossEncoderReranker', FakeReranker)
    monkeypatch.setattr(
        'seaglass.app.engine.ContactIndex.load', lambda *a, **k: ContactIndex(list(contacts))
    )
    engine = SearchEngine(str(index_db), str(chat_db))
    engine.warmup(progress=lambda name: __import__('contextlib').nullcontext())
    return engine


class TestTheRunningAppPicksThemUp:
    def test_warmup_records_when_contacts_were_read(self, tmp_path, monkeypatch):
        engine = _warmed_engine(tmp_path, monkeypatch, [])

        assert engine._contacts_loaded_at > 0, 'otherwise the first query always reloads'

    def test_a_search_refreshes_contacts_that_have_aged_out(self, tmp_path, monkeypatch):
        from seaglass.app.engine import SearchOptions
        from seaglass.app.filters import SearchFilters

        engine = _warmed_engine(tmp_path, monkeypatch, [])
        monkeypatch.setattr(
            'seaglass.app.engine.ContactIndex.load',
            lambda *a, **k: ContactIndex([_contact('Ben Gibson', '+15551111111')]),
        )
        engine._contacts_loaded_at = 0.0

        engine.search('dinner', SearchFilters(), SearchOptions())

        assert engine.contact_index.resolve_handle('+15551111111') == 'Ben Gibson'

    def test_the_refresh_happens_before_the_name_is_parsed(self, tmp_path, monkeypatch):
        """The whole point: a name becomes a handle filter during parse, so
        refreshing afterwards would still answer this query wrongly."""
        from seaglass.app.engine import SearchOptions
        from seaglass.app.filters import SearchFilters

        engine = _warmed_engine(tmp_path, monkeypatch, [])
        monkeypatch.setattr(
            'seaglass.app.engine.ContactIndex.load',
            lambda *a, **k: ContactIndex([_contact('Ben Gibson', '+15551111111')]),
        )
        engine._contacts_loaded_at = 0.0

        payload = engine.search('latest from Ben Gibson', SearchFilters(), SearchOptions())

        assert '+15551111111' in (payload['effective_filters'].get('people_participant') or [])

    def test_chat_titles_are_rebuilt_when_a_name_appears(self, tmp_path, monkeypatch):
        engine = _warmed_engine(tmp_path, monkeypatch, [])
        assert 'Ben Gibson' not in engine.chatmeta.all()[0].title

        monkeypatch.setattr(
            'seaglass.app.engine.ContactIndex.load',
            lambda *a, **k: ContactIndex([_contact('Ben Gibson', '+15551111111')]),
        )
        engine.refresh_contacts()

        assert 'Ben Gibson' in engine.chatmeta.all()[0].title

    def test_suggest_sees_a_contact_added_since_warmup(self, tmp_path, monkeypatch):
        engine = _warmed_engine(tmp_path, monkeypatch, [])
        monkeypatch.setattr(
            'seaglass.app.engine.ContactIndex.load',
            lambda *a, **k: ContactIndex([_contact('Ben Gibson', '+15551111111')]),
        )
        engine._contacts_loaded_at = 0.0

        names = [s['display_name'] for s in engine.suggest_contacts('Ben Gib')]

        assert 'Ben Gibson' in names


class TestGrantingAccessDoesNotNeedARestart:
    def test_contacts_load_after_the_grant(self, tmp_path, monkeypatch):
        """Warmup runs before any Cocoa event loop, so a first-run user is
        always denied there and `contact_index` is None."""
        engine = _warmed_engine(tmp_path, monkeypatch, [])
        engine.contact_index = None
        engine._contacts_fingerprint = None
        monkeypatch.setattr(
            'seaglass.app.engine.ContactIndex.load',
            lambda *a, **k: ContactIndex([_contact('Ben Gibson', '+15551111111')]),
        )

        assert engine.refresh_contacts() is True
        assert engine.contact_index.resolve_handle('+15551111111') == 'Ben Gibson'


class TestTheFingerprintDistinguishesRealChanges:
    def test_handles_are_part_of_the_identity(self):
        """A contact gaining a second number is a real change: queries for
        that person must start matching the new handle."""
        one = ContactIndex([Contact('x', 'Kaya', ('+1206',))])
        two = ContactIndex([Contact('x', 'Kaya', ('+1206', '+1647'))])

        assert _contacts_fingerprint(one) != _contacts_fingerprint(two)

    def test_ordering_is_not_a_change(self):
        a = ContactIndex([_contact('A', '+1'), _contact('B', '+2')])
        b = ContactIndex([_contact('B', '+2'), _contact('A', '+1')])

        assert _contacts_fingerprint(a) == _contacts_fingerprint(b)

    def test_no_contacts_is_distinct_from_no_access(self):
        assert _contacts_fingerprint(ContactIndex([])) != _contacts_fingerprint(None)
