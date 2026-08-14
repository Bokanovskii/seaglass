from __future__ import annotations

from fastapi.testclient import TestClient

from seaglass.app.server import create_app
from seaglass.app.warmup import WarmupState


class StubEngine:
    def __init__(self):
        self.contact_index = None
        self.corpus_bounds = (0, 2000000000)

    def health(self):
        return {'warnings': []}

    def status(self):
        return {'n_chunks': 1, 'n_vectors': 1, 'n_chats': 1, 'hydration_available': True, 'n_messages_since_index': 0}

    def search(self, text, filters, options):
        return {'n_sessions': 1, 'n_results': 1, 'confidence': 'high', 'sessions': [{'chat_id': 1, 'day': '2024-01-01', 'score': 1.0, 'title': 'Alice Chen', 'is_group': False, 'participant_count': 1, 'participants': ['+15551234567'], 'messages': [{'message_id': 1, 'ts': 1.0, 'sender': 'Alice Chen', 'text': text, 'has_attachment': False}], 'context_messages': []}], 'effective_filters': {'semantic': text}, 'parse_source': 'deterministic', 'timings': {}, 'elapsed_s': 0.1}

    def conversation(self, chat_id, around_ts, limit):
        return {'chat_id': chat_id, 'title': 'Alice Chen', 'is_group': False, 'participants': ['+15551234567'], 'messages': []}

    def suggest_contacts(self, q, limit=10):
        return [{'display_name': 'Alice Chen', 'handles': ['+15551234567'], 'n_handles': 1, 'score': 1.0}]

    def suggest_chats(self, q, limit=20):
        return [{'chat_id': 1, 'title': 'Alice Chen', 'is_group': False, 'participant_count': 1, 'n_chunks': 1, 'last_ts': 1.0}]


class StubConfig:
    port = 8765
    copilot_bin = None
    index_db = '/tmp/does-not-matter-index.db'
    chat_db = '/tmp/does-not-matter-chat.db'
    chat_db_source = '/tmp/does-not-matter-source.db'
    def to_dict(self):
        return {'port': self.port}


def _client():
    warmup = WarmupState(['open_index'])
    with warmup.step('open_index'):
        pass
    warmup.set_ready()
    app = create_app(StubEngine(), warmup, StubConfig(), 'token123')
    return TestClient(app)


def test_api_requires_token():
    client = _client()
    response = client.get('/api/health')
    assert response.status_code == 401


def test_api_rejects_bad_origin():
    client = _client()
    response = client.get('/api/health', headers={'Authorization': 'Bearer token123', 'Origin': 'https://evil.example', 'Host': '127.0.0.1:8765'})
    assert response.status_code == 403


def test_health_round_trip():
    client = _client()
    response = client.get('/api/health', headers={'Authorization': 'Bearer token123', 'Host': '127.0.0.1:8765'})
    assert response.status_code == 200
    assert response.json()['state'] == 'READY'


def test_search_round_trip():
    client = _client()
    response = client.post('/api/search', headers={'Authorization': 'Bearer token123', 'Host': '127.0.0.1:8765'}, json={'query': 'lease', 'filters': {}, 'options': {}, 'assist': 'off', 'request_id': 'abc'})
    assert response.status_code == 200
    assert response.json()['request_id'] == 'abc'


def test_build_status_reports_idle_by_default():
    client = _client()
    response = client.get('/api/index/build', headers={'Authorization': f'Bearer token123', 'Host': '127.0.0.1:8765'})
    assert response.status_code == 200
    body = response.json()
    assert body['running'] is False
    assert body['stage'] == 'idle'


def test_health_includes_build_state():
    client = _client()
    response = client.get('/api/health', headers={'Authorization': f'Bearer token123', 'Host': '127.0.0.1:8765'})
    assert response.status_code == 200
    assert 'build' in response.json()
    assert response.json()['build']['running'] is False


def test_start_build_reports_conflict_when_already_running():
    client = _client()
    app = client.app
    app.state.build_state.running = True
    response = client.post('/api/index/build', headers={'Authorization': f'Bearer token123', 'Host': '127.0.0.1:8765'})
    assert response.status_code == 409


def test_open_settings_rejects_unknown_pane(monkeypatch):
    """A pane name is interpolated into a URL handed to `open`, so only an
    allow-listed set may ever reach it."""
    import seaglass.app.server as server_module

    calls = []
    monkeypatch.setattr(server_module.subprocess, 'Popen', lambda *a, **k: calls.append(a))
    client = _client()
    response = client.post(
        '/api/system/open-settings',
        json={'pane': 'x"; rm -rf ~'},
        headers={'Authorization': 'Bearer token123', 'Host': '127.0.0.1:8765'},
    )
    assert response.status_code == 400
    assert calls == []


def test_open_settings_opens_known_pane(monkeypatch):
    import seaglass.app.server as server_module

    calls = []
    monkeypatch.setattr(server_module.subprocess, 'Popen', lambda *a, **k: calls.append(a[0]))
    client = _client()
    response = client.post(
        '/api/system/open-settings',
        json={'pane': 'contacts'},
        headers={'Authorization': 'Bearer token123', 'Host': '127.0.0.1:8765'},
    )
    assert response.status_code == 200
    assert response.json()['opened'] is True
    assert calls and 'Privacy_Contacts' in calls[0][-1]


def test_bundle_path_requires_launch_services_marker(monkeypatch):
    """The bundle is identified by the env var LaunchServices exports, since
    the executable path is the interpreter's either way."""
    from seaglass.app.server import _bundle_path

    monkeypatch.delenv('__CFBundleIdentifier', raising=False)
    assert _bundle_path() is None
    monkeypatch.setenv('__CFBundleIdentifier', 'com.apple.Terminal')
    assert _bundle_path() is None


def _assist_client():
    """A client whose engine records the search it was asked to run and whose
    contact index can resolve one person."""
    from seaglass.imessage.contacts import Contact, ContactIndex

    engine = StubEngine()
    engine.contact_index = ContactIndex([
        Contact(identifier='1', display_name='Alice Chen', handles=('+15551234567',)),
    ])
    engine.calls = []
    inner = engine.search

    def recording_search(text, filters, options):
        engine.calls.append((text, filters, options))
        return inner(text, filters, options)

    engine.search = recording_search
    warmup = WarmupState(['open_index'])
    with warmup.step('open_index'):
        pass
    warmup.set_ready()
    app = create_app(engine, warmup, StubConfig(), 'token123')
    return TestClient(app), app, engine


def _stub_assist(app, payload, query):
    """Pretend Copilot has already answered, without invoking it."""
    import concurrent.futures

    from seaglass.search.parse import parse_query

    manager = app.state.assist_manager
    future = concurrent.futures.Future()
    future.set_result(payload)
    manager.pending['tok'] = future
    manager.submitted['tok'] = (query, parse_query(query, contact_index=manager.engine.contact_index))
    return 'tok'


_AUTH = {'Authorization': 'Bearer token123', 'Host': '127.0.0.1:8765'}


def test_apply_assist_actually_applies_the_parse():
    # The bug: apply-assist re-ran engine.search(merged.raw, ...), so the
    # only thing the merge ever contributed was keyword expansions. The
    # banner said "people: alice, Aug 6 -> Aug 13" while the results were
    # the un-assisted ones.
    client, app, engine = _assist_client()
    token = _stub_assist(app, {
        'people': ['alice chen'],
        'date_from': '2024-01-01',
        'date_to': '2024-01-03',
        'semantic': 'lease paperwork',
        'expansions': ['contract'],
        'confidence': 0.9,
    }, 'lease paperwork with alice chen in january')
    response = client.post('/api/search/apply-assist', headers=_AUTH, json={
        'assist_token': token,
        'query': 'lease paperwork with alice chen in january',
        'filters': {},
        'options': {},
        'request_id': 'req-1',
    })
    assert response.status_code == 200
    text, filters, options = engine.calls[-1]
    assert filters.people_handles == ['+15551234567']
    assert filters.date_from is not None and filters.date_to is not None
    assert 'contract' in options.expansions
    assert 'lease' in text
    body = response.json()
    assert body['request_id'] == 'req-1'
    assert 'Alice Chen' in body['assist_description']
    # "load more" has to page the assisted query, not the original one.
    assert body['applied_query'] == text
    assert body['applied_filters']['people_handles'] == ['+15551234567']


def test_assist_status_merges_against_the_real_query():
    # It used to merge against parse_query(''), throwing away every filter
    # the deterministic parser had already found.
    client, app, engine = _assist_client()
    token = _stub_assist(app, {'people': ['alice chen'], 'semantic': 'lease'},
                         'what did alice chen say about the lease')
    response = client.get(f'/api/assist/{token}', headers=_AUTH)
    assert response.status_code == 200
    body = response.json()
    assert body['status'] == 'ready'
    assert 'Alice Chen' in body['description']
    assert body['unresolved_people'] == []


def test_assist_status_reports_a_name_it_could_not_resolve():
    client, app, engine = _assist_client()
    token = _stub_assist(app, {'people': ['Zebedee'], 'semantic': 'lease'}, 'what did zebedee say about the lease')
    body = client.get(f'/api/assist/{token}', headers=_AUTH).json()
    assert body['unresolved_people'] == ['Zebedee']
    assert 'no contact matched Zebedee' in body['description']


def test_apply_assist_rejects_a_missing_token_with_400():
    # A typo in the request body is the client's fault, not the engine's.
    # Letting the KeyError escape reported it as a 500, which sends whoever
    # is debugging into the search pipeline instead of their own payload.
    client, _app, _engine = _assist_client()
    response = client.post('/api/search/apply-assist', headers=_AUTH, json={})
    assert response.status_code == 400


def test_apply_assist_rejects_an_unknown_token_with_404():
    client, _app, _engine = _assist_client()
    response = client.post(
        '/api/search/apply-assist', headers=_AUTH, json={'assist_token': 'nope'}
    )
    assert response.status_code == 404
