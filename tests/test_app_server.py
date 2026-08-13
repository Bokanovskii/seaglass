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
