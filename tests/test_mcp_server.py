"""Unit tests for seaglass.mcp_server -- Phase 6 MCP tool surface
(search_messages, get_conversation, index_status). Synthetic chat.db +
index.db, no network/MLX dependency: monkeypatches the module's lazy
model/connection getters to return fakes, exactly like test_retrieve.py
and test_rank.py do for the underlying pipeline pieces.
"""

from __future__ import annotations

import json
import types

import pytest

from seaglass.imessage.source import connect_readonly
from seaglass.index.build import build_index, open_index_db

import seaglass.mcp_server as mcp_server

from conftest import FakeEmbeddingModel, build_fixture_chat_db


class FakeReranker:
    """Same deterministic word-overlap stand-in used in test_rank.py."""

    def score(self, pairs):
        scores = []
        for query, text in pairs:
            query_words = set(query.lower().split())
            text_words = set(text.lower().split())
            scores.append(float(len(query_words & text_words)))
        return scores


APPLE_EPOCH_START = 700000000


@pytest.fixture(autouse=True)
def no_running_app(monkeypatch):
    """Never let a desktop app that happens to be running on the dev
    machine answer for the pipeline under test."""
    monkeypatch.setattr(mcp_server, "_running_app_lock", lambda: None)


def _fixture(tmp_path):
    chats = [
        {
            "chat_id": 1,
            "handles": ["+15551110000"],
            "messages": [
                ("lisbon trip planning starts now", APPLE_EPOCH_START, False, 0),
                ("what hotel should we book in lisbon", APPLE_EPOCH_START + 30, True, 0),
                ("the alfama district looks amazing", APPLE_EPOCH_START + 60, False, 0),
                ("any plans for dinner tonight lisbon", APPLE_EPOCH_START + 90, False, 0),
            ],
        },
        {
            "chat_id": 2,
            "handles": ["+15552220000"],
            "messages": [
                ("can you send the tax documents", APPLE_EPOCH_START + 100000, False, 0),
                ("sure, emailing them over now", APPLE_EPOCH_START + 100030, True, 0),
            ],
        },
    ]
    chat_db_path = build_fixture_chat_db(tmp_path, chats)
    index_db_path = tmp_path / "index.db"
    build_index(chat_db_path, index_db_path, embedding_model=FakeEmbeddingModel(), batch_size=10)
    return chat_db_path, index_db_path


def _patch_server(monkeypatch, tmp_path):
    """Wire mcp_server's lazy getters to fixture data + fakes, bypassing
    env vars and real model loads entirely.
    """
    chat_db_path, index_db_path = _fixture(tmp_path)
    index_con = open_index_db(index_db_path)
    chat_con = connect_readonly(chat_db_path)

    monkeypatch.setattr(mcp_server, "_get_index_con", lambda: index_con)
    monkeypatch.setattr(mcp_server, "_get_chat_con", lambda: chat_con)
    monkeypatch.setattr(mcp_server, "_get_embedding_model", lambda: FakeEmbeddingModel())
    monkeypatch.setattr(mcp_server, "_get_reranker", lambda: FakeReranker())
    monkeypatch.setattr(mcp_server, "_get_contact_index", lambda: None)
    return chat_db_path, index_db_path


def test_search_messages_returns_hydrated_sessions(tmp_path, monkeypatch):
    _patch_server(monkeypatch, tmp_path)

    payload = mcp_server.search_messages("lisbon trip", max_sessions=5)

    assert payload["n_sessions"] >= 1
    assert "elapsed_s" in payload
    session = payload["sessions"][0]
    assert session["chat_id"] == 1
    assert any("lisbon" in m["text"].lower() for m in session["messages"])


def test_search_messages_no_results_when_fused_empty(tmp_path, monkeypatch):
    _patch_server(monkeypatch, tmp_path)
    # Force the "nothing survived fusion" branch directly, rather than
    # relying on a genuinely-empty index.db (retrieve() treats a *never
    # built* index, i.e. no meta.int8_absmax at all, as a hard error --
    # a distinct condition from "built index, zero matches").
    # Patched on the engine, not on this module: ranking is the engine's
    # now, so that is where retrieval happens.
    monkeypatch.setattr("seaglass.app.engine.retrieve", lambda *a, **k: [])

    # A contentful query: a filler-only one ("anything at all") never
    # reaches retrieve() -- it browses by recency instead.
    payload = mcp_server.search_messages("dinner reservation")
    assert payload["n_sessions"] == 0
    assert payload["confidence"] == "none"


def test_search_messages_degrades_without_chat_db(tmp_path, monkeypatch):
    _, index_db_path = _fixture(tmp_path)
    index_con = open_index_db(index_db_path)

    monkeypatch.setattr(mcp_server, "_get_index_con", lambda: index_con)
    monkeypatch.setattr(mcp_server, "_get_chat_con", lambda: None)
    monkeypatch.setattr(mcp_server, "_get_embedding_model", lambda: FakeEmbeddingModel())
    monkeypatch.setattr(mcp_server, "_get_reranker", lambda: FakeReranker())
    monkeypatch.setattr(mcp_server, "_get_contact_index", lambda: None)

    payload = mcp_server.search_messages("lisbon trip")
    assert "unhydrated" in payload["confidence"]
    assert payload["n_sessions"] >= 1
    assert "preview" in payload["sessions"][0]


def test_get_conversation_returns_messages_in_order(tmp_path, monkeypatch):
    _patch_server(monkeypatch, tmp_path)

    result = mcp_server.get_conversation(chat_id=1, limit=10)

    assert result["chat_id"] == 1
    assert result["n_messages"] == 4
    timestamps = [m["ts"] for m in result["messages"]]
    assert timestamps == sorted(timestamps)


def test_get_conversation_around_ts_picks_closest_messages(tmp_path, monkeypatch):
    _patch_server(monkeypatch, tmp_path)

    result = mcp_server.get_conversation(chat_id=1, around_ts=mcp_server.apple_to_unix(APPLE_EPOCH_START), limit=2)

    assert result["n_messages"] == 2
    texts = [m["text"] for m in result["messages"]]
    assert "lisbon trip planning starts now" in texts


def test_get_conversation_around_ts_finds_old_message_beyond_2000_row_window(tmp_path, monkeypatch):
    # Regression test for BUG-9: get_conversation(around_ts=...) used to
    # only fetch the newest 2000 rows before doing its distance sort, so
    # an old target message in a chat with >2000 messages was silently
    # invisible. Build a chat with an old message, a large run of recent
    # filler, and confirm around_ts pointing at the old message still
    # finds it.
    chats = [
        {
            "chat_id": 1,
            "handles": ["+15551110000"],
            "messages": (
                [("ancient lisbon trip message", APPLE_EPOCH_START, False, 0)]
                + [
                    (f"filler message {i}", APPLE_EPOCH_START + 10_000_000 + i, False, 0)
                    for i in range(2500)
                ]
            ),
        }
    ]
    chat_db_path = build_fixture_chat_db(tmp_path, chats)
    index_db_path = tmp_path / "index.db"
    build_index(chat_db_path, index_db_path, embedding_model=FakeEmbeddingModel(), batch_size=200)
    index_con = open_index_db(index_db_path)
    chat_con = connect_readonly(chat_db_path)
    monkeypatch.setattr(mcp_server, "_get_index_con", lambda: index_con)
    monkeypatch.setattr(mcp_server, "_get_chat_con", lambda: chat_con)
    monkeypatch.setattr(mcp_server, "_get_contact_index", lambda: None)

    result = mcp_server.get_conversation(
        chat_id=1, around_ts=mcp_server.apple_to_unix(APPLE_EPOCH_START), limit=5
    )

    texts = [m["text"] for m in result["messages"]]
    assert "ancient lisbon trip message" in texts


def test_get_conversation_requires_chat_db(tmp_path, monkeypatch):
    _, index_db_path = _fixture(tmp_path)
    index_con = open_index_db(index_db_path)
    monkeypatch.setattr(mcp_server, "_get_index_con", lambda: index_con)
    monkeypatch.setattr(mcp_server, "_get_chat_con", lambda: None)

    try:
        mcp_server.get_conversation(chat_id=1)
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "SEAGLASS_CHAT_DB" in str(exc)


def test_index_status_reports_counts(tmp_path, monkeypatch):
    _patch_server(monkeypatch, tmp_path)
    monkeypatch.setenv("SEAGLASS_INDEX_DB", "fake-path.db")

    status = mcp_server.index_status()

    assert status["n_chunks"] > 0
    assert status["n_vectors"] == status["n_chunks"]
    assert status["hydration_available"] is True


def test_index_status_reports_error_when_unconfigured(monkeypatch, tmp_path):
    monkeypatch.delenv("SEAGLASS_INDEX_DB", raising=False)
    # ...and no index at the app's configured location either. (The default
    # path is resolved at import time, so setting HOME here would not take.)
    monkeypatch.setattr(
        "seaglass.app.config.load_config",
        lambda *a, **k: types.SimpleNamespace(index_db=str(tmp_path / "missing.db"), chat_db=None),
    )
    monkeypatch.setattr(mcp_server, "_index_con", None)

    status = mcp_server.index_status()

    assert "error" in status


def test_resolve_path_falls_back_to_app_config(monkeypatch, tmp_path):
    """With no env var set, the MCP server uses whatever index the desktop
    app is configured with -- otherwise a pinned, stale path in the MCP
    config sends every Grogu search to a nonexistent database."""
    monkeypatch.delenv("SEAGLASS_INDEX_DB", raising=False)
    index = tmp_path / "index.db"
    index.write_bytes(b"")
    fake_config = types.SimpleNamespace(index_db=str(index), chat_db=None)
    monkeypatch.setattr(
        "seaglass.app.config.load_config", lambda *a, **k: fake_config
    )

    assert mcp_server._resolve_path("SEAGLASS_INDEX_DB", lambda c: c.index_db) == str(index)
    # A configured-but-missing path resolves to nothing rather than being
    # handed on to be silently created as an empty database.
    assert mcp_server._resolve_path("SEAGLASS_CHAT_DB", lambda c: c.chat_db) is None


def test_resolve_path_prefers_env_var(monkeypatch):
    monkeypatch.setenv("SEAGLASS_INDEX_DB", "/tmp/explicit.db")
    assert mcp_server._resolve_path("SEAGLASS_INDEX_DB", lambda c: 'ignored') == "/tmp/explicit.db"


def test_search_prefers_the_running_app(monkeypatch, tmp_path):
    """A running app already holds the models and a warm cache; loading a
    second copy here to recompute an answer it can give over loopback costs
    ~1GB and a multi-second cold start."""
    monkeypatch.setattr(mcp_server, "_running_app_lock", lambda: {"port": 1, "token": "t"})
    monkeypatch.setattr(mcp_server, "_resolve_path", lambda *a, **k: None)
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(request.full_url)
        body = (
            {"index_ready": True, "index_db": "/x/index.db"}
            if request.full_url.endswith("/api/status")
            else {"n_sessions": 1, "sessions": [], "timings": {"parse": 0.1}}
        )
        return _FakeResponse(body)

    monkeypatch.setattr(mcp_server, "urlopen", fake_urlopen)
    payload = mcp_server.search_messages("dinner")

    assert payload["n_sessions"] == 1
    assert "timings" not in payload  # UI-only field, stripped
    assert any(url.endswith("/api/search") for url in calls)


def test_search_ignores_an_app_serving_a_different_index(monkeypatch, tmp_path):
    """Answering from a different corpus than the caller configured would be
    worse than being slow."""
    monkeypatch.setattr(mcp_server, "_running_app_lock", lambda: {"port": 1, "token": "t"})
    monkeypatch.setattr(mcp_server, "_resolve_path", lambda *a, **k: "/configured/index.db")
    monkeypatch.setattr(
        mcp_server,
        "urlopen",
        lambda request, timeout=None: _FakeResponse({"index_ready": True, "index_db": "/other/index.db"}),
    )

    assert mcp_server._search_via_running_app("dinner", max_sessions=8, redact=False) is None


def test_search_ignores_an_app_that_is_not_ready(monkeypatch):
    monkeypatch.setattr(mcp_server, "_running_app_lock", lambda: {"port": 1, "token": "t"})
    monkeypatch.setattr(
        mcp_server,
        "urlopen",
        lambda request, timeout=None: _FakeResponse({"index_ready": False}),
    )

    assert mcp_server._search_via_running_app("dinner", max_sessions=8, redact=False) is None


class _FakeResponse:
    def __init__(self, body):
        self._body = json.dumps(body).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_search_messages_forwards_offset_to_the_running_app(monkeypatch):
    """Grogu pages a chronological answer. If `offset` stops here, every
    page is page one and the caller loops on the same messages."""
    seen = {}

    def fake(query, *, max_sessions, redact, offset=0):
        seen.update(query=query, max_sessions=max_sessions, offset=offset)
        return {"sessions": [], "has_more": False}

    monkeypatch.setattr(mcp_server, "_search_via_running_app", fake)

    mcp_server.search_messages("latest from sam", max_sessions=8, offset=16)

    assert seen["offset"] == 16


def test_search_messages_offset_defaults_to_the_first_page(monkeypatch):
    seen = {}

    def fake(query, *, max_sessions, redact, offset=0):
        seen["offset"] = offset
        return {"sessions": [], "has_more": False}

    monkeypatch.setattr(mcp_server, "_search_via_running_app", fake)

    mcp_server.search_messages("latest from sam")

    assert seen["offset"] == 0
