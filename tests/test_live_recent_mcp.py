"""Synthetic-only live MCP tests: no personal database, index or model loads."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import subprocess
import sys
import types
from pathlib import Path

import pytest

import seaglass.mcp_server as mcp
from seaglass.imessage import source
from seaglass.imessage.source import apple_to_unix, connect_readonly
from conftest import build_fixture_chat_db


START = 700000000


def _db(directory, messages=None):
    directory.mkdir(exist_ok=True)
    return build_fixture_chat_db(directory, [
        {"chat_id": 1, "handles": ["synthetic@example.invalid"],
         "messages": messages if messages is not None else [("synthetic body", START, False, 0)]},
    ])


@pytest.fixture
def live(tmp_path, monkeypatch):
    from seaglass.imessage.contacts import ContactIndex

    path = _db(tmp_path)
    monkeypatch.setenv("SEAGLASS_CHAT_DB_SOURCE", str(path))
    monkeypatch.setattr(ContactIndex, "load", classmethod(lambda cls: ContactIndex([])))
    monkeypatch.setattr(mcp, "_contact_index", None)
    monkeypatch.setattr(mcp, "_contact_index_attempted", False)

    def forbidden(*args, **kwargs):
        pytest.fail("A live read touched the snapshot, index, models or app bridge")

    for name in (
        "_get_index_con", "_get_chat_con", "_get_engine", "_get_embedding_model",
        "_get_reranker", "_search_via_running_app",
    ):
        monkeypatch.setattr(mcp, name, forbidden)
    return path


@pytest.mark.parametrize("reader", ["recent", "metadata", "conversation", "search"])
@pytest.mark.parametrize("contacts_available", [True, False])
def test_cold_live_reads_resolve_sender_names(live, monkeypatch, reader, contacts_available):
    from seaglass.imessage.contacts import Contact, ContactIndex, ContactsUnavailableError

    def load(cls):
        if not contacts_available:
            raise ContactsUnavailableError("synthetic contacts denied")
        return ContactIndex([Contact("synthetic", "Zorina", ("synthetic@example.invalid",))])

    monkeypatch.setattr(ContactIndex, "load", classmethod(load))
    with sqlite3.connect(live) as con:
        con.execute("INSERT INTO handle (ROWID,id) VALUES (2,'unknown@example.invalid')")
        for rowid, from_me, handle in [(2, 0, 2), (3, 1, 1)]:
            con.execute(
                "INSERT INTO message (ROWID,text,date,is_from_me,handle_id,associated_message_type) "
                "VALUES (?,'synthetic',?,?,?,0)", (rowid, START + rowid, from_me, handle),
            )
            con.execute("INSERT INTO chat_message_join VALUES (1,?)", (rowid,))
    if reader == "conversation":
        result = mcp.get_conversation(1)
    elif reader == "search":
        result = mcp.search_messages("most recent messages")
        result["messages"] = [message for session in result["sessions"] for message in session["messages"]]
    else:
        result = mcp.recent_messages(since_ts=0, include_text=reader != "metadata")
    messages = {message["message_id"]: message for message in result["messages"]}
    assert messages[1]["sender"] == ("Zorina" if contacts_available else "synthetic@example.invalid")
    assert messages[2]["sender"] == "unknown@example.invalid"
    assert messages[3]["sender"] is None
    if reader == "metadata":
        assert all("text" not in message for message in messages.values())


def test_stale_snapshot_and_real_wal_commit(live, tmp_path, monkeypatch):
    snapshot = _db(tmp_path / "snapshot")
    monkeypatch.setenv("SEAGLASS_CHAT_DB", str(snapshot))
    writer = sqlite3.connect(live)
    try:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        # Open the reader before the commit, as happens in a long-lived MCP.
        with mcp._live_chat() as reader:
            assert source.newest_message_ts(reader) == apple_to_unix(START)
        writer.execute(
            "INSERT INTO message (ROWID,text,date,is_from_me,handle_id,associated_message_type) "
            "VALUES (2,'new synthetic WAL row',?,0,1,0)", ((START + 100) * 10**9,),
        )
        writer.execute("INSERT INTO chat_message_join VALUES (1,2)")
        writer.commit()
        assert Path(str(live) + "-wal").stat().st_size > 0
        result = mcp.recent_messages(since_ts=0)
        assert result["source"] == "live_chat_db"
        assert result["newest_source_ts"] == apple_to_unix(START + 100)
        assert [row["message_id"] for row in result["messages"]] == [2, 1]
        assert mcp.search_messages("most recent messages")["sessions"][0]["messages"][0]["message_id"] == 2
        assert mcp.get_conversation(1)["messages"][-1]["message_id"] == 2
    finally:
        writer.close()


def test_backfill_mixed_units_inclusive_bounds_and_frozen_paging(live):
    with sqlite3.connect(live) as con:
        for rowid, date in [(2, (START + 200) * 10**9), (3, START + 200), (99, START - 100)]:
            con.execute(
                "INSERT INTO message (ROWID,text,date,is_from_me,associated_message_type) "
                "VALUES (?,'synthetic',?,1,0)", (rowid, date),
            )
            con.execute("INSERT INTO chat_message_join VALUES (1,?)", (rowid,))
        con.execute("INSERT INTO chat VALUES (2,45)")
        con.execute("INSERT INTO chat_message_join VALUES (2,3)")
    page = mcp.recent_messages(since_ts=apple_to_unix(START), before_ts=apple_to_unix(START + 200), limit=2)
    assert [(m["message_id"], m["chat_id"]) for m in page["messages"]] == [(3, 2), (3, 1)]
    assert page["has_more"] is True
    rest = mcp.recent_messages(since_ts=page["since_ts"], before_ts=page["before_ts"],
                               limit=2, offset=page["next_offset"])
    assert [m["message_id"] for m in rest["messages"]] == [2, 1]
    assert rest["has_more"] is False and rest["next_offset"] is None
    scoped = mcp.recent_messages(since_ts=0, chat_id=2)
    assert [(m["message_id"], m["chat_id"]) for m in scoped["messages"]] == [(3, 2)]


def test_attributed_attachment_system_and_reaction_rows(live, monkeypatch):
    with sqlite3.connect(live) as con:
        con.execute("UPDATE message SET text=NULL, attributedBody=X'0102' WHERE ROWID=1")
        for rowid, reaction in [(2, 0), (3, 2000), (4, 0)]:
            con.execute(
                "INSERT INTO message (ROWID,date,is_from_me,associated_message_type) VALUES (?,?,0,?)",
                (rowid, START + rowid, reaction),
            )
            con.execute("INSERT INTO chat_message_join VALUES (1,?)", (rowid,))
        con.execute("INSERT INTO attachment (ROWID,filename) VALUES (1,'not-read.bin')")
        con.execute("INSERT INTO message_attachment_join VALUES (2,1)")
        con.execute("UPDATE message SET text='synthetic reaction' WHERE ROWID=3")
    monkeypatch.setattr(source, "decode_attributed_body", lambda blob: "synthetic decoded body")
    rows = mcp.recent_messages(since_ts=0)["messages"]
    assert [m["message_id"] for m in rows] == [2, 1]
    assert rows[0]["text"] is None and rows[0]["has_attachment"] is True
    assert rows[1]["text"] == "synthetic decoded body"
    # Source freshness includes rows without text, even excluded system rows.
    assert mcp.recent_messages(since_ts=0)["newest_source_ts"] == apple_to_unix(START + 4)

    def forbidden(blob):
        pytest.fail("metadata-only read decoded a body")

    monkeypatch.setattr(source, "decode_attributed_body", forbidden)
    metadata = mcp.recent_messages(since_ts=0, include_text=False)
    assert all("text" not in row for row in metadata["messages"])


@pytest.mark.parametrize("kwargs", [
    {"limit": 0}, {"limit": 101}, {"limit": True}, {"offset": -1},
    {"offset": 1.5}, {"since_ts": float("nan")}, {"before_ts": float("inf")},
    {"since_ts": 2, "before_ts": 1}, {"since_ts": True},
    {"include_text": "false"}, {"chat_id": "1"},
])
def test_invalid_arguments_never_open_source(monkeypatch, kwargs):
    monkeypatch.setattr(mcp, "_live_source_path", lambda: pytest.fail("opened invalid request"))
    with pytest.raises(ValueError):
        mcp.recent_messages(**kwargs)


def test_empty_source_and_empty_window_are_distinct(live):
    assert mcp.recent_messages(since_ts=apple_to_unix(START + 1))["messages"] == []
    assert mcp.recent_messages(since_ts=apple_to_unix(START + 1))["newest_source_ts"] is not None
    with sqlite3.connect(live) as con:
        con.execute("DELETE FROM message")
    empty = mcp.recent_messages()
    assert empty["newest_source_ts"] is None and empty["messages"] == []
    assert empty["since_ts"] == empty["observed_at_ts"] - 86400
    assert empty["before_ts"] == empty["observed_at_ts"]


def test_errors_are_explicit_and_never_fall_back(live, monkeypatch):
    with sqlite3.connect(live) as con:
        con.execute("UPDATE message SET text=NULL, attributedBody=X'0102'")
    monkeypatch.setattr(source, "decode_attributed_body", lambda blob: None)
    with pytest.raises(RuntimeError, match="decoding failed"):
        mcp.recent_messages(since_ts=0)
    with sqlite3.connect(live) as con:
        con.execute("ALTER TABLE message RENAME COLUMN date TO changed")
    with pytest.raises(RuntimeError, match="SchemaDriftError"):
        mcp.recent_messages()
    monkeypatch.setenv("SEAGLASS_CHAT_DB_SOURCE", str(live.parent / "missing.db"))
    with pytest.raises(RuntimeError, match="Full Disk Access"):
        mcp.recent_messages()
    assert not (live.parent / "missing.db").exists()


def test_native_permission_and_sql_failures_are_explicit(live, monkeypatch):
    def denied(path):
        raise PermissionError("synthetic access denied")

    monkeypatch.setattr(mcp, "connect_readonly", denied)
    with pytest.raises(RuntimeError, match="actual MCP/Gateway launcher"):
        mcp.recent_messages()


def test_freshness_diagnostics_do_not_initialize_the_engine(live, monkeypatch):
    status = mcp._staleness(apple_to_unix(START - 1))
    assert status["n_messages_since_index"] == 1 and status["stale"] is True
    assert status["newest_source_ts"] == apple_to_unix(START)
    monkeypatch.setenv("SEAGLASS_CHAT_DB_SOURCE", str(live.parent / "missing.db"))
    unavailable = mcp._staleness(apple_to_unix(START - 1))
    assert unavailable["stale"] is None and unavailable["live_chat_readable"] is False
    assert "live_chat_error" in unavailable


def test_mcp_dispatch_validates_types_and_returns_explicit_errors(live):
    success = asyncio.run(mcp.server.call_tool("recent_messages", {"since_ts": 0, "include_text": False}))
    assert not success.is_error
    with pytest.raises(Exception, match="limit|integer"):
        asyncio.run(mcp.server.call_tool("recent_messages", {"limit": True}))
    with sqlite3.connect(live) as con:
        con.execute("DROP TABLE message")
    with pytest.raises(Exception, match="Live chat.db read failed"):
        asyncio.run(mcp.server.call_tool("recent_messages", {}))


def test_readonly_and_query_only_protect_source_and_memory(live):
    con = connect_readonly(live)
    try:
        assert con.execute("PRAGMA query_only").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            con.execute("DELETE FROM im.message")
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            con.execute("CREATE TABLE accidental_write (value)")
        con.execute("PRAGMA query_only=OFF")
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            con.execute("DELETE FROM im.message")
    finally:
        con.close()


def test_source_resolution_never_uses_snapshot(tmp_path, monkeypatch):
    monkeypatch.delenv("SEAGLASS_CHAT_DB_SOURCE", raising=False)
    configured = tmp_path / "configured.db"
    monkeypatch.setattr("seaglass.app.config.load_config", lambda: types.SimpleNamespace(chat_db_source=str(configured)))
    assert mcp._live_source_path() == configured
    explicit = tmp_path / "explicit.db"
    monkeypatch.setenv("SEAGLASS_CHAT_DB_SOURCE", str(explicit))
    assert mcp._live_source_path() == explicit


def test_filters_only_search_date_sender_media_and_session_paging(live):
    with sqlite3.connect(live) as con:
        for rowid, day in [(2, 1), (3, 2)]:
            con.execute(
                "INSERT INTO message (ROWID,text,date,is_from_me,associated_message_type) "
                "VALUES (?,'synthetic outgoing',?,1,0)", (rowid, START + day * 86400),
            )
            con.execute("INSERT INTO chat_message_join VALUES (1,?)", (rowid,))
    result = mcp.search_messages("latest messages from me", max_sessions=1)
    assert result["served_from"] == "live_chat_db" and result["ordering"] == "recent"
    assert result["index_stale"] is None  # not inferred without opening the index
    assert result["has_more"] is True and result["total_sessions"] == 2
    assert [m["message_id"] for m in result["sessions"][0]["messages"]] == [3]
    page = mcp.search_messages("latest messages from me", max_sessions=1, offset=1)
    assert [m["message_id"] for m in page["sessions"][0]["messages"]] == [2]
    assert page["has_more"] is False
    assert mcp.search_messages("latest photos")["n_results"] == 0


def test_named_filters_browse_live_without_models(live, monkeypatch):
    from seaglass.imessage.contacts import Contact, ContactIndex

    contacts = ContactIndex([Contact("synthetic", "Zorina", ("synthetic@example.invalid",))])
    monkeypatch.setattr(mcp, "_get_contact_index", lambda: contacts)
    result = mcp.search_messages("latest from Zorina")
    assert result["n_results"] == 1
    assert result["sessions"][0]["messages"][0]["sender"] == "Zorina"
    assert result["effective_filters"]["people_sender"] == ["synthetic@example.invalid"]


def test_date_only_search_reads_the_requested_live_month(live):
    from datetime import datetime

    month_query = datetime.fromtimestamp(apple_to_unix(START)).strftime("%B %Y")
    result = mcp.search_messages(month_query)
    assert result["n_results"] == 1 and result["served_from"] == "live_chat_db"
    assert result["effective_filters"]["date_from"] <= apple_to_unix(START)
    assert result["effective_filters"]["date_to"] >= apple_to_unix(START)


def test_cold_tool_listing_and_live_calls_cannot_import_models_or_open_index(live):
    script = r'''
import asyncio, importlib.abc, json, os, resource, sys, time
class NoHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        banned = ("seaglass.index.", "seaglass.search.rank", "seaglass.search.rerank",
                  "seaglass.search.retrieve", "seaglass.app.engine",
                  "mlx", "mlx_embeddings", "transformers", "safetensors", "sqlite_vec")
        if any(fullname == item or fullname.startswith(item) for item in banned):
            raise AssertionError("Heavy import: " + fullname)
sys.meta_path.insert(0, NoHeavyImports())
def audit(event, args):
    if event == "sqlite3.connect" and args[0] != ":memory:":
        raise AssertionError("Unexpected database connection")
    if event == "open" and isinstance(args[0], (str, bytes)):
        name = os.fsdecode(args[0])
        if name.endswith((".safetensors", ".npz", ".gguf")):
            raise AssertionError("Model weights opened")
sys.addaudithook(audit)
start = time.perf_counter()
import seaglass.mcp_server as m
tools = asyncio.run(m.server.list_tools())
assert "recent_messages" in [tool.name for tool in tools]
from seaglass.imessage.contacts import ContactIndex
ContactIndex.load = classmethod(lambda cls: ContactIndex([]))
metadata = m.recent_messages(since_ts=0, include_text=False)
assert metadata["messages"] and "text" not in metadata["messages"][0]
assert m.search_messages("most recent messages")["n_results"] == 1
assert m.get_conversation(1)["n_messages"] == 1
assert all(getattr(m, name) is None for name in
           ("_index_con", "_chat_con", "_embedding_model", "_reranker", "_engine"))
print(json.dumps({"elapsed_s": round(time.perf_counter()-start, 3),
                  "max_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  "heavy_state": False, "tool_count": len(tools)}))
'''
    env = dict(os.environ, SEAGLASS_CHAT_DB_SOURCE=str(live))
    process = subprocess.run([sys.executable, "-c", script], env=env, text=True,
                             capture_output=True, timeout=30)
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["heavy_state"] is False
    assert "loading" not in process.stderr


def test_registered_schema_exposes_contract_bounds():
    tools = asyncio.run(mcp.server.list_tools())
    schema = next(tool.input_schema for tool in tools if tool.name == "recent_messages")
    assert set(schema["properties"]) == {"since_ts", "before_ts", "limit", "offset", "chat_id", "include_text"}
    assert schema["properties"]["limit"]["maximum"] == 100
