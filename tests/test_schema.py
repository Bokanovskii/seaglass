"""Smoke test for index/schema.sql -- verifies it loads with sqlite-vec and
the shapes match what build.py will rely on. Not exhaustive; the schema's
correctness is really proven by build.py's integration path.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import sqlite_vec

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "seaglass" / "index" / "schema.sql"


def _fresh_db() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    con.executescript(SCHEMA_PATH.read_text())
    return con


def test_schema_loads_without_error():
    _fresh_db()  # must not raise


def test_expected_tables_exist():
    con = _fresh_db()
    tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for expected in ("chunks", "chunk_message", "meta", "attachment_retry", "attachment_place", "eval_candidate"):
        assert expected in tables


def test_chunks_fts_is_contentless_and_queryable():
    con = _fresh_db()
    con.execute("INSERT INTO chunks(id, chat_id, start_ts, end_ts) VALUES (1, 1, 100, 200)")
    con.execute("INSERT INTO chunks_fts(rowid, body) VALUES (1, 'hello world from lisbon')")
    rows = con.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'lisbon'").fetchall()
    assert rows == [(1,)]
    # contentless_delete=1 must allow deleting without re-supplying the text
    con.execute("DELETE FROM chunks_fts WHERE rowid = 1")
    rows = con.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'lisbon'").fetchall()
    assert rows == []


def test_chunks_vec_int8_insert_and_constrained_knn():
    con = _fresh_db()
    vec = bytes([1] * 384)
    con.execute(
        "INSERT INTO chunks_vec(rowid, embedding) VALUES (1, vec_int8(?))",
        (vec,),
    )
    rows = con.execute(
        "SELECT rowid FROM chunks_vec WHERE embedding MATCH vec_int8(?) AND k = 1",
        (vec,),
    ).fetchall()
    assert rows == [(1,)]
