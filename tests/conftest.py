"""Shared pytest fixtures/helpers for building synthetic chat.db + index.db
pairs without any live data or network dependency.
"""

from __future__ import annotations

import sqlite3

import numpy as np

from seaglass.index.embed import l2_normalize


class FakeEmbeddingModel:
    """Deterministic, network-free stand-in for EmbeddingModel. Embeds by
    hashing text to a seeded random unit vector -- stable across calls,
    distinct per distinct text.
    """

    def embed(self, texts):
        vectors = []
        for text in texts:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            vectors.append(rng.normal(size=384))
        return l2_normalize(np.array(vectors, dtype=np.float32))


def build_fixture_chat_db(tmp_path, chats):
    """Build a synthetic chat.db on disk.

    `chats` is a list of dicts:
      {"chat_id": int, "handles": [str, ...], "messages": [(text, date, is_from_me, handle_idx), ...]}

    `date` is Apple-epoch seconds (pre-nanosecond era, matching
    `apple_to_unix`'s seconds branch). `handle_idx` indexes into that
    chat's `handles` list; ignored when `is_from_me` is true.
    """
    chat_db_path = tmp_path / "chat.db"
    con = sqlite3.connect(chat_db_path)
    con.executescript(
        """
        CREATE TABLE message (
            ROWID INTEGER PRIMARY KEY,
            text TEXT,
            attributedBody BLOB,
            date INTEGER,
            date_edited INTEGER,
            date_retracted INTEGER,
            is_from_me INTEGER,
            handle_id INTEGER,
            associated_message_type INTEGER
        );
        CREATE TABLE chat (ROWID INTEGER PRIMARY KEY, style INTEGER);
        CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER);
        CREATE TABLE chat_handle_join (chat_id INTEGER, handle_id INTEGER);
        CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT);
        CREATE TABLE attachment (ROWID INTEGER PRIMARY KEY, filename TEXT);
        CREATE TABLE message_attachment_join (message_id INTEGER, attachment_id INTEGER);
        """
    )
    handle_rowid_by_string: dict = {}
    next_handle_rowid = 1
    rowid = 1
    for chat in chats:
        con.execute("INSERT INTO chat (ROWID, style) VALUES (?, 45)", (chat["chat_id"],))
        chat_handle_rowids = []
        for handle_str in chat["handles"]:
            if handle_str not in handle_rowid_by_string:
                handle_rowid_by_string[handle_str] = next_handle_rowid
                con.execute("INSERT INTO handle (ROWID, id) VALUES (?, ?)", (next_handle_rowid, handle_str))
                next_handle_rowid += 1
            handle_rowid = handle_rowid_by_string[handle_str]
            chat_handle_rowids.append(handle_rowid)
            con.execute(
                "INSERT INTO chat_handle_join (chat_id, handle_id) VALUES (?, ?)",
                (chat["chat_id"], handle_rowid),
            )
        for text, date, is_from_me, handle_idx in chat["messages"]:
            handle_rowid = None if is_from_me else chat_handle_rowids[handle_idx]
            con.execute(
                "INSERT INTO message (ROWID, text, attributedBody, date, date_edited, "
                "date_retracted, is_from_me, handle_id, associated_message_type) "
                "VALUES (?, ?, NULL, ?, NULL, NULL, ?, ?, 0)",
                (rowid, text, date, int(is_from_me), handle_rowid),
            )
            con.execute(
                "INSERT INTO chat_message_join (chat_id, message_id) VALUES (?, ?)",
                (chat["chat_id"], rowid),
            )
            rowid += 1
    con.commit()
    con.close()
    return chat_db_path
