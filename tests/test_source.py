"""Unit tests for seaglass.imessage.source -- pure logic, no live chat.db needed.

Builds a small synthetic in-memory database matching chat.db's shape so
these tests run identically in CI regardless of what's on the machine.
"""

from __future__ import annotations

import sqlite3
import time

import pytest

from seaglass.imessage.source import (
    APPLE_EPOCH_UNIX,
    SchemaDriftError,
    apple_to_unix,
    assert_schema,
    iter_messages,
)


class TestAppleToUnix:
    def test_seconds_era(self):
        # 2019-01-01 in Apple-epoch seconds (pre-nanosecond era)
        apple_seconds = 567993600
        unix = apple_to_unix(apple_seconds)
        assert unix == pytest.approx(apple_seconds + APPLE_EPOCH_UNIX)

    def test_nanoseconds_era(self):
        # Same instant, expressed in nanoseconds (modern rows)
        apple_seconds = 567993600
        apple_ns = apple_seconds * 1_000_000_000
        unix = apple_to_unix(apple_ns)
        assert unix == pytest.approx(apple_seconds + APPLE_EPOCH_UNIX)

    def test_threshold_boundary_plausible_today(self):
        # "Now", expressed in nanoseconds, should land near the real current time
        now_unix = time.time()
        apple_seconds_now = now_unix - APPLE_EPOCH_UNIX
        apple_ns_now = apple_seconds_now * 1_000_000_000
        result = apple_to_unix(apple_ns_now)
        assert result == pytest.approx(now_unix, abs=1.0)


def _build_fixture_db() -> sqlite3.Connection:
    """A minimal synthetic chat.db-shaped database, attached as `im`."""
    con = sqlite3.connect(":memory:")
    con.execute(f"ATTACH DATABASE ':memory:' AS im")
    con.executescript(
        """
        CREATE TABLE im.message (
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
        CREATE TABLE im.chat (ROWID INTEGER PRIMARY KEY, style INTEGER);
        CREATE TABLE im.chat_message_join (chat_id INTEGER, message_id INTEGER);
        CREATE TABLE im.chat_handle_join (chat_id INTEGER, handle_id INTEGER);
        CREATE TABLE im.handle (ROWID INTEGER PRIMARY KEY, id TEXT);
        CREATE TABLE im.attachment (ROWID INTEGER PRIMARY KEY, filename TEXT);
        CREATE TABLE im.message_attachment_join (message_id INTEGER, attachment_id INTEGER);
        """
    )
    return con


class TestAssertSchema:
    def test_passes_on_matching_schema(self):
        con = _build_fixture_db()
        assert_schema(con)  # must not raise

    def test_raises_on_missing_column(self):
        con = _build_fixture_db()
        con.execute("ALTER TABLE im.message RENAME COLUMN date_edited TO renamed_col")
        with pytest.raises(SchemaDriftError, match="date_edited"):
            assert_schema(con)

    def test_raises_on_missing_table(self):
        con = sqlite3.connect(":memory:")
        con.execute("ATTACH DATABASE ':memory:' AS im")
        with pytest.raises(SchemaDriftError):
            assert_schema(con)


class TestIterMessages:
    def test_filters_tapbacks_and_resolves_handle(self):
        con = _build_fixture_db()
        con.execute("INSERT INTO im.handle(ROWID, id) VALUES (1, '+15551234567')")
        con.execute(
            """
            INSERT INTO im.message
                (ROWID, text, attributedBody, date, date_edited, date_retracted,
                 is_from_me, handle_id, associated_message_type)
            VALUES
                (1, 'hello world', NULL, 700000000000000000, NULL, NULL, 0, 1, 0),
                (2, NULL, NULL, 700000001000000000, NULL, NULL, 1, NULL, 2000)
            """
        )
        con.executemany(
            "INSERT INTO im.chat_message_join(chat_id, message_id) VALUES (?, ?)",
            [(10, 1), (10, 2)],
        )
        messages = list(iter_messages(con))
        # associated_message_type=2000 (tapback) must be excluded
        assert len(messages) == 1
        assert messages[0].rowid == 1
        assert messages[0].chat_id == 10
        assert messages[0].handle == "+15551234567"
        assert messages[0].text == "hello world"
        assert messages[0].is_from_me is False

    def test_falls_back_to_attributed_body_when_text_is_null(self, monkeypatch):
        con = _build_fixture_db()
        con.execute(
            """
            INSERT INTO im.message
                (ROWID, text, attributedBody, date, date_edited, date_retracted,
                 is_from_me, handle_id, associated_message_type)
            VALUES (1, NULL, X'deadbeef', 700000000000000000, NULL, NULL, 0, NULL, 0)
            """
        )
        con.execute("INSERT INTO im.chat_message_join(chat_id, message_id) VALUES (5, 1)")

        import seaglass.imessage.source as source_mod

        monkeypatch.setattr(source_mod, "decode_attributed_body", lambda blob: "decoded text")
        messages = list(iter_messages(con))
        assert len(messages) == 1
        assert messages[0].text == "decoded text"

    def test_filters_by_chat_id(self):
        con = _build_fixture_db()
        con.execute(
            """
            INSERT INTO im.message
                (ROWID, text, attributedBody, date, date_edited, date_retracted,
                 is_from_me, handle_id, associated_message_type)
            VALUES
                (1, 'chat A', NULL, 1, NULL, NULL, 0, NULL, 0),
                (2, 'chat B', NULL, 2, NULL, NULL, 0, NULL, 0)
            """
        )
        con.executemany(
            "INSERT INTO im.chat_message_join(chat_id, message_id) VALUES (?, ?)",
            [(1, 1), (2, 2)],
        )
        messages = list(iter_messages(con, chat_id=1))
        assert len(messages) == 1
        assert messages[0].text == "chat A"

    def test_has_attachment_flag(self):
        con = _build_fixture_db()
        con.execute(
            """
            INSERT INTO im.message
                (ROWID, text, attributedBody, date, date_edited, date_retracted,
                 is_from_me, handle_id, associated_message_type)
            VALUES (1, 'photo', NULL, 1, NULL, NULL, 0, NULL, 0)
            """
        )
        con.execute("INSERT INTO im.chat_message_join(chat_id, message_id) VALUES (1, 1)")
        con.execute("INSERT INTO im.attachment(ROWID, filename) VALUES (1, 'IMG_0001.HEIC')")
        con.execute("INSERT INTO im.message_attachment_join(message_id, attachment_id) VALUES (1, 1)")
        messages = list(iter_messages(con))
        assert messages[0].has_attachment is True
