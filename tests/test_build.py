"""Unit tests for seaglass.index.build -- synthetic chat.db fixture, no
live chat.db, no MLX network dependency (a small deterministic fake
embedding model stands in for EmbeddingModel).
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from seaglass.index.build import build_index, iter_chunks_by_chat, open_index_db
from seaglass.index.embed import l2_normalize


class FakeEmbeddingModel:
    """Deterministic, network-free stand-in for EmbeddingModel. Embeds by
    hashing text to a seeded random unit vector -- stable across calls
    within a test, distinct per distinct text.
    """

    def embed(self, texts):
        vectors = []
        for text in texts:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            vectors.append(rng.normal(size=384))
        return l2_normalize(np.array(vectors, dtype=np.float32))


def _build_fixture_chat_db(tmp_path, n_chats=2, n_messages_per_chat=6):
    """A synthetic chat.db on disk (build_index opens chat_db_path via
    connect_readonly, which ATTACHes a real file path).
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
    con.execute("INSERT INTO handle (ROWID, id) VALUES (1, '+15551234567')")
    rowid = 1
    apple_epoch_seconds_start = 700000000  # arbitrary, plausible seconds-era date
    for chat_id in range(1, n_chats + 1):
        con.execute("INSERT INTO chat (ROWID, style) VALUES (?, 45)", (chat_id,))
        for i in range(n_messages_per_chat):
            date = apple_epoch_seconds_start + i * 30  # 30s apart, well within any gap threshold
            con.execute(
                "INSERT INTO message (ROWID, text, attributedBody, date, date_edited, "
                "date_retracted, is_from_me, handle_id, associated_message_type) "
                "VALUES (?, ?, NULL, ?, NULL, NULL, ?, 1, 0)",
                (rowid, f"chat {chat_id} message {i}", date, i % 2),
            )
            con.execute(
                "INSERT INTO chat_message_join (chat_id, message_id) VALUES (?, ?)",
                (chat_id, rowid),
            )
            rowid += 1
    con.commit()
    con.close()
    return chat_db_path


class TestIterChunksByChat:
    def test_yields_chunks_grouped_and_ordered_by_chat(self, tmp_path):
        chat_db_path = _build_fixture_chat_db(tmp_path, n_chats=2, n_messages_per_chat=4)
        from seaglass.imessage.source import connect_readonly

        con = connect_readonly(chat_db_path)
        results = list(iter_chunks_by_chat(con))
        chat_ids_seen = [chunk.chat_id for chunk, _ in results]
        assert chat_ids_seen == sorted(chat_ids_seen)
        assert len(results) >= 2  # at least one chunk per chat


class TestBuildIndex:
    def test_fresh_build_writes_all_chunks(self, tmp_path):
        chat_db_path = _build_fixture_chat_db(tmp_path, n_chats=2, n_messages_per_chat=5)
        index_db_path = tmp_path / "index.db"
        written = build_index(
            chat_db_path, index_db_path, embedding_model=FakeEmbeddingModel(), batch_size=3
        )
        assert written > 0

        con = open_index_db(index_db_path)
        chunk_count = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert chunk_count == written
        vec_count = con.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0]
        assert vec_count == written
        fts_count = con.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
        assert fts_count == written
        # chunk_message rows cover every message in the fixture (10 total)
        msg_count = con.execute("SELECT COUNT(*) FROM chunk_message").fetchone()[0]
        assert msg_count >= 10

    def test_meta_records_calibration_and_versions(self, tmp_path):
        chat_db_path = _build_fixture_chat_db(tmp_path, n_chats=1, n_messages_per_chat=3)
        index_db_path = tmp_path / "index.db"
        build_index(chat_db_path, index_db_path, embedding_model=FakeEmbeddingModel())

        con = open_index_db(index_db_path)
        rows = dict(con.execute("SELECT key, value FROM meta").fetchall())
        assert "int8_absmax" in rows
        assert "embed_version" in rows
        assert "semantic_format_version" in rows
        assert "lexical_format_version" in rows
        assert "build_cursor" in rows
        assert int(rows["build_cursor"]) > 0

    def test_calibration_samples_multiple_chunks_not_just_the_first(self, tmp_path, monkeypatch):
        # Regression test for IMPROVEMENT-11: calibration used to run on
        # a single-element list (the very first rendered chunk), even
        # though CALIBRATION_SAMPLE_SIZE declares a much larger target.
        # Lower the target so a small fixture corpus can still exceed it,
        # and assert _ensure_calibration is invoked with more than one
        # sample text when more than one chunk is available.
        import seaglass.index.build as build_mod

        monkeypatch.setattr(build_mod, "CALIBRATION_SAMPLE_SIZE", 5)
        seen_sample_sizes = []
        real_ensure_calibration = build_mod._ensure_calibration

        def spy(index_con, embedding_model, sample_texts):
            seen_sample_sizes.append(len(sample_texts))
            return real_ensure_calibration(index_con, embedding_model, sample_texts)

        monkeypatch.setattr(build_mod, "_ensure_calibration", spy)

        chat_db_path = _build_fixture_chat_db(tmp_path, n_chats=3, n_messages_per_chat=10)
        index_db_path = tmp_path / "index.db"
        build_index(chat_db_path, index_db_path, embedding_model=FakeEmbeddingModel(), batch_size=3)

        assert len(seen_sample_sizes) == 1  # calibration only runs once
        assert seen_sample_sizes[0] > 1

    def test_rerunning_a_completed_build_writes_nothing_new(self, tmp_path):
        chat_db_path = _build_fixture_chat_db(tmp_path, n_chats=2, n_messages_per_chat=5)
        index_db_path = tmp_path / "index.db"
        model = FakeEmbeddingModel()
        first = build_index(chat_db_path, index_db_path, embedding_model=model, batch_size=3)
        second = build_index(chat_db_path, index_db_path, embedding_model=model, batch_size=3)
        assert first > 0
        assert second == 0

        con = open_index_db(index_db_path)
        chunk_count = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert chunk_count == first  # no duplicates

    def test_resume_after_partial_batch_continues_correctly(self, tmp_path):
        chat_db_path = _build_fixture_chat_db(tmp_path, n_chats=3, n_messages_per_chat=5)
        index_db_path = tmp_path / "index.db"
        model = FakeEmbeddingModel()

        # Simulate a mid-build stop: limit_chunks caps this run short.
        first = build_index(
            chat_db_path, index_db_path, embedding_model=model, batch_size=2, limit_chunks=1
        )
        assert 0 < first

        con = open_index_db(index_db_path)
        partial_count = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert partial_count == first

        # Resume: should pick up where it left off and finish the rest.
        second = build_index(chat_db_path, index_db_path, embedding_model=model, batch_size=2)
        assert second > 0

        con = open_index_db(index_db_path)
        total_count = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        # every chunk id should be unique (no collisions/duplicates across runs)
        ids = [row[0] for row in con.execute("SELECT id FROM chunks ORDER BY id")]
        assert ids == list(range(1, total_count + 1))

    def test_new_messages_merged_into_an_existing_non_tail_chunk_are_indexed(self, tmp_path):
        """Regression test: new messages arriving in a chat whose last
        chunk was already committed used to be silently dropped, because
        the old design skipped any chunk id <= build_cursor based purely
        on global sequential *position*, with no way to notice that the
        *content* at an already-committed position had changed. Here,
        chat 1 is NOT the last chat touched (chat 2 exists after it), so
        its tail chunk is a "non-last-globally" position -- exactly the
        case that reproduced the user's "N new messages never clears"
        bug: new messages land inside an already-chunked position rather
        than appending as a brand new trailing chunk.
        """
        chat_db_path = _build_fixture_chat_db(tmp_path, n_chats=2, n_messages_per_chat=5)
        index_db_path = tmp_path / "index.db"
        model = FakeEmbeddingModel()
        first = build_index(chat_db_path, index_db_path, embedding_model=model)
        assert first > 0

        con = open_index_db(index_db_path)
        chat1_msg_count_before = con.execute(
            "SELECT COUNT(*) FROM chunk_message cm JOIN chunks c ON c.id = cm.chunk_id "
            "WHERE c.chat_id = 1"
        ).fetchone()[0]
        assert chat1_msg_count_before == 5

        # Append 2 new messages to chat 1 (not the globally-last chat),
        # close enough in time that the chunker merges them into the
        # SAME existing chunk rather than opening a new one.
        chat_con = sqlite3.connect(chat_db_path)
        apple_epoch_seconds_start = 700000000
        for i, rowid in enumerate((100, 101)):
            date = apple_epoch_seconds_start + (5 + i) * 30
            chat_con.execute(
                "INSERT INTO message (ROWID, text, attributedBody, date, date_edited, "
                "date_retracted, is_from_me, handle_id, associated_message_type) "
                "VALUES (?, ?, NULL, ?, NULL, NULL, 0, 1, 0)",
                (rowid, f"chat 1 new message {i}", date),
            )
            chat_con.execute(
                "INSERT INTO chat_message_join (chat_id, message_id) VALUES (1, ?)", (rowid,)
            )
        chat_con.commit()
        chat_con.close()

        second = build_index(chat_db_path, index_db_path, embedding_model=model)
        assert second > 0, "new messages merged into an existing chunk must still be written"

        con = open_index_db(index_db_path)
        chat1_msg_count_after = con.execute(
            "SELECT COUNT(*) FROM chunk_message cm JOIN chunks c ON c.id = cm.chunk_id "
            "WHERE c.chat_id = 1"
        ).fetchone()[0]
        assert chat1_msg_count_after == 7

        # The rewritten chunk's rows must be consistent across all four
        # derived tables (no stale duplicates, no orphans).
        chunk_ids = [
            row[0] for row in con.execute("SELECT id FROM chunks WHERE chat_id = 1")
        ]
        assert len(chunk_ids) == len(set(chunk_ids))
        for cid in chunk_ids:
            vec_count = con.execute(
                "SELECT COUNT(*) FROM chunks_vec WHERE rowid = ?", (cid,)
            ).fetchone()[0]
            assert vec_count == 1
            fts_count = con.execute(
                "SELECT COUNT(*) FROM chunks_fts WHERE rowid = ?", (cid,)
            ).fetchone()[0]
            assert fts_count == 1

        # New messages must actually be findable via FTS.
        hits = con.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'new'"
        ).fetchall()
        assert len(hits) > 0

        # A third, no-op build should now find nothing new.
        third = build_index(chat_db_path, index_db_path, embedding_model=model)
        assert third == 0

    def test_int8_embeddings_are_retrievable_via_constrained_knn(self, tmp_path):
        chat_db_path = _build_fixture_chat_db(tmp_path, n_chats=1, n_messages_per_chat=4)
        index_db_path = tmp_path / "index.db"
        build_index(chat_db_path, index_db_path, embedding_model=FakeEmbeddingModel())

        con = open_index_db(index_db_path)
        # query with an arbitrary stored vector reconstructed via a fresh fake embed
        vec = FakeEmbeddingModel().embed(["chat 1 message 0"])[0]
        from seaglass.index.embed import quantize_int8

        absmax = float(con.execute("SELECT value FROM meta WHERE key='int8_absmax'").fetchone()[0])
        q = quantize_int8(vec.reshape(1, -1), absmax)[0]
        rows = con.execute(
            "SELECT rowid FROM chunks_vec WHERE embedding MATCH vec_int8(?) AND k = 3",
            (q.tobytes(),),
        ).fetchall()
        assert len(rows) > 0


class TestAttachmentPlaceIntegration:
    """Phase 2 exif.py wiring: a geotagged attachment on disk should end
    up reverse-geocoded into attachment_place and inlined into the
    chunk's lexical body, without ever touching body_semantic.
    """

    def test_geotagged_attachment_populates_attachment_place_and_lexical_body(self, tmp_path):
        from PIL import Image
        from PIL.ExifTags import IFD

        chat_db_path = _build_fixture_chat_db(tmp_path, n_chats=1, n_messages_per_chat=1)
        con = sqlite3.connect(chat_db_path)
        photo_path = tmp_path / "IMG_0001.jpg"
        img = Image.new("RGB", (4, 4), color="red")
        exif = img.getexif()
        gps_ifd = exif.get_ifd(IFD.GPSInfo)
        gps_ifd[1] = "N"
        gps_ifd[2] = (37.0, 46.0, 26.4)
        gps_ifd[3] = "W"
        gps_ifd[4] = (122.0, 25.0, 9.6)
        exif[IFD.GPSInfo] = gps_ifd
        img.save(photo_path, exif=exif)

        con.execute("INSERT INTO attachment (ROWID, filename) VALUES (1, ?)", (str(photo_path),))
        con.execute("INSERT INTO message_attachment_join (message_id, attachment_id) VALUES (1, 1)")
        con.commit()
        con.close()

        index_db_path = tmp_path / "index.db"
        build_index(chat_db_path, index_db_path, embedding_model=FakeEmbeddingModel())

        con = open_index_db(index_db_path)
        place_row = con.execute("SELECT place FROM attachment_place WHERE attachment_id = 1").fetchone()
        assert place_row is not None
        assert "United States" in place_row[0]

        lexical_match = con.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?",
            (f'"{place_row[0]}"',),
        ).fetchall()
        assert len(lexical_match) > 0

    def test_missing_attachment_file_recorded_in_attachment_retry(self, tmp_path):
        chat_db_path = _build_fixture_chat_db(tmp_path, n_chats=1, n_messages_per_chat=1)
        con = sqlite3.connect(chat_db_path)
        con.execute(
            "INSERT INTO attachment (ROWID, filename) VALUES (1, ?)",
            (str(tmp_path / "does_not_exist.jpg"),),
        )
        con.execute("INSERT INTO message_attachment_join (message_id, attachment_id) VALUES (1, 1)")
        con.commit()
        con.close()

        index_db_path = tmp_path / "index.db"
        build_index(chat_db_path, index_db_path, embedding_model=FakeEmbeddingModel())

        con = open_index_db(index_db_path)
        retry_row = con.execute(
            "SELECT attachment_id FROM attachment_retry WHERE attachment_id = 1"
        ).fetchone()
        assert retry_row is not None
        place_row = con.execute("SELECT * FROM attachment_place WHERE attachment_id = 1").fetchone()
        assert place_row is None
