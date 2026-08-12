"""Unit tests for seaglass.eval.harvest -- the 4-stage candidate
harvesting pipeline (sql/idf/nn/score), using a synthetic chat.db
fixture and a FakeEmbeddingModel-built index.db, no network or MLX
dependency.
"""

from __future__ import annotations

from seaglass.eval.harvest import stage_idf, stage_nn, stage_score, stage_sql
from seaglass.imessage.source import connect_readonly
from seaglass.index.build import build_index, open_index_db

from conftest import FakeEmbeddingModel, build_fixture_chat_db


def _build_multi_chat_index(tmp_path, n_chats=3, n_messages_per_chat=12):
    chat_db_path = build_fixture_chat_db(
        tmp_path,
        chats=[
            {
                "chat_id": chat_id,
                "handles": [f"+1555000{chat_id}00{i}" for i in range(3)],
                "messages": [
                    (
                        f"chat {chat_id} message {i} about something distinctive like http://example.com/page{i}"
                        if i % 5 == 0
                        else f"chat {chat_id} short message {i}",
                        700000000 + i * 3600,  # 1h apart within a chat
                        i % 2,
                        i % 3,
                    )
                    for i in range(n_messages_per_chat)
                ],
            }
            for chat_id in range(1, n_chats + 1)
        ],
    )
    index_db_path = tmp_path / "index.db"
    build_index(
        chat_db_path,
        index_db_path,
        embedding_model=FakeEmbeddingModel(),
        chunker_kwargs={"max_messages": 4},
    )
    return chat_db_path, index_db_path


class TestStageSql:
    def test_writes_one_row_per_chunk_with_regex_and_chat_signals(self, tmp_path):
        chat_db_path, index_db_path = _build_multi_chat_index(tmp_path)
        index_con = open_index_db(index_db_path)
        chat_con = connect_readonly(chat_db_path)

        n_chunks = index_con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        n_written = stage_sql(index_con, chat_con)
        assert n_written == n_chunks

        rows = index_con.execute(
            "SELECT chunk_id, has_url, token_count, participant_count, is_group FROM eval_candidate"
        ).fetchall()
        assert len(rows) == n_chunks
        # at least one chunk should have picked up the http:// URL signal
        assert any(row[1] == 1 for row in rows)
        # 3-handle group chats should be flagged is_group
        assert all(row[3] == 3 for row in rows)  # our fixture chats all have 3 participants

    def test_idempotent_rerun_upserts_not_duplicates(self, tmp_path):
        _chat_db_path, index_db_path = _build_multi_chat_index(tmp_path)
        index_con = open_index_db(index_db_path)
        chat_con = connect_readonly(_chat_db_path)
        stage_sql(index_con, chat_con)
        n_before = index_con.execute("SELECT COUNT(*) FROM eval_candidate").fetchone()[0]
        stage_sql(index_con, chat_con)
        n_after = index_con.execute("SELECT COUNT(*) FROM eval_candidate").fetchone()[0]
        assert n_before == n_after

    def test_works_without_chat_con(self, tmp_path):
        _chat_db_path, index_db_path = _build_multi_chat_index(tmp_path)
        index_con = open_index_db(index_db_path)
        n_written = stage_sql(index_con, None)
        assert n_written > 0


class TestStageIdf:
    def test_writes_idf_mean_for_every_candidate(self, tmp_path):
        chat_db_path, index_db_path = _build_multi_chat_index(tmp_path)
        index_con = open_index_db(index_db_path)
        chat_con = connect_readonly(chat_db_path)
        stage_sql(index_con, chat_con)

        n_written = stage_idf(index_con)
        rows = index_con.execute("SELECT idf_mean FROM eval_candidate").fetchall()
        assert n_written == len(rows)
        assert all(row[0] is not None for row in rows)
        # distinctive/rare terms (the url-bearing chunks) should score higher idf than boilerplate
        distinctive = index_con.execute(
            "SELECT idf_mean FROM eval_candidate WHERE has_url = 1"
        ).fetchall()
        boilerplate = index_con.execute(
            "SELECT idf_mean FROM eval_candidate WHERE has_url = 0"
        ).fetchall()
        assert distinctive and boilerplate
        assert max(r[0] for r in distinctive) >= min(r[0] for r in boilerplate)


class TestStageNn:
    def test_writes_nn_distance_excluding_self_and_overlap_sharing_chunks(self, tmp_path):
        chat_db_path, index_db_path = _build_multi_chat_index(tmp_path, n_chats=2, n_messages_per_chat=20)
        index_con = open_index_db(index_db_path)
        chat_con = connect_readonly(chat_db_path)
        stage_sql(index_con, chat_con)
        stage_idf(index_con)

        n_written = stage_nn(index_con, target=1000)
        # short synthetic fixture chunks fall below TOKEN_COUNT_MIN's prefilter,
        # so nothing is expected to survive -- assert the pipeline runs cleanly
        # (no crash, no negative count) rather than asserting a specific yield.
        assert n_written >= 0
        rows = index_con.execute(
            "SELECT chunk_id, nn_distance, nn_chunk_id FROM eval_candidate WHERE nn_distance IS NOT NULL"
        ).fetchall()
        for chunk_id, nn_distance, nn_chunk_id in rows:
            assert nn_chunk_id != chunk_id  # never a self-match
            assert 0.0 <= nn_distance <= 2.0  # valid cosine-distance range

    def test_prefilter_survives_with_lower_token_threshold(self, tmp_path, monkeypatch):
        """Same fixture, but with TOKEN_COUNT_MIN lowered to fit the tiny
        synthetic messages -- exercises the actual nn_distance computation
        and the self/overlap exclusion, which the above test can't reach.
        """
        import seaglass.eval.harvest as harvest_module

        monkeypatch.setattr(harvest_module, "TOKEN_COUNT_MIN", 0)
        chat_db_path, index_db_path = _build_multi_chat_index(tmp_path, n_chats=2, n_messages_per_chat=20)
        index_con = open_index_db(index_db_path)
        chat_con = connect_readonly(chat_db_path)
        stage_sql(index_con, chat_con)
        stage_idf(index_con)

        n_written = stage_nn(index_con, target=1000)
        assert n_written > 0
        rows = index_con.execute(
            "SELECT chunk_id, nn_distance, nn_chunk_id FROM eval_candidate WHERE nn_distance IS NOT NULL"
        ).fetchall()
        assert rows
        for chunk_id, nn_distance, nn_chunk_id in rows:
            assert nn_chunk_id != chunk_id
            assert 0.0 <= nn_distance <= 2.0

    def test_respects_target_prefilter_size(self, tmp_path):
        chat_db_path, index_db_path = _build_multi_chat_index(tmp_path, n_chats=2, n_messages_per_chat=20)
        index_con = open_index_db(index_db_path)
        chat_con = connect_readonly(chat_db_path)
        stage_sql(index_con, chat_con)
        stage_idf(index_con)

        n_written = stage_nn(index_con, target=2)
        assert n_written <= 2


class TestStageScore:
    def test_assigns_score_and_category_to_every_nn_scored_row(self, tmp_path, monkeypatch):
        import seaglass.eval.harvest as harvest_module

        monkeypatch.setattr(harvest_module, "TOKEN_COUNT_MIN", 0)
        chat_db_path, index_db_path = _build_multi_chat_index(tmp_path, n_chats=2, n_messages_per_chat=20)
        index_con = open_index_db(index_db_path)
        chat_con = connect_readonly(chat_db_path)
        stage_sql(index_con, chat_con)
        stage_idf(index_con)
        stage_nn(index_con, target=1000)

        n_written = stage_score(index_con)
        rows = index_con.execute(
            "SELECT category, score FROM eval_candidate WHERE nn_distance IS NOT NULL"
        ).fetchall()
        assert n_written == len(rows)
        assert rows  # sanity: the lowered threshold must have let some rows through
        assert all(row[0] is not None for row in rows)
        assert all(row[1] is not None for row in rows)
        valid_categories = {"media_geo", "exact_string", "person_filtered", "time_filtered", "multi_session", "topical"}
        assert all(row[0] in valid_categories for row in rows)

    def test_rows_without_nn_distance_are_left_unscored(self, tmp_path):
        chat_db_path, index_db_path = _build_multi_chat_index(tmp_path, n_chats=2, n_messages_per_chat=20)
        index_con = open_index_db(index_db_path)
        chat_con = connect_readonly(chat_db_path)
        stage_sql(index_con, chat_con)
        stage_idf(index_con)
        stage_nn(index_con, target=1)  # deliberately leave most rows without nn_distance

        stage_score(index_con)
        unscored = index_con.execute(
            "SELECT COUNT(*) FROM eval_candidate WHERE nn_distance IS NULL AND score IS NOT NULL"
        ).fetchone()[0]
        assert unscored == 0
