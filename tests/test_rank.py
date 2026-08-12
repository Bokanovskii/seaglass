"""Unit tests for seaglass.search.rank -- rerank scoring, session
aggregation (dedup + group-by-(chat_id, day) + score summation), and
±2 time-adjacent context expansion.
"""

from __future__ import annotations

from seaglass.index.build import build_index, open_index_db
from seaglass.search.rank import (
    RankedChunk,
    aggregate_sessions,
    expand_sessions,
    rerank_candidates,
)
from seaglass.search.retrieve import RetrievalResult

from conftest import FakeEmbeddingModel, build_fixture_chat_db


class FakeReranker:
    """Deterministic stand-in for CrossEncoderReranker: scores a pair by
    how many query words appear in the candidate text, so tests can
    assert on ordering without loading MLX.
    """

    def score(self, pairs):
        scores = []
        for query, text in pairs:
            query_words = set(query.lower().split())
            text_words = set(text.lower().split())
            scores.append(float(len(query_words & text_words)))
        return scores


def _one_chat_index(tmp_path, n_messages=20):
    chat_db_path = build_fixture_chat_db(
        tmp_path,
        chats=[
            {
                "chat_id": 1,
                "handles": ["+15551234567"],
                "messages": [
                    (f"message number {i} about dinner plans", 700000000 + i * 30, i % 2, 0)
                    for i in range(n_messages)
                ],
            }
        ],
    )
    index_db_path = tmp_path / "index.db"
    build_index(
        chat_db_path,
        index_db_path,
        embedding_model=FakeEmbeddingModel(),
        chunker_kwargs={"max_messages": 3},  # force multiple chunks
    )
    return index_db_path


class TestRerankCandidates:
    def test_orders_by_word_overlap_and_truncates_to_top_k(self, tmp_path):
        index_db_path = _one_chat_index(tmp_path)
        index_con = open_index_db(index_db_path)
        all_ids = [row[0] for row in index_con.execute("SELECT id FROM chunks").fetchall()]
        fused = [RetrievalResult(chunk_id=cid, rrf_score=1.0) for cid in all_ids]

        ranked = rerank_candidates(index_con, "dinner plans", fused, FakeReranker(), top_k=3)
        assert len(ranked) == 3
        assert all(isinstance(r, RankedChunk) for r in ranked)
        # descending by rerank_score
        scores = [r.rerank_score for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_empty_fused_list_returns_empty(self, tmp_path):
        index_db_path = _one_chat_index(tmp_path)
        index_con = open_index_db(index_db_path)
        assert rerank_candidates(index_con, "anything", [], FakeReranker()) == []


class TestAggregateSessions:
    def test_groups_by_chat_and_day_summing_scores(self):
        ranked = [
            RankedChunk(chunk_id=1, chat_id=1, start_ts=700000000, rerank_score=2.0),
            RankedChunk(chunk_id=2, chat_id=1, start_ts=700000030, rerank_score=3.0),  # same day
            RankedChunk(chunk_id=3, chat_id=1, start_ts=700000000 + 86400 * 5, rerank_score=1.0),  # different day
            RankedChunk(chunk_id=4, chat_id=2, start_ts=700000000, rerank_score=5.0),  # different chat
        ]
        sessions = aggregate_sessions(ranked, max_sessions=8)
        # chat 2's single-chunk session (score 5.0) should outrank chat 1's fused day (score 5.0 too -- tie broken by insertion order is fine)
        by_chat = {s.chat_id: s for s in sessions}
        assert set(by_chat[1].hit_chunk_ids) == {1, 2} or set(by_chat[1].hit_chunk_ids) == {3}
        total_chat1_score = sum(s.score for s in sessions if s.chat_id == 1)
        assert total_chat1_score == 6.0  # 2.0+3.0 (same day) + 1.0 (different day) split into 2 sessions
        assert by_chat[2].score == 5.0

    def test_caps_at_max_sessions(self):
        ranked = [
            RankedChunk(chunk_id=i, chat_id=i, start_ts=700000000, rerank_score=float(i))
            for i in range(1, 11)
        ]
        sessions = aggregate_sessions(ranked, max_sessions=8)
        assert len(sessions) == 8
        # highest scores kept
        assert sessions[0].score == 10.0

    def test_dedup_by_chunk_id(self):
        ranked = [
            RankedChunk(chunk_id=1, chat_id=1, start_ts=700000000, rerank_score=2.0),
            RankedChunk(chunk_id=1, chat_id=1, start_ts=700000000, rerank_score=2.0),  # duplicate
        ]
        sessions = aggregate_sessions(ranked)
        assert len(sessions) == 1
        assert sessions[0].score == 2.0
        assert sessions[0].hit_chunk_ids == [1]


class TestExpandSessions:
    def test_pulls_up_to_radius_neighbors_same_chat(self, tmp_path):
        index_db_path = _one_chat_index(tmp_path, n_messages=30)
        index_con = open_index_db(index_db_path)
        all_ids = sorted(row[0] for row in index_con.execute("SELECT id FROM chunks").fetchall())
        assert len(all_ids) >= 5  # need enough chunks for a meaningful expansion test

        mid_id = all_ids[len(all_ids) // 2]
        from seaglass.search.rank import Session

        session = Session(chat_id=1, day="2022-03-14", score=1.0, hit_chunk_ids=[mid_id], context_chunk_ids=[])
        expand_sessions(index_con, [session], radius=2)

        pos = all_ids.index(mid_id)
        expected_neighbors = set(all_ids[max(0, pos - 2): pos]) | set(all_ids[pos + 1: pos + 3])
        assert set(session.context_chunk_ids) == expected_neighbors
        assert mid_id not in session.context_chunk_ids

    def test_edge_of_chat_truncates_gracefully(self, tmp_path):
        index_db_path = _one_chat_index(tmp_path, n_messages=30)
        index_con = open_index_db(index_db_path)
        all_ids = sorted(row[0] for row in index_con.execute("SELECT id FROM chunks").fetchall())
        first_id = all_ids[0]

        from seaglass.search.rank import Session

        session = Session(chat_id=1, day="2022-03-14", score=1.0, hit_chunk_ids=[first_id], context_chunk_ids=[])
        expand_sessions(index_con, [session], radius=2)
        # no crash, no negative-index wraparound; only forward neighbors
        assert set(session.context_chunk_ids).issubset(set(all_ids))
        assert first_id not in session.context_chunk_ids
