"""Unit tests for seaglass.search.rank -- rerank scoring, session
aggregation (dedup + group-by-(chat_id, day) + score summation), and
±2 time-adjacent context expansion.
"""

from __future__ import annotations

import pytest

from seaglass.index.build import build_index, open_index_db
from seaglass.search.rank import (
    RankedChunk,
    aggregate_sessions,
    order_sessions,
    expand_sessions,
    rerank_candidates,
)
from seaglass.search.retrieve import RetrievalResult

from conftest import FakeEmbeddingModel, build_fixture_chat_db


class FakeReranker:
    """Deterministic stand-in for CrossEncoderReranker: scores a pair by
    how many query words appear in the candidate text, so tests can
    assert on ordering without loading MLX.

    Deliberately maps overlap counts into the same negative-logit range
    the real cross-encoder actually produces (median ~-7 over real
    queries, ~98% negative) rather than a nonnegative overlap count --
    a nonnegative FakeReranker previously masked a real aggregate_sessions
    bug (raw-logit summation penalising multi-hit sessions) because the
    fake never exercised the negative regime the bug lived in.
    """

    def score(self, pairs):
        scores = []
        for query, text in pairs:
            query_words = set(query.lower().split())
            text_words = set(text.lower().split())
            overlap = len(query_words & text_words)
            scores.append(float(overlap) - 8.0)  # e.g. 0 overlap -> -8, 3 overlap -> -5
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
        from seaglass.search.rank import _sigmoid

        ranked = [
            RankedChunk(chunk_id=1, chat_id=1, start_ts=700000000, rerank_score=2.0),
            RankedChunk(chunk_id=2, chat_id=1, start_ts=700000030, rerank_score=3.0),  # same day
            RankedChunk(chunk_id=3, chat_id=1, start_ts=700000000 + 86400 * 5, rerank_score=1.0),  # different day
            RankedChunk(chunk_id=4, chat_id=2, start_ts=700000000, rerank_score=5.0),  # different chat
        ]
        sessions = aggregate_sessions(ranked, max_sessions=8)
        # chat 2's single-chunk session should outrank either of chat 1's split-day sessions
        by_chat = {s.chat_id: s for s in sessions}
        assert set(by_chat[1].hit_chunk_ids) == {1, 2} or set(by_chat[1].hit_chunk_ids) == {3}
        total_chat1_score = sum(s.score for s in sessions if s.chat_id == 1)
        # scores are sigmoid-mapped before summing (see rank.py's aggregate_sessions
        # docstring: raw-logit summation penalises multi-hit sessions), so the
        # same-day pair (2.0, 3.0) sums their *sigmoids*, not the raw logits.
        assert total_chat1_score == pytest.approx(_sigmoid(2.0) + _sigmoid(3.0) + _sigmoid(1.0))
        assert by_chat[2].score == pytest.approx(_sigmoid(5.0))

    def test_caps_at_max_sessions(self):
        from seaglass.search.rank import _sigmoid

        ranked = [
            RankedChunk(chunk_id=i, chat_id=i, start_ts=700000000, rerank_score=float(i))
            for i in range(1, 11)
        ]
        sessions = aggregate_sessions(ranked, max_sessions=8)
        assert len(sessions) == 8
        # highest scores kept
        assert sessions[0].score == pytest.approx(_sigmoid(10.0))

    def test_dedup_by_chunk_id(self):
        from seaglass.search.rank import _sigmoid

        ranked = [
            RankedChunk(chunk_id=1, chat_id=1, start_ts=700000000, rerank_score=2.0),
            RankedChunk(chunk_id=1, chat_id=1, start_ts=700000000, rerank_score=2.0),  # duplicate
        ]
        sessions = aggregate_sessions(ranked)
        assert len(sessions) == 1
        assert sessions[0].score == pytest.approx(_sigmoid(2.0))
        assert sessions[0].hit_chunk_ids == [1]

    def test_more_relevant_hits_never_hurt_a_session(self):
        # Regression test for the raw-logit-summation bug: a session with
        # two decent (less-negative) hits must never score below a
        # single-hit session whose one hit is worse, even though summing
        # *raw* negative logits would invert that (e.g. -5 + -6 = -11 <
        # -9). Scores here mimic the real cross-encoder's mostly-negative
        # logit range.
        two_hit_session = [
            RankedChunk(chunk_id=1, chat_id=1, start_ts=700000000, rerank_score=-5.0),
            RankedChunk(chunk_id=2, chat_id=1, start_ts=700000030, rerank_score=-6.0),
        ]
        one_hit_session = [
            RankedChunk(chunk_id=3, chat_id=2, start_ts=700000000, rerank_score=-9.0),
        ]
        sessions = aggregate_sessions(two_hit_session + one_hit_session, max_sessions=8)
        by_chat = {s.chat_id: s for s in sessions}
        assert by_chat[1].score > by_chat[2].score


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


class TestRecencyShaping:
    """Recency is a first-class ranking signal (see rank.py's
    RECENCY_MAX_BOOST / RECENT_SESSION_SLOTS / RERANK_RECENT_SLOTS).
    """

    def test_multiplier_decays_from_max_boost_to_one(self):
        from seaglass.search.rank import (
            RECENCY_HALF_LIFE_DAYS,
            RECENCY_MAX_BOOST,
            _recency_multiplier,
        )

        now = 1_700_000_000.0
        day = 86400.0
        assert _recency_multiplier(now, now) == pytest.approx(1 + RECENCY_MAX_BOOST)
        half = _recency_multiplier(now - RECENCY_HALF_LIFE_DAYS * day, now)
        assert half == pytest.approx(1 + RECENCY_MAX_BOOST / 2)
        # far past decays toward a no-op multiplier, never below 1.0
        ancient = _recency_multiplier(now - 100 * 365 * day, now)
        assert 1.0 <= ancient < 1.01
        # future timestamps (clock skew) must not amplify beyond the cap
        assert _recency_multiplier(now + 10 * day, now) <= 1 + RECENCY_MAX_BOOST

    def test_recent_session_wins_ties_but_not_blowouts(self):
        now = 1_700_000_000.0
        day = 86400.0
        old = int(now - 3000 * day)
        recent = int(now - day)

        # comparable relevance -> recency decides
        tied = [
            RankedChunk(chunk_id=1, chat_id=1, start_ts=old, rerank_score=1.0),
            RankedChunk(chunk_id=2, chat_id=2, start_ts=recent, rerank_score=1.0),
        ]
        ordered = aggregate_sessions(tied, now=now, recent_slots=0)
        assert ordered[0].chat_id == 2

        # a far stronger old match still wins: recency is bounded
        lopsided = [
            RankedChunk(chunk_id=1, chat_id=1, start_ts=old, rerank_score=8.0),
            RankedChunk(chunk_id=2, chat_id=2, start_ts=recent, rerank_score=-8.0),
        ]
        ordered = aggregate_sessions(lopsided, now=now, recent_slots=0)
        assert ordered[0].chat_id == 1

    def test_reserved_slots_admit_recent_low_scoring_session(self):
        now = 1_700_000_000.0
        day = 86400.0
        # 8 strong old sessions would fill every slot on relevance alone
        ranked = [
            RankedChunk(chunk_id=i, chat_id=i, start_ts=int(now - (2000 + i) * day), rerank_score=6.0)
            for i in range(1, 9)
        ]
        today = RankedChunk(chunk_id=99, chat_id=99, start_ts=int(now), rerank_score=-6.0)

        without = aggregate_sessions(ranked + [today], now=now, recent_slots=0)
        assert 99 not in {s.chat_id for s in without}

        with_slots = aggregate_sessions(ranked + [today], now=now, recent_slots=2)
        assert 99 in {s.chat_id for s in with_slots}
        # bounded: the reservation never costs more than `slots` results,
        # and the result set stays the same size and duplicate-free
        assert len(with_slots) == 8
        chat_ids = [s.chat_id for s in with_slots]
        assert len(set(chat_ids)) == len(chat_ids)

    def test_reservation_is_a_noop_when_nothing_is_displaced(self):
        now = 1_700_000_000.0
        ranked = [
            RankedChunk(chunk_id=i, chat_id=i, start_ts=int(now), rerank_score=float(i))
            for i in range(1, 4)
        ]
        assert aggregate_sessions(ranked, now=now, recent_slots=2) == aggregate_sessions(
            ranked, now=now, recent_slots=0
        )

    def test_rerank_cut_reserves_slots_for_newest_candidates(self):
        from seaglass.search.rank import select_reranked_head

        # score-descending, with the newest chunks at the *bottom* by score
        scored = [
            RankedChunk(chunk_id=i, chat_id=1, start_ts=i * 1000, rerank_score=10.0 - i)
            for i in range(10)
        ]
        picked = select_reranked_head(scored, top_k=5, recent_slots=2)
        ids = [c.chunk_id for c in picked]
        assert len(ids) == 5 and len(set(ids)) == 5
        assert ids[:3] == [0, 1, 2]      # top-(top_k - slots) purely by score
        assert set(ids[3:]) == {9, 8}    # newest of the remainder

        # degenerate cases fall back to the plain top-k
        assert select_reranked_head(scored, top_k=5, recent_slots=0) == scored[:5]
        assert select_reranked_head(scored[:3], top_k=5, recent_slots=2) == scored[:3]


class TestSessionPagination:
    """order_sessions returns a stable full ordering whose first page is
    byte-identical to the unpaginated result (see rank.py's order_sessions).
    """

    @staticmethod
    def _ranked(n, now):
        # descending relevance, ascending age -> recency and relevance disagree
        return [
            RankedChunk(
                chunk_id=i,
                chat_id=i,
                start_ts=int(now - (n - i) * 86400),
                rerank_score=float(n - i),
            )
            for i in range(n)
        ]

    def test_first_page_matches_the_unpaginated_result(self):
        now = 1_700_000_000.0
        ranked = self._ranked(30, now)
        full = order_sessions(ranked, head_size=8, now=now)
        page1 = aggregate_sessions(ranked, max_sessions=8, now=now)
        assert [(s.chat_id, s.day) for s in full[:8]] == [(s.chat_id, s.day) for s in page1]

    def test_pages_are_disjoint_and_cover_everything(self):
        now = 1_700_000_000.0
        ranked = self._ranked(30, now)
        full = order_sessions(ranked, head_size=8, now=now)

        keys = [(s.chat_id, s.day) for s in full]
        assert len(set(keys)) == len(keys), "a session must never appear on two pages"

        # walking pages reconstructs the full list exactly, with no gaps
        walked = []
        for offset in range(0, len(full), 8):
            walked.extend(full[offset: offset + 8])
        assert [(s.chat_id, s.day) for s in walked] == keys

    def test_tail_is_ordered_by_score(self):
        now = 1_700_000_000.0
        full = order_sessions(self._ranked(30, now), head_size=8, now=now)
        tail_scores = [s.score for s in full[8:]]
        assert tail_scores == sorted(tail_scores, reverse=True)

    def test_returns_everything_when_it_all_fits_in_the_head(self):
        now = 1_700_000_000.0
        ranked = self._ranked(3, now)
        assert len(order_sessions(ranked, head_size=8, now=now)) == 3


class TestLexicalDominance:
    """A recent verbatim match beats a stronger-but-older semantic match
    (see rank.py's LEXICAL_MATCH_BOOST / LEXICAL_HALF_LIFE_DAYS).
    """

    def test_recent_verbatim_match_outranks_stronger_old_semantic(self):
        now = 1_700_000_000.0
        day = 86400.0
        chunks = [
            # strong cross-encoder score, but months old
            RankedChunk(chunk_id=1, chat_id=1, start_ts=int(now - 180 * day), rerank_score=3.0),
            # weak score -- the query word is one message in a long chunk --
            # but someone literally typed it today
            RankedChunk(chunk_id=2, chat_id=2, start_ts=int(now - 3600), rerank_score=-0.8),
        ]
        ordered = aggregate_sessions(chunks, now=now, recent_slots=0, lexical_chunk_ids={2})
        assert ordered[0].chat_id == 2

        # without the lexical signal the old semantic match wins, which is
        # exactly the behaviour being corrected
        plain = aggregate_sessions(chunks, now=now, recent_slots=0)
        assert plain[0].chat_id == 1

    def test_old_verbatim_match_earns_almost_nothing(self):
        now = 1_700_000_000.0
        day = 86400.0
        chunks = [
            RankedChunk(chunk_id=1, chat_id=1, start_ts=int(now - 10 * day), rerank_score=2.0),
            RankedChunk(chunk_id=2, chat_id=2, start_ts=int(now - 900 * day), rerank_score=1.0),
        ]
        ordered = aggregate_sessions(chunks, now=now, recent_slots=0, lexical_chunk_ids={2})
        assert ordered[0].chat_id == 1

    def test_no_lexical_ids_leaves_ordering_untouched(self):
        now = 1_700_000_000.0
        chunks = [
            RankedChunk(chunk_id=1, chat_id=1, start_ts=int(now - 86400), rerank_score=2.0),
            RankedChunk(chunk_id=2, chat_id=2, start_ts=int(now - 86400), rerank_score=1.0),
        ]
        base = aggregate_sessions(chunks, now=now, recent_slots=0)
        empty = aggregate_sessions(chunks, now=now, recent_slots=0, lexical_chunk_ids=set())
        assert [s.score for s in base] == [s.score for s in empty]

    def test_freshness_uses_the_newest_verbatim_chunk_in_the_session(self):
        from seaglass.search.rank import LEXICAL_MATCH_BOOST

        now = 1_700_000_000.0
        day = 86400.0
        # same session, two verbatim matches: the recent one sets freshness
        chunks = [
            RankedChunk(chunk_id=1, chat_id=1, start_ts=int(now - 400 * day), rerank_score=0.0),
            RankedChunk(chunk_id=2, chat_id=1, start_ts=int(now), rerank_score=0.0),
        ]
        ordered = aggregate_sessions(
            chunks, now=now, recent_slots=0, lexical_chunk_ids={1, 2}, max_sessions=8
        )
        boosted = [s for s in ordered if s.chat_id == 1]
        # two same-day sessions may split by day; the newest must carry ~full boost
        assert max(s.score for s in boosted) > LEXICAL_MATCH_BOOST


class TestExactPhraseLookup:
    def test_matches_are_restricted_to_the_candidate_pool(self, tmp_path):
        import sqlite3

        from seaglass.search.retrieve import exact_phrase_chunk_ids

        con = sqlite3.connect(":memory:")
        con.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(body)")
        con.execute("INSERT INTO chunks_fts(rowid, body) VALUES (1, 'that was a classic move')")
        con.execute("INSERT INTO chunks_fts(rowid, body) VALUES (2, 'classic')")
        con.execute("INSERT INTO chunks_fts(rowid, body) VALUES (3, 'nothing relevant here')")

        assert exact_phrase_chunk_ids(con, "classic", [1, 2, 3]) == {1, 2}
        assert exact_phrase_chunk_ids(con, "classic", [3]) == set()
        assert exact_phrase_chunk_ids(con, "classic", []) == set()

    def test_phrase_is_matched_in_order_not_as_loose_terms(self, tmp_path):
        import sqlite3

        from seaglass.search.retrieve import exact_phrase_chunk_ids

        con = sqlite3.connect(":memory:")
        con.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(body)")
        con.execute("INSERT INTO chunks_fts(rowid, body) VALUES (1, 'how was golf today')")
        con.execute("INSERT INTO chunks_fts(rowid, body) VALUES (2, 'golf was fun how about you')")

        assert exact_phrase_chunk_ids(con, "how was golf", [1, 2]) == {1}

    def test_malformed_query_fails_open_instead_of_raising(self):
        import sqlite3

        from seaglass.search.retrieve import exact_phrase_chunk_ids

        con = sqlite3.connect(":memory:")
        con.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(body)")
        con.execute("INSERT INTO chunks_fts(rowid, body) VALUES (1, 'he said \"classic\" again')")

        # embedded quotes are escaped, not left to break the MATCH syntax
        assert exact_phrase_chunk_ids(con, '"classic"', [1]) == {1}
        assert exact_phrase_chunk_ids(con, '*', [1]) == set()
