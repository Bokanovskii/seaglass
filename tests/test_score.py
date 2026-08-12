"""Unit tests for seaglass.eval.score -- Wilson intervals, McNemar's
statistic, and the end-to-end scoring pipeline (recall@50/@12/@final)
against a synthetic chat.db + index.db, using the deterministic
FakeReranker (no MLX load in unit tests).
"""

from __future__ import annotations

from seaglass.eval.score import (
    mcnemar_chi2,
    score_golden_set,
    summarize,
    wilson_interval,
)
from seaglass.imessage.source import connect_readonly
from seaglass.index.build import build_index, open_index_db

from conftest import FakeEmbeddingModel, build_fixture_chat_db


class FakeReranker:
    """Same deterministic word-overlap stand-in as tests/test_rank.py."""

    def score(self, pairs):
        scores = []
        for query, text in pairs:
            query_words = set(query.lower().split())
            text_words = set(text.lower().split())
            scores.append(float(len(query_words & text_words)))
        return scores


class TestWilsonInterval:
    def test_point_estimate_matches_naive_proportion(self):
        p, lo, hi = wilson_interval(hits=17, n=20)
        assert abs(p - 0.85) < 1e-9
        assert lo < p < hi

    def test_zero_n_returns_zeros(self):
        assert wilson_interval(hits=0, n=0) == (0.0, 0.0, 0.0)

    def test_perfect_score_has_nonzero_width_unlike_wald(self):
        # Wald would give [1.0, 1.0] here (zero width); Wilson must not.
        _, lo, hi = wilson_interval(hits=20, n=20)
        assert lo < 1.0
        assert hi <= 1.0

    def test_larger_n_gives_narrower_interval_at_same_proportion(self):
        _, lo_small, hi_small = wilson_interval(hits=17, n=20)
        _, lo_big, hi_big = wilson_interval(hits=170, n=200)
        assert (hi_big - lo_big) < (hi_small - lo_small)


class TestMcnemar:
    def test_no_discordance_returns_none(self):
        assert mcnemar_chi2(0, 0) is None

    def test_symmetric_discordance_gives_low_chi2(self):
        assert mcnemar_chi2(10, 10) == 0.0

    def test_asymmetric_discordance_gives_higher_chi2(self):
        assert mcnemar_chi2(20, 2) > mcnemar_chi2(11, 9)


def _build_scored_fixture(tmp_path):
    chat_db_path = build_fixture_chat_db(
        tmp_path,
        chats=[
            {
                "chat_id": 1,
                "handles": ["+15551234567"],
                "messages": [
                    ("hey are we still on for dinner tonight", 700000000, 0, 0),
                    ("yes see you at 7 sharp", 700000030, 1, 0),
                    ("great bringing wine and dessert", 700000060, 0, 0),
                    ("completely unrelated chat about the weather forecast", 700003600, 1, 0),
                    ("yeah it looks like rain all week honestly", 700003630, 0, 0),
                ],
            }
        ],
    )
    index_db_path = tmp_path / "index.db"
    build_index(chat_db_path, index_db_path, embedding_model=FakeEmbeddingModel())
    return chat_db_path, index_db_path


class TestScoreGoldenSet:
    def test_finds_positive_and_reports_hits(self, tmp_path):
        chat_db_path, index_db_path = _build_scored_fixture(tmp_path)
        index_con = open_index_db(index_db_path)
        chat_con = connect_readonly(chat_db_path)

        # message_id 1 ("yes see you at 7 sharp") should be inside the one chunk
        # covering this small fixture, so a query naming its content should hit.
        golden = [
            {
                "id": "ev-1",
                "query": "dinner tonight",
                "positive_msg_ids": [1],
                "alt_positive_msg_ids": [],
                "category": "topical",
                "nn_distance": 0.3,
            }
        ]
        results = score_golden_set(index_con, chat_con, FakeEmbeddingModel(), FakeReranker(), golden)
        assert len(results) == 1
        r = results[0]
        assert r["id"] == "ev-1"
        assert isinstance(r["recall_50"], bool)
        assert isinstance(r["recall_final"], bool)
        # recall@50 implies membership in the fused candidate pool -- since this
        # is the only chat/chunk in the fixture, it must always be found.
        assert r["recall_50"] is True

    def test_summarize_produces_per_category_and_overall_rows(self, tmp_path):
        chat_db_path, index_db_path = _build_scored_fixture(tmp_path)
        index_con = open_index_db(index_db_path)
        chat_con = connect_readonly(chat_db_path)

        golden = [
            {"id": "ev-1", "query": "dinner plans", "positive_msg_ids": [0], "category": "topical", "nn_distance": 0.2},
            {"id": "ev-2", "query": "weather talk", "positive_msg_ids": [3], "category": "topical", "nn_distance": 0.5},
        ]
        results = score_golden_set(index_con, chat_con, FakeEmbeddingModel(), FakeReranker(), golden)
        report = summarize(results)

        assert "topical" in report["by_category"]
        assert report["by_category"]["topical"]["n"] == 2
        assert report["overall"]["n"] == 2
        assert 0.0 <= report["overall"]["recall_50"] <= 1.0
        assert "recall_50_ci" in report["overall"]

    def test_no_positives_never_crashes(self, tmp_path):
        chat_db_path, index_db_path = _build_scored_fixture(tmp_path)
        index_con = open_index_db(index_db_path)
        chat_con = connect_readonly(chat_db_path)
        golden = [{"id": "ev-1", "query": "something that matches nothing at all", "positive_msg_ids": [], "category": "topical", "nn_distance": None}]
        results = score_golden_set(index_con, chat_con, FakeEmbeddingModel(), FakeReranker(), golden)
        assert results[0]["recall_50"] is False
        assert results[0]["filter_killed"] is False  # no positives to be killed
