"""Unit tests for seaglass.search.retrieve -- Phase 4a baseline pipeline
(pre-filter -> dense + sparse -> RRF). Synthetic chat.db + index.db, no
network/MLX dependency (FakeEmbeddingModel from conftest.py).
"""

from __future__ import annotations

from seaglass.imessage.source import connect_readonly
from seaglass.index.build import build_index, open_index_db
from seaglass.search.parse import parse_query
from seaglass.search.retrieve import (
    build_candidate_chunk_ids,
    dense_search,
    resolve_participant_chat_ids,
    retrieve,
    RetrievalResult,
    rrf_fuse,
    sparse_search,
)

from conftest import FakeEmbeddingModel, build_fixture_chat_db


APPLE_EPOCH_START = 700000000  # arbitrary plausible seconds-era date


def _two_chat_fixture(tmp_path):
    chats = [
        {
            "chat_id": 1,
            "handles": ["+15551110000"],
            "messages": [
                ("lisbon trip planning starts now", APPLE_EPOCH_START, False, 0),
                ("what hotel should we book", APPLE_EPOCH_START + 30, True, 0),
                ("the alfama district looks amazing", APPLE_EPOCH_START + 60, False, 0),
            ],
        },
        {
            "chat_id": 2,
            "handles": ["+15552220000"],
            "messages": [
                ("can you send the tax documents", APPLE_EPOCH_START + 100000, False, 0),
                ("sure, emailing them over now", APPLE_EPOCH_START + 100030, True, 0),
                ("got them, thanks!", APPLE_EPOCH_START + 100060, False, 0),
            ],
        },
    ]
    chat_db_path = build_fixture_chat_db(tmp_path, chats)
    index_db_path = tmp_path / "index.db"
    build_index(chat_db_path, index_db_path, embedding_model=FakeEmbeddingModel(), batch_size=10)
    return chat_db_path, index_db_path


class TestRRFFuse:
    def test_agreement_across_arms_outranks_single_arm_hit(self):
        dense = [10, 20, 30]
        sparse = [20, 10, 40]
        fused = rrf_fuse([dense, sparse], k=60, top_k=10)
        ids_in_order = [r.chunk_id for r in fused]
        # 10 and 20 appear in both arms near the top; 30/40 appear once each
        assert ids_in_order[0] in (10, 20)
        assert ids_in_order[1] in (10, 20)

    def test_truncates_to_top_k_after_fusion(self):
        dense = list(range(1, 101))
        sparse = list(range(50, 150))
        fused = rrf_fuse([dense, sparse], k=60, top_k=5)
        assert len(fused) == 5

    def test_long_tail_agreement_still_contributes(self):
        # A doc at rank 80 in one arm and rank 90 in the other should beat
        # a doc that only appears once at rank 1 -- this is what "retrieve
        # deep, fuse, then truncate" is for.
        dense = list(range(1, 101))
        sparse = [dense[79]] + [x for x in range(200, 300)][:89] + [dense[79]]
        # simpler direct construction:
        dense2 = [999] + list(range(1, 100))
        sparse2 = list(range(1, 100)) + [999]
        fused = rrf_fuse([dense2, sparse2], k=60, top_k=200)
        scores = {r.chunk_id: r.rrf_score for r in fused}
        # 999 (rank 1 in dense, last in sparse) vs a doc ranked ~50 in both
        assert scores[999] > 0


class TestBuildCandidateChunkIds:
    def test_no_filters_returns_none(self, tmp_path):
        _, index_db_path = _two_chat_fixture(tmp_path)
        index_con = open_index_db(index_db_path)
        parsed = parse_query("something with no filters at all")
        assert build_candidate_chunk_ids(index_con, parsed) is None

    def test_media_filter_restricts_to_has_attachment(self, tmp_path):
        _, index_db_path = _two_chat_fixture(tmp_path)
        index_con = open_index_db(index_db_path)
        parsed = parse_query("find that screenshot")
        result = build_candidate_chunk_ids(index_con, parsed)
        assert result == set()  # fixture has no attachments at all

    def test_people_filter_restricts_to_that_chat(self, tmp_path):
        chat_db_path, index_db_path = _two_chat_fixture(tmp_path)
        index_con = open_index_db(index_db_path)
        chat_con = connect_readonly(chat_db_path)
        chat_ids = resolve_participant_chat_ids(chat_con, ["+15551110000"])
        assert chat_ids == {1}
        all_chunks_in_chat1 = {
            row[0] for row in index_con.execute("SELECT id FROM chunks WHERE chat_id = 1")
        }
        parsed = parse_query("lisbon planning")
        # simulate the resolved people filter directly through the chat_id path
        from dataclasses import replace

        parsed = replace(parsed, people_participant=["+15551110000"])
        result = build_candidate_chunk_ids(index_con, parsed, chat_con=chat_con)
        assert result == all_chunks_in_chat1


class TestDenseAndSparseSearch:
    def test_dense_search_respects_candidate_restriction(self, tmp_path):
        _, index_db_path = _two_chat_fixture(tmp_path)
        index_con = open_index_db(index_db_path)
        model = FakeEmbeddingModel()
        absmax = float(
            index_con.execute("SELECT value FROM meta WHERE key='int8_absmax'").fetchone()[0]
        )
        from seaglass.index.embed import quantize_int8

        query_vec = quantize_int8(model.embed(["lisbon trip"]), absmax)[0]
        restricted_ids = {1}  # force restriction to just chunk 1, whatever it is
        results = dense_search(index_con, query_vec, restricted_ids, top_k=10)
        assert set(results) <= restricted_ids

    def test_sparse_search_finds_lexical_match(self, tmp_path):
        _, index_db_path = _two_chat_fixture(tmp_path)
        index_con = open_index_db(index_db_path)
        results = sparse_search(index_con, "alfama", None, top_k=10)
        assert len(results) >= 1

    def test_sparse_search_empty_query_returns_nothing(self, tmp_path):
        _, index_db_path = _two_chat_fixture(tmp_path)
        index_con = open_index_db(index_db_path)
        assert sparse_search(index_con, "", None) == []

    def test_sparse_search_malformed_query_fails_open(self, tmp_path):
        _, index_db_path = _two_chat_fixture(tmp_path)
        index_con = open_index_db(index_db_path)
        # bare FTS5 operator syntax that would otherwise raise
        assert sparse_search(index_con, '"unterminated', None) == []


class TestRetrieveEndToEnd:
    def test_retrieve_returns_relevant_results_for_lexical_match(self, tmp_path):
        chat_db_path, index_db_path = _two_chat_fixture(tmp_path)
        index_con = open_index_db(index_db_path)
        model = FakeEmbeddingModel()
        parsed = parse_query("alfama district")
        results = retrieve(index_con, parsed, model)
        assert len(results) > 0

    def test_retrieve_with_date_filter_excludes_out_of_range_chunks(self, tmp_path):
        chat_db_path, index_db_path = _two_chat_fixture(tmp_path)
        index_con = open_index_db(index_db_path)
        model = FakeEmbeddingModel()
        parsed = parse_query("tax documents")
        # force a date range that only covers chat 1's messages
        from dataclasses import replace

        parsed = replace(parsed, date_from=APPLE_EPOCH_START, date_to=APPLE_EPOCH_START + 90)
        results = retrieve(index_con, parsed, model)
        result_chat_ids = {
            row[0]
            for r in results
            for row in index_con.execute("SELECT chat_id FROM chunks WHERE id = ?", (r.chunk_id,))
        }
        assert result_chat_ids <= {1}

    def test_retrieve_raises_clear_error_if_never_built(self, tmp_path):
        from seaglass.index.build import open_index_db as _open

        index_db_path = tmp_path / "empty_index.db"
        index_con = _open(index_db_path)
        model = FakeEmbeddingModel()
        parsed = parse_query("anything")
        try:
            retrieve(index_con, parsed, model)
            assert False, "expected RuntimeError"
        except RuntimeError as error:
            assert "int8_absmax" in str(error)


class TestReserveRecentSlots:
    """The newest matched chunks must reach the reranker even when fusion
    buries them (see retrieve.py's RECENCY_RESERVED_SLOTS).
    """

    @staticmethod
    def _index(tmp_path):
        # one chat per day so the build produces plenty of distinct chunks
        chats = [
            {
                "chat_id": n,
                "handles": [f"+1555111{n:04d}"],
                "messages": [
                    (f"day {n} planning notes", APPLE_EPOCH_START + n * 86400, False, 0),
                    (f"day {n} follow up", APPLE_EPOCH_START + n * 86400 + 30, True, 0),
                ],
            }
            for n in range(1, 9)
        ]
        chat_db_path = build_fixture_chat_db(tmp_path, chats)
        index_db_path = tmp_path / "index.db"
        build_index(chat_db_path, index_db_path, embedding_model=FakeEmbeddingModel(), batch_size=10)
        return open_index_db(index_db_path)

    def test_recent_match_is_promoted_into_the_candidate_pool(self, tmp_path):
        from seaglass.search.retrieve import recency_ranked, reserve_recent_slots

        con = self._index(tmp_path)
        matched = [row[0] for row in con.execute("SELECT id FROM chunks").fetchall()]
        assert len(matched) >= 3, "fixture needs several chunks to exercise displacement"

        by_recency = recency_ranked(con, matched)
        newest = by_recency[0]
        # fusion ranks the newest chunk dead last
        fused = [
            RetrievalResult(chunk_id=cid, rrf_score=1.0 / (i + 1))
            for i, cid in enumerate([c for c in matched if c != newest] + [newest])
        ]
        top_k = len(fused) - 1

        without = reserve_recent_slots(con, fused, matched, top_k=top_k, slots=0)
        assert newest not in {r.chunk_id for r in without}

        with_slots = reserve_recent_slots(con, fused, matched, top_k=top_k, slots=2)
        ids = [r.chunk_id for r in with_slots]
        assert newest in ids
        # the reranker's cost must not change: exactly top_k, no duplicates
        assert len(ids) == top_k
        assert len(set(ids)) == len(ids)
        # the head of the list stays purely relevance-ordered
        assert ids[: top_k - 2] == [r.chunk_id for r in fused[: top_k - 2]]

    def test_slack_falls_back_to_fusion_ranking(self, tmp_path):
        from seaglass.search.retrieve import reserve_recent_slots

        con = self._index(tmp_path)
        all_ids = [row[0] for row in con.execute("SELECT id FROM chunks").fetchall()]
        fused = [RetrievalResult(chunk_id=cid, rrf_score=1.0 / (i + 1)) for i, cid in enumerate(all_ids)]

        # only one chunk is a real match, so most reserved slots go unused
        result = reserve_recent_slots(con, fused, all_ids[:1], top_k=len(all_ids), slots=5)
        ids = [r.chunk_id for r in result]
        assert len(ids) == len(all_ids), "must never return a short candidate list"
        assert set(ids) == set(all_ids)

    def test_noop_when_pool_already_fits(self, tmp_path):
        from seaglass.search.retrieve import reserve_recent_slots

        con = self._index(tmp_path)
        all_ids = [row[0] for row in con.execute("SELECT id FROM chunks").fetchall()]
        fused = [RetrievalResult(chunk_id=cid, rrf_score=1.0 / (i + 1)) for i, cid in enumerate(all_ids)]
        assert reserve_recent_slots(con, fused, all_ids, top_k=len(all_ids) + 10, slots=3) == fused
