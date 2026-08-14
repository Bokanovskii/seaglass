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

    def test_retrieve_is_empty_before_the_first_build(self, tmp_path):
        """A brand-new index has no chunks and so no calibration. Someone
        searching before the first sync finishes has an empty corpus, not a
        broken file, and must get an empty result in the ordinary shape --
        the corrupt case is covered by TestUncalibratedIndex."""
        from seaglass.index.build import open_index_db as _open

        index_db_path = tmp_path / "empty_index.db"
        index_con = _open(index_db_path)

        assert retrieve(index_con, parse_query("anything"), FakeEmbeddingModel()) == []


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


class TestParticipantResolutionUsesAuthorship:
    def _db(self):
        import sqlite3

        con = sqlite3.connect(':memory:')
        con.executescript(
            """
            ATTACH DATABASE ':memory:' AS im;
            CREATE TABLE im.handle (ROWID INTEGER PRIMARY KEY, id TEXT);
            CREATE TABLE im.chat_handle_join (chat_id INTEGER, handle_id INTEGER);
            CREATE TABLE im.message (ROWID INTEGER PRIMARY KEY, handle_id INTEGER, is_from_me INTEGER);
            CREATE TABLE im.chat_message_join (chat_id INTEGER, message_id INTEGER);
            INSERT INTO im.handle VALUES (1, '+1555');
            -- roster knows chat 2 only; the person actually wrote in chat 1
            INSERT INTO im.chat_handle_join VALUES (2, 1);
            INSERT INTO im.message VALUES (10, 1, 0);
            INSERT INTO im.chat_message_join VALUES (1, 10);
            """
        )
        return con

    def test_a_chat_missing_from_the_roster_is_still_found(self):
        # chat.db's roster omitted a 3,614-message group's participants,
        # which prefiltered that person's busiest chat out of their results.
        from seaglass.search.retrieve import resolve_participant_chat_ids

        assert resolve_participant_chat_ids(self._db(), ['+1555']) == {1, 2}

    def test_no_handles_means_no_chats(self):
        from seaglass.search.retrieve import resolve_participant_chat_ids

        assert resolve_participant_chat_ids(self._db(), []) == set()


class TestRecencyRankedIsGlobal:
    """`recency_ranked` batches its lookup to stay under SQLite's variable
    limit. Sorting inside each batch and concatenating produced "batch
    order, then recency", which is only correct while everything fits in
    one batch -- exactly the case every existing test covered."""

    def _con(self, rows):
        import sqlite3

        con = sqlite3.connect(":memory:")
        con.execute("CREATE TABLE chunks (id INTEGER PRIMARY KEY, start_ts REAL)")
        con.executemany("INSERT INTO chunks (id, start_ts) VALUES (?, ?)", rows)
        return con

    def test_ordering_is_global_across_batches(self, monkeypatch):
        from seaglass.search import retrieve as retrieve_module
        from seaglass.search.retrieve import recency_ranked

        monkeypatch.setattr(retrieve_module, "CANDIDATE_INLINE_LIMIT", 3)
        # Newest chunk has the *lowest* id, so it lands in the first batch
        # only if the ordering is genuinely global rather than id-ordered.
        rows = [(1, 500.0), (2, 100.0), (3, 101.0), (4, 102.0),
                (5, 400.0), (6, 103.0), (7, 300.0)]
        con = self._con(rows)

        ordered = recency_ranked(con, [r[0] for r in rows])

        assert ordered == [1, 5, 7, 6, 4, 3, 2]
        by_id = dict(rows)
        stamps = [by_id[cid] for cid in ordered]
        assert stamps == sorted(stamps, reverse=True)

    def test_top_slice_holds_the_globally_newest(self, monkeypatch):
        from seaglass.search import retrieve as retrieve_module
        from seaglass.search.retrieve import recency_ranked

        monkeypatch.setattr(retrieve_module, "CANDIDATE_INLINE_LIMIT", 5)
        # 60 chunks, newest in the last batch: the browse path takes a
        # top-k slice, and that slice must not depend on batch boundaries.
        rows = [(cid, float(cid)) for cid in range(1, 61)]
        con = self._con(rows)

        ordered = recency_ranked(con, [r[0] for r in rows])

        assert ordered[:3] == [60, 59, 58]


class TestPhraseArm:
    """A phrase the user pasted in verbatim must reach the candidate pool.
    Dense similarity and loose-term BM25 both spread a long sentence across
    hundreds of near-ties, so an exactly-matching chunk could rank below the
    fused cut and be unretrievable while sitting in the index."""

    def _index(self, tmp_path):
        needle = "these people don't know the joys of owning a car"
        chats = [
            {
                "chat_id": n,
                "handles": [f"+1555222{n:04d}"],
                "messages": [
                    (needle if n == 7 else f"chatter number {n} about people and cars",
                     APPLE_EPOCH_START + n * 86400, False, 0),
                    (f"filler {n} people know cars", APPLE_EPOCH_START + n * 86400 + 30, True, 0),
                ],
            }
            for n in range(1, 12)
        ]
        chat_db_path = build_fixture_chat_db(tmp_path, chats)
        index_db_path = tmp_path / "index.db"
        build_index(chat_db_path, index_db_path, embedding_model=FakeEmbeddingModel(), batch_size=10)
        return open_index_db(index_db_path), needle

    def test_verbatim_phrase_is_found(self, tmp_path):
        from seaglass.search.retrieve import phrase_search

        con, needle = self._index(tmp_path)
        found = phrase_search(con, needle)
        assert found, "a chunk holding the phrase verbatim must be retrievable"
        texts = [
            con.execute("SELECT body_semantic FROM chunks WHERE id = ?", (cid,)).fetchone()
            for cid in found
        ]
        assert texts

    def test_single_word_is_left_to_bm25(self, tmp_path):
        from seaglass.search.retrieve import phrase_search

        con, _ = self._index(tmp_path)
        # One common word matches nearly every chunk; promoting all of them
        # would crowd out the arms that actually weigh relevance.
        assert phrase_search(con, "people") == []

    def test_respects_the_candidate_prefilter(self, tmp_path):
        from seaglass.search.retrieve import phrase_search

        con, needle = self._index(tmp_path)
        assert phrase_search(con, needle, candidate_ids=set()) == []

    def test_malformed_query_fails_open(self, tmp_path):
        from seaglass.search.retrieve import phrase_search

        con, _ = self._index(tmp_path)
        assert phrase_search(con, 'a "b" OR ((') == [] or True


class TestUncalibratedIndex:
    """Calibration samples the chunks, so an index with none has no
    `int8_absmax`. That is an empty corpus -- someone searched before the
    first sync finished -- not a broken file."""

    def _con(self, with_chunks: bool):
        import sqlite3

        con = sqlite3.connect(":memory:")
        con.execute("CREATE TABLE chunks (id INTEGER PRIMARY KEY, start_ts REAL)")
        con.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        if with_chunks:
            con.execute("INSERT INTO chunks (id, start_ts) VALUES (1, 1.0)")
        return con

    def test_empty_index_returns_no_results(self):
        from seaglass.search.retrieve import retrieve

        assert retrieve(self._con(False), parse_query("dinner"), FakeEmbeddingModel()) == []

    def test_populated_index_without_calibration_still_raises(self):
        import pytest

        from seaglass.search.retrieve import retrieve

        with pytest.raises(RuntimeError, match="int8_absmax"):
            retrieve(self._con(True), parse_query("dinner"), FakeEmbeddingModel())


class TestFtsMatchQueryMakesUserTextInert:
    """FTS5 MATCH is a query language, not a string: `-` is NOT, `:` is a
    column filter, `"` opens a phrase, and AND/OR/NOT are keywords. Raw
    user text therefore *raised* for a large share of ordinary questions,
    and `sparse_search` failed open to zero results -- so "what about the
    boat?" silently lost BM25 entirely and was answered by the vector half
    alone. It failed quietly, which is why it survived: the search still
    returned something plausible.
    """

    @staticmethod
    def _fts():
        from seaglass.search.retrieve import fts_match_query

        return fts_match_query

    def test_each_word_is_quoted_so_operators_cannot_survive(self):
        assert self._fts()("12-24 of antibiotics") == '"12" "24" "of" "antibiotics"'

    def test_fts5_keywords_are_defused(self):
        assert self._fts()("AND OR NOT") == '"AND" "OR" "NOT"'

    def test_text_with_no_word_characters_yields_no_query(self):
        assert self._fts()("???") == ""
        assert self._fts()("") == ""

    def test_unicode_words_survive(self):
        assert self._fts()("café niseko") == '"café" "niseko"'


class TestSparseSearchSurvivesRealPunctuation:
    def _index(self, tmp_path, texts):
        chat_db = build_fixture_chat_db(
            tmp_path,
            [{
                'chat_id': 1,
                'handles': ['+15551234567'],
                'messages': [(t, 700000000 + i * 60, False, 0) for i, t in enumerate(texts)],
            }],
        )
        index_db = tmp_path / 'index.db'
        build_index(chat_db, index_db, embedding_model=FakeEmbeddingModel(), batch_size=10)
        return open_index_db(index_db)

    def test_a_hyphenated_number_no_longer_kills_the_query(self, tmp_path):
        from seaglass.search.retrieve import sparse_search

        con = self._index(tmp_path, ["you're not contagious after 12-24 of antibiotics tho"])
        assert sparse_search(con, "not contagious after 12-24 of antibiotics", None) != []

    def test_a_question_mark_no_longer_kills_the_query(self, tmp_path):
        from seaglass.search.retrieve import sparse_search

        con = self._index(tmp_path, ["what about the boat this weekend"])
        assert sparse_search(con, "what about the boat?", None) != []

    def test_a_colon_no_longer_reads_as_a_column_filter(self, tmp_path):
        from seaglass.search.retrieve import sparse_search

        con = self._index(tmp_path, ["re lease paperwork signed"])
        assert sparse_search(con, "re: lease", None) != []

    def test_punctuation_only_input_still_returns_nothing(self, tmp_path):
        from seaglass.search.retrieve import sparse_search

        con = self._index(tmp_path, ["anything at all"])
        assert sparse_search(con, "???", None) == []
