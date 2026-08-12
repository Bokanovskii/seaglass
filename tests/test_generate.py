"""Unit tests for seaglass.eval.generate -- stratified selection, vocab-
overlap filtering, and the end-to-end generate() pipeline with a mocked
ghcp client (no real network/LLM calls in tests).
"""

from __future__ import annotations

import json

from seaglass.eval.generate import (
    _content_words,
    _select_stratified,
    _vocab_overlap_pct,
    generate,
)
from seaglass.eval.harvest import stage_idf, stage_nn, stage_score, stage_sql
from seaglass.imessage.source import connect_readonly
from seaglass.index.build import build_index, open_index_db

from conftest import FakeEmbeddingModel, build_fixture_chat_db


class TestVocabOverlap:
    def test_no_overlap_is_zero(self):
        assert _vocab_overlap_pct("what did we decide about the plan", "completely different words entirely here") == 0.0

    def test_full_overlap_is_one(self):
        assert _vocab_overlap_pct("boat radio thoughts", "boat radio thoughts") == 1.0

    def test_stopwords_excluded_from_content_words(self):
        assert _content_words("the a an and we you") == set()


class TestSelectStratified:
    def _harvested_index(self, tmp_path, monkeypatch, n_chats=2, n_messages_per_chat=20):
        import seaglass.eval.harvest as harvest_module

        monkeypatch.setattr(harvest_module, "TOKEN_COUNT_MIN", 0)
        chat_db_path = build_fixture_chat_db(
            tmp_path,
            chats=[
                {
                    "chat_id": chat_id,
                    "handles": [f"+1555000{chat_id}00{i}" for i in range(3)],
                    "messages": [
                        (f"chat {chat_id} message number {i} discussing something", 700000000 + i * 3600, i % 2, i % 3)
                        for i in range(n_messages_per_chat)
                    ],
                }
                for chat_id in range(1, n_chats + 1)
            ],
        )
        index_db_path = tmp_path / "index.db"
        build_index(chat_db_path, index_db_path, embedding_model=FakeEmbeddingModel(), chunker_kwargs={"max_messages": 4})
        index_con = open_index_db(index_db_path)
        chat_con = connect_readonly(chat_db_path)
        stage_sql(index_con, chat_con)
        stage_idf(index_con)
        stage_nn(index_con, target=1000)
        stage_score(index_con)
        return chat_db_path, index_db_path

    def test_excludes_bottom_band_and_respects_target(self, tmp_path, monkeypatch):
        _chat_db_path, index_db_path = self._harvested_index(tmp_path, monkeypatch)
        index_con = open_index_db(index_db_path)
        n_scored = index_con.execute("SELECT COUNT(*) FROM eval_candidate WHERE nn_distance IS NOT NULL").fetchone()[0]
        assert n_scored > 0

        selected = _select_stratified(index_con, target=n_scored)
        # bottom band excluded, so selection should never exceed the scored pool
        assert len(selected) <= n_scored
        assert len(set(selected)) == len(selected)  # no duplicates

    def test_returns_empty_when_no_candidates_scored(self, tmp_path):
        chat_db_path = build_fixture_chat_db(
            tmp_path,
            chats=[{"chat_id": 1, "handles": ["+15551234567"], "messages": [("hi", 700000000, 0, 0)]}],
        )
        index_db_path = tmp_path / "index.db"
        build_index(chat_db_path, index_db_path, embedding_model=FakeEmbeddingModel())
        index_con = open_index_db(index_db_path)
        assert _select_stratified(index_con) == []


class TestGenerateEndToEnd:
    def test_generate_writes_filtered_entries(self, tmp_path, monkeypatch):
        import seaglass.eval.harvest as harvest_module
        import seaglass.eval.generate as generate_module

        monkeypatch.setattr(harvest_module, "TOKEN_COUNT_MIN", 0)
        chat_db_path = build_fixture_chat_db(
            tmp_path,
            chats=[
                {
                    "chat_id": 1,
                    "handles": ["+15551234567", "+15557654321"],
                    "messages": [
                        (f"message number {i} about weekend plans and travel logistics", 700000000 + i * 3600, i % 2, i % 2)
                        for i in range(20)
                    ],
                }
            ],
        )
        index_db_path = tmp_path / "index.db"
        build_index(chat_db_path, index_db_path, embedding_model=FakeEmbeddingModel(), chunker_kwargs={"max_messages": 4})
        index_con = open_index_db(index_db_path)
        chat_con = connect_readonly(chat_db_path)
        stage_sql(index_con, chat_con)
        stage_idf(index_con)
        stage_nn(index_con, target=1000)
        stage_score(index_con)

        def fake_ghcp_json(prompt, timeout_s=180):
            # Extract snippet ids from the prompt and return one bland,
            # non-overlapping question per id -- deterministic and network-free.
            import re

            ids = re.findall(r"^(\d+) \(category", prompt, re.MULTILINE)
            return [{"id": int(i), "question": f"what did we end up figuring out that time (item {i})"} for i in ids]

        monkeypatch.setattr(generate_module, "call_ghcp_json", fake_ghcp_json)
        monkeypatch.setattr(generate_module, "EmbeddingModel", FakeEmbeddingModel)

        output_path = tmp_path / "candidates_for_review.jsonl"
        n_written = generate(index_db_path, chat_db_path, output_path, target=50, batch_size=5)

        assert output_path.exists()
        lines = output_path.read_text().strip().splitlines()
        assert len(lines) == n_written
        if lines:
            entry = json.loads(lines[0])
            assert entry["origin"] == "harvested"
            assert entry["reviewed"] is False
            assert "query" in entry and entry["query"]
            assert "positive_msg_ids" in entry and entry["positive_msg_ids"]
            assert "_vocab_overlap_pct" in entry
            assert entry["_vocab_overlap_pct"] <= 0.40

    def test_generate_returns_zero_with_no_harvested_candidates(self, tmp_path):
        chat_db_path = build_fixture_chat_db(
            tmp_path,
            chats=[{"chat_id": 1, "handles": ["+15551234567"], "messages": [("hi", 700000000, 0, 0)]}],
        )
        index_db_path = tmp_path / "index.db"
        build_index(chat_db_path, index_db_path, embedding_model=FakeEmbeddingModel())
        output_path = tmp_path / "out.jsonl"
        n_written = generate(index_db_path, chat_db_path, output_path)
        assert n_written == 0
