"""Unit tests for seaglass.eval.review -- the human review CLI's decision
logic (accept/edit/reject/multi-positive/skip), driven by a scripted
fake input function instead of real stdin.
"""

from __future__ import annotations

import json
import sqlite3

from seaglass.eval.review import _load_jsonl, review


def _write_jsonl(path, entries):
    with path.open("w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def _draft_entry(entry_id, query="what happened that day", trivially_hard=False):
    return {
        "id": entry_id,
        "query": query,
        "positive_msg_ids": [1, 2],
        "alt_positive_msg_ids": [],
        "chat_id": 1,
        "category": "topical",
        "source_chunk_id": 5,
        "nn_distance": 0.42,
        "origin": "harvested",
        "reviewed": False,
        "split": "dev",
        "_vocab_overlap_pct": 0.1,
        "_sparse_rank1_is_source": False,
        "_trivially_hard": trivially_hard,
        "_top5_other_chunk_ids": [6, 7],
    }


class _ScriptedInput:
    def __init__(self, responses):
        self.responses = list(responses)

    def __call__(self, prompt=""):
        return self.responses.pop(0)


class TestReview:
    def test_accept_writes_to_golden_without_internal_fields(self, tmp_path):
        input_path = tmp_path / "candidates.jsonl"
        _write_jsonl(input_path, [_draft_entry("ev-1")])
        golden_path = tmp_path / "golden.jsonl"
        rejected_path = tmp_path / "rejected.jsonl"

        review(
            input_path, golden_path, rejected_path,
            already_decided_ids=set(),
            interactive_input=_ScriptedInput(["a"]),
        )

        golden = _load_jsonl(golden_path)
        assert len(golden) == 1
        assert golden[0]["id"] == "ev-1"
        assert golden[0]["reviewed"] is True
        assert not any(k.startswith("_") for k in golden[0])
        assert _load_jsonl(rejected_path) == []

    def test_reject_writes_to_rejected_not_golden(self, tmp_path):
        input_path = tmp_path / "candidates.jsonl"
        _write_jsonl(input_path, [_draft_entry("ev-1")])
        golden_path = tmp_path / "golden.jsonl"
        rejected_path = tmp_path / "rejected.jsonl"

        review(
            input_path, golden_path, rejected_path,
            already_decided_ids=set(),
            interactive_input=_ScriptedInput(["r"]),
        )
        assert _load_jsonl(golden_path) == []
        assert len(_load_jsonl(rejected_path)) == 1

    def test_edit_overwrites_query_text(self, tmp_path):
        input_path = tmp_path / "candidates.jsonl"
        _write_jsonl(input_path, [_draft_entry("ev-1", query="old question")])
        golden_path = tmp_path / "golden.jsonl"
        rejected_path = tmp_path / "rejected.jsonl"

        review(
            input_path, golden_path, rejected_path,
            already_decided_ids=set(),
            interactive_input=_ScriptedInput(["e", "new better question"]),
        )
        golden = _load_jsonl(golden_path)
        assert golden[0]["query"] == "new better question"

    def test_edit_with_blank_keeps_original_query(self, tmp_path):
        input_path = tmp_path / "candidates.jsonl"
        _write_jsonl(input_path, [_draft_entry("ev-1", query="keep me")])
        golden_path = tmp_path / "golden.jsonl"
        rejected_path = tmp_path / "rejected.jsonl"

        review(
            input_path, golden_path, rejected_path,
            already_decided_ids=set(),
            interactive_input=_ScriptedInput(["e", ""]),
        )
        golden = _load_jsonl(golden_path)
        assert golden[0]["query"] == "keep me"

    def test_multi_positive_records_alt_ids(self, tmp_path):
        input_path = tmp_path / "candidates.jsonl"
        _write_jsonl(input_path, [_draft_entry("ev-1")])
        golden_path = tmp_path / "golden.jsonl"
        rejected_path = tmp_path / "rejected.jsonl"

        review(
            input_path, golden_path, rejected_path,
            already_decided_ids=set(),
            interactive_input=_ScriptedInput(["m", "99, 100"]),
        )
        golden = _load_jsonl(golden_path)
        assert golden[0]["alt_positive_msg_ids"] == [99, 100]

    def test_skip_leaves_entry_undecided(self, tmp_path):
        input_path = tmp_path / "candidates.jsonl"
        _write_jsonl(input_path, [_draft_entry("ev-1"), _draft_entry("ev-2")])
        golden_path = tmp_path / "golden.jsonl"
        rejected_path = tmp_path / "rejected.jsonl"

        review(
            input_path, golden_path, rejected_path,
            already_decided_ids=set(),
            interactive_input=_ScriptedInput(["s", "a"]),
        )
        golden = _load_jsonl(golden_path)
        assert len(golden) == 1
        assert golden[0]["id"] == "ev-2"

    def test_already_decided_entries_are_not_shown_again(self, tmp_path):
        input_path = tmp_path / "candidates.jsonl"
        _write_jsonl(input_path, [_draft_entry("ev-1"), _draft_entry("ev-2")])
        golden_path = tmp_path / "golden.jsonl"
        rejected_path = tmp_path / "rejected.jsonl"

        # only ev-2 should be prompted; a script with just one response proves ev-1 was skipped
        review(
            input_path, golden_path, rejected_path,
            already_decided_ids={"ev-1"},
            interactive_input=_ScriptedInput(["a"]),
        )
        golden = _load_jsonl(golden_path)
        assert len(golden) == 1
        assert golden[0]["id"] == "ev-2"

    def test_review_accepts_chat_con_and_still_decides_correctly(self, tmp_path, capsys):
        # A fake chat.db, attached under the "im" alias review.py expects
        # (matching seaglass.imessage.source.connect_readonly's real
        # ATTACH DATABASE ... AS im), so review() can print real message
        # text instead of bare integer ids for a human to actually verify.
        chat_con = sqlite3.connect(":memory:")
        chat_con.execute("ATTACH DATABASE ':memory:' AS im")
        chat_con.execute(
            "CREATE TABLE im.message (ROWID INTEGER PRIMARY KEY, text TEXT, "
            "attributedBody BLOB, date INTEGER, is_from_me INTEGER, handle_id INTEGER)"
        )
        chat_con.execute("CREATE TABLE im.handle (ROWID INTEGER PRIMARY KEY, id TEXT)")
        chat_con.execute("INSERT INTO im.handle VALUES (1, '+15551234567')")
        chat_con.execute(
            "INSERT INTO im.message VALUES (1, 'hey are we still on for dinner', NULL, 0, 0, 1)"
        )
        chat_con.execute(
            "INSERT INTO im.message VALUES (2, 'yeah see you at 7', NULL, 0, 1, NULL)"
        )
        chat_con.commit()

        input_path = tmp_path / "candidates.jsonl"
        _write_jsonl(input_path, [_draft_entry("ev-1")])
        golden_path = tmp_path / "golden.jsonl"
        rejected_path = tmp_path / "rejected.jsonl"

        review(
            input_path, golden_path, rejected_path,
            already_decided_ids=set(),
            interactive_input=_ScriptedInput(["a"]),
            chat_con=chat_con,
        )

        golden = _load_jsonl(golden_path)
        assert len(golden) == 1
        assert golden[0]["id"] == "ev-1"

        printed = capsys.readouterr().out
        assert "hey are we still on for dinner" in printed
        assert "yeah see you at 7" in printed

