"""Unit tests for seaglass.index.chunker -- pure logic, synthetic Message
objects only, single chat per test (matching the module's contract that
chunks never span chat boundaries).
"""

from __future__ import annotations

from seaglass.imessage.source import Message
from seaglass.index.chunker import chunk_messages


def _msg(rowid, ts, chat_id=1, text="hello", has_attachment=False, is_from_me=False):
    return Message(
        rowid=rowid,
        chat_id=chat_id,
        ts=ts,
        is_from_me=is_from_me,
        handle=None if is_from_me else "+15551234567",
        text=text,
        date_edited=None,
        date_retracted=None,
        has_attachment=has_attachment,
    )


class TestChunkMessages:
    def test_empty_input_yields_nothing(self):
        assert list(chunk_messages([])) == []

    def test_single_message_is_one_chunk(self):
        messages = [_msg(1, ts=1000.0)]
        chunks = list(chunk_messages(messages))
        assert len(chunks) == 1
        assert chunks[0].msg_ids == (1,)
        assert chunks[0].chat_id == 1
        assert chunks[0].start_ts == 1000.0
        assert chunks[0].end_ts == 1000.0

    def test_small_contiguous_conversation_stays_one_chunk(self):
        messages = [_msg(i, ts=1000.0 + i * 30) for i in range(1, 6)]
        chunks = list(chunk_messages(messages))
        assert len(chunks) == 1
        assert chunks[0].msg_ids == (1, 2, 3, 4, 5)

    def test_gap_over_threshold_splits_into_new_chunk(self):
        messages = [
            _msg(1, ts=1000.0),
            _msg(2, ts=1010.0),
            _msg(3, ts=1010.0 + 3600),  # 1hr later, over 45min default gap
            _msg(4, ts=1010.0 + 3610),
        ]
        chunks = list(chunk_messages(messages, overlap=0))
        assert len(chunks) == 2
        assert chunks[0].msg_ids == (1, 2)
        assert chunks[1].msg_ids == (3, 4)

    def test_overlap_carries_last_n_messages_into_next_chunk(self):
        messages = [
            _msg(1, ts=1000.0),
            _msg(2, ts=1010.0),
            _msg(3, ts=1010.0 + 3600),
            _msg(4, ts=1010.0 + 3610),
        ]
        chunks = list(chunk_messages(messages, overlap=1))
        assert len(chunks) == 2
        assert chunks[0].msg_ids == (1, 2)
        # chunk 2 carries message 2 (the last of chunk 1) plus 3, 4
        assert chunks[1].msg_ids == (2, 3, 4)

    def test_max_messages_forces_a_split(self):
        messages = [_msg(i, ts=1000.0 + i) for i in range(1, 11)]
        chunks = list(chunk_messages(messages, max_messages=4, overlap=0))
        assert len(chunks) == 3
        assert chunks[0].msg_ids == (1, 2, 3, 4)
        assert chunks[1].msg_ids == (5, 6, 7, 8)
        assert chunks[2].msg_ids == (9, 10)

    def test_token_target_forces_a_split(self):
        long_text = " ".join(f"word{i}" for i in range(50))
        messages = [_msg(i, ts=1000.0 + i, text=long_text) for i in range(1, 12)]
        # 50 tokens/message; token_target=120 should split roughly every 2-3 msgs
        chunks = list(chunk_messages(messages, token_target=120, overlap=0))
        assert len(chunks) > 1
        all_ids = tuple(mid for c in chunks for mid in c.msg_ids)
        assert all_ids == tuple(range(1, 12))

    def test_has_attachment_true_if_any_message_has_one(self):
        messages = [
            _msg(1, ts=1000.0, has_attachment=False),
            _msg(2, ts=1010.0, has_attachment=True),
        ]
        chunks = list(chunk_messages(messages))
        assert len(chunks) == 1
        assert chunks[0].has_attachment is True

    def test_chunks_cover_all_messages_no_gaps_no_duplicates_beyond_overlap(self):
        messages = [_msg(i, ts=1000.0 + i * 20) for i in range(1, 21)]
        chunks = list(chunk_messages(messages, max_messages=5, overlap=2))
        # every message id 1..20 appears in at least one chunk
        all_ids = set(mid for c in chunks for mid in c.msg_ids)
        assert all_ids == set(range(1, 21))
        # each chunk is contiguous and non-empty
        for c in chunks:
            assert list(c.msg_ids) == list(range(c.msg_ids[0], c.msg_ids[-1] + 1))
        # consecutive chunks overlap by exactly `overlap` messages
        for prev, nxt in zip(chunks, chunks[1:]):
            assert nxt.msg_ids[0] == prev.msg_ids[-1] - 1
