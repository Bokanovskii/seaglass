"""Unit tests for seaglass.search.hydrate -- step 7: pulling raw messages
back for the final surviving sessions, split into hit vs context
messages, with sender resolution.
"""

from __future__ import annotations

from seaglass.imessage.source import connect_readonly
from seaglass.index.build import build_index, open_index_db
from seaglass.search.hydrate import hydrate_sessions
from seaglass.search.rank import Session

from conftest import FakeEmbeddingModel, build_fixture_chat_db


def _built_fixture(tmp_path):
    chat_db_path = build_fixture_chat_db(
        tmp_path,
        chats=[
            {
                "chat_id": 1,
                "handles": ["+15551234567"],
                "messages": [
                    ("hey are we still on for dinner", 700000000, 0, 0),
                    ("yes see you at 7", 700000030, 1, 0),
                    ("great, bringing wine", 700000060, 0, 0),
                ],
            }
        ],
    )
    index_db_path = tmp_path / "index.db"
    build_index(chat_db_path, index_db_path, embedding_model=FakeEmbeddingModel())
    return chat_db_path, index_db_path


class TestHydrateSessions:
    def test_returns_hit_and_context_messages_with_text(self, tmp_path):
        chat_db_path, index_db_path = _built_fixture(tmp_path)
        index_con = open_index_db(index_db_path)
        chat_con = connect_readonly(chat_db_path)
        chunk_id = index_con.execute("SELECT id FROM chunks LIMIT 1").fetchone()[0]

        session = Session(chat_id=1, day="2022-03-14", score=3.0, hit_chunk_ids=[chunk_id], context_chunk_ids=[])
        hydrated = hydrate_sessions(index_con, chat_con, [session])

        assert len(hydrated) == 1
        h = hydrated[0]
        assert h.chat_id == 1
        assert h.score == 3.0
        assert len(h.hit_messages) == 3  # all 3 messages are in the one chunk
        assert h.context_messages == []
        texts = [m.text for m in h.hit_messages]
        assert "hey are we still on for dinner" in texts

    def test_resolves_is_from_me_sender_to_none(self, tmp_path):
        chat_db_path, index_db_path = _built_fixture(tmp_path)
        index_con = open_index_db(index_db_path)
        chat_con = connect_readonly(chat_db_path)
        chunk_id = index_con.execute("SELECT id FROM chunks LIMIT 1").fetchone()[0]
        session = Session(chat_id=1, day="2022-03-14", score=1.0, hit_chunk_ids=[chunk_id], context_chunk_ids=[])
        hydrated = hydrate_sessions(index_con, chat_con, [session])[0]

        from_me_msgs = [m for m in hydrated.hit_messages if m.is_from_me]
        others_msgs = [m for m in hydrated.hit_messages if not m.is_from_me]
        assert all(m.sender is None for m in from_me_msgs)
        assert all(m.sender == "+15551234567" for m in others_msgs)

    def test_empty_chunk_ids_yields_empty_message_list(self, tmp_path):
        chat_db_path, index_db_path = _built_fixture(tmp_path)
        index_con = open_index_db(index_db_path)
        chat_con = connect_readonly(chat_db_path)
        session = Session(chat_id=1, day="2022-03-14", score=0.0, hit_chunk_ids=[], context_chunk_ids=[])
        hydrated = hydrate_sessions(index_con, chat_con, [session])[0]
        assert hydrated.hit_messages == []
        assert hydrated.context_messages == []
