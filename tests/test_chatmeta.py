from __future__ import annotations

from seaglass.app.chatmeta import ChatMetadataCache, classify_chat
from seaglass.imessage.source import connect_readonly

from conftest import build_fixture_chat_db


def test_classify_chat_uses_style_45_as_one_to_one():
    assert classify_chat(45, 1) is False
    assert classify_chat(43, 3) is True


def test_classify_chat_falls_back_to_participant_count_when_style_missing():
    assert classify_chat(None, 1) is False
    assert classify_chat(None, 2) is True


def test_chatmeta_cache_builds_titles_and_group_flags(tmp_path):
    chat_db = build_fixture_chat_db(
        tmp_path,
        [
            {'chat_id': 1, 'handles': ['+15551234567'], 'messages': [('hello', 700000000, False, 0)]},
            {'chat_id': 2, 'handles': ['+15550000001', '+15550000002'], 'messages': [('group hi', 700000010, False, 0)]},
        ],
    )
    import sqlite3
    writable = sqlite3.connect(chat_db)
    writable.execute('UPDATE chat SET style = 43 WHERE ROWID = 2')
    writable.commit()
    writable.close()
    con = connect_readonly(chat_db)
    cache = ChatMetadataCache.build(con)
    assert cache.get(1).title == '+15551234567'
    assert cache.get(1).is_group is False
    assert cache.get(2).is_group is True


def test_chatmeta_cache_resolves_titles_via_contact_index(tmp_path):
    from seaglass.imessage.contacts import Contact, ContactIndex

    chat_db = build_fixture_chat_db(
        tmp_path,
        [
            {'chat_id': 1, 'handles': ['+15551234567'], 'messages': [('hello', 700000000, False, 0)]},
        ],
    )
    con = connect_readonly(chat_db)
    contact_index = ContactIndex([Contact(identifier='abc', display_name='Alice Chen', handles=('+15551234567',))])
    cache = ChatMetadataCache.build(con, contact_index=contact_index)
    assert cache.get(1).title == 'Alice Chen'


def test_chatmeta_cache_falls_back_to_handle_when_contact_unresolved(tmp_path):
    from seaglass.imessage.contacts import Contact, ContactIndex

    chat_db = build_fixture_chat_db(
        tmp_path,
        [
            {'chat_id': 1, 'handles': ['+15559999999'], 'messages': [('hello', 700000000, False, 0)]},
        ],
    )
    con = connect_readonly(chat_db)
    contact_index = ContactIndex([Contact(identifier='abc', display_name='Alice Chen', handles=('+15551234567',))])
    cache = ChatMetadataCache.build(con, contact_index=contact_index)
    assert cache.get(1).title == '+15559999999'
