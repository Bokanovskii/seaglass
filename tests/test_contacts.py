"""Unit tests for seaglass.imessage.contacts -- exercises the pure-logic
parts (normalisation, fuzzy matching over an in-memory roster) without
touching the real Contacts framework.
"""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from seaglass.imessage import contacts as contacts_module
from seaglass.imessage.contacts import (
    Contact,
    ContactIndex,
    ContactsUnavailableError,
    _load_address_book_contacts,
    _normalise_handle,
)


def test_normalise_handle_lowercases_email():
    assert _normalise_handle("Alice@Example.com") == "alice@example.com"


def test_normalise_handle_formats_phone_e164():
    import phonenumbers

    example = phonenumbers.example_number("US")
    national_format = phonenumbers.format_number(example, phonenumbers.PhoneNumberFormat.NATIONAL)
    normalised = _normalise_handle(national_format, region="US")
    assert normalised.startswith("+1")


def _sample_index() -> ContactIndex:
    contacts = [
        Contact(identifier="1", display_name="Alice Chen", handles=("+15551234567", "alice@example.com")),
        Contact(identifier="2", display_name="Bob Smith", handles=("+15559876543",)),
    ]
    return ContactIndex(contacts)


class TestContactIndex:
    def test_resolve_handle_returns_display_name(self):
        index = _sample_index()
        assert index.resolve_handle("+15551234567") == "Alice Chen"

    def test_resolve_handle_returns_none_for_unknown(self):
        index = _sample_index()
        assert index.resolve_handle("+19998887777") is None

    def test_fuzzy_match_finds_close_name(self):
        index = _sample_index()
        matches = index.fuzzy_match("alice chen", threshold=80.0)
        names = {c.display_name for c in matches}
        assert "Alice Chen" in names

    def test_fuzzy_match_respects_threshold(self):
        index = _sample_index()
        # A wildly unrelated query should not match at a high threshold
        matches = index.fuzzy_match("zzz totally unrelated qqq", threshold=95.0)
        assert matches == []

    def test_handle_ids_for_names(self):
        index = _sample_index()
        handles = index.handle_ids_for_names("alice chen", threshold=80.0)
        assert "+15551234567" in handles
        assert "alice@example.com" in handles


def _index():
    return ContactIndex([
        Contact(identifier='1', display_name='Kaya Makivic', handles=('+15551234567',)),
        Contact(identifier='2', display_name='Rogers Mom', handles=('+15557654321',)),
        Contact(identifier='3', display_name='Home', handles=('+15550000000',)),
    ])


def test_fuzzy_match_ignores_case():
    # "kaya" scored 0 and "Kaya" scored 86, because rapidfuzz applies no
    # processor by default -- so every lower-cased name lost its filter.
    index = _index()
    assert index.handle_ids_for_names('kaya', threshold=85.0) == ['+15551234567']


def test_exact_name_matches_the_given_name_but_not_any_token():
    # Matching any token read "from my mom" as the contact "Rogers Mom".
    index = _index()
    assert index.handle_ids_for_exact_name('kaya') == ['+15551234567']
    assert index.handle_ids_for_exact_name('kaya makivic') == ['+15551234567']
    assert index.handle_ids_for_exact_name('mom') == []


def test_similar_given_name_accepts_one_transposition():
    index = _index()
    assert index.handle_ids_for_similar_given_name('kaay') == ['+15551234567']


def test_similar_given_name_rejects_distant_and_short_words():
    index = _index()
    assert index.handle_ids_for_similar_given_name('kayaking') == []
    assert index.handle_ids_for_similar_given_name('maya') == []  # different first letter
    assert index.handle_ids_for_similar_given_name('kay') == []   # too short to be sure


_ADDRESS_BOOK = "AddressBook-v22.abcddb"


def _write_address_book(path, records, phones=(), emails=()):
    """A Contacts database with only the columns seaglass reads.
    `records` rows are (Z_PK, ZUNIQUEID, ZFIRSTNAME, ZLASTNAME, ZNICKNAME, ZNAME)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE ZABCDRECORD (Z_PK INTEGER PRIMARY KEY, ZUNIQUEID TEXT, ZFIRSTNAME TEXT,
                                  ZLASTNAME TEXT, ZNICKNAME TEXT, ZNAME TEXT);
        CREATE TABLE ZABCDPHONENUMBER (Z_PK INTEGER PRIMARY KEY, ZOWNER INTEGER, ZFULLNUMBER TEXT);
        CREATE TABLE ZABCDEMAILADDRESS (Z_PK INTEGER PRIMARY KEY, ZOWNER INTEGER, ZADDRESS TEXT);
        """
    )
    con.executemany("INSERT INTO ZABCDRECORD VALUES (?, ?, ?, ?, ?, ?)", records)
    con.executemany("INSERT INTO ZABCDPHONENUMBER VALUES (?, ?, ?)", phones)
    con.executemany("INSERT INTO ZABCDEMAILADDRESS VALUES (?, ?, ?)", emails)
    con.commit()
    con.close()


def test_address_book_reads_every_source_with_the_framework_name_rule(tmp_path):
    import phonenumbers

    example = phonenumbers.example_number("US")
    national = phonenumbers.format_number(example, phonenumbers.PhoneNumberFormat.NATIONAL)
    e164 = phonenumbers.format_number(example, phonenumbers.PhoneNumberFormat.E164)
    _write_address_book(tmp_path / _ADDRESS_BOOK, [(1, "uid-local", "Local", "Person", None, None)])
    _write_address_book(
        tmp_path / "Sources" / "A" / _ADDRESS_BOOK,
        [
            (1, "uid-1", "Alice", "Chen", None, None),
            (2, "uid-2", "Robert", "Smith", "Bob", None),
            (3, "uid-group", None, None, None, "Family"),  # a group, not a person
        ],
        phones=[(1, 1, national)],
        emails=[(1, 1, "Alice@Example.com")],
    )

    contacts = _load_address_book_contacts(root=tmp_path)

    assert [(c.identifier, c.display_name, c.handles) for c in contacts] == [
        ("uid-local", "Local Person", ()),
        ("uid-1", "Alice Chen", (e164, "alice@example.com")),
        ("uid-2", "Bob", ()),
    ]


def test_address_book_is_unavailable_when_nothing_can_be_read(tmp_path):
    assert _load_address_book_contacts(root=tmp_path) is None
    (tmp_path / _ADDRESS_BOOK).write_bytes(b"not a database")
    assert _load_address_book_contacts(root=tmp_path) is None


def _denied_framework(initial_status):
    """A Contacts framework that denies access, as macOS does for an
    unbundled interpreter run by launchd: the request fails without a prompt."""
    state = {"status": initial_status}

    def request_access(_entity, completion):
        state["status"] = 2
        completion(False, None)

    store = SimpleNamespace(requestAccessForEntityType_completionHandler_=request_access)
    return SimpleNamespace(
        CNEntityTypeContacts=0,
        CNContactStore=SimpleNamespace(
            authorizationStatusForEntityType_=lambda _entity: state["status"],
            alloc=lambda: SimpleNamespace(init=lambda: store),
        ),
    )


@pytest.mark.parametrize("initial_status", [0, 2], ids=["request-denied", "already-denied"])
def test_load_falls_back_to_the_database_when_the_framework_is_denied(
    tmp_path, monkeypatch, initial_status
):
    _write_address_book(
        tmp_path / "Sources" / "A" / _ADDRESS_BOOK,
        [(1, "uid-1", "Alice", "Chen", None, None)],
        emails=[(1, 1, "alice@example.com")],
    )
    monkeypatch.setattr(contacts_module, "_Contacts", _denied_framework(initial_status))
    monkeypatch.setattr(contacts_module, "ADDRESS_BOOK_DIR", tmp_path)

    assert ContactIndex.load().resolve_handle("alice@example.com") == "Alice Chen"


def test_load_reports_the_denial_when_the_database_is_unreadable(tmp_path, monkeypatch):
    monkeypatch.setattr(contacts_module, "_Contacts", _denied_framework(2))
    monkeypatch.setattr(contacts_module, "ADDRESS_BOOK_DIR", tmp_path)

    with pytest.raises(ContactsUnavailableError, match="status=2"):
        ContactIndex.load()
