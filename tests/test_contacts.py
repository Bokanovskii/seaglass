"""Unit tests for seaglass.imessage.contacts -- exercises the pure-logic
parts (normalisation, fuzzy matching over an in-memory roster) without
touching the real Contacts framework.
"""

from __future__ import annotations

from seaglass.imessage.contacts import Contact, ContactIndex, _normalise_handle


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
