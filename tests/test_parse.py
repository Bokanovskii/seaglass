"""Unit tests for seaglass.search.parse -- deterministic query parsing.
No network, no MLX; ContactIndex is built in-memory (same fixture style
as test_contacts.py).
"""

from __future__ import annotations

import time

from seaglass.imessage.contacts import Contact, ContactIndex
from seaglass.search.parse import DATE_PAD_DAYS, parse_query


def _sample_index() -> ContactIndex:
    contacts = [
        Contact(identifier="1", display_name="Alice Chen", handles=("+15551234567", "alice@example.com")),
        Contact(identifier="2", display_name="Bob Smith", handles=("+15559876543",)),
    ]
    return ContactIndex(contacts)


class TestMediaFilter:
    def test_detects_photo_keyword(self):
        parsed = parse_query("find that photo of the sunset")
        assert parsed.has_media is True

    def test_detects_screenshot_keyword(self):
        parsed = parse_query("the screenshot alice sent")
        assert parsed.has_media is True

    def test_no_media_keyword_leaves_it_false(self):
        parsed = parse_query("what restaurant did we pick")
        assert parsed.has_media is False


class TestDateExtraction:
    def test_extracts_absolute_month_range(self):
        parsed = parse_query("lisbon trip plans in march 2024")
        assert parsed.date_from is not None
        assert parsed.date_to is not None
        assert parsed.date_from < parsed.date_to

    def test_no_date_leaves_both_none(self):
        parsed = parse_query("what did we say about the lisbon trip")
        assert parsed.date_from is None
        assert parsed.date_to is None

    def test_bare_pronoun_false_positive_is_rejected(self):
        # Regression test: dateparser.search_dates misreads "we" as "Wed".
        parsed = parse_query("the trip we took together")
        assert parsed.date_from is None
        assert "we" in parsed.semantic

    def test_date_range_is_padded(self):
        parsed = parse_query("plans on march 15 2024")
        assert parsed.date_to - parsed.date_from > DATE_PAD_DAYS * 86400 * 1.5


class TestParticipantExtraction:
    def test_from_preposition_extracts_participant_handle(self):
        index = _sample_index()
        parsed = parse_query("photos from Alice Chen last week", contact_index=index)
        assert "+15551234567" in parsed.people_participant

    def test_with_preposition_extracts_participant_handle(self):
        index = _sample_index()
        parsed = parse_query("dinner plans with Bob Smith", contact_index=index)
        assert "+15559876543" in parsed.people_participant

    def test_about_preposition_does_not_extract_a_filter(self):
        # "about Alice" is a mention, not participation -- must stay in the
        # semantic residual per PLAN.md's from/with vs about/re heuristic.
        index = _sample_index()
        parsed = parse_query("what did we say about Alice Chen", contact_index=index)
        assert parsed.people_participant == []
        assert "Alice" in parsed.semantic

    def test_no_contact_index_never_raises_and_extracts_nothing(self):
        parsed = parse_query("photos from Alice Chen")
        assert parsed.people_participant == []

    def test_unmatched_name_fails_open_stays_in_residual(self):
        index = _sample_index()
        parsed = parse_query("dinner with Zach Totally Unknown", contact_index=index)
        assert parsed.people_participant == []
        assert "Zach" in parsed.semantic


class TestSemanticResidual:
    def test_residual_excludes_matched_date_and_person_substrings(self):
        index = _sample_index()
        parsed = parse_query("lisbon trip photos from Alice Chen in march 2024", contact_index=index)
        assert "Alice Chen" not in parsed.semantic
        assert "lisbon" in parsed.semantic.lower()

    def test_residual_never_empty(self):
        # An entirely-filter query (hypothetically) must still leave
        # something for the embedder/FTS -- fail open, never fail closed.
        parsed = parse_query("march 2024")
        assert parsed.semantic.strip() != ""

    def test_raw_is_preserved_unmodified(self):
        text = "photos from Alice in march 2024"
        parsed = parse_query(text)
        assert parsed.raw == text
