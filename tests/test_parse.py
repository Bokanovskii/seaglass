"""Unit tests for seaglass.search.parse -- deterministic query parsing.
No network, no MLX; ContactIndex is built in-memory (same fixture style
as test_contacts.py).
"""

from __future__ import annotations

import datetime as dt

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

    def test_may_as_modal_verb_is_not_a_date_false_positive(self):
        # Regression test for BUG-10: "may" is both a month name and a
        # common modal verb -- a bare occurrence with no other date
        # signal (digit, second date word) must not trigger a date filter.
        parsed = parse_query("may i borrow the car")
        assert parsed.date_from is None
        assert parsed.date_to is None
        assert "may" in parsed.semantic

    def test_may_with_a_day_number_is_still_a_real_date(self):
        parsed = parse_query("what did we plan for may 5th")
        assert parsed.date_from is not None
        assert parsed.date_to is not None


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

    def test_lowercase_word_after_preposition_is_not_mistaken_for_a_name(self):
        # Regression test for BUG-10: the participant regex's own
        # re.IGNORECASE flag defeated its capitalized-name heuristic,
        # letting ordinary lowercase words after from/with (e.g. "the",
        # "yesterday") match as if they were a person's name.
        index = _sample_index()
        parsed = parse_query("what did we plan with the plumber", contact_index=index)
        assert parsed.people_participant == []
        parsed2 = parse_query("what happened from yesterday onward", contact_index=index)
        assert parsed2.people_participant == []

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

    def test_filter_only_query_leaves_nothing_to_embed(self):
        # "march 2024" is a *time*, not a topic. Embedding the leftover
        # words would rank on the noise vector of a date; an empty semantic
        # routes the engine to its recency browse instead.
        parsed = parse_query("march 2024")
        assert parsed.semantic.strip() == ""

    def test_raw_is_preserved_unmodified(self):
        text = "photos from Alice in march 2024"
        parsed = parse_query(text)
        assert parsed.raw == text


class TestRelativeRanges:
    """A phrase naming a *span* of time must produce that span.

    dateparser resolves "last month" to a single instant a month ago, which
    the +/- pad then turns into a six-day window around an arbitrary day in
    the middle of it -- so "golf last month" searched six days, found
    nothing, and looked like a retrieval failure rather than a parsing one.
    """

    @staticmethod
    def _span_days(query):
        parsed = parse_query(query)
        assert parsed.date_from is not None, query
        return (parsed.date_to - parsed.date_from) / 86400.0

    def test_last_month_spans_a_month(self):
        assert 28 <= self._span_days('golf last month') <= 32

    def test_last_week_spans_a_week(self):
        assert 6.5 <= self._span_days('texts from last week') <= 8

    def test_last_year_spans_a_year(self):
        assert 360 <= self._span_days('stuff last year') <= 370

    def test_this_week_is_understood(self):
        # Previously produced no date filter at all.
        assert 6.5 <= self._span_days('plans this week') <= 8

    def test_named_day_is_one_day_and_does_not_run_into_the_future(self):
        parsed = parse_query('anything yesterday')
        assert 0.9 <= (parsed.date_to - parsed.date_from) / 86400.0 <= 1.1
        assert parsed.date_to <= time.time() + 1

    def test_named_month_spans_the_month(self):
        parsed = parse_query('photos from March 2024')
        assert dt.datetime.fromtimestamp(parsed.date_from).strftime('%Y-%m-%d') == '2024-03-01'
        assert 28 <= (parsed.date_to - parsed.date_from) / 86400.0 <= 32

    def test_ambiguous_month_words_are_still_not_dates(self):
        # The bare-modal guard must survive the new month handling.
        assert parse_query('may i borrow the car').date_from is None
        assert parse_query('march madness bracket').date_from is None

    def test_range_phrase_is_removed_from_the_semantic_residual(self):
        parsed = parse_query('texts from last week')
        # 'texts' is filler, not a topic -- nothing to embed, so browse.
        assert parsed.semantic == ''

    def test_explicit_day_count(self):
        assert 3 <= self._span_days('dinner in the past 3 days') <= 4.5


class _StubContacts:
    """Minimal ContactIndex stand-in: one contact, two handles."""

    def handle_ids_for_names(self, name, threshold=None):
        return ['+15550001111', 'kaya@example.com'] if name.lower() == 'kaya' else []

    def handle_ids_for_exact_name(self, name):
        return self.handle_ids_for_names(name)

    def handle_ids_for_similar_given_name(self, name):
        return []

    def resolve_handle(self, handle):
        return 'Kaya'


class TestSenderExtraction:
    def test_from_someone_is_a_sender_filter_not_just_a_participant_one(self):
        # The bug this guards: "messages from Jakie" narrowed to chats Jakie
        # is in and then answered with a *different* person's messages.
        parsed = parse_query('messages from Kaya', contact_index=_StubContacts())
        assert parsed.people_sender == ['+15550001111', 'kaya@example.com']

    def test_a_sender_is_also_a_participant(self):
        # Chunk candidates can only be narrowed by chat, so the sender must
        # appear in the participant filter or there is nothing to filter.
        parsed = parse_query('latest from Kaya', contact_index=_StubContacts())
        assert set(parsed.people_sender) <= set(parsed.people_participant)

    def test_past_tense_phrasing_names_the_sender(self):
        for text in ('what did Kaya say', 'the last thing Kaya sent me', 'Kaya texted about it'):
            parsed = parse_query(text, contact_index=_StubContacts())
            assert parsed.people_sender, text

    def test_with_someone_is_not_a_sender_filter(self):
        # "conversation with Kaya" wants both halves of the exchange.
        parsed = parse_query('conversation with Kaya', contact_index=_StubContacts())
        assert parsed.people_sender == []
        assert parsed.people_participant


class TestContentlessQueries:
    def test_recency_words_alone_route_to_browse(self):
        for text in ('recent messages from Kaya', 'latest from Kaya', 'what did Kaya say recently'):
            parsed = parse_query(text, contact_index=_StubContacts())
            assert parsed.semantic == '', text

    def test_a_real_topic_still_searches(self):
        for text in ('what did Kaya say about dinner', 'the last thing Kaya sent about golf'):
            parsed = parse_query(text, contact_index=_StubContacts())
            assert parsed.semantic.strip() != '', text

    def test_contractions_are_filler_too(self):
        parsed = parse_query("what's the last thing Kaya sent me", contact_index=_StubContacts())
        assert parsed.semantic == ''


class TestFirstPersonSender:
    def test_what_did_i_say_filters_to_me(self):
        parsed = parse_query('what did I say about the lease')
        assert parsed.from_me is True
        assert 'lease' in parsed.semantic

    def test_my_messages_filters_to_me(self):
        assert parse_query('my messages about dinner').from_me is True

    def test_a_named_sender_wins_over_the_pronoun(self):
        # "what did Kaya say when I asked about rent" is about Kaya.
        parsed = parse_query('what did Kaya say when I asked about rent')
        assert parsed.from_me is None

    def test_a_plain_question_is_not_a_self_filter(self):
        assert parse_query('what did we decide about dinner').from_me is None

    def test_the_pronoun_is_never_treated_as_a_contact_name(self):
        # "I" is capitalised like a name and was fuzzy-matched to contacts.
        parsed = parse_query('what did I say about rent')
        assert parsed.people_sender == []


class TestNameSurfaceForms:
    """TEST-EVAL-PLAN-V2.md §4. The suite only ever typed `Kaya`, so the
    parser's requirement of a leading capital went unmeasured and
    "recent messages from kaya" answered with a stranger's messages."""

    def test_lower_case_name_still_resolves(self):
        parsed = parse_query('recent messages from kaya', contact_index=_StubContacts())
        assert parsed.people_sender

    def test_upper_case_name_still_resolves(self):
        assert parse_query('messages from KAYA', contact_index=_StubContacts()).people_sender

    def test_quoted_name_still_resolves(self):
        assert parse_query('messages from "kaya"', contact_index=_StubContacts()).people_sender

    def test_possessive_is_a_sender_query(self):
        parsed = parse_query("kaya's latest messages", contact_index=_StubContacts())
        assert parsed.people_sender

    def test_trailing_punctuation_is_not_part_of_the_name(self):
        assert parse_query('messages from kaya?', contact_index=_StubContacts()).people_sender

    def test_a_date_word_is_not_a_person(self):
        # The capital requirement used to be what stopped this.
        for text in ('texts from yesterday', 'messages from monday', 'photos from june'):
            parsed = parse_query(text, contact_index=_StubContacts())
            assert not parsed.people_participant, text
            assert not parsed.people_sender, text

    def test_an_ordinary_word_is_not_a_person(self):
        for text in ('messages i sent from work', 'pictures from the wedding', 'notes from the meeting'):
            parsed = parse_query(text, contact_index=_StubContacts())
            assert not parsed.people_participant, text

    def test_the_extra_word_is_not_swallowed_into_the_name(self):
        parsed = parse_query('messages from kaya yesterday', contact_index=_StubContacts())
        assert parsed.people_sender
        assert parsed.date_from is not None
