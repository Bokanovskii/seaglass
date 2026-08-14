"""The behaviour suite's properties must be able to fail.

A full 208-case run once reported 1.00 on every property and 0 failing
queries. That was not a result: three properties were structurally
incapable of failing (`"key" in payload`), and the one that read ordering
was handed a payload the scorer had already sorted into the expected
shape. Two real defects shipped in Grogu underneath that clean report.

Every test here feeds `check_properties` a payload that violates exactly
one property and asserts that property says so. A property that stops
discriminating fails these tests instead of quietly reading 1.00 forever.
"""
from types import SimpleNamespace

import pytest

from seaglass.eval.behavior import Case, _flat_in_caller_order, check_properties


def _parsed(**overrides):
    base = dict(
        people_sender=[], people_participant=[], date_from=None, date_to=None,
        from_me=False, has_media=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _oracle(rows=()):
    return SimpleNamespace(messages_from=lambda *a, **k: list(rows))


def _payload(**overrides):
    base = {
        "sessions": [],
        "index_stale": False,
        "n_messages_since_index": 0,
        "ordering": "relevance",
    }
    base.update(overrides)
    return base


def _props(payload, case=None, parsed=None, oracle=None):
    return check_properties(
        case or Case("q", "test"), payload, parsed or _parsed(), oracle or _oracle()
    )


class TestFreshnessIsJudgedNotCounted:
    """`freshness_declared` was `"index_stale" in payload`, so it passed on
    every case ever run. Staleness is a claim about messages the answer
    could not see, so it has to agree with the count of them."""

    def test_an_honest_payload_passes(self):
        assert _props(_payload())["freshness_declared"] is True

    def test_stale_claimed_with_nothing_behind_fails(self):
        assert _props(_payload(index_stale=True))["freshness_declared"] is False

    def test_messages_behind_but_not_marked_stale_fails(self):
        payload = _payload(n_messages_since_index=12)
        assert _props(payload)["freshness_declared"] is False

    def test_a_missing_freshness_field_fails(self):
        payload = _payload()
        del payload["index_stale"]
        assert _props(payload)["freshness_declared"] is False


class TestOrderingIsJudgedNotCounted:
    """`ordering_declared` was `"ordering" in payload`. A declaration a
    caller acts on -- Grogu re-sorts on it -- has to be true."""

    def test_a_known_ordering_passes(self):
        assert _props(_payload())["ordering_declared"] is True

    def test_an_unknown_ordering_value_fails(self):
        assert _props(_payload(ordering="whatever"))["ordering_declared"] is False

    def test_recent_declared_but_emitted_oldest_first_fails(self):
        payload = _payload(
            ordering="recent",
            _caller_order=[{"ts": 1.0, "_kind": "hit"}, {"ts": 9.0, "_kind": "hit"}],
        )
        assert _props(payload)["ordering_declared"] is False

    def test_recent_declared_and_actually_newest_first_passes(self):
        payload = _payload(
            ordering="recent",
            _caller_order=[{"ts": 9.0, "_kind": "hit"}, {"ts": 1.0, "_kind": "hit"}],
        )
        assert _props(payload)["ordering_declared"] is True


class TestContextNeverOutranksAMatch:
    """The property this replaced asked whether the first session was
    non-empty -- which `no_empty_sessions` already answers -- and so read
    1.00 while Grogu shipped a session's context above a later session's
    match."""

    def test_context_trailing_every_hit_passes(self):
        payload = _payload(_caller_order=[
            {"ts": 3.0, "_kind": "hit"},
            {"ts": 2.0, "_kind": "hit"},
            {"ts": 1.0, "_kind": "context"},
        ])
        assert _props(payload)["context_after_hits"] is True

    def test_context_above_a_later_hit_fails(self):
        payload = _payload(_caller_order=[
            {"ts": 3.0, "_kind": "hit"},
            {"ts": 1.0, "_kind": "context"},
            {"ts": 2.0, "_kind": "hit"},
        ])
        assert _props(payload)["context_after_hits"] is False

    def test_a_nested_payload_is_not_graded(self):
        """A target that returns sessions and leaves flattening to its
        caller has no single order to be wrong about."""
        payload = _payload(sessions=[{"messages": [{"ts": 1.0}], "context_messages": []}])
        assert _props(payload)["context_after_hits"] is None


class TestTheScorerReadsTheOrderItWasGiven:
    """`_flat_in_caller_order` rebuilt the caller's view as "every
    session's hits, then every session's context". For a target that emits
    its own flat list that is not an observation, it is a repair: it sorts
    the answer into the shape the properties expect before grading it."""

    def test_an_emitted_order_is_read_back_verbatim(self):
        order = [
            {"message_id": 1, "_kind": "hit"},
            {"message_id": 9, "_kind": "context"},
            {"message_id": 2, "_kind": "hit"},
        ]
        flat = _flat_in_caller_order(_payload(_caller_order=order))
        assert [m["message_id"] for m in flat] == [1, 9, 2]

    def test_a_nested_payload_is_still_modelled_as_the_caller_will_flatten_it(self):
        payload = _payload(sessions=[
            {"messages": [{"message_id": 1}], "context_messages": [{"message_id": 9}]},
            {"messages": [{"message_id": 2}], "context_messages": []},
        ])
        flat = _flat_in_caller_order(payload)
        assert [m["message_id"] for m in flat] == [1, 2, 9]


class TestVerbatimRecallIsGradedWidely:
    """`lexical_presence` has caught more real defects than any other
    property and was being graded on 6 of 208 cases."""

    def test_a_missing_phrase_fails(self):
        case = Case("q", "lexical", lexical="the boat")
        payload = _payload(sessions=[{"messages": [{"text": "hi hi"}], "context_messages": []}])
        assert _props(payload, case=case)["lexical_presence"] is False

    def test_a_present_phrase_passes(self):
        case = Case("q", "lexical", lexical="the boat")
        payload = _payload(sessions=[
            {"messages": [{"text": "how was the Boat!"}], "context_messages": []}
        ])
        assert _props(payload, case=case)["lexical_presence"] is True

    def test_the_phrase_is_looked_for_in_the_order_a_caller_reads(self):
        """Including context, which is where a match demoted by one
        session and shipped by another ends up."""
        case = Case("q", "lexical", lexical="the boat")
        payload = _payload(_caller_order=[
            {"text": "hi hi", "_kind": "hit"},
            {"text": "how was the boat", "_kind": "context"},
        ], sessions=[{"messages": [{"text": "hi hi"}], "context_messages": []}])
        assert _props(payload, case=case)["lexical_presence"] is True


def test_the_suite_draws_verbatim_phrases_from_more_than_one_week():
    """The phrases were all taken from the newest 600 messages and capped
    at six, so verbatim recall was measured on one week of one
    conversation."""
    import inspect

    from seaglass.eval import suites

    signature = inspect.signature(suites._corpus_phrases)
    assert signature.parameters["limit"].default >= 24
    source = inspect.getsource(suites._corpus_phrases)
    assert "stride" in source, "phrases must be spread across the sample, not taken from its head"
