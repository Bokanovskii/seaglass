"""Unit tests for seaglass.search.format -- Phase 5 payload shaping:
confidence signal, max_sessions truncation, and redaction.
"""

from __future__ import annotations

from seaglass.search.format import format_search_result
from seaglass.search.hydrate import HydratedMessage, HydratedSession


def _session(chat_id, score, texts, senders=None):
    senders = senders or [None] * len(texts)
    messages = [
        HydratedMessage(
            message_id=i,
            ts=700000000.0 + i,
            is_from_me=(sender is None),
            sender=sender,
            text=text,
            has_attachment=False,
        )
        for i, (text, sender) in enumerate(zip(texts, senders))
    ]
    return HydratedSession(chat_id=chat_id, day="2022-03-14", score=score, hit_messages=messages, context_messages=[])


class TestFormatSearchResult:
    def test_shape_and_counts(self):
        sessions = [_session(1, 5.0, ["hello", "world"]), _session(2, 3.0, ["hi"])]
        payload = format_search_result(sessions)
        assert payload["n_sessions"] == 2
        assert payload["n_results"] == 3
        assert len(payload["sessions"]) == 2
        assert payload["sessions"][0]["chat_id"] == 1
        assert payload["sessions"][0]["messages"][0]["message_id"] == 0

    def test_confidence_none_high_low_thresholds(self):
        assert format_search_result([])["confidence"] == "none"
        assert format_search_result([_session(1, 1.0, ["x"])])["confidence"] == "low"
        three = [_session(i, 1.0, ["x"]) for i in range(3)]
        assert format_search_result(three)["confidence"] == "high"

    def test_max_sessions_truncates(self):
        sessions = [_session(i, float(10 - i), ["x"]) for i in range(5)]
        payload = format_search_result(sessions, max_sessions=2)
        assert payload["n_sessions"] == 2
        assert [s["chat_id"] for s in payload["sessions"]] == [0, 1]

    def test_redaction_strips_phone_and_email(self):
        sessions = [
            _session(
                1,
                1.0,
                ["call me at 555-123-4567 or email me at a@b.com"],
                senders=["+15559998888"],
            )
        ]
        payload = format_search_result(sessions, redact=True)
        text = payload["sessions"][0]["messages"][0]["text"]
        assert "555-123-4567" not in text
        assert "a@b.com" not in text
        assert "[phone]" in text
        assert "[email]" in text
        sender = payload["sessions"][0]["messages"][0]["sender"]
        assert sender == "[phone]"

    def test_no_redaction_by_default(self):
        sessions = [_session(1, 1.0, ["call 555-123-4567"])]
        payload = format_search_result(sessions)
        assert "555-123-4567" in payload["sessions"][0]["messages"][0]["text"]
