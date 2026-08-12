"""Unit tests for seaglass.index.render -- pure logic, synthetic Message
objects only.
"""

from __future__ import annotations

from seaglass.imessage.source import AttachmentRow, Message
from seaglass.index.render import (
    approx_token_count,
    format_lexical,
    format_semantic,
)


def _msg(rowid, chat_id=1, ts=1000.0, is_from_me=False, handle="+15551234567",
         text=None, has_attachment=False):
    return Message(
        rowid=rowid,
        chat_id=chat_id,
        ts=ts,
        is_from_me=is_from_me,
        handle=None if is_from_me else handle,
        text=text,
        date_edited=None,
        date_retracted=None,
        has_attachment=has_attachment,
    )


class TestFormatSemantic:
    def test_roles_labeled_me_and_them_for_dm(self):
        messages = [
            _msg(1, is_from_me=False, text="hey are you free tonight?"),
            _msg(2, is_from_me=True, text="yeah what's up"),
        ]
        rendered = format_semantic(messages)
        assert "Them: hey are you free tonight?" in rendered
        assert "Me: yeah what's up" in rendered

    def test_group_chat_gets_letter_labels_not_names(self):
        messages = [
            _msg(1, handle="+15550001111", text="who's driving"),
            _msg(2, handle="+15552223333", text="I can"),
            _msg(3, handle="+15550001111", text="great see you at 6"),
        ]
        rendered = format_semantic(messages)
        assert "A: who's driving" in rendered
        assert "B: I can" in rendered
        assert "A: great see you at 6" in rendered
        # never leak the raw handle/phone number as a label
        assert "+15550001111" not in rendered

    def test_urls_collapsed_to_domain_only(self):
        messages = [_msg(1, text="check this out https://www.example.com/path?x=1")]
        rendered = format_semantic(messages)
        assert "[link:example.com]" in rendered
        assert "https://" not in rendered

    def test_attachment_becomes_bare_placeholder(self):
        messages = [_msg(1, text=None, has_attachment=True)]
        rendered = format_semantic(messages)
        assert rendered.strip().endswith("[attachment]")

    def test_message_with_no_text_and_no_attachment_is_skipped(self):
        messages = [
            _msg(1, text=None, has_attachment=False),
            _msg(2, text="hello", has_attachment=False),
        ]
        rendered = format_semantic(messages)
        assert rendered.strip() == "Them: hello"

    def test_truncates_with_middle_drop_when_over_cap(self):
        # 600 tokens, one per message-line word so we can track survivors
        words = [f"tok{i}" for i in range(600)]
        messages = [_msg(1, text=" ".join(words))]
        rendered = format_semantic(messages, max_tokens=100)
        survivors = rendered.split()
        assert len(survivors) == 100
        # opening preserved (role label + first content tokens)
        assert survivors[0] == "Them:"
        assert "tok0" in survivors
        # closing preserved
        assert "tok599" in rendered

    def test_no_place_names_in_semantic_render(self):
        messages = [_msg(1, text=None, has_attachment=True)]
        rendered = format_semantic(messages)
        assert "Lisbon" not in rendered


class TestFormatLexical:
    def test_urls_kept_verbatim(self):
        messages = [_msg(1, text="see https://example.com/foo?bar=1")]
        rendered = format_lexical(messages)
        assert "https://example.com/foo?bar=1" in rendered

    def test_no_role_labels(self):
        messages = [
            _msg(1, is_from_me=False, text="hello there"),
            _msg(2, is_from_me=True, text="hi"),
        ]
        rendered = format_lexical(messages)
        assert "Me:" not in rendered
        assert "Them:" not in rendered
        assert "hello there" in rendered
        assert "hi" in rendered

    def test_attachment_placeholder_includes_place_and_filename(self):
        messages = [_msg(1, text=None, has_attachment=True)]
        attachments_by_msg = {1: [AttachmentRow(attachment_id=42, message_id=1, filename="sunset.jpg")]}
        places_by_attachment = {42: "Lisbon Alfama Lisboa Portugal"}
        rendered = format_lexical(messages, attachments_by_msg, places_by_attachment)
        assert "Lisbon Alfama Lisboa Portugal" in rendered
        assert "sunset.jpg" in rendered
        assert "[attachment" in rendered

    def test_attachment_with_no_place_or_filename_falls_back_to_bare(self):
        messages = [_msg(1, text=None, has_attachment=True)]
        rendered = format_lexical(messages)
        assert "[attachment]" in rendered

    def test_no_length_cap(self):
        words = [f"tok{i}" for i in range(2000)]
        messages = [_msg(1, text=" ".join(words))]
        rendered = format_lexical(messages)
        assert "tok0" in rendered
        assert "tok1999" in rendered
        assert len(rendered.split()) == 2000


class TestApproxTokenCount:
    def test_counts_whitespace_separated_tokens(self):
        assert approx_token_count("hello there friend") == 3

    def test_empty_string_is_zero(self):
        assert approx_token_count("") == 0
