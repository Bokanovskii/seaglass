"""Unit tests for seaglass.imessage.attributedbody."""

from __future__ import annotations

from seaglass.imessage.attributedbody import decode_attributed_body


def test_returns_none_for_garbage_blob():
    assert decode_attributed_body(b"not a real typedstream blob") is None


def test_returns_none_for_empty_blob():
    assert decode_attributed_body(b"") is None
