"""Decode the legacy `attributedBody` blob stored on `message` rows.

⚠️ Terminology matters (PLAN.md §6 Phase 1): this blob is a legacy
**`typedstream`** (`NSArchiver`) serialisation, *not* an `NSKeyedArchiver`
keyed archive. Reaching for `plistlib` or a keyed-archive decoder will fail
silently or raise confusingly. Use the `typedstream` wire format directly.

This is called out in the plan as the **highest-risk extraction component**:
validate against messages you can visually confirm in Messages.app before
trusting it over a full corpus.
"""

from __future__ import annotations

from typing import Optional

from typedstream.stream import TypedStreamReader


def decode_attributed_body(blob: bytes) -> Optional[str]:
    """Best-effort extraction of the plain-text body from an `NSAttributedString`
    typedstream blob.

    Returns `None` if no text payload could be found (e.g. a tapback or a
    blob typedstream cannot parse) rather than raising -- callers are
    expected to fall back to `message.text` and then to a regex scan
    (PLAN.md §6 Phase 1).

    The body surfaces as a `bytes` event in the stream (the raw NSString
    payload) rather than a reconstructed Python `str`; typedstream does not
    do that reconstruction for us. The first non-empty `bytes` event is the
    string body in every sample observed so far -- there is exactly one
    `NSString` per plain-text `NSAttributedString` produced by Messages.
    """
    try:
        events = list(TypedStreamReader.from_data(blob))
    except Exception:
        return None
    for event in events:
        if isinstance(event, bytes) and event:
            try:
                return event.decode("utf-8")
            except UnicodeDecodeError:
                return event.decode("utf-8", errors="replace")
    return None
