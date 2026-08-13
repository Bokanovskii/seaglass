"""`search/format.py` — Phase 5 (PLAN.md §6 Phase 5): shapes the final
hydrated sessions into the payload an MCP tool call returns. There is no
generation step here — the calling model reasons over what this returns,
so the payload is structured fields, not prose.

Design choices straight from PLAN.md §6 Phase 5:
- citations are `message_id` integers, never `message://` links (that's
  Mail.app's URL scheme and will not open iMessage)
- a `confidence`/`n_results` signal so a thin retrieval doesn't get
  reasoned over as if it were rich
- `max_sessions` is caller-controlled, trading recall against the
  consuming model's context budget
- optional redaction of phone numbers / email addresses, since this
  payload may be forwarded upstream by the calling agent
"""

from __future__ import annotations

import re
from typing import List, Optional

from seaglass.search.hydrate import HydratedMessage, HydratedSession

_PHONE_RE = re.compile(r"\+?\d[\d\-\s().]{7,}\d")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def _redact(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    text = _EMAIL_RE.sub("[email]", text)
    text = _PHONE_RE.sub("[phone]", text)
    return text


_WORD_RE = re.compile(r"[\w']+")

# Words too common to tell a matching message from its neighbours; without
# this every message containing "the" scores as a match.
_STOPWORDS = frozenset(
    """a an and are as at be but by can did do does for from had has have he her him his
    how i if in into is it its me my no not of on or our so that the their them then there
    these they this to up was we were what when where which who will with would you your""".split()
)


def query_terms(query: str) -> frozenset:
    """Content words from a query, for marking which messages matched."""
    return frozenset(
        word for word in _WORD_RE.findall((query or "").lower())
        if word not in _STOPWORDS and len(word) > 1
    )


def _match_score(text: Optional[str], terms: frozenset) -> int:
    if not terms or not text:
        return 0
    found = {word for word in _WORD_RE.findall(text.lower())}
    return len(terms & found)


def _format_message(message: HydratedMessage, redact: bool, terms: frozenset = frozenset()) -> dict:
    return {
        "message_id": message.message_id,
        "ts": message.ts,
        "is_from_me": message.is_from_me,
        "sender": _redact(message.sender) if redact else message.sender,
        "text": _redact(message.text) if redact else message.text,
        "has_attachment": message.has_attachment,
        "attachment_kind": message.attachment_kind,
        # How many of the query's content words this message contains.
        # A chunk is a ~22-message window, so every message in it is
        # returned as a "hit" -- without this a caller reading the first
        # few gets the matched message's neighbours instead of the match,
        # which for a one-line answer is the whole result.
        "match_score": _match_score(message.text, terms),
    }


def _format_session(session: HydratedSession, redact: bool, terms: frozenset = frozenset()) -> dict:
    return {
        "chat_id": session.chat_id,
        "day": session.day,
        "score": session.score,
        "messages": [_format_message(m, redact, terms) for m in session.hit_messages],
        "context_messages": [_format_message(m, redact, terms) for m in session.context_messages],
    }


def format_search_result(
    sessions: List[HydratedSession],
    *,
    max_sessions: Optional[int] = None,
    redact: bool = False,
    query: str = "",
) -> dict:
    """Shape hydrated sessions into the tool-result payload. `max_sessions`
    truncates (sessions already arrive best-first, by summed rerank
    score); `redact` strips phone numbers/emails from every message body
    and sender field.
    """
    terms = query_terms(query)
    truncated = sessions if max_sessions is None else sessions[:max_sessions]
    n_results = sum(len(s.hit_messages) for s in truncated)
    return {
        "n_sessions": len(truncated),
        "n_results": n_results,
        # thin retrieval (0-1 sessions, or a lone weak match) should read
        # differently to the calling model than a rich multi-session hit
        "confidence": "high" if len(truncated) >= 3 else ("low" if truncated else "none"),
        "sessions": [_format_session(s, redact, terms) for s in truncated],
    }
