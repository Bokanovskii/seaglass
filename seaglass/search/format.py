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


def _format_message(message: HydratedMessage, redact: bool) -> dict:
    return {
        "message_id": message.message_id,
        "ts": message.ts,
        "is_from_me": message.is_from_me,
        "sender": _redact(message.sender) if redact else message.sender,
        "text": _redact(message.text) if redact else message.text,
        "has_attachment": message.has_attachment,
    }


def _format_session(session: HydratedSession, redact: bool) -> dict:
    return {
        "chat_id": session.chat_id,
        "day": session.day,
        "score": session.score,
        "messages": [_format_message(m, redact) for m in session.hit_messages],
        "context_messages": [_format_message(m, redact) for m in session.context_messages],
    }


def format_search_result(
    sessions: List[HydratedSession],
    *,
    max_sessions: Optional[int] = None,
    redact: bool = False,
) -> dict:
    """Shape hydrated sessions into the tool-result payload. `max_sessions`
    truncates (sessions already arrive best-first, by summed rerank
    score); `redact` strips phone numbers/emails from every message body
    and sender field.
    """
    truncated = sessions if max_sessions is None else sessions[:max_sessions]
    n_results = sum(len(s.hit_messages) for s in truncated)
    return {
        "n_sessions": len(truncated),
        "n_results": n_results,
        # thin retrieval (0-1 sessions, or a lone weak match) should read
        # differently to the calling model than a rich multi-session hit
        "confidence": "high" if len(truncated) >= 3 else ("low" if truncated else "none"),
        "sessions": [_format_session(s, redact) for s in truncated],
    }
