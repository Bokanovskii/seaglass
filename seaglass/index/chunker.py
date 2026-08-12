"""`index/chunker.py` — group a single chat's messages into session windows.

See development-plans/PLAN.md §6 Phase 2. Operates on one chat's messages,
already ordered by timestamp (the caller -- build.py / sync.py -- is
responsible for grouping by `chat_id`; chunks must never span chat
boundaries, so this module never needs to know about more than one chat at
a time).

Rules, in order of precedence (any one closes the current chunk):

* gap > `gap_threshold_s` since the previous message (default 45 min,
  "tune from Phase 0" per the plan -- kept as a parameter, not hardcoded,
  for the sweep in EVALUATION.md §9)
* ~`token_target` tokens accumulated (approximate count; see
  index/render.py's docstring on why this is a stand-in for the real
  tokenizer)
* `max_messages` messages (default 40)

Adjacent chunks overlap by `overlap` messages (default 2).
"""

from __future__ import annotations

import dataclasses
from typing import Iterator, List, Sequence

from seaglass.imessage.source import Message
from seaglass.index.render import approx_token_count

DEFAULT_GAP_THRESHOLD_S = 45 * 60
DEFAULT_TOKEN_TARGET = 400
DEFAULT_MAX_MESSAGES = 40
DEFAULT_OVERLAP = 2


@dataclasses.dataclass(frozen=True)
class Chunk:
    chat_id: int
    start_ts: float
    end_ts: float
    msg_ids: tuple  # message ROWIDs, in order -- the chunk_message rows to write
    has_attachment: bool


def _close_chunk(chat_id: int, messages: Sequence[Message]) -> Chunk:
    return Chunk(
        chat_id=chat_id,
        start_ts=messages[0].ts,
        end_ts=messages[-1].ts,
        msg_ids=tuple(m.rowid for m in messages),
        has_attachment=any(m.has_attachment for m in messages),
    )


def chunk_messages(
    messages: Sequence[Message],
    *,
    gap_threshold_s: float = DEFAULT_GAP_THRESHOLD_S,
    token_target: int = DEFAULT_TOKEN_TARGET,
    max_messages: int = DEFAULT_MAX_MESSAGES,
    overlap: int = DEFAULT_OVERLAP,
) -> Iterator[Chunk]:
    """Chunk one chat's messages, already ordered by `ts` ascending.

    Yields `Chunk` objects covering the whole input, in order. The tail
    chunk (the last one yielded) is the one Phase 7 sync must be able to
    reopen and extend -- see development-plans/PLAN.md §6 Phase 7 and
    ADDENDUM.md §4 for why "reopen the tail" needs care under backfill.
    """
    if not messages:
        return

    current: List[Message] = []
    token_count = 0

    def flush() -> Chunk:
        return _close_chunk(messages[0].chat_id, current)

    for msg in messages:
        if current:
            gap = msg.ts - current[-1].ts
            is_gap_close = gap > gap_threshold_s
            projected_tokens = token_count + approx_token_count(msg.text or "")
            should_close = (
                is_gap_close
                or projected_tokens > token_target
                or len(current) >= max_messages
            )
            if should_close:
                yield flush()
                # Overlap: the next chunk starts with the last `overlap`
                # messages of the one just closed -- but ONLY when the
                # close was a token/message-count limit, never a gap
                # close. A gap close means the conversation genuinely
                # paused (default 45min); carrying the pre-gap messages
                # into the next chunk corrupts that chunk's start_ts with
                # a timestamp from before the gap (sometimes years
                # earlier -- BUG-4, see ADDENDUM), mislabels its
                # (chat_id, day) session key, and pollutes its embedding
                # with content from an unrelated conversation.
                carried = current[-overlap:] if (overlap > 0 and not is_gap_close) else []
                current = list(carried)
                token_count = sum(approx_token_count(m.text or "") for m in current)
        current.append(msg)
        token_count += approx_token_count(msg.text or "")

    if current:
        yield flush()
