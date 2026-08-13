"""`search/rank.py` — Phase 4b, steps 5-6 of PLAN.md §6 Phase 4: rerank
the fused candidates with the cross-encoder, then aggregate into
`(chat_id, day)` sessions and expand each session with ±2 time-adjacent
context chunks.

Deliberately separate from `search/retrieve.py` (Phase 4a, baseline
retriever with no reranker) so the two phases stay independently
switchable per PLAN.md's explicit instruction: "Phase 4a (baseline
retriever) → Phase 3.5 (golden set) → Phase 4b (rerank, expansion,
tuning against real numbers)".

Session "day" boundaries use the local system timezone (not UTC) — day
grouping is meant to approximate how a person remembers "that day we
talked about X", and `chat.db` timestamps have no stored timezone of
their own, so local wall-clock day is the closest approximation
available. Not explicitly specified in PLAN.md; recorded here and in
ADDENDUM.md as a judgment call.
"""

from __future__ import annotations

import dataclasses
import time
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import zstandard

from seaglass.search.rerank import CrossEncoderReranker
from seaglass.search.retrieve import RetrievalResult

RERANK_TOP_K = 12
# Slots (out of RERANK_TOP_K) reserved for the newest candidates. Every
# fused candidate is scored in the same batched forward pass regardless of
# the cut, so widening this costs no extra model compute -- it only decides
# which scored chunks are allowed to reach session aggregation. Without it
# the recency shaping below is unreachable for any chunk the cross-encoder
# happens to rate poorly (see RECENT_SESSION_SLOTS).
RERANK_RECENT_SLOTS = 3
EXPANSION_RADIUS = 2
MAX_SESSIONS = 8

# Recency shaping for the final session ordering. A session's relevance
# score is multiplied by 1 + RECENCY_MAX_BOOST * 0.5**(age / half-life),
# i.e. a session from today is worth up to 1.6x one from the distant past.
# Bounded on purpose: recency should break ties between comparably
# relevant sessions, not let a weak recent match outrank a strong old one.
RECENCY_HALF_LIFE_DAYS = 365.0
RECENCY_MAX_BOOST = 0.6

# Sessions (out of MAX_SESSIONS) reserved for the most recent matches.
#
# The cross-encoder scores a whole conversation *chunk*, so a query that
# matches one short message inside a long window scores terribly -- a
# message sent today reading exactly "Sweet" sits in a 22-message chunk
# that scores -4.89 for the query "sweet". Relevance alone therefore can
# never surface it, no matter how large a recency multiplier is applied,
# and the user concludes search simply cannot find what they just said.
#
# The sparse/BM25 arm *did* find that chunk -- that signal is what gets
# thrown away at ranking time. Reserving a small number of slots for the
# newest matched sessions puts it back, bounded so the majority of results
# stay purely relevance-ordered.
RECENT_SESSION_SLOTS = 2

_dctx = zstandard.ZstdDecompressor()


@dataclasses.dataclass(frozen=True)
class RankedChunk:
    chunk_id: int
    chat_id: int
    start_ts: int
    rerank_score: float


@dataclasses.dataclass
class Session:
    chat_id: int
    day: str  # local-timezone "YYYY-MM-DD", the (chat_id, day) group key
    score: float  # sum of member chunks' rerank_score -- PLAN.md §6 step 6
    hit_chunk_ids: List[int]  # chunks that actually contributed to ranking
    context_chunk_ids: List[int]  # ±2 expansion chunks -- zero ranking weight


def _day_key(start_ts: int) -> str:
    return datetime.fromtimestamp(start_ts).strftime("%Y-%m-%d")


def rerank_candidates(
    index_con,
    query: str,
    fused: Sequence[RetrievalResult],
    reranker: CrossEncoderReranker,
    top_k: int = RERANK_TOP_K,
    *,
    recent_slots: int = RERANK_RECENT_SLOTS,
) -> List[RankedChunk]:
    """Step 5: score every fused candidate against `query` in one batched
    forward pass, return the top `top_k` by rerank score (descending).
    Reads `body_semantic` for the candidates only (PLAN.md §6 step 5 --
    this is why `body_semantic` is stored uncompressed-per-chunk).
    """
    if not fused:
        return []
    ids = [r.chunk_id for r in fused]
    placeholders = ",".join("?" for _ in ids)
    rows = index_con.execute(
        f"SELECT id, chat_id, start_ts, body_semantic FROM chunks WHERE id IN ({placeholders})",
        ids,
    ).fetchall()
    by_id = {row[0]: row for row in rows}

    pairs: List[Tuple[str, str]] = []
    ordered_ids: List[int] = []
    for chunk_id in ids:
        row = by_id.get(chunk_id)
        if row is None:
            continue  # defensive: a candidate id that vanished between fuse and rerank
        _, _, _, compressed = row
        text = _dctx.decompress(compressed).decode("utf-8")
        pairs.append((query, text))
        ordered_ids.append(chunk_id)

    if not pairs:
        return []

    scores = reranker.score(pairs)
    by_score = sorted(zip(ordered_ids, scores), key=lambda pair: pair[1], reverse=True)
    ranked = _reserve_recent_chunks(by_score, by_id, top_k=top_k, slots=recent_slots)

    result: List[RankedChunk] = []
    for chunk_id, score in ranked:
        _, chat_id, start_ts, _ = by_id[chunk_id]
        result.append(
            RankedChunk(chunk_id=chunk_id, chat_id=chat_id, start_ts=start_ts, rerank_score=score)
        )
    return result


def _reserve_recent_chunks(
    by_score: Sequence[Tuple[int, float]],
    by_id: Dict[int, tuple],
    *,
    top_k: int,
    slots: int,
) -> List[Tuple[int, float]]:
    """Take the top `top_k - slots` candidates by rerank score, then fill
    the remaining slots with the newest of whatever is left. Always returns
    exactly min(top_k, len(by_score)) distinct candidates.
    """
    if slots <= 0 or len(by_score) <= top_k:
        return list(by_score[:top_k])

    keep = max(0, top_k - slots)
    selected = list(by_score[:keep])
    chosen = {chunk_id for chunk_id, _ in selected}

    remainder = [pair for pair in by_score[keep:] if pair[0] not in chosen]
    remainder.sort(key=lambda pair: by_id[pair[0]][2], reverse=True)
    selected.extend(remainder[: top_k - len(selected)])
    return selected


def _sigmoid(logit: float) -> float:
    # Cross-encoder logits are raw (Identity activation, see rerank.py's
    # score() docstring) and in practice mostly negative (median ~-7 over
    # real queries) -- summing raw logits across a session's hits
    # penalises sessions with *more* relevant hits (two decent -5/-6 hits
    # sums to -11, worse than one mediocre -9 hit), which silently drops
    # well-corroborated multi-hit sessions during the max_sessions cutoff.
    # Map to (0, 1) before summing so more relevant hits always help.
    import math

    return 1.0 / (1.0 + math.exp(-logit))


def _recency_multiplier(ts: float, now: float) -> float:
    """Exponential-decay boost in (1, 1 + RECENCY_MAX_BOOST]. Clamped at
    age >= 0 so a message with a slightly future timestamp (clock skew
    between devices is common in chat.db) can't earn an outsized boost.
    """
    age_days = max(0.0, (now - ts) / 86400.0)
    return 1.0 + RECENCY_MAX_BOOST * (0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS))


def aggregate_sessions(
    ranked: Sequence[RankedChunk],
    max_sessions: int = MAX_SESSIONS,
    *,
    now: Optional[float] = None,
    recent_slots: int = RECENT_SESSION_SLOTS,
) -> List[Session]:
    """Step 6a: dedup by chunk_id (a chunk can only appear once in `ranked`
    already, since it came from a single top-k rerank pass, but this stays
    defensive) then group by `(chat_id, day)`, summing each chunk's
    sigmoid-mapped rerank score (see `_sigmoid`), then scaling by how
    recent the session is (see `_recency_multiplier`). Returns sessions
    sorted by score descending, capped at `max_sessions`.

    `now` is injectable so tests are deterministic.
    """
    now = time.time() if now is None else now
    groups: Dict[Tuple[int, str], Session] = {}
    latest_ts: Dict[Tuple[int, str], float] = {}
    seen_chunk_ids: set = set()
    for chunk in ranked:
        if chunk.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk.chunk_id)
        key = (chunk.chat_id, _day_key(chunk.start_ts))
        session = groups.get(key)
        if session is None:
            session = Session(
                chat_id=chunk.chat_id, day=key[1], score=0.0, hit_chunk_ids=[], context_chunk_ids=[]
            )
            groups[key] = session
        session.score += _sigmoid(chunk.rerank_score)
        session.hit_chunk_ids.append(chunk.chunk_id)
        latest_ts[key] = max(latest_ts.get(key, 0.0), float(chunk.start_ts))

    for key, session in groups.items():
        session.score *= _recency_multiplier(latest_ts[key], now)

    sessions = sorted(groups.values(), key=lambda s: s.score, reverse=True)
    return _reserve_recent_sessions(
        sessions, latest_ts, max_sessions=max_sessions, slots=recent_slots
    )


def _reserve_recent_sessions(
    sessions: List[Session],
    latest_ts: Dict[Tuple[int, str], float],
    *,
    max_sessions: int,
    slots: int,
) -> List[Session]:
    """Keep the top relevance-ranked sessions, but guarantee the newest
    matches a place in the results. See RECENT_SESSION_SLOTS.
    """
    if slots <= 0 or len(sessions) <= max_sessions:
        return sessions[:max_sessions]

    keep = max(0, max_sessions - slots)
    selected = sessions[:keep]
    selected_keys = {(s.chat_id, s.day) for s in selected}

    newest = sorted(
        (s for s in sessions if (s.chat_id, s.day) not in selected_keys),
        key=lambda s: latest_ts[(s.chat_id, s.day)],
        reverse=True,
    )
    for session in newest[: max_sessions - len(selected)]:
        selected.append(session)
        selected_keys.add((session.chat_id, session.day))

    # Re-sort so the reserved entries slot into a sensible order rather
    # than always appearing last.
    return sorted(selected, key=lambda s: s.score, reverse=True)[:max_sessions]


def expand_sessions(index_con, sessions: Sequence[Session], radius: int = EXPANSION_RADIUS) -> None:
    """Step 6b: mutate `sessions` in place, populating `context_chunk_ids`
    with up to `radius` time-adjacent chunks on each side of every hit
    chunk, within the same chat. Context chunks contribute **zero** to
    ranking (PLAN.md §6 step 6) -- this function never touches `.score`.

    Adjacency is by `id` ordering within a chat, not raw id arithmetic
    across chats: `build_index` assigns ids in (chat_id ascending,
    chronological) order, so same-chat chunks are id-contiguous, but
    verifying that defensively -- rather than assuming id±1 is always the
    same chat -- costs one extra query per session and is worth it.
    """
    for session in sessions:
        rows = index_con.execute(
            "SELECT id FROM chunks WHERE chat_id = ? ORDER BY id", (session.chat_id,)
        ).fetchall()
        ordered_ids = [row[0] for row in rows]
        position_by_id = {cid: pos for pos, cid in enumerate(ordered_ids)}

        context_ids: set = set()
        for hit_id in session.hit_chunk_ids:
            pos = position_by_id.get(hit_id)
            if pos is None:
                continue
            lo, hi = max(0, pos - radius), min(len(ordered_ids), pos + radius + 1)
            for neighbor_id in ordered_ids[lo:hi]:
                if neighbor_id not in session.hit_chunk_ids:
                    context_ids.add(neighbor_id)
        session.context_chunk_ids = sorted(context_ids)
