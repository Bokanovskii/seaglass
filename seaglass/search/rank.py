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
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

import zstandard

from seaglass.search.rerank import CrossEncoderReranker
from seaglass.search.retrieve import RetrievalResult

RERANK_TOP_K = 12
EXPANSION_RADIUS = 2
MAX_SESSIONS = 8

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
    ranked = sorted(zip(ordered_ids, scores), key=lambda pair: pair[1], reverse=True)[:top_k]

    result: List[RankedChunk] = []
    for chunk_id, score in ranked:
        _, chat_id, start_ts, _ = by_id[chunk_id]
        result.append(
            RankedChunk(chunk_id=chunk_id, chat_id=chat_id, start_ts=start_ts, rerank_score=score)
        )
    return result


def aggregate_sessions(ranked: Sequence[RankedChunk], max_sessions: int = MAX_SESSIONS) -> List[Session]:
    """Step 6a: dedup by chunk_id (a chunk can only appear once in `ranked`
    already, since it came from a single top-k rerank pass, but this stays
    defensive) then group by `(chat_id, day)`, summing rerank scores.
    Returns sessions sorted by score descending, capped at `max_sessions`.
    """
    groups: Dict[Tuple[int, str], Session] = {}
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
        session.score += chunk.rerank_score
        session.hit_chunk_ids.append(chunk.chunk_id)

    sessions = sorted(groups.values(), key=lambda s: s.score, reverse=True)
    return sessions[:max_sessions]


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
