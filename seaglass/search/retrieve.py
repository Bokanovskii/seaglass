"""`search/retrieve.py` — Phase 4a baseline retriever: pre-filter → dense
(int8 KNN) + sparse (FTS5 BM25) → RRF fuse. No reranker/aggregation/
expansion yet -- those are Phase 4b, deliberately built and evaluated
separately (PLAN.md §6: "Phase 4a (baseline retriever) → Phase 3.5
(golden set) → Phase 4b").

**Known simplification vs PLAN.md's exact §6 Phase 4 spec:** PLAN.md
distinguishes "from X" (sender -- `message.handle_id` -> `chunk_message`)
from "with X" (participant -- current `chat_handle_join` membership,
including people who later left the chat). `search/parse.py` doesn't yet
distinguish the two prepositions in its output (`people_participant` is a
single list), so this module applies the broader "with" semantics
(current chat membership) to every extracted handle. This can only ever
be *too permissive* (a chunk from a chat the person is a current member
of, even if the specific message wasn't from them), never miss a
relevant chunk -- an acceptable Phase 4a simplification, revisit if the
golden-set eval (Phase 3.5) shows it hurts precision.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np

from seaglass.index.embed import EmbeddingModel, quantize_int8
from seaglass.search.parse import ParsedQuery

# BGE is an asymmetric embedder: queries need this instruction prefix,
# documents never do. Omitting it is a measurable quality regression
# (PLAN.md §6 Phase 4).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

DENSE_TOP_K = 200
SPARSE_TOP_K = 200
RRF_K = 60
FUSED_TOP_K = 50
CANDIDATE_INLINE_LIMIT = 2000

# Candidate slots (out of FUSED_TOP_K) reserved for the most recent
# matching chunks. Lexical/semantic scores are near-identical across
# hundreds of hits for a common word -- "sweet" matches 228 chunks, and
# BM25's ordering among them is essentially arbitrary -- so a message sent
# today can land at sparse rank 171 and never reach the reranker at all.
# That reads to the user as "search can't find what I just said".
#
# Reserving slots (rather than weighting recency into the fusion score)
# keeps relevance ordering intact and the reranker's workload identical:
# recent matches are guaranteed a *hearing*, and the cross-encoder still
# decides whether they deserve to be shown.
RECENCY_RESERVED_SLOTS = 10


@dataclasses.dataclass(frozen=True)
class RetrievalResult:
    chunk_id: int
    rrf_score: float


def resolve_participant_chat_ids(chat_con, handle_ids: Sequence[str]) -> Set[int]:
    """"with X" semantics: chat_ids where any of these handles is a
    CURRENT member (chat_handle_join), per PLAN.md §6 Phase 4.
    """
    if not handle_ids:
        return set()
    placeholders = ",".join("?" for _ in handle_ids)
    query = f"""
        SELECT DISTINCT chj.chat_id
        FROM im.chat_handle_join chj
        JOIN im.handle h ON h.ROWID = chj.handle_id
        WHERE h.id IN ({placeholders})
    """
    return {row[0] for row in chat_con.execute(query, list(handle_ids))}




def resolve_group_chat_ids(chat_con, is_group: bool) -> Set[int]:
    rows = chat_con.execute(
        """
        SELECT c.ROWID, c.style, COUNT(DISTINCT chj.handle_id)
        FROM im.chat c
        LEFT JOIN im.chat_handle_join chj ON chj.chat_id = c.ROWID
        GROUP BY c.ROWID, c.style
        """
    ).fetchall()
    matched = set()
    for chat_id, style, participant_count in rows:
        chat_is_group = int((style or 0) not in (45,)) if style is not None else participant_count > 1
        if bool(chat_is_group) is bool(is_group):
            matched.add(chat_id)
    return matched

def build_candidate_chunk_ids(
    index_con,
    parsed_query: ParsedQuery,
    chat_con=None,
) -> Optional[Set[int]]:
    """The pre-filter entry point: combines direct `chunks` column filters
    (dates, media) with a people-participant filter (resolved via
    `chat_con`, if provided) into one candidate chunk id set. Returns
    `None` if no filter applies at all -- callers must treat `None` as
    "no constraint", not "empty set".
    """
    conditions: List[str] = []
    params: List[object] = []

    if parsed_query.date_from is not None:
        conditions.append("start_ts >= ?")
        params.append(parsed_query.date_from)
    if parsed_query.date_to is not None:
        conditions.append("end_ts <= ?")
        params.append(parsed_query.date_to)
    if parsed_query.has_media:
        conditions.append("has_attachment = 1")

    chat_id_filter: Optional[Set[int]] = None
    if chat_con is not None:
        filters: list[Set[int]] = []
        if parsed_query.people_participant:
            filters.append(resolve_participant_chat_ids(chat_con, parsed_query.people_participant))
        if getattr(parsed_query, "is_group", None) is not None:
            filters.append(resolve_group_chat_ids(chat_con, parsed_query.is_group))
        if getattr(parsed_query, "chat_ids", None):
            filters.append(set(parsed_query.chat_ids))
        if filters:
            chat_id_filter = filters[0]
            for subset in filters[1:]:
                chat_id_filter &= subset

    if not conditions and chat_id_filter is None:
        return None

    candidate_ids: Optional[Set[int]] = None
    if conditions:
        where_clause = " AND ".join(conditions)
        rows = index_con.execute(f"SELECT id FROM chunks WHERE {where_clause}", params).fetchall()
        candidate_ids = {row[0] for row in rows}

    if chat_id_filter is not None:
        if not chat_id_filter:
            # A people filter that resolved to zero chats: fail closed
            # here (an explicit "with Bob" that matches no chat should
            # return nothing), rather than silently ignoring the filter.
            return set()
        placeholders = ",".join("?" for _ in chat_id_filter)
        rows = index_con.execute(
            f"SELECT id FROM chunks WHERE chat_id IN ({placeholders})", list(chat_id_filter)
        ).fetchall()
        people_candidate_ids = {row[0] for row in rows}
        candidate_ids = (
            people_candidate_ids if candidate_ids is None else (candidate_ids & people_candidate_ids)
        )

    return candidate_ids


def dense_search(
    index_con,
    query_vector_int8: np.ndarray,
    candidate_ids: Optional[Set[int]],
    top_k: int = DENSE_TOP_K,
) -> List[int]:
    """int8 KNN, constrained to `candidate_ids` if given. Returns chunk ids
    ranked best-first.
    """
    if candidate_ids is not None:
        if not candidate_ids:
            return []
        if len(candidate_ids) > CANDIDATE_INLINE_LIMIT:
            base = dense_search(index_con, query_vector_int8, None, top_k=max(top_k * 3, top_k))
            return [chunk_id for chunk_id in base if chunk_id in candidate_ids][:top_k]
        placeholders = ",".join("?" for _ in candidate_ids)
        query = (
            f"SELECT rowid FROM chunks_vec WHERE rowid IN ({placeholders}) "
            "AND embedding MATCH vec_int8(?) AND k = ?"
        )
        params = list(candidate_ids) + [query_vector_int8.tobytes(), top_k]
    else:
        query = "SELECT rowid FROM chunks_vec WHERE embedding MATCH vec_int8(?) AND k = ?"
        params = [query_vector_int8.tobytes(), top_k]
    return [row[0] for row in index_con.execute(query, params)]


def sparse_search(
    index_con,
    query_text: str,
    candidate_ids: Optional[Set[int]],
    top_k: int = SPARSE_TOP_K,
) -> List[int]:
    """BM25 over `chunks_fts`, constrained to `candidate_ids` if given.
    ⚠️ FTS5 bm25() returns negative values; best matches sort ascending
    (PLAN.md §6 Phase 4).
    """
    if not query_text.strip():
        return []
    if candidate_ids is not None:
        if not candidate_ids:
            return []
        if len(candidate_ids) > CANDIDATE_INLINE_LIMIT:
            base = sparse_search(index_con, query_text, None, top_k=max(top_k * 3, top_k))
            return [chunk_id for chunk_id in base if chunk_id in candidate_ids][:top_k]
        placeholders = ",".join("?" for _ in candidate_ids)
        query = (
            f"SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? AND rowid IN ({placeholders}) "
            "ORDER BY bm25(chunks_fts) ASC LIMIT ?"
        )
        params = [query_text, *candidate_ids, top_k]
    else:
        query = "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) ASC LIMIT ?"
        params = [query_text, top_k]
    try:
        return [row[0] for row in index_con.execute(query, params)]
    except Exception:
        # FTS5 MATCH raises on malformed query syntax (bare punctuation,
        # unbalanced quotes, etc.) -- fail open to "no sparse results"
        # rather than let a raw-text query ever error out the whole
        # retrieval.
        return []


def rrf_fuse(
    ranked_lists: Sequence[Sequence[int]],
    k: int = RRF_K,
    top_k: int = FUSED_TOP_K,
    weights: Optional[Sequence[float]] = None,
) -> List[RetrievalResult]:
    """Reciprocal Rank Fusion over any number of ranked (best-first) id
    lists. ⚠️ Retrieve deep, fuse, THEN truncate -- truncating each arm to
    top_k before fusion discards the long-tail corroboration signal RRF
    depends on (PLAN.md §6 Phase 4).

    `weights` scales each list's contribution (default 1.0 each), so a
    supporting signal like recency can inform the fusion without carrying
    the same authority as the dense/sparse relevance arms.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    scores: Dict[int, float] = {}
    for ranked_list, weight in zip(ranked_lists, weights):
        for rank, chunk_id in enumerate(ranked_list, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank)
    ordered = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return [RetrievalResult(chunk_id=cid, rrf_score=score) for cid, score in ordered[:top_k]]


def recency_ranked(index_con, chunk_ids: Iterable[int]) -> List[int]:
    """Order already-matched chunks newest-first.

    Used as a third RRF arm. Lexical/semantic scores are near-identical
    across hundreds of hits for a common word ("sweet" matches 228 chunks
    with essentially arbitrary BM25 ordering among them), so without this
    a message from today can land at sparse rank 171 and never reach the
    candidate pool at all -- the search silently can't find things the
    user just said. Restricted to chunks that already matched, so it adds
    *ordering*, never new unrelated results.
    """
    ids = list(chunk_ids)
    if not ids:
        return []
    ordered: List[int] = []
    for start in range(0, len(ids), CANDIDATE_INLINE_LIMIT):
        batch = ids[start:start + CANDIDATE_INLINE_LIMIT]
        placeholders = ",".join("?" for _ in batch)
        ordered.extend(
            row[0]
            for row in index_con.execute(
                f"SELECT id FROM chunks WHERE id IN ({placeholders}) ORDER BY start_ts DESC",
                batch,
            )
        )
    return ordered


def retrieve(
    index_con,
    parsed_query: ParsedQuery,
    embedding_model: EmbeddingModel,
    *,
    chat_con=None,
    dense_top_k: int = DENSE_TOP_K,
    sparse_top_k: int = SPARSE_TOP_K,
    rrf_k: int = RRF_K,
    fused_top_k: int = FUSED_TOP_K,
    extra_sparse_queries: Sequence[str] = (),
    recency_slots: int = RECENCY_RESERVED_SLOTS,
) -> List[RetrievalResult]:
    """The Phase 4a baseline pipeline: pre-filter -> dense + sparse ->
    RRF fuse -> top `fused_top_k`. No reranker (Phase 4b).
    """
    candidate_ids = build_candidate_chunk_ids(index_con, parsed_query, chat_con=chat_con)

    absmax_row = index_con.execute("SELECT value FROM meta WHERE key = 'int8_absmax'").fetchone()
    if absmax_row is None:
        raise RuntimeError("index.db has no meta.int8_absmax -- has build_index() ever run?")
    absmax = float(absmax_row[0])

    query_vector = embedding_model.embed([QUERY_PREFIX + parsed_query.semantic])
    query_vector_int8 = quantize_int8(query_vector, absmax)[0]

    dense_ids = dense_search(index_con, query_vector_int8, candidate_ids, top_k=dense_top_k)
    sparse_ids = sparse_search(index_con, parsed_query.semantic, candidate_ids, top_k=sparse_top_k)
    ranked_lists = [dense_ids, sparse_ids]
    for extra_query in extra_sparse_queries:
        extra_ids = sparse_search(index_con, extra_query, candidate_ids, top_k=sparse_top_k)
        if extra_ids:
            ranked_lists.append(extra_ids)

    fused = rrf_fuse(ranked_lists, k=rrf_k, top_k=fused_top_k)
    matched = list(dict.fromkeys([cid for arm in ranked_lists for cid in arm]))
    return reserve_recent_slots(
        index_con, fused, matched, top_k=fused_top_k, slots=recency_slots
    )


def reserve_recent_slots(
    index_con,
    fused: Sequence[RetrievalResult],
    matched_ids: Sequence[int],
    *,
    top_k: int = FUSED_TOP_K,
    slots: int = RECENCY_RESERVED_SLOTS,
) -> List[RetrievalResult]:
    """Guarantee that the newest matching chunks reach the reranker.

    Keeps the best `top_k - slots` candidates by fusion score, then fills
    the remainder with the most recent matched chunks not already present
    (falling back to more fusion-ranked candidates if there aren't enough
    recent ones). The result is still exactly `top_k` candidates, so the
    reranker's cost is unchanged. See RECENCY_RESERVED_SLOTS.
    """
    if slots <= 0 or not fused:
        return list(fused[:top_k])

    kept = list(fused[:max(0, top_k - slots)])
    kept_ids = {r.chunk_id for r in kept}
    room = top_k - len(kept)

    recent_pool = [cid for cid in recency_ranked(index_con, matched_ids) if cid not in kept_ids]
    additions: List[RetrievalResult] = []
    scores = {r.chunk_id: r.rrf_score for r in fused}
    for chunk_id in recent_pool[:room]:
        additions.append(RetrievalResult(chunk_id=chunk_id, rrf_score=scores.get(chunk_id, 0.0)))
        kept_ids.add(chunk_id)

    if len(kept) + len(additions) < top_k:
        # Not enough distinct recent matches -- give the slack back to the
        # fusion ranking rather than returning a short candidate list.
        for result in fused[len(kept):]:
            if result.chunk_id in kept_ids:
                continue
            additions.append(result)
            kept_ids.add(result.chunk_id)
            if len(kept) + len(additions) >= top_k:
                break

    return kept + additions
