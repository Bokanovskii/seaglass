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

import re

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
    """"with X" semantics: chat_ids where these handles are a member.

    Membership is the union of two sources, not just `chat_handle_join`.
    That table is chat.db's *current* roster and it is not complete: on this
    machine a 3,614-message SMS group listed none of its long-standing
    participants, so "messages from Sachu" prefiltered his busiest chat
    away and answered from the scraps that survived. Someone who wrote
    messages in a chat was in that chat, whatever the roster says.
    """
    if not handle_ids:
        return set()
    placeholders = ",".join("?" for _ in handle_ids)
    roster = f"""
        SELECT DISTINCT chj.chat_id
        FROM im.chat_handle_join chj
        JOIN im.handle h ON h.ROWID = chj.handle_id
        WHERE h.id IN ({placeholders})
    """
    authored = f"""
        SELECT DISTINCT cmj.chat_id
        FROM im.message m
        JOIN im.handle h ON h.ROWID = m.handle_id
        JOIN im.chat_message_join cmj ON cmj.message_id = m.ROWID
        WHERE h.id IN ({placeholders}) AND m.is_from_me = 0
    """
    params = list(handle_ids)
    found = {row[0] for row in chat_con.execute(roster, params)}
    found |= {row[0] for row in chat_con.execute(authored, params)}
    return found


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

def resolve_chat_id_filter(chat_con, parsed_query) -> Optional[Set[int]]:
    """Which chats a query is allowed to answer from, or None for "any".

    Shared rather than inlined because the filters-only path answers from
    chat.db directly and must scope to exactly the same chats as the index
    would. Two copies of this would drift, and the drift would be silent:
    the answer would still look plausible, just drawn from a chat the user
    excluded.

    An empty set is meaningful -- it means "a filter resolved to no chats"
    and callers must fail closed on it, not treat it as unfiltered.
    """
    if chat_con is None:
        return None
    filters: list[Set[int]] = []
    if parsed_query.people_participant:
        filters.append(resolve_participant_chat_ids(chat_con, parsed_query.people_participant))
    if getattr(parsed_query, "is_group", None) is not None:
        filters.append(resolve_group_chat_ids(chat_con, parsed_query.is_group))
    if getattr(parsed_query, "chat_ids", None):
        filters.append(set(parsed_query.chat_ids))
    if not filters:
        return None
    chat_id_filter = filters[0]
    for subset in filters[1:]:
        chat_id_filter &= subset
    return chat_id_filter


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

    chat_id_filter = resolve_chat_id_filter(chat_con, parsed_query)

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


_FTS_TOKEN = re.compile(r"\w+", re.UNICODE)


def fts_match_query(query_text: str) -> str:
    """Turn arbitrary user text into an FTS5 MATCH expression.

    FTS5 MATCH is a *query language*, not a string: `-` is NOT, `:` is a
    column filter, `"` opens a phrase, and AND/OR/NOT are keywords. Passing
    raw user text meant "what about the boat?" and "re: lease" and
    "12-24" all raised, and the caller failed open to zero results -- so
    an ordinary question silently lost BM25 entirely and was answered by
    the vector half alone. It failed *quietly*, which is why it survived:
    the search still returned something plausible.

    Each word token is extracted and quoted individually, which keeps the
    implicit-AND semantics the unquoted form had while making every input
    syntactically inert. Returns "" when there is no word character to
    search for, which the caller treats as "no sparse results" -- the one
    case where that answer is actually correct.
    """
    tokens = _FTS_TOKEN.findall(query_text)
    return " ".join('"' + token + '"' for token in tokens)


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
    match = fts_match_query(query_text)
    if not match:
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
        params = [match, *candidate_ids, top_k]
    else:
        query = "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) ASC LIMIT ?"
        params = [match, top_k]
    try:
        return [row[0] for row in index_con.execute(query, params)]
    except Exception:
        # FTS5 MATCH raises on malformed query syntax (bare punctuation,
        # unbalanced quotes, etc.) -- fail open to "no sparse results"
        # rather than let a raw-text query ever error out the whole
        # retrieval.
        return []


def exact_phrase_chunk_ids(
    index_con, phrase: str, chunk_ids: Sequence[int]
) -> Set[int]:
    """Which of `chunk_ids` contain `phrase` as a literal phrase.

    Distinct from `sparse_search`, which ranks by BM25 over loose terms:
    this asks the yes/no question "did someone actually type this?". The
    phrase is wrapped in double quotes so FTS5 treats it as a phrase
    rather than as a bag of terms, with any embedded quotes doubled to
    escape them.

    Fails open to the empty set: a lexical bonus is an enhancement, and a
    malformed query must never break the search that would otherwise work.
    """
    phrase = phrase.strip()
    if not phrase or not chunk_ids:
        return set()

    matched: Set[int] = set()
    escaped = '"' + phrase.replace('"', '""') + '"'
    ids = list(chunk_ids)
    # Chunked to stay under SQLite's variable limit for large candidate pools.
    for start in range(0, len(ids), CANDIDATE_INLINE_LIMIT):
        batch = ids[start: start + CANDIDATE_INLINE_LIMIT]
        placeholders = ",".join("?" for _ in batch)
        try:
            rows = index_con.execute(
                f"SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? AND rowid IN ({placeholders})",
                [escaped, *batch],
            ).fetchall()
        except Exception:
            return set()
        matched.update(row[0] for row in rows)
    return matched


def phrase_search(
    index_con, phrase: str, candidate_ids: Optional[Set[int]] = None, top_k: int = 20
) -> List[int]:
    """Chunks containing `phrase` verbatim, best BM25 first.

    A fourth retrieval arm, and the only one that can answer "find the
    message where someone said exactly this". Dense similarity spreads a
    long sentence across hundreds of near-ties and BM25 over loose terms
    does the same, so a phrase the user pasted in could rank below the
    fused cut and never reach the reranker at all: "these people don't
    know the joys" sat in the index, matched FTS, and was unretrievable.

    Restricted to multi-word phrases -- a single word is what `sparse_search`
    already ranks, and promoting every chunk containing one common word
    would crowd out the arms that weigh relevance.
    """
    phrase = phrase.strip()
    if len(phrase.split()) < 2:
        return []
    escaped = '"' + phrase.replace('"', '""') + '"'
    sql = "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?"
    params: List[object] = [escaped, top_k]
    if candidate_ids is not None:
        if not candidate_ids:
            return []
        ids = list(candidate_ids)
        if len(ids) <= CANDIDATE_INLINE_LIMIT:
            placeholders = ",".join("?" for _ in ids)
            sql = (
                f"SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? "
                f"AND rowid IN ({placeholders}) ORDER BY rank LIMIT ?"
            )
            params = [escaped, *ids, top_k]
    try:
        rows = index_con.execute(sql, params).fetchall()
    except Exception:
        # A phrase bonus is an enhancement; a query FTS5 cannot parse must
        # never break the search that would otherwise work.
        return []
    matched = [row[0] for row in rows]
    if candidate_ids is not None and len(candidate_ids) > CANDIDATE_INLINE_LIMIT:
        matched = [cid for cid in matched if cid in candidate_ids]
    return matched


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
    # Each batch must be merged into one global ordering, not appended.
    # Sorting inside the batch and concatenating makes the result "batch
    # order, then recency", so a caller taking the top 50 got the newest
    # chunks *of whichever batch happened to come first* -- and batch order
    # comes from set iteration, i.e. chunk id. With 3,011 candidates,
    # "what did I tell Vamski" answered with yesterday's conversation and
    # dropped today's entirely.
    rows: List[tuple] = []
    for start in range(0, len(ids), CANDIDATE_INLINE_LIMIT):
        batch = ids[start:start + CANDIDATE_INLINE_LIMIT]
        placeholders = ",".join("?" for _ in batch)
        rows.extend(
            index_con.execute(
                f"SELECT id, start_ts FROM chunks WHERE id IN ({placeholders})",
                batch,
            )
        )
    rows.sort(key=lambda row: row[1], reverse=True)
    return [row[0] for row in rows]


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
    term_ids_out: Optional[List[int]] = None,
) -> List[RetrievalResult]:
    """The Phase 4a baseline pipeline: pre-filter -> dense + sparse ->
    RRF fuse -> top `fused_top_k`. No reranker (Phase 4b).
    """
    candidate_ids = build_candidate_chunk_ids(index_con, parsed_query, chat_con=chat_con)

    absmax_row = index_con.execute("SELECT value FROM meta WHERE key = 'int8_absmax'").fetchone()
    if absmax_row is None:
        # An index with no chunks has never been calibrated, because
        # calibration samples the chunks. That is an empty corpus, not a
        # broken file: someone searched before the first sync finished, and
        # they should get "no results yet" in the ordinary payload shape
        # rather than a stack trace. A *populated* index with no absmax is
        # still corrupt and still says so.
        if index_con.execute("SELECT 1 FROM chunks LIMIT 1").fetchone() is None:
            return []
        raise RuntimeError("index.db has no meta.int8_absmax -- has build_index() ever run?")
    absmax = float(absmax_row[0])

    query_vector = embedding_model.embed([QUERY_PREFIX + parsed_query.semantic])
    query_vector_int8 = quantize_int8(query_vector, absmax)[0]

    dense_ids = dense_search(index_con, query_vector_int8, candidate_ids, top_k=dense_top_k)
    sparse_ids = sparse_search(index_con, parsed_query.semantic, candidate_ids, top_k=sparse_top_k)
    ranked_lists = [dense_ids, sparse_ids]
    phrase_ids = phrase_search(index_con, parsed_query.semantic, candidate_ids)
    if phrase_ids:
        ranked_lists.append(phrase_ids)
    for extra_query in extra_sparse_queries:
        extra_ids = sparse_search(index_con, extra_query, candidate_ids, top_k=sparse_top_k)
        if extra_ids:
            ranked_lists.append(extra_ids)

    if term_ids_out is not None:
        # Which candidates matched on the query's terms at all. Session
        # ranking must not *score* on this -- BM25 matches "the" -- but the
        # recency reservation needs it to tell a recent session that is
        # about the query from one that is merely recent.
        term_ids_out.extend(dict.fromkeys(sparse_ids + phrase_ids))

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
