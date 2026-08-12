"""`eval/generate.py` — EVALUATION.md §4: from `eval_candidate` rows to
draft golden-set entries awaiting human review (`eval/review.py`).

Pipeline:
  1. Stratified selection: drop the bottom nn_distance band (§3.3 --
     boilerplate, ambiguous positives), sample the rest by band quota
     (30% band 1, 40% band 2, 30% band 3) and by category, ~300 total.
  2. Batched question generation via `eval/ghcp_client.py` (never one
     call per chunk -- see that module's docstring).
  3. Automatic filters (§4.2): vocabulary-leakage (>40% content-word
     overlap -- reject), ambiguity check via real retrieval (multi-
     positive flag, or discard if >3 plausible answers), trivially-hard
     flag (source outside top 200 -- flag for review, never auto-drop).

Output: `candidates_for_review.jsonl`, one draft entry per surviving
question, in the golden.jsonl shape (EVALUATION.md §4.4) plus review-only
fields (`_vocab_overlap_pct`, `_sparse_rank1_is_source`,
`_trivially_hard`, `_top5_other_chunk_ids`) that `eval/review.py` reads
and `eval/harvest.py`'s consumers should strip before scoring.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from seaglass.imessage.contacts import ContactIndex, ContactsUnavailableError
from seaglass.imessage.source import connect_readonly
from seaglass.index.build import open_index_db
from seaglass.index.embed import EmbeddingModel
from seaglass.eval.ghcp_client import call_ghcp_json
from seaglass.search.parse import parse_query
from seaglass.search.retrieve import retrieve

BATCH_SIZE = 10  # chunks per ghcp call -- balances prompt size against overhead
SAMPLE_TARGET = 300
VOCAB_OVERLAP_REJECT_THRESHOLD = 0.40
TRIVIALLY_HARD_RANK = 200
_TOKEN_RE = re.compile(r"\w+")

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be", "been",
    "to", "of", "in", "on", "at", "for", "with", "about", "we", "us", "i", "you",
    "did", "do", "does", "what", "where", "when", "who", "how", "that", "this",
    "it", "was", "our", "their", "them", "he", "she", "his", "her", "me",
}

_PROMPT_TEMPLATE = """You will be given {n} numbered text snippets, each from a private group-chat conversation. For EACH snippet, write ONE terse search query that a person might type months later trying to recall that specific conversation.

Rules for every query:
- Phrase it the way someone actually types into a search box: terse, a fragment or a few keywords, NOT a full grammatical question. Do NOT write complete sentences like "what was" or "wasn't there something about" -- drop the question words and helper verbs entirely (e.g. "dinner plans with sam" not "what were the dinner plans with sam", "that thing about the flight" not "wasn't there something about a flight").
- Keep it vague, partial, uncertain -- like a fuzzy memory, not a precise recall of the snippet's content.
- Do NOT reuse distinctive words or phrases from the snippet.
- Do NOT include any dates or names, UNLESS the snippet's category (given after each snippet) is "time_filtered" or "person_filtered" -- for those specific categories, including a rough date or a name is fine and expected.
- Output the query text only, no preamble.

Output ONLY a JSON array, one object per snippet, in this exact shape and nothing else:
[{{"id": <snippet id>, "question": "..."}}, ...]

Snippets:
{snippets}
"""


def _select_stratified(index_con, target: int = SAMPLE_TARGET) -> List[int]:
    """§3.3: drop the bottom nn_distance band, sample the rest by band
    quota (30/40/30 across bands 1/2/3), filling from whichever category
    is thinnest first so no one category dominates the sample.
    """
    rows = index_con.execute(
        "SELECT chunk_id, nn_distance, category, score FROM eval_candidate "
        "WHERE nn_distance IS NOT NULL AND category IS NOT NULL"
    ).fetchall()
    if not rows:
        return []

    nn_distances = sorted(r[1] for r in rows)
    q1 = nn_distances[len(nn_distances) // 4]

    survivors = [r for r in rows if r[1] > q1]  # excludes bottom 25% (band 0)
    if not survivors:
        return []

    q2 = nn_distances[len(nn_distances) // 2]
    q3 = nn_distances[3 * len(nn_distances) // 4]

    def band_of(nn_distance: float) -> int:
        if nn_distance <= q2:
            return 1
        if nn_distance <= q3:
            return 2
        return 3

    band_quota = {1: 0.30, 2: 0.40, 3: 0.30}
    by_band: Dict[int, List[tuple]] = {1: [], 2: [], 3: []}
    for row in survivors:
        by_band[band_of(row[1])].append(row)
    for band in by_band:
        by_band[band].sort(key=lambda r: r[3], reverse=True)  # highest composite score first

    selected: List[int] = []
    for band, quota_share in band_quota.items():
        n_take = int(round(target * quota_share))
        selected.extend(r[0] for r in by_band[band][:n_take])
    return selected


def _content_words(text: str) -> set:
    return {t.lower() for t in _TOKEN_RE.findall(text) if t.lower() not in _STOPWORDS and len(t) > 2}


def _vocab_overlap_pct(question: str, source_text: str) -> float:
    q_words = _content_words(question)
    s_words = _content_words(source_text)
    if not q_words:
        return 0.0
    return len(q_words & s_words) / len(q_words)


def _generate_questions_for_batch(chunk_ids: Sequence[int], texts_by_id: Dict[int, str], categories_by_id: Dict[int, str]) -> Dict[int, str]:
    snippet_lines = []
    for i, chunk_id in enumerate(chunk_ids, start=1):
        snippet_lines.append(f"{i} (category: {categories_by_id[chunk_id]}): {texts_by_id[chunk_id]}")
    prompt = _PROMPT_TEMPLATE.format(n=len(chunk_ids), snippets="\n".join(snippet_lines))
    parsed = call_ghcp_json(prompt)
    if parsed is None:
        return {}
    result: Dict[int, str] = {}
    for item in parsed:
        try:
            local_id = int(item["id"])
            question = str(item["question"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if 1 <= local_id <= len(chunk_ids) and question:
            result[chunk_ids[local_id - 1]] = question
    return result


def generate(
    index_db_path: Path,
    chat_db_path: Optional[Path],
    output_path: Path,
    *,
    target: int = SAMPLE_TARGET,
    batch_size: int = BATCH_SIZE,
) -> int:
    import zstandard

    index_con = open_index_db(index_db_path)
    dctx = zstandard.ZstdDecompressor()

    chunk_ids = _select_stratified(index_con, target=target)
    if not chunk_ids:
        print("[generate] no candidates survived stratified selection -- has harvest.py run?", file=sys.stderr)
        return 0

    placeholders = ",".join("?" for _ in chunk_ids)
    rows = index_con.execute(
        f"SELECT c.id, c.chat_id, c.body_semantic, ec.category, ec.nn_distance "
        f"FROM chunks c JOIN eval_candidate ec ON ec.chunk_id = c.id WHERE c.id IN ({placeholders})",
        chunk_ids,
    ).fetchall()
    texts_by_id = {row[0]: dctx.decompress(row[2]).decode("utf-8") for row in rows}
    chat_by_id = {row[0]: row[1] for row in rows}
    category_by_id = {row[0]: row[3] for row in rows}
    nn_distance_by_id = {row[0]: row[4] for row in rows}

    msg_ids_by_chunk: Dict[int, List[int]] = {}
    for chunk_id, msg_id in index_con.execute(
        f"SELECT chunk_id, msg_id FROM chunk_message WHERE chunk_id IN ({placeholders}) ORDER BY msg_id", chunk_ids
    ):
        msg_ids_by_chunk.setdefault(chunk_id, []).append(msg_id)

    chat_con = connect_readonly(chat_db_path) if chat_db_path else None
    contact_index = None
    if chat_db_path is not None:
        try:
            contact_index = ContactIndex.load()
        except ContactsUnavailableError:
            pass
    embedding_model = EmbeddingModel()

    questions_by_id: Dict[int, str] = {}
    for i in range(0, len(chunk_ids), batch_size):
        batch = chunk_ids[i : i + batch_size]
        t0 = time.time()
        batch_questions = _generate_questions_for_batch(batch, texts_by_id, category_by_id)
        questions_by_id.update(batch_questions)
        print(f"[generate] batch {i // batch_size + 1}: {len(batch_questions)}/{len(batch)} questions in {time.time() - t0:.1f}s")

    n_written = 0
    with output_path.open("w") as f:
        for chunk_id in chunk_ids:
            question = questions_by_id.get(chunk_id)
            if not question:
                continue
            source_text = texts_by_id[chunk_id]
            overlap = _vocab_overlap_pct(question, source_text)
            if overlap > VOCAB_OVERLAP_REJECT_THRESHOLD:
                continue  # §4.2 vocabulary-leakage auto-reject; not a review flag

            parsed_query = parse_query(question, contact_index=contact_index)
            fused = retrieve(index_con, parsed_query, embedding_model, chat_con=chat_con, fused_top_k=200)
            ranked_ids = [r.chunk_id for r in fused]
            source_rank = ranked_ids.index(chunk_id) + 1 if chunk_id in ranked_ids else None
            trivially_hard = source_rank is None or source_rank > TRIVIALLY_HARD_RANK

            top5_other = [cid for cid in ranked_ids[:5] if cid != chunk_id][:5]
            sparse_ids = []  # populated below only if we need the rank-1 review flag
            sparse_rank1_is_source = False
            try:
                from seaglass.search.retrieve import sparse_search

                sparse_ids = sparse_search(index_con, parsed_query.semantic, None, top_k=1)
                sparse_rank1_is_source = bool(sparse_ids) and sparse_ids[0] == chunk_id
            except Exception:
                pass

            entry = {
                "id": f"ev-{chunk_id:05d}",
                "query": question,
                "positive_msg_ids": msg_ids_by_chunk.get(chunk_id, []),
                "alt_positive_msg_ids": [],
                "chat_id": chat_by_id[chunk_id],
                "category": category_by_id[chunk_id],
                "source_chunk_id": chunk_id,
                "nn_distance": nn_distance_by_id[chunk_id],
                "origin": "harvested",
                "reviewed": False,
                "split": "dev",
                "_vocab_overlap_pct": overlap,
                "_sparse_rank1_is_source": sparse_rank1_is_source,
                "_trivially_hard": trivially_hard,
                "_top5_other_chunk_ids": top5_other,
            }
            f.write(json.dumps(entry) + "\n")
            n_written += 1

    print(f"[generate] wrote {n_written} draft entries to {output_path}")
    return n_written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index_db", help="path to index.db (harvest.py must have run)")
    parser.add_argument("output", help="output path for candidates_for_review.jsonl")
    parser.add_argument("--chat-db", default=None, help="chat.db snapshot, needed for ambiguity-check retrieval's people filters")
    parser.add_argument("--target", type=int, default=SAMPLE_TARGET)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args(argv)
    generate(
        Path(args.index_db), Path(args.chat_db) if args.chat_db else None, Path(args.output),
        target=args.target, batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
