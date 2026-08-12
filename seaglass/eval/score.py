"""`eval/score.py` — EVALUATION.md §7: score a golden set against the
full retrieval pipeline, reporting recall@50 (post-RRF, pre-rerank),
recall@12 (post-rerank), and recall@final (the actual MCP payload,
post-aggregation/expansion/max_sessions), per category and overall, plus
the two diagnostic gaps and the filter kill rate (§8.3).

⚠️ Scores by `chunk_message`, never a `message.ROWID` range test — a
range spans other conversations, since `message.ROWID` is global and
chronological across all chats (§7's explicit warning, DESIGN-NOTES.md
§9).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

from seaglass.imessage.contacts import ContactIndex, ContactsUnavailableError
from seaglass.imessage.source import connect_readonly
from seaglass.index.build import open_index_db
from seaglass.index.embed import EmbeddingModel
from seaglass.search.parse import parse_query
from seaglass.search.rank import aggregate_sessions, expand_sessions, rerank_candidates
from seaglass.search.rerank import CrossEncoderReranker
from seaglass.search.retrieve import build_candidate_chunk_ids, retrieve

RERANK_TOP_K = 12
MAX_SESSIONS = 8


def wilson_interval(hits: int, n: int, z: float = 1.96) -> "tuple[float, float, float]":
    """95% Wilson score interval (not the Wald interval -- Wald gives
    zero width at p=1.0 and can extend outside [0,1], per EVALUATION.md
    §6.2). Returns (point_estimate, lower, upper).
    """
    if n == 0:
        return (0.0, 0.0, 0.0)
    p_hat = hits / n
    denom = 1 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
    return (p_hat, max(0.0, center - half), min(1.0, center + half))


def mcnemar_chi2(n10: int, n01: int) -> Optional[float]:
    """§6.1: paired comparison statistic over discordant pairs only. `None`
    when there's no discordance to test (both zero).
    """
    if n10 + n01 == 0:
        return None
    return ((n10 - n01) ** 2) / (n10 + n01)


def _chunks_containing(index_con, msg_ids: Sequence[int]) -> Set[int]:
    if not msg_ids:
        return set()
    placeholders = ",".join("?" for _ in msg_ids)
    rows = index_con.execute(
        f"SELECT DISTINCT chunk_id FROM chunk_message WHERE msg_id IN ({placeholders})", list(msg_ids)
    ).fetchall()
    return {row[0] for row in rows}


def score_entry(
    index_con,
    chat_con,
    embedding_model: EmbeddingModel,
    reranker: CrossEncoderReranker,
    entry: dict,
    contact_index: Optional[ContactIndex] = None,
) -> dict:
    """Run one golden entry through the full pipeline, returning
    recall@50/@12/@final hit booleans plus the filter-kill diagnostic.
    """
    positives = set(entry.get("positive_msg_ids", [])) | set(entry.get("alt_positive_msg_ids", []))
    positive_chunk_ids = _chunks_containing(index_con, sorted(positives))

    parsed = parse_query(entry["query"], contact_index=contact_index)
    candidate_ids = build_candidate_chunk_ids(index_con, parsed, chat_con=chat_con)
    filter_killed = candidate_ids is not None and not (candidate_ids & positive_chunk_ids) and bool(positive_chunk_ids)

    fused = retrieve(index_con, parsed, embedding_model, chat_con=chat_con, fused_top_k=50)
    recall_50 = any(r.chunk_id in positive_chunk_ids for r in fused)

    ranked = rerank_candidates(index_con, parsed.semantic, fused, reranker, top_k=RERANK_TOP_K)
    recall_12 = any(rc.chunk_id in positive_chunk_ids for rc in ranked)

    sessions = aggregate_sessions(ranked, max_sessions=MAX_SESSIONS)
    expand_sessions(index_con, sessions)
    final_chunk_ids = set()
    for session in sessions:
        final_chunk_ids.update(session.hit_chunk_ids)
        final_chunk_ids.update(session.context_chunk_ids)  # context counts as a hit per §7
    recall_final = any(cid in positive_chunk_ids for cid in final_chunk_ids)

    return {
        "id": entry["id"],
        "category": entry.get("category", "topical"),
        "nn_distance": entry.get("nn_distance"),
        "recall_50": recall_50,
        "recall_12": recall_12,
        "recall_final": recall_final,
        "filter_killed": filter_killed,
    }


def score_golden_set(
    index_con,
    chat_con,
    embedding_model: EmbeddingModel,
    reranker: CrossEncoderReranker,
    entries: Sequence[dict],
    contact_index: Optional[ContactIndex] = None,
) -> List[dict]:
    return [score_entry(index_con, chat_con, embedding_model, reranker, e, contact_index) for e in entries]


def _nn_band(nn_distance: Optional[float], quartiles) -> Optional[str]:
    if nn_distance is None or quartiles is None:
        return None
    q1, q2, q3 = quartiles
    if nn_distance <= q1:
        return "Q1"
    if nn_distance <= q2:
        return "Q2"
    if nn_distance <= q3:
        return "Q3"
    return "Q4"


def summarize(results: Sequence[dict]) -> dict:
    """Aggregate per-category and overall recall@50/@12/@final, gaps,
    filter kill rate, and by-nn_band recall -- the report shape in
    EVALUATION.md §7's "Output shape".
    """
    by_category: Dict[str, List[dict]] = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r)

    def _row(subset: Sequence[dict]) -> dict:
        n = len(subset)
        r50 = sum(1 for r in subset if r["recall_50"])
        r12 = sum(1 for r in subset if r["recall_12"])
        rfin = sum(1 for r in subset if r["recall_final"])
        kills = sum(1 for r in subset if r["filter_killed"])
        p50, lo50, hi50 = wilson_interval(r50, n)
        p12, _, _ = wilson_interval(r12, n)
        pfin, _, _ = wilson_interval(rfin, n)
        return {
            "n": n,
            "recall_50": p50,
            "recall_50_ci": (lo50, hi50),
            "recall_12": p12,
            "recall_final": pfin,
            "gap1": p12 - p50,
            "gap2": pfin - p12,
            "kill_rate": (kills / n) if n else 0.0,
        }

    report = {"by_category": {cat: _row(rows) for cat, rows in by_category.items()}, "overall": _row(results)}

    nn_distances = sorted(r["nn_distance"] for r in results if r["nn_distance"] is not None)
    if nn_distances:
        q1 = nn_distances[len(nn_distances) // 4]
        q2 = nn_distances[len(nn_distances) // 2]
        q3 = nn_distances[3 * len(nn_distances) // 4]
        by_band: Dict[str, List[dict]] = {}
        for r in results:
            band = _nn_band(r["nn_distance"], (q1, q2, q3))
            if band:
                by_band.setdefault(band, []).append(r)
        report["by_nn_band"] = {band: _row(rows)["recall_final"] for band, rows in sorted(by_band.items())}
    else:
        report["by_nn_band"] = {}

    return report


def print_report(report: dict) -> None:
    print(f"{'category':<18}{'n':>5}{'r@50':>8}{'r@12':>8}{'r@fin':>8}{'gap1':>8}{'gap2':>8}{'kill':>7}")
    for cat, row in sorted(report["by_category"].items()):
        print(
            f"{cat:<18}{row['n']:>5}{row['recall_50']:>8.2f}{row['recall_12']:>8.2f}"
            f"{row['recall_final']:>8.2f}{row['gap1']:>8.2f}{row['gap2']:>8.2f}{row['kill_rate']:>7.2f}"
        )
    print("-" * 74)
    overall = report["overall"]
    print(
        f"{'OVERALL':<18}{overall['n']:>5}{overall['recall_50']:>8.2f}{overall['recall_12']:>8.2f}"
        f"{overall['recall_final']:>8.2f}{overall['gap1']:>8.2f}{overall['gap2']:>8.2f}{overall['kill_rate']:>7.2f}"
    )
    lo, hi = overall["recall_50_ci"]
    print(f"recall@50 95% Wilson CI: [{lo:.2f}, {hi:.2f}]  n={overall['n']}")
    if report["by_nn_band"]:
        band_str = "  ".join(f"{band} {v:.2f}" for band, v in report["by_nn_band"].items())
        print(f"by nn_band (recall@final): {band_str}   <- Q2 is the honest number")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index_db", help="path to index.db")
    parser.add_argument("golden", help="path to golden.jsonl")
    parser.add_argument("--chat-db", default=None, help="chat.db snapshot, needed for people filters")
    args = parser.parse_args(argv)

    index_con = open_index_db(Path(args.index_db))
    chat_con = connect_readonly(Path(args.chat_db)) if args.chat_db else None
    contact_index = None
    if args.chat_db:
        try:
            contact_index = ContactIndex.load()
        except ContactsUnavailableError:
            pass

    entries = []
    with open(args.golden) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    embedding_model = EmbeddingModel()
    reranker = CrossEncoderReranker()
    results = score_golden_set(index_con, chat_con, embedding_model, reranker, entries, contact_index)
    report = summarize(results)
    print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
