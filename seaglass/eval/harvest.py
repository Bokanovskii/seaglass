"""`eval/harvest.py` — Phase 3.5 candidate harvesting (EVALUATION.md §3).
Entirely a post-build job: reads only `index.db` (+ `chat.db` for
`chat_handle_join`/`chat.style`), touches nothing on indexing's hot path.

Four resumable, independently re-runnable stages, per EVALUATION.md §3:
  harvest.py --stage sql     regex + chunks + chat + temporal signals
  harvest.py --stage idf     one FTS5 statistics query
  harvest.py --stage nn      nearest-neighbour distance (the expensive one)
  harvest.py --stage score   bands, composite score, category

Populates `eval_candidate` (schema.sql), a fully-derived, droppable
build artifact -- consistent with DESIGN-NOTES.md §12's "own only
derived data" principle.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import zstandard

from seaglass.imessage.source import connect_readonly
from seaglass.index.build import open_index_db

_dctx = zstandard.ZstdDecompressor()

_ENTITY_RE = re.compile(r"(?<!^)(?<![.!?]\s)\b[A-Z][a-z]+\b")
_URL_RE = re.compile(r"\[link:")  # render.py collapses URLs to "[link:domain]"
_NUMBER_RE = re.compile(r"\d")
_LONG_TOKEN_RE = re.compile(r"\b\w{15,}\b")
_TOKEN_RE = re.compile(r"\w+")

# §3.2: prefilter down to ~5000 candidates before the expensive nn_distance pass.
PREFILTER_TARGET = 5000
TOKEN_COUNT_MIN = 40
TOKEN_COUNT_MAX = 900


def _regex_signals(text: str) -> Dict[str, int]:
    return {
        "entity_count": len(_ENTITY_RE.findall(text)),
        "has_url": int(bool(_URL_RE.search(text))),
        "has_number": int(bool(_NUMBER_RE.search(text))),
        "has_long_token": int(bool(_LONG_TOKEN_RE.search(text))),
        "token_count": len(_TOKEN_RE.findall(text)),
    }


def stage_sql(index_con: sqlite3.Connection, chat_con: Optional[sqlite3.Connection]) -> int:
    """Regex signals over body_semantic + chunks/chunk_message/chat-derived
    signals (has_attachment, msg_count, is_group, participant_count,
    temporal_isolation). Everything except idf_mean and nn_distance.
    """
    rows = index_con.execute("SELECT id, chat_id, start_ts, has_attachment, body_semantic FROM chunks").fetchall()

    # temporal_isolation: days to nearest *other* chunk in the same chat, by start_ts.
    by_chat: Dict[int, List[Tuple[int, int]]] = {}
    for chunk_id, chat_id, start_ts, _has_attachment, _body in rows:
        by_chat.setdefault(chat_id, []).append((start_ts, chunk_id))
    isolation_by_chunk: Dict[int, float] = {}
    for chat_id, entries in by_chat.items():
        entries.sort()
        n = len(entries)
        for i, (ts, chunk_id) in enumerate(entries):
            neighbours = []
            if i > 0:
                neighbours.append(abs(ts - entries[i - 1][0]))
            if i < n - 1:
                neighbours.append(abs(entries[i + 1][0] - ts))
            isolation_by_chunk[chunk_id] = (min(neighbours) / 86400.0) if neighbours else 9999.0

    msg_count_by_chunk = {
        row[0]: row[1]
        for row in index_con.execute(
            "SELECT chunk_id, COUNT(*) FROM chunk_message GROUP BY chunk_id"
        ).fetchall()
    }

    # attachment_place is populated by the not-yet-built index/exif.py (ADDENDUM.md
    # §11/§13); short-circuit the has_geo lookup entirely while it's always empty,
    # rather than paying an N+1 query cost per chunk for a signal that can't fire yet.
    any_geo_data = bool(index_con.execute("SELECT 1 FROM attachment_place LIMIT 1").fetchone())

    participant_count_by_chat: Dict[int, int] = {}
    is_group_by_chat: Dict[int, int] = {}
    if chat_con is not None:
        for row in chat_con.execute("SELECT chat_id, COUNT(DISTINCT handle_id) FROM im.chat_handle_join GROUP BY chat_id"):
            participant_count_by_chat[row[0]] = row[1]
        for row in chat_con.execute("SELECT ROWID, style FROM im.chat"):
            is_group_by_chat[row[0]] = int((row[1] or 0) not in (45,))  # 45 == 1:1 DM in Apple's schema

    n_written = 0
    with index_con:
        for chunk_id, chat_id, start_ts, has_attachment, body in rows:
            text = _dctx.decompress(body).decode("utf-8")
            signals = _regex_signals(text)
            has_geo = 0
            if chat_con is not None and any_geo_data:
                msg_ids = [r[0] for r in index_con.execute(
                    "SELECT msg_id FROM chunk_message WHERE chunk_id = ?", (chunk_id,)
                ).fetchall()]
                if msg_ids:
                    placeholders = ",".join("?" for _ in msg_ids)
                    attachment_ids = [
                        r[0] for r in chat_con.execute(
                            f"SELECT attachment_id FROM im.message_attachment_join "
                            f"WHERE message_id IN ({placeholders})",
                            msg_ids,
                        ).fetchall()
                    ]
                    if attachment_ids:
                        aplaceholders = ",".join("?" for _ in attachment_ids)
                        has_geo = int(bool(index_con.execute(
                            f"SELECT 1 FROM attachment_place WHERE attachment_id IN ({aplaceholders}) LIMIT 1",
                            attachment_ids,
                        ).fetchone()))
            participant_count = participant_count_by_chat.get(chat_id, 1 if not chat_con else 0)
            is_group = is_group_by_chat.get(chat_id, int(participant_count > 1))
            index_con.execute(
                """
                INSERT INTO eval_candidate
                    (chunk_id, entity_count, has_url, has_number, has_long_token,
                     token_count, has_geo, has_attachment, is_group,
                     participant_count, temporal_isolation, msg_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    entity_count=excluded.entity_count, has_url=excluded.has_url,
                    has_number=excluded.has_number, has_long_token=excluded.has_long_token,
                    token_count=excluded.token_count, has_geo=excluded.has_geo,
                    has_attachment=excluded.has_attachment, is_group=excluded.is_group,
                    participant_count=excluded.participant_count,
                    temporal_isolation=excluded.temporal_isolation, msg_count=excluded.msg_count
                """,
                (
                    chunk_id, signals["entity_count"], signals["has_url"], signals["has_number"],
                    signals["has_long_token"], signals["token_count"], has_geo, has_attachment or 0,
                    is_group, participant_count, isolation_by_chunk.get(chunk_id, 9999.0),
                    msg_count_by_chunk.get(chunk_id, 0),
                ),
            )
            n_written += 1
    return n_written


def stage_idf(index_con: sqlite3.Connection) -> int:
    """idf_mean: mean IDF of the 5 rarest terms in each candidate chunk's
    body, using FTS5's own statistics (bm25's idf, via the auxiliary
    `fts5vocab` shadow table -- avoids hand-rolling document frequency).
    """
    index_con.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts_vocab USING fts5vocab('chunks_fts', 'row')"
    )
    n_docs_row = index_con.execute("SELECT COUNT(*) FROM chunks").fetchone()
    n_docs = max(n_docs_row[0], 1)
    df_by_term: Dict[str, int] = {
        term: doc for term, doc, _cnt in index_con.execute("SELECT term, doc, cnt FROM chunks_fts_vocab")
    }

    rows = index_con.execute(
        "SELECT ec.chunk_id, c.body_semantic FROM eval_candidate ec JOIN chunks c ON c.id = ec.chunk_id"
    ).fetchall()
    n_written = 0
    with index_con:
        for chunk_id, body in rows:
            text = _dctx.decompress(body).decode("utf-8")
            terms = {t.lower() for t in _TOKEN_RE.findall(text)}
            idfs = []
            for term in terms:
                df = df_by_term.get(term)
                if df:
                    idfs.append(np.log(n_docs / df))
            idf_mean = float(np.mean(sorted(idfs, reverse=True)[:5])) if idfs else 0.0
            index_con.execute("UPDATE eval_candidate SET idf_mean = ? WHERE chunk_id = ?", (idf_mean, chunk_id))
            n_written += 1
    return n_written


def _prefilter_ids(index_con: sqlite3.Connection, target: int = PREFILTER_TARGET) -> List[int]:
    """§3.2 step 1: narrow to ~`target` candidates on cheap signals before
    the expensive nn_distance pass, ranked by a quick proxy so the
    prefilter itself favours question-able content.
    """
    rows = index_con.execute(
        """
        SELECT chunk_id, idf_mean, entity_count FROM eval_candidate
        WHERE token_count BETWEEN ? AND ?
        """,
        (TOKEN_COUNT_MIN, TOKEN_COUNT_MAX),
    ).fetchall()
    if len(rows) <= target:
        return [row[0] for row in rows]
    median_idf = float(np.median([row[1] for row in rows]))
    filtered = [row for row in rows if row[1] >= median_idf and row[2] >= 2]
    filtered.sort(key=lambda row: (row[1], row[2]), reverse=True)
    if len(filtered) >= target:
        return [row[0] for row in filtered[:target]]
    # not enough survive both cuts -- fall back to ranking everything by idf
    rows.sort(key=lambda row: row[1], reverse=True)
    return [row[0] for row in rows[:target]]


def stage_nn(index_con: sqlite3.Connection, target: int = PREFILTER_TARGET) -> int:
    """§3.2 step 2: nn_distance for the prefiltered candidates only, as one
    batched NumPy pass over the loaded int8 vectors (not N separate KNN
    SQL queries) -- same arithmetic, far less per-call overhead.

    Excludes self-matches and any chunk sharing a message with the
    source (adjacent chunks overlap by design and would always be
    near-neighbours, per EVALUATION.md §3.2).
    """
    candidate_ids = _prefilter_ids(index_con, target)
    if not candidate_ids:
        return 0

    all_rows = index_con.execute("SELECT rowid, embedding FROM chunks_vec").fetchall()
    all_ids = np.array([row[0] for row in all_rows], dtype=np.int64)
    all_vecs = np.stack([np.frombuffer(row[1], dtype=np.int8).astype(np.float32) for row in all_rows])
    id_to_row = {cid: i for i, cid in enumerate(all_ids)}

    # cosine distance == 1 - cosine similarity; normalise once.
    norms = np.linalg.norm(all_vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit_vecs = all_vecs / norms

    # messages shared with the source chunk exclude a neighbour from
    # counting as "other" -- adjacent overlap chunks would otherwise
    # always win as the nearest neighbour by construction.
    msg_ids_by_chunk: Dict[int, set] = {}
    for chunk_id, msg_id in index_con.execute("SELECT chunk_id, msg_id FROM chunk_message"):
        msg_ids_by_chunk.setdefault(chunk_id, set()).add(msg_id)

    n_written = 0
    with index_con:
        for chunk_id in candidate_ids:
            row_idx = id_to_row.get(chunk_id)
            if row_idx is None:
                continue
            sims = unit_vecs @ unit_vecs[row_idx]
            distances = 1.0 - sims
            own_msgs = msg_ids_by_chunk.get(chunk_id, set())
            best_dist, best_id = None, None
            order = np.argsort(distances)
            for idx in order:
                other_id = int(all_ids[idx])
                if other_id == chunk_id:
                    continue
                if own_msgs & msg_ids_by_chunk.get(other_id, set()):
                    continue  # excludes overlap-sharing neighbours (chunker.py overlap)
                best_dist, best_id = float(distances[idx]), other_id
                break
            if best_dist is None:
                continue
            index_con.execute(
                "UPDATE eval_candidate SET nn_distance = ?, nn_chunk_id = ? WHERE chunk_id = ?",
                (best_dist, best_id, chunk_id),
            )
            n_written += 1
    return n_written


def _norm(values: List[float]) -> Dict[int, float]:
    if not values:
        return {}
    lo, hi = min(values), max(values)
    if hi == lo:
        return {i: 0.5 for i in range(len(values))}
    return {i: (v - lo) / (hi - lo) for i, v in enumerate(values)}


def _band(nn_distance: float, quartiles: Tuple[float, float, float]) -> int:
    q1, q2, q3 = quartiles
    if nn_distance <= q1:
        return 0  # bottom 25%, boilerplate
    if nn_distance <= q2:
        return 1  # 25-50%, hard cases
    if nn_distance <= q3:
        return 2  # 50-75%
    return 3  # top 25%, isolated


def _category(row: dict) -> str:
    """EVALUATION.md §3.4 automatic category assignment."""
    if row["has_geo"] or row["has_attachment"]:
        return "media_geo"
    if row["has_url"] or row["has_long_token"]:
        return "exact_string"
    if row["is_group"] and row["participant_count"] >= 3:
        return "person_filtered"
    if row["_time_isolation_pctl"] is not None and row["_time_isolation_pctl"] >= 0.8:
        return "time_filtered"
    if row["_multi_session_count"] >= 3:
        return "multi_session"
    return "topical"


def stage_score(index_con: sqlite3.Connection, weights: Optional[Dict[str, float]] = None) -> int:
    """§3.3 composite score + §3.4 category assignment. Requires stage_nn
    to have run for a chunk (rows with `nn_distance IS NULL` are scored
    but never assigned a `category` other than via non-nn signals, since
    band(nn_distance) can't be computed -- they simply won't surface in
    a stratified sample by nn band).
    """
    weights = weights or {"nn_band": 1.0, "idf": 1.0, "entity": 1.0, "category_bonus": 0.5, "token_penalty": 1.0}

    rows = index_con.execute(
        "SELECT chunk_id, nn_distance, idf_mean, entity_count, has_url, has_long_token, "
        "has_geo, has_attachment, is_group, participant_count, temporal_isolation, token_count "
        "FROM eval_candidate WHERE nn_distance IS NOT NULL"
    ).fetchall()
    if not rows:
        return 0

    nn_distances = sorted(r[1] for r in rows)
    q1 = nn_distances[len(nn_distances) // 4]
    q2 = nn_distances[len(nn_distances) // 2]
    q3 = nn_distances[3 * len(nn_distances) // 4]

    idf_norm = _norm([r[2] for r in rows])
    entity_norm = _norm([r[3] for r in rows])

    isolations = sorted(r[10] for r in rows if r[10] < 9999.0)

    def isolation_pctl(value: float) -> Optional[float]:
        if value >= 9999.0 or not isolations:
            return None
        idx = np.searchsorted(isolations, value)
        return idx / len(isolations)

    # multi_session: >=3 high-idf chunks in the same chat within 24h -- approximate
    # via chunk_id's chat_id + start_ts, reusing the chunks table.
    chat_start_by_chunk = {
        row[0]: (row[1], row[2]) for row in index_con.execute("SELECT id, chat_id, start_ts FROM chunks")
    }
    by_chat_ts: Dict[int, List[int]] = {}
    for chunk_id, (chat_id, start_ts) in chat_start_by_chunk.items():
        by_chat_ts.setdefault(chat_id, []).append(start_ts)
    for chat_id in by_chat_ts:
        by_chat_ts[chat_id].sort()

    def multi_session_count(chunk_id: int) -> int:
        chat_id, start_ts = chat_start_by_chunk.get(chunk_id, (None, None))
        if chat_id is None:
            return 0
        window = [t for t in by_chat_ts[chat_id] if abs(t - start_ts) <= 86400]
        return len(window)

    n_written = 0
    with index_con:
        for i, row in enumerate(rows):
            (chunk_id, nn_distance, idf_mean, entity_count, has_url, has_long_token,
             has_geo, has_attachment, is_group, participant_count, temporal_isolation, token_count) = row

            band = _band(nn_distance, (q1, q2, q3))
            token_penalty = 0.0
            if token_count < 150:
                token_penalty = (150 - token_count) / 150.0
            elif token_count > 600:
                token_penalty = (token_count - 600) / 600.0

            category = _category(
                {
                    "has_geo": has_geo, "has_attachment": has_attachment, "has_url": has_url,
                    "has_long_token": has_long_token, "is_group": is_group,
                    "participant_count": participant_count,
                    "_time_isolation_pctl": isolation_pctl(temporal_isolation),
                    "_multi_session_count": multi_session_count(chunk_id),
                }
            )
            category_bonus = 0.0  # filled in by a stratification pass, not per-row scoring

            score = (
                weights["nn_band"] * (band / 3.0)
                + weights["idf"] * idf_norm[i]
                + weights["entity"] * entity_norm[i]
                + weights["category_bonus"] * category_bonus
                - weights["token_penalty"] * token_penalty
            )
            index_con.execute(
                "UPDATE eval_candidate SET category = ?, score = ? WHERE chunk_id = ?",
                (category, score, chunk_id),
            )
            n_written += 1
    return n_written


def run(index_db_path: Path, chat_db_path: Optional[Path], stages: List[str]) -> None:
    index_con = open_index_db(index_db_path)
    chat_con = connect_readonly(chat_db_path) if chat_db_path else None

    stage_fns = {
        "sql": lambda: stage_sql(index_con, chat_con),
        "idf": lambda: stage_idf(index_con),
        "nn": lambda: stage_nn(index_con),
        "score": lambda: stage_score(index_con),
    }
    for stage in stages:
        t0 = time.time()
        n = stage_fns[stage]()
        print(f"[harvest] stage={stage} rows={n} elapsed={time.time() - t0:.1f}s")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index_db", help="path to index.db")
    parser.add_argument("--chat-db", default=None, help="chat.db snapshot (needed for the sql stage's is_group/participant_count)")
    parser.add_argument(
        "--stage", action="append", choices=["sql", "idf", "nn", "score", "all"], default=None,
        help="which stage(s) to run (repeatable); default: all in order",
    )
    args = parser.parse_args(argv)
    stages = args.stage or ["all"]
    if "all" in stages:
        stages = ["sql", "idf", "nn", "score"]
    run(Path(args.index_db), Path(args.chat_db) if args.chat_db else None, stages)
    return 0


if __name__ == "__main__":
    sys.exit(main())
