"""`eval/behavior.py` — QUERY-EVAL-PLAN.md: score the query classes whose
correctness is decided by filters, ordering and freshness rather than by
embedding similarity.

Two scoring modes, per QUERY-EVAL-PLAN.md §2 and §4:

* **Oracle** — a filter query ("everything Kaya sent last week") has exactly
  one correct answer set, and it is a SQL query away in chat.db. No
  labelling, no LLM, and it stays correct as the corpus grows. This is what
  makes the suite cheap; `EVALUATION.md` §2's "a miss is not a miss"
  objection applies to topical queries, not to these.
* **Property** — statements that must hold of any payload ("no hit was
  written by someone other than the named sender"). One bug is caught by
  whichever query trips it first.

Drives `SearchEngine` directly -- the same object the app and the MCP server
use. `eval/score.py` reimplements the pipeline, which is how browse mode
ended up shipped but never measured (§7).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from seaglass.app.engine import SearchEngine, SearchOptions
from seaglass.app.filters import SearchFilters
from seaglass.imessage.source import APPLE_EPOCH_UNIX, NS_VS_S_THRESHOLD
from seaglass.search.parse import parse_query

# --------------------------------------------------------------------------
# Oracle: the correct answer set, straight from chat.db
# --------------------------------------------------------------------------

# `message.date` is Apple-epoch, seconds on older macOS and nanoseconds on
# Big Sur+ -- the same magnitude heuristic as imessage/source.apple_to_unix.
_TS_EXPR = f"(CASE WHEN m.date > {NS_VS_S_THRESHOLD} THEN m.date / 1e9 ELSE m.date END)"


def _unix(apple_ts: float) -> float:
    return float(apple_ts) + APPLE_EPOCH_UNIX


@dataclass
class Oracle:
    """Ground truth for filter-only queries, computed from chat.db."""

    con: object

    def messages_from(
        self,
        handles: Sequence[str],
        *,
        since: Optional[float] = None,
        until: Optional[float] = None,
        with_media: bool = False,
        limit: int = 500,
    ) -> List[dict]:
        """Messages *written by* one of `handles`, newest first.

        `is_from_me = 0` is not redundant: chat.db stamps outgoing 1:1
        messages with the *recipient's* handle_id, so a handle test alone
        returns both halves of the conversation.
        """
        if not handles:
            return []
        marks = ",".join("?" for _ in handles)
        where = [f"h.id IN ({marks})", "m.is_from_me = 0", "m.associated_message_type = 0"]
        params: List[object] = list(handles)
        if since is not None:
            where.append(f"{_TS_EXPR} >= ?")
            params.append(since - APPLE_EPOCH_UNIX)
        if until is not None:
            where.append(f"{_TS_EXPR} <= ?")
            params.append(until - APPLE_EPOCH_UNIX)
        if with_media:
            where.append(
                "EXISTS (SELECT 1 FROM message_attachment_join maj WHERE maj.message_id = m.ROWID)"
            )
        params.append(limit)
        rows = self.con.execute(
            f"""
            SELECT m.ROWID, {_TS_EXPR} AS ts, m.text, m.attributedBody IS NOT NULL
            FROM message m
            JOIN handle h ON m.handle_id = h.ROWID
            JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
            WHERE {' AND '.join(where)}
            ORDER BY m.date DESC LIMIT ?
            """,
            params,
        ).fetchall()
        # Empty rows (system messages, attachment-only) are dropped by
        # hydration, so the oracle must drop them too or every comparison
        # is off by however many the window happens to contain.
        return [
            {"message_id": r[0], "ts": _unix(r[1])}
            for r in rows
            if (r[2] or "").strip() or r[3]
        ]

    def newest_from(self, handles: Sequence[str]) -> Optional[dict]:
        found = self.messages_from(handles, limit=1)
        return found[0] if found else None


# --------------------------------------------------------------------------
# Properties
# --------------------------------------------------------------------------


@dataclass
class Case:
    query: str
    cls: str
    # Names to resolve for oracle scoring; None means "whatever the parser found".
    expects_results: bool = True
    lexical: Optional[str] = None
    notes: str = ""


@dataclass
class Result:
    case: Case
    payload: dict
    parsed: object
    elapsed_s: float
    properties: Dict[str, Optional[bool]] = field(default_factory=dict)
    oracle: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None


def _hits(payload: dict) -> List[dict]:
    return [m for s in payload.get("sessions", []) for m in s.get("messages", [])]


def _flat_in_caller_order(payload: dict) -> List[dict]:
    """What a caller honouring `limit` actually reads: every session's hits,
    then every session's context (the order grogu_imessage flattens to)."""
    hits, context = [], []
    for session in payload.get("sessions", []):
        hits.extend(session.get("messages", []))
        context.extend(session.get("context_messages", []))
    return hits + context


def check_properties(case: Case, payload: dict, parsed, oracle: Oracle) -> Dict[str, Optional[bool]]:
    """Each property returns True/False, or None when it does not apply."""
    props: Dict[str, Optional[bool]] = {}
    sessions = payload.get("sessions", [])
    hits = _hits(payload)

    # "Did it find anything" is only a fair question when there is something
    # to find. "what did Sraddha say yesterday" returns nothing because
    # Sraddha last wrote in 2021 -- scoring that as a miss punished the
    # engine for being right, and hid the cases where it is wrong.
    expected = case.expects_results
    if expected and parsed.people_sender:
        expected = bool(
            oracle.messages_from(
                parsed.people_sender, since=parsed.date_from, until=parsed.date_to,
                with_media=parsed.has_media, limit=1,
            )
        )
    props["nonempty"] = bool(sessions) if expected else None
    props["no_empty_sessions"] = (
        all(s.get("messages") for s in sessions) if sessions else None
    )

    if parsed.people_sender and hits:
        wanted = {h.lower() for h in parsed.people_sender}
        names = {
            (m.get("sender") or "").lower() for m in hits if not m.get("is_from_me")
        }
        # Hydration resolves handles to display names, so purity is checked
        # on "did anything come back from someone we did not ask for" via
        # is_from_me plus a single-name check.
        props["no_self_in_sender"] = not any(m.get("is_from_me") for m in hits)
        props["sender_purity"] = len(names) <= 1
        del wanted
    else:
        props["no_self_in_sender"] = None
        props["sender_purity"] = None

    if (parsed.date_from is not None or parsed.date_to is not None) and hits:
        lo = parsed.date_from or float("-inf")
        hi = parsed.date_to or float("inf")
        props["date_containment"] = all(lo <= (m.get("ts") or 0) <= hi for m in hits)
    else:
        props["date_containment"] = None

    if payload.get("ordering") == "recent" and len(hits) > 1:
        # Two invariants, because results are grouped into sessions: hits
        # inside a session run newest-first, and sessions themselves run
        # newest-first by their newest hit. A single global ordering is not
        # achievable while grouping, and pretending otherwise made this
        # property unfixable rather than informative.
        per_session = []
        newest_per_session = []
        for session in sessions:
            stamps = [m.get("ts") or 0 for m in session.get("messages", [])]
            if not stamps:
                continue
            per_session.append(stamps == sorted(stamps, reverse=True))
            newest_per_session.append(max(stamps))
        props["recency_order"] = all(per_session)
        props["recency_session_order"] = (
            newest_per_session == sorted(newest_per_session, reverse=True)
            if len(newest_per_session) > 1 else None
        )
    else:
        props["recency_order"] = None
        props["recency_session_order"] = None

    if sessions:
        first = sessions[0]
        props["hits_before_context"] = bool(first.get("messages"))
    else:
        props["hits_before_context"] = None

    if case.lexical and sessions:
        needle = case.lexical.lower()
        props["lexical_presence"] = any(
            needle in (m.get("text") or "").lower()
            for m in _flat_in_caller_order(payload)
        )
    else:
        props["lexical_presence"] = None

    props["freshness_declared"] = "index_stale" in payload
    props["ordering_declared"] = "ordering" in payload
    return props


def score_against_oracle(
    parsed, payload: dict, oracle: Oracle, index_horizon: Optional[float] = None,
    index_con=None,
) -> Dict[str, float]:
    """Precision/recall of the returned hits against chat.db's answer, for
    queries whose answer set is exactly computable."""
    # Only filter-only queries have a computable answer set. "what did Kaya
    # say about dinner" is answered by similarity, so the newest message is
    # not the right answer and scoring it as one would punish the engine for
    # being correct (QUERY-EVAL-PLAN.md §3: classes 3, 4, 6, 7, 9, 11).
    if not parsed.people_sender or parsed.semantic.strip():
        return {}
    truth = oracle.messages_from(
        parsed.people_sender,
        since=parsed.date_from,
        until=parsed.date_to,
        with_media=parsed.has_media,
    )
    if not truth:
        return {}

    # A message the index has never seen cannot be ranked, and counting it
    # as a ranking miss hides the two failures behind one number: "the
    # engine ordered badly" and "the engine was asked a question about
    # messages that arrived after the last build" need opposite fixes
    # (QUERY-EVAL-PLAN.md §6).
    scores_extra: Dict[str, float] = {}
    if index_horizon is not None:
        indexed = [m for m in truth if m["ts"] <= index_horizon]
        scores_extra["newest_is_indexed"] = 1.0 if truth and truth[0]["ts"] <= index_horizon else 0.0
        if not indexed:
            return scores_extra
        truth = indexed

    # A message can predate the index horizon and still be absent from it:
    # this chat.db holds 109,665 messages from one contact of which only
    # 180 are attached to a chat, so the rest were never chunked. That is a
    # corpus-coverage failure, and charging it to ranking would send the
    # next person tuning the ranker at a problem that is not there. So it
    # gets its own number, and recall is measured over what the index could
    # actually return.
    head = [m["message_id"] for m in truth[:20]]
    reachable = head
    if index_con is not None and head:
        marks = ",".join("?" for _ in head)
        present = {
            row[0]
            for row in index_con.execute(
                f"SELECT DISTINCT msg_id FROM chunk_message WHERE msg_id IN ({marks})", head
            )
        }
        scores_extra["indexed_coverage"] = len(present) / len(head)
        reachable = [mid for mid in head if mid in present] or head

    truth_ids = {m["message_id"] for m in truth}
    got_ids = {m.get("message_id") for m in _hits(payload)}
    if not got_ids:
        return {**scores_extra, "precision": 0.0, "recall_top": 0.0,
                "newest_is_true_newest": 0.0, "newest_present": 0.0}

    overlap = len(truth_ids & got_ids)
    scores = {
        "precision": overlap / len(got_ids),
        # Recall over the whole history is meaningless for a paged result --
        # what matters is whether the newest messages, the ones a "latest"
        # query is asking for, made it in.
        "recall_top": len(got_ids & set(reachable)) / len(reachable) if reachable else 1.0,
    }
    newest = truth[0]["message_id"]
    flat = _flat_in_caller_order(payload)
    scores["newest_is_true_newest"] = 1.0 if flat and flat[0].get("message_id") == newest else 0.0
    scores["newest_present"] = 1.0 if newest in got_ids else 0.0
    scores.update(scores_extra)
    return scores


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def run_case(engine: SearchEngine, oracle: Oracle, case: Case, max_sessions: int = 8,
             index_horizon: Optional[float] = None, index_con=None) -> Result:
    parsed = parse_query(case.query, contact_index=engine.contact_index)
    started = time.time()
    try:
        payload = engine.search(
            case.query, SearchFilters(), SearchOptions(max_sessions=max_sessions)
        )
    except Exception as error:  # noqa: BLE001 - a crash is a result, not a stop
        return Result(case, {}, parsed, time.time() - started, error=repr(error))
    elapsed = time.time() - started
    return Result(
        case,
        payload,
        parsed,
        elapsed,
        properties=check_properties(case, payload, parsed, oracle),
        oracle=score_against_oracle(parsed, payload, oracle, index_horizon, index_con),
    )


def summarize(results: Sequence[Result]) -> dict:
    by_class: Dict[str, List[Result]] = {}
    for result in results:
        by_class.setdefault(result.case.cls, []).append(result)

    property_names = sorted(
        {name for r in results for name in r.properties}
    )
    report: dict = {"classes": {}, "properties": {}, "failures": []}

    for cls, rows in sorted(by_class.items()):
        latencies = [r.elapsed_s for r in rows if r.error is None]
        report["classes"][cls] = {
            "n": len(rows),
            "errors": sum(1 for r in rows if r.error),
            "p50_s": round(statistics.median(latencies), 3) if latencies else None,
            "p95_s": round(max(latencies), 3) if latencies else None,
            "pass_rate": _pass_rate([r for r in rows]),
        }

    for name in property_names:
        checked = [r for r in results if r.properties.get(name) is not None]
        passed = [r for r in checked if r.properties[name]]
        report["properties"][name] = {
            "checked": len(checked),
            "passed": len(passed),
            "rate": round(len(passed) / len(checked), 3) if checked else None,
        }

    oracle_names = sorted({k for r in results for k in r.oracle})
    report["oracle"] = {
        name: round(
            statistics.mean([r.oracle[name] for r in results if name in r.oracle]), 3
        )
        for name in oracle_names
        if any(name in r.oracle for r in results)
    }

    for result in results:
        failed = [n for n, v in result.properties.items() if v is False]
        # An oracle miss is a failure too. Listing only property failures
        # meant recall_top could sit at 0.72 with the report announcing
        # "0 failing queries", which is how a metric stops being read.
        failed += [
            name for name in ("recall_top", "newest_present", "newest_is_true_newest")
            if result.oracle.get(name) is not None and result.oracle[name] < 0.9
        ]
        if failed or result.error:
            report["failures"].append(
                {
                    "query": result.case.query,
                    "class": result.case.cls,
                    "failed": failed,
                    "error": result.error,
                    "oracle": {k: round(v, 3) for k, v in result.oracle.items()},
                }
            )
    return report


def _pass_rate(rows: Sequence[Result]) -> Optional[float]:
    checked = [v for r in rows for v in r.properties.values() if v is not None]
    return round(sum(1 for v in checked if v) / len(checked), 3) if checked else None


def print_report(report: dict) -> None:
    print(f"\n{'class':<22}{'n':>4}{'err':>5}{'pass':>8}{'p50':>8}{'p95':>8}")
    for cls, row in report["classes"].items():
        print(
            f"{cls:<22}{row['n']:>4}{row['errors']:>5}"
            f"{(row['pass_rate'] if row['pass_rate'] is not None else 0):>8.2f}"
            f"{(row['p50_s'] or 0):>8.2f}{(row['p95_s'] or 0):>8.2f}"
        )
    print(f"\n{'property':<24}{'checked':>9}{'passed':>8}{'rate':>8}")
    for name, row in report["properties"].items():
        rate = row["rate"]
        print(f"{name:<24}{row['checked']:>9}{row['passed']:>8}{(rate if rate is not None else -1):>8.2f}")
    if report["oracle"]:
        print(f"\n{'oracle metric':<24}{'mean':>8}")
        for name, value in report["oracle"].items():
            print(f"{name:<24}{value:>8.2f}")
    if report["failures"]:
        print(f"\n{len(report['failures'])} failing quer{'y' if len(report['failures']) == 1 else 'ies'}:")
        for row in report["failures"]:
            detail = row["error"] or ", ".join(row["failed"])
            print(f"  [{row['class']}] {row['query']!r}: {detail}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-db", required=True)
    parser.add_argument("--chat-db", required=True)
    parser.add_argument("--chat-db-source", default=None, help="live chat.db, for freshness")
    parser.add_argument("--suite", default=None, help="path to a JSON suite; default is built in")
    parser.add_argument("--only", default=None, help="run one class")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    from seaglass.eval.suites import build_suite

    # A running app holds its own copy of the embedding and rerank models.
    # Two copies on one machine contend for memory and GPU, and a rerank
    # that takes 2.6s alone took 96s alongside the app -- so latency
    # gathered this way says nothing about the product.
    lock = Path.home() / ".seaglass" / "app.lock"
    if lock.exists():
        print(
            "WARNING: Seaglass appears to be running (~/.seaglass/app.lock).\n"
            "         Quality numbers are still valid; latency numbers are not.\n",
            file=sys.stderr,
        )

    engine = SearchEngine(args.index_db, args.chat_db, chat_db_source=args.chat_db_source)
    engine.warmup(progress=lambda name: __import__("contextlib").nullcontext())
    # The oracle reads the *live* chat.db when one is configured: an oracle
    # bounded by the same snapshot the index was built from can never
    # observe staleness, which is one of the failures the suite exists to
    # separate out.
    oracle_con = engine.chat_con
    if args.chat_db_source:
        oracle_con = sqlite3.connect(f"file:{args.chat_db_source}?mode=ro&immutable=1", uri=True)
    oracle = Oracle(oracle_con)
    index_horizon = engine.index_con.execute("SELECT MAX(end_ts) FROM chunks").fetchone()[0]

    cases = build_suite(engine, path=Path(args.suite) if args.suite else None)
    if args.only:
        cases = [c for c in cases if c.cls == args.only]

    results = [
        run_case(engine, oracle, case, index_horizon=index_horizon, index_con=engine.index_con)
        for case in cases
    ]
    report = summarize(results)
    print_report(report)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
