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
import contextlib
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
from seaglass.imessage.contacts import ContactIndex
from seaglass.imessage.source import connect_readonly
from seaglass.search.parse import ParsedQuery, parse_query

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
        from_me: bool = False,
        limit: int = 500,
    ) -> List[dict]:
        """Messages *written by* one of `handles`, newest first -- or, when
        `from_me` is set, messages *I* wrote to one of them.

        The direction test is not redundant: chat.db stamps outgoing 1:1
        messages with the *recipient's* handle_id, so a handle test alone
        returns both halves of the conversation. Which half is the answer
        depends on the question -- "what did I tell Kaya" is asking for the
        outgoing half, and scoring it against Kaya's own messages marks the
        engine wrong for being right.
        """
        if not handles:
            return []
        marks = ",".join("?" for _ in handles)
        where = [
            f"h.id IN ({marks})",
            f"m.is_from_me = {1 if from_me else 0}",
            "m.associated_message_type = 0",
        ]
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
        # Which join defines "a message between me and X" depends on the
        # direction. Incoming messages carry X's handle_id directly. My
        # *outgoing* messages in a group chat carry handle_id 0 -- there is
        # no single recipient -- so a handle join sees only the 1:1 half and
        # declares yesterday's 1:1 message the newest thing I ever told
        # them, while the engine correctly returns today's group message.
        # Scoping the outgoing half by chat membership is the same
        # definition the engine applies: messages in a conversation X is
        # part of, written by me.
        join = (
            "JOIN chat_message_join cmj ON cmj.message_id = m.ROWID "
            "JOIN chat_handle_join chj ON chj.chat_id = cmj.chat_id "
            "JOIN handle h ON h.ROWID = chj.handle_id"
            if from_me
            else "JOIN handle h ON m.handle_id = h.ROWID "
                 "JOIN chat_message_join cmj ON cmj.message_id = m.ROWID"
        )
        rows = self.con.execute(
            f"""
            SELECT DISTINCT m.ROWID, {_TS_EXPR} AS ts, m.text, m.attributedBody IS NOT NULL
            FROM message m
            {join}
            WHERE {' AND '.join(where)}
            ORDER BY ts DESC LIMIT ?
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
    # TEST-EVAL-PLAN-V2.md §3. What the case was *built* to ask for, which
    # is not the same as what the engine thought it was asked. Judging a
    # payload against its own `effective_filters` is circular: a parse that
    # extracts nothing makes every filter property inapplicable, and the
    # case passes vacuously. That is exactly how "recent messages from
    # kaya" scored a clean pass while returning a stranger's messages.
    expect_handles: List[str] = field(default_factory=list)
    expect_person: Optional[str] = None
    expect_date: bool = False
    expect_from_me: bool = False
    expect_media: bool = False
    form: str = "plain"
    # A filter-only case: chat.db can compute its exact answer set. Declared
    # by the suite rather than inferred from the parse, because a parse that
    # failed leaves the name in the residual and would look "topical".
    oracle_scored: bool = False
    # Is the named person the *author* of the wanted messages, or just a
    # participant in the conversation? Other people speaking in a
    # conversation you asked to see is correct; other people answering
    # "what did Kaya say" is not.
    expect_sender: bool = True


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
    """What a caller honouring `limit` actually reads.

    When the target emits a flat list of its own (grogu does), that list
    *is* the caller order and gets read back verbatim. Rebuilding it as
    "every session's hits, then every session's context" instead quietly
    repaired the ordering before grading it: grogu shipped a session's
    context above a later session's match for two commits while this
    harness reported the property it broke at 1.00.

    For a target that returns nested sessions and leaves flattening to
    its caller, that hits-then-context reconstruction is still the right
    model, because it is what the caller will do.
    """
    order = payload.get("_caller_order")
    if order is not None:
        return list(order)
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

    # --- declared expectations (TEST-EVAL-PLAN-V2.md §3) -------------------
    # These are the only properties that can fail when the parse comes back
    # empty, which is precisely the failure mode the rest of this function
    # cannot see.
    if case.expect_handles:
        wanted = {h.lower() for h in case.expect_handles}
        found = {h.lower() for h in list(parsed.people_sender) + list(parsed.people_participant)}
        props["person_filter_applied"] = bool(found & wanted)
        if hits and case.expect_person and case.expect_sender:
            # Hydration resolves handles to display names, so the check is
            # on the name: every message we were handed should be from the
            # person asked for (or from me, in a conversation with them).
            wanted_name = case.expect_person.lower()
            senders = {
                (m.get("sender") or "").lower() for m in hits if not m.get("is_from_me")
            }
            props["sender_is_expected"] = all(
                wanted_name in sender or sender in wanted_name for sender in senders
            ) if senders else None
        else:
            props["sender_is_expected"] = None
    else:
        props["person_filter_applied"] = None
        props["sender_is_expected"] = None

    props["date_filter_applied"] = (
        (parsed.date_from is not None or parsed.date_to is not None) if case.expect_date else None
    )
    props["self_filter_applied"] = bool(parsed.from_me) if case.expect_from_me else None
    props["media_filter_applied"] = bool(parsed.has_media) if case.expect_media else None

    # "Did it find anything" is only a fair question when there is something
    # to find. "what did Sraddha say yesterday" returns nothing because
    # Sraddha last wrote in 2021 -- scoring that as a miss punished the
    # engine for being right, and hid the cases where it is wrong.
    expected = case.expects_results
    handles = case.expect_handles or parsed.people_sender
    if expected and handles:
        expected = bool(
            oracle.messages_from(
                handles, since=parsed.date_from, until=parsed.date_to,
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

    # Only a target that emits its own flat list can get this wrong; a
    # nested payload has no single order to be wrong about. The old check
    # here asked whether the first session was non-empty, which is what
    # `no_empty_sessions` already answers -- so it read 1.00 on 206 of 208
    # cases while grogu was shipping context above a later session's match.
    caller_order = payload.get("_caller_order")
    if caller_order:
        kinds = [row.get("_kind", "hit") for row in caller_order]
        first_context = next((i for i, k in enumerate(kinds) if k == "context"), None)
        props["context_after_hits"] = (
            True if first_context is None
            else all(k == "context" for k in kinds[first_context:])
        )
    else:
        props["context_after_hits"] = None

    if case.lexical and sessions:
        needle = case.lexical.lower()
        props["lexical_presence"] = any(
            needle in (m.get("text") or "").lower()
            for m in _flat_in_caller_order(payload)
        )
    else:
        props["lexical_presence"] = None

    # These two were `"key" in payload` -- structurally true on every case
    # that ever ran, so they contributed two guaranteed 1.00 rows to the
    # report and could never fail. Judge the value instead of its presence.
    stale = payload.get("index_stale")
    behind = payload.get("n_messages_since_index")
    props["freshness_declared"] = (
        isinstance(stale, bool)
        and isinstance(behind, int)
        and behind >= 0
        # "stale" has to mean something: it is a claim about messages the
        # answer could not see, so it must agree with the count.
        and (behind > 0) == bool(stale)
    ) if "index_stale" in payload else False

    ordering = payload.get("ordering")
    if ordering not in ("recent", "relevance"):
        props["ordering_declared"] = False
    elif ordering == "recent" and payload.get("_caller_order"):
        # A declared chronological answer must actually be chronological
        # for a target that emits one flat list -- that declaration is
        # exactly what grogu acts on when it re-sorts. A nested payload
        # groups by session and has no single order to check, which is
        # what recency_order and recency_session_order grade instead.
        stamps = [m.get("ts") or 0 for m in payload["_caller_order"]
                  if m.get("_kind", "hit") == "hit"]
        props["ordering_declared"] = (
            stamps == sorted(stamps, reverse=True) if len(stamps) > 1 else True
        )
    else:
        props["ordering_declared"] = True
    return props


def score_against_oracle(
    parsed, payload: dict, oracle: Oracle, index_horizon: Optional[float] = None,
    index_con=None, case: Optional[Case] = None,
    reachable_ids: Optional[set] = None,
) -> Dict[str, float]:
    """Precision/recall of the returned hits against chat.db's answer, for
    queries whose answer set is exactly computable."""
    # Keyed on what the case *asked for* where that is declared, not on what
    # the engine decided it was asked (TEST-EVAL-PLAN-V2.md §3). Keying on
    # the engine's own parse meant a dropped person filter skipped oracle
    # scoring entirely instead of scoring zero.
    handles = list(case.expect_handles) if case is not None and case.expect_handles else list(parsed.people_sender)
    # Only filter-only queries have a computable answer set. "what did Kaya
    # say about dinner" is answered by similarity, so the newest message is
    # not the right answer and scoring it as one would punish the engine for
    # being correct (QUERY-EVAL-PLAN.md §3: classes 3, 4, 6, 7, 9, 11).
    topical = parsed.semantic.strip() and not (case is not None and case.oracle_scored)
    if not handles or topical:
        return {}
    truth = oracle.messages_from(
        handles,
        since=parsed.date_from,
        until=parsed.date_to,
        with_media=parsed.has_media,
        from_me=bool(case.expect_from_me) if case is not None else parsed.from_me,
    )
    if not truth:
        return {}

    # A message the index has never seen cannot be ranked, and counting it
    # as a ranking miss hides the two failures behind one number: "the
    # engine ordered badly" and "the engine was asked a question about
    # messages that arrived after the last build" need opposite fixes
    # (QUERY-EVAL-PLAN.md §6).
    scores_extra: Dict[str, float] = {}
    # ...unless the engine reached past the index to answer. A filters-only
    # query is served from chat.db directly, so "the index has never seen
    # it" no longer implies "the engine could not return it". Clipping the
    # truth here anyway scored the engine against the newest *indexed*
    # message while it correctly answered with a newer unindexed one, and
    # marked it wrong for the improvement: 30 person-recency cases failed
    # newest_is_true_newest the first time the tail ran.
    if payload.get("unindexed_included"):
        index_horizon = None
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
        # Same allowance as the horizon above: "the index never chunked it"
        # only bounds what the engine can return while the index is the
        # only thing it reads. Once the tail is live the newest messages
        # are reachable *because* they are unindexed, so clipping recall to
        # the indexed subset demands the older half of the head and marks a
        # correctly-newest answer as a recall miss -- 31 grogu cases, whose
        # limit of 20 messages was entirely filled by newer, real results.
        if not payload.get("unindexed_included"):
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
        # query is asking for, made it in -- across the pages a user can
        # actually reach, since page 1 holds a fixed number of sessions.
        "recall_top": (
            len((reachable_ids if reachable_ids is not None else got_ids) & set(reachable))
            / len(reachable)
        ) if reachable else 1.0,
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


class Corpus:
    """The databases and contact list the harness needs to *judge* an
    answer: the oracle, the index horizon, and the names the suite is
    generated from. Deliberately model-free -- none of this needs an
    embedder, and loading one here is what made the harness compete with
    the thing it was measuring.
    """

    def __init__(self, index_db: str, chat_db: str, chat_db_source: Optional[str] = None):
        self.index_con = sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)
        self.chat_con = connect_readonly(Path(chat_db))
        # The oracle reads the *live* chat.db when there is one: an oracle
        # bounded by the same snapshot the index was built from can never
        # observe staleness, which is one of the failures the suite exists
        # to tell apart.
        # `connect_readonly`, not a bare `immutable=1` connect. `immutable`
        # asserts the file cannot change, so SQLite skips the WAL -- on a
        # database Messages is actively writing that raises "database disk
        # image is malformed" at best and, per imessage.source, returns
        # silently corrupt reads at worst. Silently corrupt is the real
        # hazard here: the oracle is what decides whether the engine was
        # right, so a bad read marks a correct answer wrong.
        self.oracle_con = (
            connect_readonly(Path(chat_db_source))
            if chat_db_source else self.chat_con
        )
        self.oracle = Oracle(self.oracle_con)
        self.index_horizon = self.index_con.execute("SELECT MAX(end_ts) FROM chunks").fetchone()[0]
        try:
            self.contact_index = ContactIndex.load()
        except Exception:  # noqa: BLE001 - names are for generating cases, not judging them
            self.contact_index = None


class AppSearcher:
    """Ask the running desktop app, over the same loopback path Grogu uses.

    This is the whole point of not warning and walking away: the app
    already holds the models, so reusing it costs no memory, contends with
    nothing, and -- more usefully -- measures the code path that actually
    answers a user's query in production, rather than a second copy of the
    engine that only the harness ever runs.
    """

    name = "running app"

    def __init__(self, max_sessions: int = 8):
        self.max_sessions = max_sessions

    @staticmethod
    def available() -> bool:
        from seaglass.mcp_server import _running_app_lock

        return _running_app_lock() is not None

    def search(self, query: str, offset: int = 0) -> dict:
        from seaglass.mcp_server import _search_via_running_app

        payload = _search_via_running_app(
            query, max_sessions=self.max_sessions, redact=False, offset=offset
        )
        if payload is None:
            raise RuntimeError("the running app stopped answering mid-run")
        return payload


class EngineSearcher:
    """Load our own engine. Correct when no app is running, and the only
    way to measure a cold start."""

    name = "in-process engine"

    def __init__(self, index_db, chat_db, chat_db_source=None, max_sessions: int = 8):
        self.engine = SearchEngine(index_db, chat_db, chat_db_source=chat_db_source)
        self.engine.warmup(progress=lambda _name: contextlib.nullcontext())
        self.max_sessions = max_sessions

    def search(self, query: str, offset: int = 0) -> dict:
        return self.engine.search(
            query,
            SearchFilters(),
            SearchOptions(max_sessions=self.max_sessions, offset=offset),
        )


GROGU_SRC = Path.home() / "src" / "grogu" / "src"


def _import_grogu():
    """Import Grogu's real modules. Never reimplement its flattening here:
    the whole question this target answers is what *Grogu's own code* does
    to our payload before a user sees it."""
    if str(GROGU_SRC) not in sys.path:
        sys.path.insert(0, str(GROGU_SRC))
    import grogu_imessage
    import grogu_mcp

    return grogu_imessage, grogu_mcp


def as_grogu_shows_it(pages, flat: Sequence[dict]) -> dict:
    """Rebuild a scoreable payload from the rows Grogu actually returns.

    Grogu's public shape is a flat list, so that is what gets modelled:
    one session, messages in Grogu's own order. Anything else would judge
    an arrangement no caller of Grogu ever receives -- and the order is
    exactly what is under test, since Grogu re-sorts when seaglass
    declares a chronological answer.

    Rows Grogu labels `context` go to `context_messages`, matching the
    distinction the app draws, so both targets are judged on their
    matches. Before Grogu labelled them, everything was a match, which is
    precisely what made its sender purity 0.23.

    The top-level fields (`effective_filters`, `ordering`, `freshness`)
    come through untouched: they describe the search, which both targets
    share.
    """
    if isinstance(pages, dict):
        pages = [pages]
    pages = list(pages) or [{}]
    payload = pages[0]
    by_id, session_of = {}, {}
    for page, page_payload in enumerate(pages):
        for index, session in enumerate(page_payload.get("sessions", [])):
            for message in session.get("messages", []) + session.get("context_messages", []):
                identifier = message.get("message_id")
                by_id.setdefault(identifier, message)
                session_of.setdefault(identifier, (page, index))
    hits, context, sources, seen = [], [], set(), set()
    for row in flat:
        identifier = row.get("id")
        if identifier not in by_id or identifier in seen:
            continue
        seen.add(identifier)
        sources.add(session_of.get(identifier))
        (context if row.get("kind") == "context" else hits).append(by_id[identifier])

    out = {k: v for k, v in payload.items() if k != "sessions"}
    out["sessions"] = (
        [{"chat_id": None, "messages": hits, "context_messages": context}]
        if hits or context else []
    )
    # Grogu cannot ask for more; saying otherwise would let the oracle
    # score it on messages no caller of Grogu can ever reach.
    out["has_more"] = False
    out["_source_sessions"] = len(sources)
    # The order grogu actually emitted, before this function split it into
    # hits and context. Without it the split silently sorts grogu's answer
    # into the shape the properties expect, and an ordering defect in the
    # code under test becomes unobservable.
    out["_caller_order"] = [
        dict(by_id[row["id"]], _kind=row.get("kind") or "hit")
        for row in flat
        if row.get("id") in by_id
    ]
    return out


class GroguSearcher:
    """Grogu's path end to end: its MCP call, its flatten, its limit.

    Runs against the same running app as `AppSearcher` -- the MCP server
    forwards to it -- so this measures Grogu's *handling* of the answer,
    not a second engine's ranking of it.
    """

    name = "grogu (MCP -> running app)"

    def __init__(self, max_sessions: int = 8, limit: int = 20):
        self.imessage, self.mcp = _import_grogu()
        self.max_sessions = max_sessions
        self.limit = limit

    @staticmethod
    def available() -> bool:
        try:
            imessage, _ = _import_grogu()
        except Exception:  # noqa: BLE001 - grogu simply is not installed
            return False
        return bool(imessage.seaglass_available())

    def search(self, query: str, offset: int = 0) -> dict:
        # Grogu's own paging is inside search_via_seaglass, driven by the
        # ordering seaglass declares. An offset from the harness would be
        # a second, contradictory pager measuring nothing real.
        if offset:
            raise RuntimeError("grogu pages internally; it takes no offset")
        # The public entry point, not the flattener underneath it: paging,
        # limit and ordering are all decided here, and a harness that
        # reaches past them measures code no caller runs.
        started = time.time()
        rows = self.imessage.search_via_seaglass(query, limit=self.limit)
        elapsed = time.time() - started
        payload = as_grogu_shows_it(self._pages_covering(query, rows), rows)
        # Hydrating for the oracle costs another round trip or three, and
        # a caller pays none of it. Report what Grogu itself took.
        payload["_elapsed_s"] = elapsed
        return payload

    def _pages_covering(self, query: str, rows: Sequence[dict]) -> List[dict]:
        """The pages Grogu's rows came from.

        Grogu returns `{id, text, date, handle, kind}`; judging a filter
        needs `is_from_me` and `has_attachment` too, which only the
        payload carries. Grogu pages a chronological answer, so following
        the same pages is what makes every returned row resolvable --
        stopping at page one would silently drop the rows paging was
        added to reach and score Grogu for a gap it no longer has.
        """
        wanted = {row.get("id") for row in rows}
        pages, offset = [], 0
        for _ in range(getattr(self.imessage, "SEAGLASS_MAX_PAGES", 3)):
            payload = self.mcp.call_tool(
                self.imessage.SEAGLASS_SERVER_NAME,
                "search_messages",
                query=query,
                max_sessions=self.max_sessions,
                offset=offset,
            )
            if not isinstance(payload, dict):
                break
            pages.append(payload)
            wanted -= {
                m.get("message_id")
                for session in payload.get("sessions", [])
                for m in session.get("messages", []) + session.get("context_messages", [])
            }
            if not wanted or not payload.get("has_more"):
                break
            offset = payload.get("next_offset") or offset + self.max_sessions
        return pages

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.mcp.close_all()


def parsed_from_payload(payload: dict, query: str, contact_index=None) -> ParsedQuery:
    """Recover what the engine decided the query meant.

    Re-parsing locally would test the harness's copy of the parser, not the
    one that produced this answer -- and against the running app they can
    be different builds. `effective_filters` is the engine saying what it
    did, so judge it on that.
    """
    filters = payload.get("effective_filters")
    if not filters:
        return parse_query(query, contact_index=contact_index)
    return ParsedQuery(
        raw=query,
        semantic=filters.get("semantic") or "",
        people_participant=list(filters.get("people_participant") or []),
        people_sender=list(filters.get("people_sender") or []),
        date_from=filters.get("date_from"),
        date_to=filters.get("date_to"),
        has_media=bool(filters.get("has_media")),
        is_group=filters.get("is_group"),
        chat_ids=filters.get("chat_ids"),
        from_me=filters.get("from_me"),
    )


ORACLE_EXTRA_PAGES = 2


def _paged_message_ids(searcher, case: Case, payload: dict) -> set:
    """Message ids reachable across the first few pages.

    Page 1 holds a fixed number of day-sessions, and the 20 newest messages
    from a chatty contact routinely span more. Scoring "are the newest
    messages retrievable" against one page measures the page size, not the
    engine: three of Sraddha's newest 20 were on page 2 the whole time.
    Latency is still judged on page 1 alone.
    """
    ids = {hit.get("message_id") for hit in _hits(payload)}
    seen = payload
    for page in range(1, ORACLE_EXTRA_PAGES + 1):
        if not seen.get("has_more"):
            break
        try:
            seen = searcher.search(case.query, offset=seen.get("next_offset") or page * 8)
        except (TypeError, RuntimeError):
            break
        ids |= {hit.get("message_id") for hit in _hits(seen)}
    return ids


def run_case(searcher, corpus: Corpus, case: Case) -> Result:
    started = time.time()
    try:
        payload = searcher.search(case.query)
    except Exception as error:  # noqa: BLE001 - a crash is a result, not a stop
        parsed = parse_query(case.query, contact_index=corpus.contact_index)
        return Result(case, {}, parsed, time.time() - started, error=repr(error))
    elapsed = payload.get("_elapsed_s", time.time() - started)
    parsed = parsed_from_payload(payload, case.query, corpus.contact_index)
    reachable_ids = None
    # Any browse-ordered answer for a declared person is oracle-scored, not
    # just the classes flagged `oracle_scored` -- "pictures Kaya sent" has
    # an exactly computable answer set too, and gating on the flag left it
    # scored against page 1 alone.
    if case.expect_handles and payload.get("ordering") == "recent":
        reachable_ids = _paged_message_ids(searcher, case, payload)
    return Result(
        case,
        payload,
        parsed,
        elapsed,
        properties=check_properties(case, payload, parsed, corpus.oracle),
        oracle=score_against_oracle(
            parsed, payload, corpus.oracle, corpus.index_horizon, corpus.index_con,
            case, reachable_ids,
        ),
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
    if report.get("target"):
        print(f"\nanswered by: {report['target']}")
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


def run_comparison(args, corpus) -> int:
    """The same cases through both front doors.

    Grogu and the app share an engine, so every difference here is
    something Grogu does to the answer on its way to the caller. Running
    them in one process against one app keeps that honest: same index,
    same models, same warm caches, no second engine to blame.
    """
    from seaglass.eval.suites import build_suite

    if not AppSearcher.available():
        print("No running app; --compare needs one.", file=sys.stderr)
        return 2
    if not GroguSearcher.available():
        print("grogu has no seaglass MCP server configured.", file=sys.stderr)
        return 2

    cases = build_suite(corpus, path=Path(args.suite) if args.suite else None)
    if args.only:
        cases = [c for c in cases if c.cls == args.only]

    targets = {
        "app": AppSearcher(max_sessions=args.max_sessions),
        "grogu": GroguSearcher(max_sessions=args.max_sessions, limit=args.grogu_limit),
    }
    reports = {}
    per_case = {}
    for label, searcher in targets.items():
        print(f"\nrunning {len(cases)} cases through {searcher.name}...", file=sys.stderr)
        results = [run_case(searcher, corpus, case) for case in cases]
        report = summarize(results)
        report["target"] = searcher.name
        reports[label] = report
        # Keyed by class *and* query: the same text appears in several
        # classes (surface forms overlap person classes), and keying on
        # the query alone silently reported one case's numbers twice.
        per_case[label] = {(r.case.cls, r.case.query): r for r in results}
        print_report(report)
        if hasattr(searcher, "close"):
            searcher.close()

    print_comparison(reports, per_case, cases)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"reports": reports, "cases": _case_deltas(per_case, cases)}, indent=2
        ))
    return 0


def _case_deltas(per_case: dict, cases: Sequence[Case]) -> list:
    rows = []
    for case in cases:
        app = per_case["app"].get((case.cls, case.query))
        grogu = per_case["grogu"].get((case.cls, case.query))
        if not app or not grogu:
            continue
        rows.append({
            "query": case.query,
            "class": case.cls,
            "app_hits": len(_hits(app.payload)),
            "grogu_hits": len(_hits(grogu.payload)),
            "app_sessions": len(app.payload.get("sessions", [])),
            "grogu_sessions": grogu.payload.get(
                "_source_sessions", len(grogu.payload.get("sessions", []))
            ),
            "app_latency_s": round(app.elapsed_s, 3),
            "grogu_latency_s": round(grogu.elapsed_s, 3),
            "app_oracle": {k: round(v, 3) for k, v in app.oracle.items()},
            "grogu_oracle": {k: round(v, 3) for k, v in grogu.oracle.items()},
            "app_failed": [n for n, v in app.properties.items() if v is False],
            "grogu_failed": [n for n, v in grogu.properties.items() if v is False],
        })
    return rows


def print_comparison(reports: dict, per_case: dict, cases: Sequence[Case]) -> None:
    app, grogu = reports["app"], reports["grogu"]
    print("\n" + "=" * 68)
    print("SIDE BY SIDE: the app's page vs what grogu hands its caller")
    print("=" * 68)

    names = sorted(set(app["properties"]) | set(grogu["properties"]))
    print(f"\n{'property':<24}{'app':>8}{'grogu':>8}{'delta':>8}")
    for name in names:
        a = (app["properties"].get(name) or {}).get("rate")
        g = (grogu["properties"].get(name) or {}).get("rate")
        delta = f"{g - a:+.2f}" if a is not None and g is not None else "--"
        print(f"{name:<24}{_fmt(a):>8}{_fmt(g):>8}{delta:>8}")

    names = sorted(set(app["oracle"]) | set(grogu["oracle"]))
    print(f"\n{'oracle metric':<24}{'app':>8}{'grogu':>8}{'delta':>8}")
    for name in names:
        a, g = app["oracle"].get(name), grogu["oracle"].get(name)
        delta = f"{g - a:+.2f}" if a is not None and g is not None else "--"
        print(f"{name:<24}{_fmt(a):>8}{_fmt(g):>8}{delta:>8}")

    rows = _case_deltas(per_case, cases)
    if rows:
        print(f"\n{'coverage':<24}{'app':>8}{'grogu':>8}")
        for label, key in (("sessions shown (mean)", "sessions"), ("messages shown (mean)", "hits")):
            a = statistics.mean(r[f"app_{key}"] for r in rows)
            g = statistics.mean(r[f"grogu_{key}"] for r in rows)
            print(f"{label:<24}{a:>8.1f}{g:>8.1f}")
        dropped = [r for r in rows if r["grogu_sessions"] < r["app_sessions"]]
        print(f"{'queries losing sessions':<24}{'':>8}{len(dropped):>8}  of {len(rows)}")

    print(f"\n{'latency (s)':<24}{'app':>8}{'grogu':>8}")
    for label, fn in (("p50", statistics.median), ("p95", max)):
        a = fn([r["app_latency_s"] for r in rows])
        g = fn([r["grogu_latency_s"] for r in rows])
        print(f"{label:<24}{a:>8.2f}{g:>8.2f}")

    print(f"\nfailing queries{'':<9}{len(app['failures']):>8}{len(grogu['failures']):>8}")
    only_grogu = (
        {f["query"] for f in grogu["failures"]} - {f["query"] for f in app["failures"]}
    )
    if only_grogu:
        print(f"\n{len(only_grogu)} quer{'y' if len(only_grogu) == 1 else 'ies'} that only grogu fails:")
        for row in grogu["failures"]:
            if row["query"] in only_grogu:
                print(f"  [{row['class']}] {row['query']!r}: "
                      f"{row['error'] or ', '.join(row['failed'])}")


def _fmt(value) -> str:
    return "--" if value is None else f"{value:.2f}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-db", required=True)
    parser.add_argument("--chat-db", required=True)
    parser.add_argument("--chat-db-source", default=None, help="live chat.db, for freshness")
    parser.add_argument("--suite", default=None, help="path to a JSON suite; default is built in")
    parser.add_argument("--only", default=None, help="run one class")
    parser.add_argument("--json-out", default=None)
    parser.add_argument(
        "--target", choices=("auto", "app", "in-process", "grogu"), default="auto",
        help="auto: reuse the running app if there is one, else load an engine",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="run the app and grogu on the same cases and print the delta",
    )
    parser.add_argument(
        "--grogu-limit", type=int, default=20,
        help="message limit grogu applies (its `imessage search --limit`)",
    )
    parser.add_argument("--max-sessions", type=int, default=8)
    args = parser.parse_args(argv)

    from seaglass.eval.suites import build_suite

    corpus = Corpus(args.index_db, args.chat_db, args.chat_db_source)

    # Reuse the app's engine when the app is running. Loading a second copy
    # of the models beside it costs a gigabyte and makes both slower: a
    # rerank that takes 2.6s alone took 96s alongside the app, which made
    # every latency number meaningless. Reusing it also measures the path
    # that actually answers a user's query.
    if args.compare:
        return run_comparison(args, corpus)

    if args.target == "grogu":
        if not GroguSearcher.available():
            print("grogu has no seaglass MCP server configured.", file=sys.stderr)
            return 2
        searcher = GroguSearcher(max_sessions=args.max_sessions, limit=args.grogu_limit)
    elif args.target == "app" or (args.target == "auto" and AppSearcher.available()):
        if not AppSearcher.available():
            print("No running app to target (--target app).", file=sys.stderr)
            return 2
        searcher = AppSearcher(max_sessions=args.max_sessions)
    else:
        searcher = EngineSearcher(
            args.index_db, args.chat_db, args.chat_db_source, max_sessions=args.max_sessions
        )
    print(f"target: {searcher.name}", file=sys.stderr)

    cases = build_suite(corpus, path=Path(args.suite) if args.suite else None)
    if args.only:
        cases = [c for c in cases if c.cls == args.only]

    results = [run_case(searcher, corpus, case) for case in cases]
    report = summarize(results)
    report["target"] = searcher.name
    print_report(report)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
