"""`seaglass/mcp_server.py` — PLAN.md §6 Phase 6: the MCP server that
lets Grogu (or any MCP client) search iMessage history through seaglass.

**Deviation from PLAN.md's Phase 6 design, recorded here and in
ADDENDUM.md §16**: PLAN.md specifies a daemon + thin-shim architecture
(a long-lived `imsearchd` holding warm models behind a Unix socket, with
a tiny per-session shim process). Per the user's explicit decision, we
are deferring that: this module *is* the whole server, single-process,
started directly by the MCP client (ghcp) over stdio. It still gets a
meaningful share of the daemon's benefit for free, because an MCP
server process is itself long-lived for the duration of a client
session -- models are loaded lazily on first use and then held warm in
module-level globals for every subsequent tool call within that
session. What it does *not* get: sharing those warm models *across*
concurrent ghcp sessions (each spawns its own server process, each pays
its own model-load latency once), and the daemon's serialised-GPU-work
guarantee under concurrent load. Both are exactly the "how much does
this actually cost" measurements the user asked to defer rather than
build for up front; if real per-session cold-start latency turns out to
be a problem once the corpus is at full size, revisit the daemon design
then.

⚠️ Never write to stdout outside of the MCP transport itself -- stdio
*is* the protocol stream here (PLAN.md's own warning, and still true
even without a separate shim). All logging in this module goes to
stderr.

Tools implemented (PLAN.md §6 table, minus `sync_index` -- there is no
`index/sync.py` yet; Phase 7 incremental sync is still unbuilt, so
re-indexing is the only update path for now, exactly as the user agreed
to when told the machine is mid-iCloud-backfill):

- `search_messages(query, max_sessions=8, redact=False)` -- the main
  tool: full pre-filter -> RRF -> rerank -> aggregate -> expand ->
  hydrate -> format pipeline.
- `get_conversation(chat_id, around_ts=None, limit=50)` -- drill-down
  into a specific conversation the model has already seen cited, for
  when it wants more surrounding context than the citation gave it.
- `index_status()` -- chunk/vector counts and index paths, so the
  calling model (or a human debugging a session) can tell whether the
  index is stale or missing before trusting a search result.

Configuration is by environment variable (there being no daemon config
file / launchd plist yet, per the above deviation):

- `SEAGLASS_INDEX_DB` (required) -- path to a built `index.db`.
- `SEAGLASS_CHAT_DB` (optional) -- path to a `chat.db` snapshot for
  hydration and people-filtering; if unset, `search_messages` still
  works but returns un-hydrated chunk previews instead of real messages,
  and `get_conversation` is unavailable.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
import time
import zstandard
from pathlib import Path
from typing import List, Optional
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from mcp.server import MCPServer

from seaglass.imessage.contacts import ContactIndex, ContactsUnavailableError
from seaglass.imessage.source import apple_to_unix, connect_readonly
from seaglass.index.build import open_index_db
from seaglass.index.embed import EmbeddingModel
from seaglass.search.format import format_search_result
from seaglass.search.hydrate import hydrate_sessions
from seaglass.search.conversation import fetch_conversation
from seaglass.search.parse import parse_query
from seaglass.search.rank import aggregate_sessions, expand_sessions, rerank_candidates
from seaglass.search.rerank import CrossEncoderReranker
from seaglass.search.retrieve import retrieve

server = MCPServer(
    name="seaglass",
    version="0.1.0",
    instructions=(
        "Search the user's iMessage history by topic/content. Good for: "
        "\"what did we say about X\", \"find the message where Y\", topical "
        "recall over past conversations. Bad at: exhaustive enumeration "
        "(\"list every time we talked about X\") or counting -- this is a "
        "semantic + keyword retriever over the top handful of matches, not "
        "a full-corpus scan. `person` in search_messages matches chat "
        "*participants* (who was in the conversation), not people merely "
        "mentioned in the text."
    ),
)

# Lazily-initialized, process-lifetime-warm globals. See module docstring:
# this is the "get some of the daemon's benefit for free" mechanism -- an
# MCP server process persists for a whole client session, so paying the
# MLX model-load cost once here (not once per tool call) already avoids
# the worst of the "seconds of cold start on every call" failure mode,
# even without a separate always-on daemon process.
_index_con: Optional[sqlite3.Connection] = None
_chat_con: Optional[sqlite3.Connection] = None
_embedding_model: Optional[EmbeddingModel] = None
_reranker: Optional[CrossEncoderReranker] = None
_contact_index: Optional[ContactIndex] = None
_contact_index_attempted = False
_engine = None  # seaglass.app.engine.SearchEngine, built lazily

# The MCP SDK dispatches sync tool functions via a thread pool (not one
# fixed thread per session), so any lazy-init global here can race:
# concurrent tool calls could each see `_embedding_model is None` and each
# load a full MLX model, and SQLite connections opened with the default
# check_same_thread=True raise ProgrammingError when used from a
# different pool thread than the one that opened them. This lock
# serializes lazy-init AND the underlying model calls (MLX itself has no
# cross-thread safety guarantee, unlike SQLite which is safe per
# connection once check_same_thread=False is set).
_init_lock = threading.Lock()


def _log(msg: str) -> None:
    # Never print() here -- stdout is the MCP transport (see module docstring).
    print(f"[seaglass-mcp] {msg}", file=sys.stderr, flush=True)


def _resolve_path(env_var: str, from_config) -> Optional[str]:
    """The env var if set, else whatever the desktop app is configured to use.

    Pinning absolute paths in the MCP config is how this drifts: an index
    built for a one-off experiment gets wired in, the app later builds its
    real index somewhere else, and every Grogu search silently falls back
    to a substring scan because the configured path no longer exists.
    Sharing the app's own config resolution means Grogu and the app can't
    disagree about which index is current.
    """
    value = os.environ.get(env_var)
    if value:
        return value
    try:
        from seaglass.app.config import load_config

        value = from_config(load_config())
    except Exception:  # noqa: BLE001 - the env var is still the contract
        return None
    return value if value and Path(value).exists() else None


def _get_index_con() -> sqlite3.Connection:
    global _index_con
    if _index_con is None:
        with _init_lock:
            if _index_con is None:
                index_db = _resolve_path("SEAGLASS_INDEX_DB", lambda c: c.index_db)
                if not index_db:
                    raise RuntimeError(
                        "SEAGLASS_INDEX_DB is not set and no index was found at the "
                        "default location -- build one from the Seaglass app, or "
                        "point SEAGLASS_INDEX_DB at a built index.db."
                    )
                index_path = Path(index_db)
                if not index_path.exists():
                    raise RuntimeError(
                        f"SEAGLASS_INDEX_DB={index_db} does not exist -- check the path "
                        "(a typo'd path would otherwise silently create an empty index.db)."
                    )
                _log(f"opening index.db: {index_db}")
                _index_con = open_index_db(index_path, check_same_thread=False, create=False)
    return _index_con


def _get_chat_con() -> Optional[sqlite3.Connection]:
    global _chat_con
    if _chat_con is None:
        with _init_lock:
            if _chat_con is None:
                chat_db = _resolve_path("SEAGLASS_CHAT_DB", lambda c: c.chat_db)
                if not chat_db:
                    return None
                _log(f"opening chat.db snapshot: {chat_db}")
                _chat_con = connect_readonly(Path(chat_db))
    return _chat_con


def _get_embedding_model() -> EmbeddingModel:
    global _embedding_model
    if _embedding_model is None:
        with _init_lock:
            if _embedding_model is None:
                _log("loading embedding model (first call this session -- cold start)")
                t0 = time.time()
                _embedding_model = EmbeddingModel()
                _log(f"embedding model warm in {time.time() - t0:.2f}s")
    return _embedding_model


def _get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        with _init_lock:
            if _reranker is None:
                _log("loading reranker (first call this session -- cold start)")
                t0 = time.time()
                _reranker = CrossEncoderReranker()
                _log(f"reranker warm in {time.time() - t0:.2f}s")
    return _reranker


def _get_contact_index() -> Optional[ContactIndex]:
    global _contact_index, _contact_index_attempted
    if not _contact_index_attempted:
        with _init_lock:
            if not _contact_index_attempted:
                _contact_index_attempted = True
                try:
                    _contact_index = ContactIndex.load()
                except ContactsUnavailableError:
                    _log("Contacts unavailable -- names will degrade to raw handles")
    return _contact_index


# Separate from _init_lock: this serializes actual MLX inference calls
# (embedding + reranking), since MLX gives no cross-thread safety
# guarantee for concurrent GPU work, unlike SQLite (safe per-connection
# once check_same_thread=False). The MCP SDK dispatches sync tools via a
# thread pool, so two concurrent search_messages calls could otherwise
# both hit the GPU at once.
_pipeline_lock = threading.Lock()


@server.tool()
def search_messages(
    query: str,
    max_sessions: int = 8,
    redact: bool = False,
) -> dict:
    """Search the user's iMessage history by topic or content and return
    ranked, hydrated conversation sessions with message_id citations.

    Good for topical recall ("what did we decide about the trip", "find
    where she mentioned the vet appointment"). Not good for exhaustive
    enumeration or counting -- it returns the most relevant handful of
    conversation sessions, reranked for quality, not every match in the
    corpus. Any date range or participant name mentioned in `query` is
    parsed automatically (e.g. "last week", "with Sam") -- `person`
    matches chat participants, not people merely mentioned in message text.
    """
    with _pipeline_lock:
        return _search_messages_impl(query, max_sessions=max_sessions, redact=redact)


def _strip_ui_fields(payload: dict) -> dict:
    """Drop fields that only mean something to the desktop UI."""
    for key in ("timings", "parse_source", "request_id", "candidate_count"):
        payload.pop(key, None)
    return payload


def _get_engine():
    """The desktop app's SearchEngine, borrowing this module's connections.

    Ranking used to be reimplemented here -- retrieve, rerank, aggregate --
    which meant every ranking improvement made in the app silently skipped
    Grogu. The recency reservation and the recent-verbatim-match boost both
    landed that way, so the same query answered differently depending on
    which door it came in through, with the MCP answer being the worse of
    the two and nothing saying so.

    The handles are rebound on every call rather than captured once,
    because they are created lazily (and swapped wholesale by tests).
    """
    global _engine
    if _engine is None:
        with _init_lock:
            if _engine is None:
                from seaglass.app.engine import SearchEngine

                _engine = SearchEngine(index_db="", chat_db=None)
    # Staleness must be judged against the *live* chat.db. Without this the
    # engine falls back to the snapshot, which is by definition exactly as
    # stale as the index, so every result claims to be up to date.
    if not _engine.chat_db_source:
        try:
            from seaglass.app.config import load_config

            _engine.chat_db_source = load_config().chat_db_source
        except Exception:  # noqa: BLE001 - no config is not a search failure
            pass
    _engine.index_con = _get_index_con()
    _engine.chat_con = _get_chat_con()
    _engine.embedding_model = _get_embedding_model()
    _engine.reranker = _get_reranker()
    _engine.contact_index = _get_contact_index()
    return _engine


def _running_app_lock() -> Optional[dict]:
    """Connection details for a live desktop app, if one is running."""
    try:
        from seaglass.app.config import LOCK_PATH

        lock = json.loads(Path(LOCK_PATH).read_text())
        os.kill(int(lock["pid"]), 0)  # signal 0: liveness check only
    except Exception:  # noqa: BLE001 - no app, stale lock, or unreadable
        return None
    return lock


def _search_via_running_app(query: str, *, max_sessions: int, redact: bool) -> Optional[dict]:
    """Answer from the already-running desktop app, if there is one.

    The app holds the embedding model, the cross-encoder and a warm SQLite
    cache. Loading a second copy in this process costs about a gigabyte of
    memory and several seconds of cold start to compute an answer that
    already exists a few milliseconds away over loopback -- and it does it
    on the first Grogu search after every reboot, which is precisely when
    someone is waiting.

    Returns None (never raises) if there is no app, it isn't ready yet, or
    it is serving a different index than this process is configured for --
    every one of which means the in-process pipeline should answer instead.
    """
    lock = _running_app_lock()
    if lock is None:
        return None
    base = f"http://127.0.0.1:{lock['port']}"
    headers = {
        "Authorization": f"Bearer {lock['token']}",
        "Host": f"127.0.0.1:{lock['port']}",
        "Content-Type": "application/json",
    }

    def _call(path: str, body: Optional[dict] = None, timeout: float = 5.0):
        request = Request(
            base + path,
            data=None if body is None else json.dumps(body).encode(),
            headers=headers,
            method="GET" if body is None else "POST",
        )
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())

    try:
        status = _call("/api/status")
        if not status.get("index_ready"):
            return None
        # Answering from a different corpus than the caller configured
        # would be worse than being slow.
        wanted = _resolve_path("SEAGLASS_INDEX_DB", lambda c: c.index_db)
        if wanted and status.get("index_db") and Path(status["index_db"]) != Path(wanted):
            return None
        payload = _call(
            "/api/search",
            {
                "query": query,
                "filters": {},
                "options": {"max_sessions": max_sessions, "redact": redact},
                "assist": "off",
            },
            timeout=120.0,
        )
    except Exception as error:  # noqa: BLE001 - the local pipeline is the fallback
        _log(f"running app unavailable ({error}); answering in-process")
        return None
    _log("answered from the running Seaglass app")
    return payload


def _search_messages_impl(query: str, *, max_sessions: int, redact: bool) -> dict:
    t0 = time.time()

    payload = _search_via_running_app(query, max_sessions=max_sessions, redact=redact)
    if payload is not None:
        return _strip_ui_fields(payload)

    chat_con = _get_chat_con()

    if chat_con is not None:
        from seaglass.app.engine import SearchOptions
        from seaglass.app.filters import SearchFilters

        payload = _get_engine().search(
            query, SearchFilters(), SearchOptions(max_sessions=max_sessions, redact=redact)
        )
        return _strip_ui_fields(payload)

    # No chat.db to hydrate from: fall back to un-hydrated chunk previews
    # rather than failing the tool call outright. This path can't use the
    # engine, which has nothing to show without message bodies.
    index_con = _get_index_con()
    contact_index = _get_contact_index()
    parsed = parse_query(query, contact_index=contact_index)
    model = _get_embedding_model()
    fused = retrieve(index_con, parsed, model, chat_con=chat_con, fused_top_k=50)

    if not fused:
        return {
            "n_sessions": 0,
            "n_results": 0,
            "confidence": "none",
            "sessions": [],
            "elapsed_s": round(time.time() - t0, 2),
        }

    reranker = _get_reranker()
    ranked = rerank_candidates(index_con, parsed.semantic, fused, reranker)
    sessions = aggregate_sessions(ranked, max_sessions=max_sessions)
    expand_sessions(index_con, sessions)

    if True:
        # No chat.db configured -- degrade to un-hydrated chunk previews
        # rather than failing the tool call outright.
        dctx = zstandard.ZstdDecompressor()
        previews = []
        for session in sessions:
            texts = []
            for chunk_id in session.hit_chunk_ids:
                row = index_con.execute(
                    "SELECT body_semantic FROM chunks WHERE id = ?", (chunk_id,)
                ).fetchone()
                if row is not None:
                    texts.append(dctx.decompress(row[0]).decode("utf-8"))
            previews.append(
                {"chat_id": session.chat_id, "day": session.day, "score": session.score, "preview": texts}
            )
        return {
            "n_sessions": len(previews),
            "n_results": sum(len(p["preview"]) for p in previews),
            "confidence": "unhydrated (no SEAGLASS_CHAT_DB configured)",
            "sessions": previews,
            "elapsed_s": round(time.time() - t0, 2),
        }




@server.tool()
def get_conversation(chat_id: int, around_ts: Optional[float] = None, limit: int = 50) -> dict:
    """Fetch raw messages from a specific conversation (chat_id), for
    follow-up drill-down when the model wants more of a thread it has
    already seen cited by search_messages. If around_ts (unix seconds) is
    given, returns the `limit` messages closest in time to it; otherwise
    returns the most recent `limit` messages in the chat.
    """
    chat_con = _get_chat_con()
    if chat_con is None:
        raise RuntimeError(
            "get_conversation requires SEAGLASS_CHAT_DB to be set -- "
            "no chat.db snapshot is configured for this server."
        )
    return fetch_conversation(
        chat_con,
        chat_id=chat_id,
        around_ts=around_ts,
        limit=limit,
        contact_index=_get_contact_index(),
    )



@server.tool()
def index_status() -> dict:
    """Report index size, freshness and hydration support.

    Freshness is the part that matters to a caller: the index is a
    snapshot, so anything said since the last build is simply invisible to
    search. Reporting only chunk counts (as this once did) meant a caller
    could not distinguish "she never mentioned it" from "she mentioned it
    after the last sync", which are opposite answers. `n_messages_since_index`
    is that gap; `stale` is the same thing as a yes/no.
    """
    status: dict = {
        "index_db": _resolve_path("SEAGLASS_INDEX_DB", lambda c: c.index_db),
        "chat_db": _resolve_path("SEAGLASS_CHAT_DB", lambda c: c.chat_db),
    }

    live = _app_status()
    if live is not None:
        live["stale"] = bool(live.get("n_messages_since_index"))
        live["served_by"] = "app"
        return live

    try:
        index_con = _get_index_con()
    except RuntimeError as exc:
        status["error"] = str(exc)
        return status

    (n_chunks,) = index_con.execute("SELECT COUNT(*) FROM chunks").fetchone()
    (n_vectors,) = index_con.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()
    (max_date,) = index_con.execute("SELECT MAX(end_ts) FROM chunks").fetchone()
    status.update(
        {
            "n_chunks": n_chunks,
            "n_vectors": n_vectors,
            "most_recent_chunk_date_end": max_date,
            "most_recent_chunk_ts": max_date,
            "hydration_available": _get_chat_con() is not None,
            "contacts_available": _get_contact_index() is not None,
            "served_by": "mcp",
        }
    )
    status.update(_staleness(max_date))
    return status


def _staleness(most_recent_chunk_ts) -> dict:
    """How many messages have arrived since the index was last built.

    Delegated to the engine rather than reimplemented: it reads the *live*
    chat.db (the snapshot is by definition as stale as the index) and it
    already handles chat.db's seconds-or-nanoseconds `date` column, which
    is exactly the sort of detail a second copy gets subtly wrong.
    """
    result = {"n_messages_since_index": 0, "live_chat_readable": False, "stale": False}
    try:
        engine = _get_engine()  # already points at the live chat.db
        _, newer = engine._cached_freshness(most_recent_chunk_ts)
    except Exception as error:  # noqa: BLE001 - usually Full Disk Access
        result["live_chat_error"] = str(error)
        return result
    result.update(
        {"n_messages_since_index": int(newer), "live_chat_readable": True, "stale": bool(newer)}
    )
    return result


def _app_status() -> Optional[dict]:
    """The running desktop app's own status, if there is one."""
    lock = _running_app_lock()
    if lock is None:
        return None
    try:
        request = Request(
            f"http://127.0.0.1:{lock['port']}/api/status",
            headers={
                "Authorization": f"Bearer {lock['token']}",
                "Host": f"127.0.0.1:{lock['port']}",
            },
        )
        with urlopen(request, timeout=5.0) as response:
            return json.loads(response.read().decode())
    except Exception:  # noqa: BLE001 - fall back to reading the files directly
        return None


@server.tool()
def sync_index(wait: bool = False) -> dict:
    """Bring the search index up to date with messages sent since it was
    last built.

    Call this when `index_status` reports `stale`: anything said after the
    last build is invisible to `search_messages`, which is indistinguishable
    from it never having been said. A sync re-snapshots chat.db and indexes
    only what is new. Pass `wait=true` to block until it finishes (minutes
    for a large backlog); otherwise it returns immediately and progress can
    be polled with `index_status`.
    """
    lock = _running_app_lock()
    if lock is None:
        return {
            "started": False,
            "detail": (
                "Syncing needs the Seaglass app running -- open it and it will "
                "sync, or call this again once it is open."
            ),
        }
    headers = {
        "Authorization": f"Bearer {lock['token']}",
        "Host": f"127.0.0.1:{lock['port']}",
        "Content-Type": "application/json",
    }
    base = f"http://127.0.0.1:{lock['port']}"

    def _call(path, method="GET", timeout=10.0):
        request = Request(base + path, data=b"{}" if method == "POST" else None,
                          headers=headers, method=method)
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())

    try:
        _call("/api/index/build", method="POST")
    except HTTPError as error:
        if error.code == 409:
            return {"started": False, "detail": "A sync is already running."}
        return {"started": False, "detail": str(error)}
    except Exception as error:  # noqa: BLE001
        return {"started": False, "detail": str(error)}

    if not wait:
        return {"started": True, "detail": "Sync started; poll index_status."}

    deadline = time.time() + 1800
    while time.time() < deadline:
        time.sleep(2.0)
        try:
            state = _call("/api/index/build")
        except Exception:  # noqa: BLE001 - a restart mid-build is not a failure
            continue
        if not state.get("running"):
            return {"started": True, "finished": True, "state": state}
    return {"started": True, "finished": False, "detail": "Still running after 30 minutes."}


def main() -> None:
    import asyncio

    asyncio.run(server.run_stdio_async())


if __name__ == "__main__":
    main()
