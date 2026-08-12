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

import os
import sqlite3
import sys
import time
import zstandard
from pathlib import Path
from typing import Optional

from mcp.server import MCPServer

from seaglass.imessage.attributedbody import decode_attributed_body
from seaglass.imessage.contacts import ContactIndex, ContactsUnavailableError
from seaglass.imessage.source import apple_to_unix, connect_readonly
from seaglass.index.build import open_index_db
from seaglass.index.embed import EmbeddingModel
from seaglass.search.format import format_search_result
from seaglass.search.hydrate import HydratedMessage, hydrate_sessions, _resolve_sender
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


def _log(msg: str) -> None:
    # Never print() here -- stdout is the MCP transport (see module docstring).
    print(f"[seaglass-mcp] {msg}", file=sys.stderr, flush=True)


def _get_index_con() -> sqlite3.Connection:
    global _index_con
    if _index_con is None:
        index_db = os.environ.get("SEAGLASS_INDEX_DB")
        if not index_db:
            raise RuntimeError(
                "SEAGLASS_INDEX_DB is not set -- point it at a built index.db "
                "(see `seaglass build`)."
            )
        _log(f"opening index.db: {index_db}")
        _index_con = open_index_db(Path(index_db))
    return _index_con


def _get_chat_con() -> Optional[sqlite3.Connection]:
    global _chat_con
    if _chat_con is None:
        chat_db = os.environ.get("SEAGLASS_CHAT_DB")
        if not chat_db:
            return None
        _log(f"opening chat.db snapshot: {chat_db}")
        _chat_con = connect_readonly(Path(chat_db))
    return _chat_con


def _get_embedding_model() -> EmbeddingModel:
    global _embedding_model
    if _embedding_model is None:
        _log("loading embedding model (first call this session -- cold start)")
        t0 = time.time()
        _embedding_model = EmbeddingModel()
        _log(f"embedding model warm in {time.time() - t0:.2f}s")
    return _embedding_model


def _get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        _log("loading reranker (first call this session -- cold start)")
        t0 = time.time()
        _reranker = CrossEncoderReranker()
        _log(f"reranker warm in {time.time() - t0:.2f}s")
    return _reranker


def _get_contact_index() -> Optional[ContactIndex]:
    global _contact_index, _contact_index_attempted
    if not _contact_index_attempted:
        _contact_index_attempted = True
        try:
            _contact_index = ContactIndex.load()
        except ContactsUnavailableError:
            _log("Contacts unavailable -- names will degrade to raw handles")
    return _contact_index


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
    t0 = time.time()
    index_con = _get_index_con()
    chat_con = _get_chat_con()
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

    if chat_con is None:
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

    hydrated = hydrate_sessions(index_con, chat_con, sessions, contact_index=contact_index)
    payload = format_search_result(hydrated, max_sessions=max_sessions, redact=redact)
    payload["elapsed_s"] = round(time.time() - t0, 2)
    return payload


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
    contact_index = _get_contact_index()

    # `date` mixes seconds/nanoseconds depending on macOS version at write
    # time (see apple_to_unix's own warning) -- rather than trying to invert
    # that ambiguity in SQL, pull a bounded window ordered by the raw column
    # (monotonic either way within one chat) and do the actual around_ts
    # distance sort in Python, in already-converted unix seconds.
    window = max(limit * 20, 2000)
    query = """
        SELECT m.ROWID, m.text, m.attributedBody, m.date, m.is_from_me, h.id
        FROM im.message m
        JOIN im.chat_message_join cmj ON cmj.message_id = m.ROWID
        LEFT JOIN im.handle h ON h.ROWID = m.handle_id
        WHERE cmj.chat_id = ?
        ORDER BY m.date DESC
        LIMIT ?
    """
    rows = chat_con.execute(query, (chat_id, window)).fetchall()

    messages = []
    for rowid, text, attributed_body, date, is_from_me, handle in rows:
        if not text and attributed_body:
            text = decode_attributed_body(attributed_body)
        sender = _resolve_sender(is_from_me, handle, contact_index)
        messages.append(
            HydratedMessage(
                message_id=rowid,
                ts=apple_to_unix(date),
                is_from_me=bool(is_from_me),
                sender=sender,
                text=text,
                has_attachment=False,
            )
        )

    if around_ts is not None:
        messages.sort(key=lambda m: abs(m.ts - around_ts))
        messages = messages[:limit]
    else:
        messages = messages[:limit]
    messages.sort(key=lambda m: m.ts)
    return {
        "chat_id": chat_id,
        "n_messages": len(messages),
        "messages": [
            {
                "message_id": m.message_id,
                "ts": m.ts,
                "is_from_me": m.is_from_me,
                "sender": m.sender,
                "text": m.text,
            }
            for m in messages
        ],
    }


@server.tool()
def index_status() -> dict:
    """Report the current index's chunk/vector counts and configured
    paths, so a caller can tell whether the index exists, is stale, or
    is missing hydration support before trusting search_messages.
    """
    status: dict = {
        "index_db": os.environ.get("SEAGLASS_INDEX_DB"),
        "chat_db": os.environ.get("SEAGLASS_CHAT_DB"),
    }
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
            "hydration_available": _get_chat_con() is not None,
            "contacts_available": _get_contact_index() is not None,
        }
    )
    return status


def main() -> None:
    import asyncio

    asyncio.run(server.run_stdio_async())


if __name__ == "__main__":
    main()
