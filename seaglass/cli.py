"""`seaglass` CLI — the "earliest useful checkpoint" throwaway interface
(PLAN.md §6: "After Phase 4a the system is queryable from a ... CLI").
Two subcommands: `build` (run/resume index_build.py against a chat.db
snapshot) and `search` (Phase 4a baseline retrieval: pre-filter -> dense +
sparse -> RRF, no reranker yet).

Not the production interface -- that's the MCP server (a later phase).
This exists to validate the pipeline end to end against real data without
waiting for the MCP server to be built.
"""

from __future__ import annotations

import argparse
import sys
import time
import zstandard
from pathlib import Path

from seaglass.imessage.contacts import ContactIndex, ContactsUnavailableError
from seaglass.imessage.source import connect_readonly
from seaglass.index.build import build_index, open_index_db
from seaglass.index.embed import EmbeddingModel
from seaglass.search.parse import parse_query
from seaglass.search.retrieve import retrieve


def _cmd_build(args: argparse.Namespace) -> int:
    t0 = time.time()
    written = build_index(
        Path(args.chat_db),
        Path(args.index_db),
        batch_size=args.batch_size,
        limit_chunks=args.limit_chunks,
    )
    elapsed = time.time() - t0
    print(f"Wrote {written} chunks in {elapsed:.1f}s ({args.index_db})")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    index_con = open_index_db(Path(args.index_db))
    chat_con = connect_readonly(Path(args.chat_db)) if args.chat_db else None

    contact_index = None
    try:
        contact_index = ContactIndex.load()
    except ContactsUnavailableError:
        pass  # people-name filtering degrades gracefully to "none extracted"

    t0 = time.time()
    parsed = parse_query(args.query, contact_index=contact_index)
    model = EmbeddingModel()
    results = retrieve(index_con, parsed, model, chat_con=chat_con, fused_top_k=args.top_k)
    elapsed = time.time() - t0

    print(f"query: {args.query!r}")
    print(f"parsed: semantic={parsed.semantic!r} date_from={parsed.date_from} "
          f"date_to={parsed.date_to} has_media={parsed.has_media} "
          f"people={parsed.people_participant}")
    print(f"{len(results)} results in {elapsed:.2f}s\n")

    dctx = zstandard.ZstdDecompressor()
    for rank, result in enumerate(results[: args.show], start=1):
        row = index_con.execute(
            "SELECT chat_id, start_ts, body_semantic FROM chunks WHERE id = ?", (result.chunk_id,)
        ).fetchone()
        if row is None:
            continue
        chat_id, start_ts, compressed = row
        text = dctx.decompress(compressed).decode("utf-8")
        preview = text.replace("\n", " ⏎ ")[:200]
        print(f"#{rank} chunk={result.chunk_id} chat={chat_id} rrf={result.rrf_score:.4f}")
        print(f"    {preview}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="seaglass", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="build or resume an index.db from a chat.db snapshot")
    build_parser.add_argument("chat_db", help="path to a chat.db SNAPSHOT (never the live file)")
    build_parser.add_argument("index_db", help="path to index.db (created if missing)")
    build_parser.add_argument("--batch-size", type=int, default=200)
    build_parser.add_argument("--limit-chunks", type=int, default=None)
    build_parser.set_defaults(func=_cmd_build)

    search_parser = subparsers.add_parser("search", help="Phase 4a baseline retrieval (no reranker yet)")
    search_parser.add_argument("index_db", help="path to an existing index.db")
    search_parser.add_argument("query", help="free-text search query")
    search_parser.add_argument("--chat-db", default=None, help="chat.db, for people-filter chat resolution")
    search_parser.add_argument("--top-k", type=int, default=50, help="RRF fused result count")
    search_parser.add_argument("--show", type=int, default=10, help="how many results to print")
    search_parser.set_defaults(func=_cmd_search)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
