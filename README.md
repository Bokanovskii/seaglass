# seaglass

Local semantic + keyword search over your personal iMessage history, exposed
as an [MCP](https://modelcontextprotocol.io/) server so an AI agent (in
particular [Grogu](https://github.com/Bokanovskii/grogu)) can search your
message history the way you'd search your own memory of it.

Everything — indexing, storage, retrieval — runs entirely on-device. No
message content, embeddings, or queries are sent anywhere during index
construction or search. See `development-plans/PLAN.md` §1 for the full
privacy boundary and its two consequences.

## Status

Working end-to-end: index build, hybrid retrieval, reranking, session
aggregation, and the MCP server are implemented and tested (see the
[interactive architecture reference](https://bokanovskii.github.io/seaglass/architecture.html)
for a detailed breakdown of every component). Grogu integrates with seaglass over MCP and prefers
it over its legacy SQL `LIKE` fallback whenever seaglass is configured and
reachable.

See `development-plans/` for the design docs driving this build:

- **`PLAN.md`** — the implementation plan (architecture, schema, build order).
- **`DESIGN-NOTES.md`** — why each decision was made, and what was tried and rejected.
- **`EVALUATION.md`** — how retrieval quality, latency, and correctness are measured.
- **`ADDENDUM.md`** — running log of decisions and findings as the build progressed.

There is currently no persistent daemon — each invocation loads the
embedding model and reranker fresh and exits. This was a deliberate initial
choice to measure real cold-start cost before building a daemon/shim; see
the architecture reference's performance section for the measured numbers.

## Requirements

- macOS (Apple Silicon), Full Disk Access granted to whatever runs this
- Python 3.11+ — **use a Homebrew or python.org build**, not the macOS system
  Python (`/usr/bin/python3`), which typically links an older SQLite lacking
  FTS5's `contentless_delete` support. Verified working with Homebrew's
  `python3.14`.

## Quick start

```bash
git clone https://github.com/Bokanovskii/seaglass.git
cd seaglass
./setup.sh
```

This script creates a venv, installs everything, runs a capability
preflight (catches missing Full Disk Access etc. with a clear message),
and launches the desktop app. It does **not** build the search index for
you — indexing reads and embeds your full message history and can take a
while on a large history, so it's a deliberate, user-initiated step: the
app itself shows a "Build index now" screen on first launch with a clear
time-cost warning, and a live progress indicator while it runs. Once
built, the app's status bar always shows how in-sync the index is (e.g.
"N new messages since last build") with a one-click "Sync now" button.
See the comments at the top of `setup.sh` for how to override paths
(e.g. `SEAGLASS_CHAT_DB_SRC`, `SEAGLASS_HOME`).

## Manual setup

If you'd rather run the steps yourself (e.g. for the MCP server only,
without the desktop app):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

You can also build/update the index from the CLI instead of the app's UI
(both call the same idempotent build — safe to re-run any time):

```bash
seaglass build <chat_db_snapshot> <index_db>
```

## Development

```bash
# capability preflight (safe to run anytime; read-only)
python -m seaglass.probe

# unit tests (no live chat.db required)
python -m pytest tests/ -v
```

## Repository layout

```
seaglass/
  probe.py              # capability preflight (+ backfill-progress signal)
  imessage/
    source.py            # the ONLY module aware of Apple's chat.db schema
    attributedbody.py    # typedstream decoding for the attributedBody blob
    contacts.py           # PyObjC Contacts resolution
  index/                 # chunking, embedding, build, sync
  search/                # query parsing, retrieval, reranking
  eval/                  # golden set + evaluation harness
tests/
docs/
  architecture.html       # interactive architecture reference (see Status above)
development-plans/       # design docs (see Status above)
```

## Desktop app

The quickest way to get the desktop app running is `./setup.sh` (see
"Quick start" above) — it handles the venv, install, and launch. Building
the index is a separate, explicit step you trigger from inside the app
(or via the CLI), since it can take a while on a large message history.

To run it manually instead: install the app extras in the same venv you
already use for seaglass, then point it at where you want the index and
chat.db snapshot to live (they don't need to exist yet):

```bash
source .venv/bin/activate
pip install -e ".[app,dev]"
seaglass-app --index-db /path/to/index.db --chat-db /path/to/chat_snapshot.db --chat-db-source /path/to/chat.db
```

By default this opens a native `pywebview` window. Use `--browser` if you want a browser tab for debugging. On first run, if no index exists yet, the app shows a "Build index now" screen instead of the warmup screen — building runs in the background with live progress, and the app automatically warms up and becomes searchable as soon as it finishes. Once ready, the app shows a real warmup screen while it loads model weights, warms SQLite, and prepares contacts, and its status bar always reports how in-sync the index is, with a "Sync now" button to pick up new messages any time. If `chat.db` or Full Disk Access is unavailable, the app explains that hydration is limited and tells you how to fix it. If GitHub Copilot CLI is missing, search still works normally; only optional query assist stays off.

