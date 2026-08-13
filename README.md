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
aggregation, and the MCP server are implemented and tested (see
[`docs/architecture.html`](docs/architecture.html) for a detailed,
interactive breakdown of every component — GitHub Pages isn't available on
this private repo's plan, so download/open that file locally rather than
following a live link). Grogu integrates with seaglass over MCP and prefers
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

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
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

Install the app extras in the same venv you already use for seaglass:

```bash
source .venv/bin/activate
pip install -e ".[app,dev]"
```

Build an index first, then launch the desktop app:

```bash
seaglass-app --index-db /path/to/index.db --chat-db /path/to/chat.db
```

By default this opens a native `pywebview` window. Use `--browser` if you want a browser tab for debugging. On first run, the app shows a real warmup screen while it loads model weights, warms SQLite, and prepares contacts. If no index exists yet, you'll get a friendly prompt to run `seaglass build ...` first. If `chat.db` or Full Disk Access is unavailable, the app explains that hydration is limited and tells you how to fix it. If GitHub Copilot CLI is missing, search still works normally; only optional query assist stays off.
