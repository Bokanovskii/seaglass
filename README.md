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

Early implementation. See `development-plans/` for the design docs driving
this build:

- **`PLAN.md`** — the implementation plan (architecture, schema, phased build order).
- **`DESIGN-NOTES.md`** — why each decision was made, and what was tried and rejected.
- **`EVALUATION.md`** — how retrieval quality, latency, and correctness are measured.
- **`ADDENDUM.md`** — decisions and findings from the first review pass, including
  a required change to the Phase 7 sync design driven by this machine's active
  iCloud backfill.

**Current phase:** Phase 1 (extraction layer) in progress. Phase 0's
capability preflight checks pass (`python -m seaglass.probe`); the
volume/sizing spikes are intentionally deferred until iCloud backfill
settles (see `ADDENDUM.md` §7) — this machine is a freshly set up Mac still
catching up on message history, so message counts and other size estimates
would be stale within days.

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
# Phase 0 capability preflight (safe to run anytime; read-only)
python -m seaglass.probe

# unit tests (no live chat.db required)
python -m pytest tests/ -v
```

## Repository layout

Mirrors `development-plans/PLAN.md` §7, adjusted to the actual package name:

```
seaglass/
  probe.py              # Phase 0 capability preflight (+ backfill-progress signal)
  imessage/
    source.py            # the ONLY module aware of Apple's chat.db schema
    attributedbody.py    # typedstream decoding for the attributedBody blob
    contacts.py           # PyObjC Contacts resolution
  index/                 # chunking, embedding, build, sync (Phase 2-3, 7)
  search/                # query parsing, retrieval, reranking (Phase 4)
  eval/                  # golden set + evaluation harness (Phase 3.5, 8)
tests/
development-plans/       # design docs (see Status above)
```
