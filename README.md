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
for a detailed breakdown of every component). Seaglass is the provider for
iMessage retrieval; harnesses such as Grogu load its skill and MCP server
without implementing a separate Messages database fallback.

See `development-plans/` for the design docs driving this build:

- **`PLAN.md`** — the implementation plan (architecture, schema, build order).
- **`DESIGN-NOTES.md`** — why each decision was made, and what was tried and rejected.
- **`EVALUATION.md`** — how retrieval quality, latency, and correctness are measured.
- **`ADDENDUM.md`** — running log of decisions and findings as the build progressed.

There is no shared model daemon. Each MCP process loads semantic models only
when a topical search needs them, then keeps them warm for that client session.
Listing tools, recent-message reads, conversation reads and filters-only searches
do not open the semantic index or load model weights.

## Live recent iMessages over MCP

Use `recent_messages` (projected as `seaglass__recent_messages` by OpenClaw)
for current messages, without building or syncing the index:

```json
{"since_ts": null, "before_ts": null, "limit": 50, "offset": 0, "chat_id": null, "include_text": true}
```

`since_ts` and `before_ts` are inclusive Unix-seconds bounds. Null defaults
mean the last 24 hours through the observation time. `limit` is 1–100 and
`offset` is a nonnegative page offset; `chat_id` optionally narrows to one chat.
Set `include_text=false` for freshness/setup checks: the `text` key is omitted
and attributed bodies are not decoded. Attachment bytes/files are never loaded.

The response contains `source: "live_chat_db"`, `observed_at_ts`,
`newest_source_ts` (global live newest timestamp, independent of filters or
index coverage), resolved `since_ts`/`before_ts`, `limit`, `offset`, `has_more`,
`next_offset` (null on the last page), and `messages`. Each message has
`message_id`, `chat_id`, `ts`, `is_from_me`, contact-resolved `sender`,
`has_attachment`, and, when requested, resolved `text` or null for attachments.
`sender` is the contact display name when available, the raw handle for an
unknown sender, or null for outgoing messages. Contact resolution also applies
to metadata-only reads and cold `get_conversation` calls. Unavailable Contacts
access logs a warning and leaves raw handles rather than guessing names.
Rows are newest first with descending message/chat ID tie-breakers. Reuse
the returned time bounds with `next_offset` to page the same window. Bounds
freeze time, not database membership: late iCloud backfill within that window
can still shift offset pages.

The live source is resolved from `SEAGLASS_CHAT_DB_SOURCE`, then the app's
`chat_db_source`, then `~/Library/Messages/chat.db`. Connections use `mode=ro`
and `query_only`, never `immutable=1`, so committed WAL messages are visible.
No stale snapshot fallback is allowed: missing access, schema changes, SQLite
errors and undecodable attributed bodies produce explicit tool errors.
Grant Full Disk Access to the **actual MCP/Gateway launcher** and recreate its
MCP process after permission/source changes; a successful terminal read does
not prove the Gateway's child has permission.

`search_messages` preserves its session-shaped response but routes empty,
recency-only and filters-only queries directly to this live path before
consulting the desktop app or opening the index/models. Its `offset` remains
a **session** offset, unlike `recent_messages`' message offset. The result
reports `served_from: "live_chat_db"` and the live newest timestamp; index
coverage fields are null because the index was deliberately not inspected.
Live reads load Contacts for sender names, but not search models. `get_conversation`
also reads live, including historical `around_ts` drill-downs. Explicit
topical/semantic searches still use the app/index and its coverage diagnostics;
only those searches need a built index and its normal sync lifecycle.

## Requirements

- macOS (Apple Silicon), Full Disk Access granted to whatever runs this
- Contacts access, for sender names. Where macOS cannot ask for it -- an
  unbundled interpreter run by launchd, such as a headless gateway, gets no
  prompt and no Settings toggle -- seaglass reads the Contacts database
  directly, which the Full Disk Access above already covers.
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

## Copilot capability plugin

Seaglass owns its iMessage-specific agent guidance rather than requiring an AI
harness to bundle macOS behavior. This repository is a Copilot plugin with a
read-only `imessage` skill backed by the Seaglass MCP tools.

Grogu users can add a local Seaglass checkout to their personal capability set:

```bash
grogu capability add /path/to/seaglass
grogu capability list
```

Grogu then loads this repository with `--plugin-dir` on normal launches. The
plugin registers its repository-local MCP entry point automatically; run
`./setup.sh` first so `.venv` and the Seaglass package exist. Seaglass remains
responsible for its own macOS requirements, Full Disk Access, index freshness,
privacy boundary, and skill behavior.

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
    contacts.py           # Contacts resolution (PyObjC, database fallback)
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

Once set up, use `./start.sh` to launch — it skips the dependency install
and preflight that `setup.sh` redoes every time. (There is deliberately no
`./seaglass`: `seaglass/` is the Python package directory, so no file of
that name can sit beside it.)

`./start.sh --install-app` additionally installs `~/Applications/Seaglass.app`,
a thin wrapper around this checkout, so seaglass can be launched from
Spotlight and pinned to the Dock like any other app. **A bundle is a
separate app identity as far as macOS privacy is concerned**, so grant it
Full Disk Access and Contacts (System Settings › Privacy & Security) —
launching from a terminal inherited *Terminal's* grants, and without its
own the app shows raw phone numbers instead of names and can't detect new
messages. The status bar says so when either is missing.

To run it manually instead: install the app extras in the same venv you
already use for seaglass, then point it at where you want the index and
chat.db snapshot to live (they don't need to exist yet):

```bash
source .venv/bin/activate
pip install -e ".[app,dev]"
seaglass-app --index-db /path/to/index.db --chat-db /path/to/chat_snapshot.db --chat-db-source /path/to/chat.db
```

By default this opens a native `pywebview` window. Use `--browser` if you want a browser tab for debugging. On first run, if no index exists yet, the app shows a "Build index now" screen instead of the warmup screen — building runs in the background with live progress, and the app automatically warms up and becomes searchable as soon as it finishes. Once ready, the app shows a real warmup screen while it loads model weights, warms SQLite, and prepares contacts, and its status bar always reports how in-sync the index is, with a "Sync now" button to pick up new messages any time. A Sync button also sits permanently in the status bar, so a re-index can be forced at any point rather than only when new messages happen to be detected. If `chat.db` or Full Disk Access is unavailable, the app explains that hydration is limited and tells you how to fix it. If GitHub Copilot CLI is missing, search still works normally; only optional query assist stays off.
