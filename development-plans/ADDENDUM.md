# Addendum — decisions and findings from first review (2026-08-12)

> Companion to `PLAN.md`, `DESIGN-NOTES.md`, `EVALUATION.md`. This file
> records decisions made *after* the initial documents were written, and one
> new finding that changes a load-bearing assumption in Phase 7. Treat it as
> an amendment, not a replacement — the original docs are unchanged.

## 1. No daemon/shim for v1

`PLAN.md` Phase 6 describes a daemon + Unix-socket + launchd-shim
architecture to share warm models/page-cache across sessions. **Deferred.**

Reasoning: Grogu's existing MCP client bridge (`src/grogu_mcp.py` in the
`grogu` repo) already spawns a configured local stdio MCP server once and
keeps one persistent session alive for the lifetime of a `codemode exec`
run (the same mechanism used for `playwright` today). That gives seaglass a
warm-for-the-duration-of-one-run process for free, with no extra moving
parts. Start with a single long-lived MCP server process, spawned directly
by Grogu's stdio client — no socket, no shim binary, no launchd unit.

**Action:** register seaglass as a `local` entry in
`~/.copilot/mcp-config.json` (`mcpServers`), same shape as other configured
local servers. Measure real cold-start latency once `search_messages` works
end-to-end. Only build the daemon/shim split (Phase 6 as originally written)
if that measured latency is actually a problem in practice — this is a
reversible, additive change, not a foundation the rest of the system depends on.

## 2. All LLM calls route through Copilot CLI (ghcp) — no separate provider

Wherever the plan called for an LLM (golden-set question generation in
`EVALUATION.md` §4.1; the deterministic query parser has no LLM step by
design, so this mainly affects eval tooling): use Copilot CLI itself, not a
separate API key/SDK. This removes a dependency from `eval/generate.py` and
keeps every off-machine call inside the trust boundary the user already
relies on for everyday Copilot CLI use. Functionally this is the same
"deliberate, bounded exception to the local-only boundary" `PLAN.md` §1
already called out for question generation — it just names the mechanism.

## 3. Grogu integration — this is the intended production consumer

The stated goal: **seaglass becomes the way Grogu searches iMessages.**
Concretely, once installed and indexed:

- `search_messages` (and the other Phase 6 tools) become the **preferred**
  search path for Grogu.
- Grogu already has a much cruder search today: `src/grogu_imessage.py`
  (`MacOSIMessageAdapter.search`) does a plain `LIKE '%query%'` scan over
  `message.text`/`handle.id`, gated by the `imessage` skill
  (`.github/skills/imessage/SKILL.md`, opt-in, confirmation-gated for
  sends). That adapter's **search** path is superseded when seaglass is
  installed and its index exists; it remains the **fallback** when seaglass
  isn't present. The **send/draft/confirmation** flow in that file is
  unrelated to search and is untouched by this project.
- Tool descriptions (`PLAN.md` Phase 6, "Tool descriptions are load-bearing")
  should be written with Grogu specifically as the calling model, not a
  generic MCP client — check Grogu's actual tool-calling conventions/limits
  when Phase 6 is implemented, not just MCP spec generalities.

## 4. New finding: this machine is mid-backfill, and it breaks a Phase 7 assumption

**Measured directly against the live `chat.db` (read-only) on 2026-08-12:**

```
total messages:                 330,136
date range:                     spans the full history (oldest to newest)
adjacent-ROWID date inversions: 165,899 of 330,313 adjacent pairs (~50%)
```

i.e. for roughly half of all `(ROWID, ROWID+1)` pairs, the later ROWID has
an **earlier** timestamp. This is Messages-in-iCloud backfilling old
conversations onto a freshly set-up Mac: old messages are being inserted
now, so they get fresh, high `ROWID`s but old `date` values.

**Why this matters:** `PLAN.md` Phase 7 sync design assumes
`frontier := MAX(msg_id)` approximates "processed everything so far," and
that any message with `ROWID > frontier` is new-and-recent enough to belong
in the *tail* chunk of its chat (`first_new_msg.ts - tail.end_ts < GAP_THRESHOLD`
decides open-vs-closed). That assumption silently breaks under active
backfill: a newly-appeared message can have a high ROWID but a timestamp
from years before the chat's current tail, and the existing logic has no
path for it — it would either wrongly merge unrelated old text into the
current tail chunk (huge negative gap, but the code as written only checks
one branch of that comparison) or leave the message effectively un-chunked.

**Required fix to Phase 7 (not just a caveat — a new code path):**

```
for each newly-seen msg_id (ROWID > frontier) in a chat:
    if msg.date falls after that chat's current tail.end_ts (within GAP_THRESHOLD):
        -> genuine tail append (existing logic is correct here)
    else:
        -> historical insert: locate the chunk whose [start_ts, end_ts]
           should contain msg.date (or the gap between two chunks where a
           new chunk belongs), and rebuild/insert there — not at the tail.
           This is a new operation distinct from both "append" and
           "reseal an edited/retracted message" in the original design.
```

This should be folded into `index/sync.py` design before Phase 7 is
implemented. It does not affect Phase 3 (initial bulk build), which chunks
whatever is present in one pass regardless of insertion order — only
*incremental* sync needs the new branch.

## 6. Second finding: `chat_message_join` lags message insertion during backfill

**Measured directly (2026-08-12), same session as the ROWID/date-inversion
finding in §4:**

```
total message rows:              374,333
associated_message_type = 0:     332,291
chat_message_join rows:          125,063   (only ~33% of message rows have a join row)
```

Sampled the unlinked ROWIDs (`message.ROWID NOT IN (SELECT message_id FROM
chat_message_join)`): they span the **entire** date range, from 2020 through
messages as recent as 2026-08-09 (three days before this check) — not
confined to old backfilled history. This means `chat_message_join`
population is currently lagging `message` insertion broadly, not just for
historical backfill.

**Why this matters:** `imessage/source.py`'s `iter_messages()` joins through
`chat_message_join` (correctly — that's the only reliable way to know which
chat a message belongs to). Right now that means **~67% of message rows are
invisible to extraction**, including some from days ago, not just years ago.

**This is not a bug in `source.py`.** A message without a `chat_message_join`
row genuinely isn't attributable to a chat yet, so skipping it is correct
behavior — and it's self-healing: once local sync catches up and the join
row appears, a later extraction run (or Phase 7 incremental sync) picks it
up naturally. No code change needed here.

**It does reinforce the §5 sequencing decision below**, and adds a concrete
number to it: building the real index today would `omit ~67%` of everything
currently in `message`, on top of the corpus itself still growing. Treat
both the total-message-count *and* the `chat_message_join` coverage ratio as
signals to watch for backfill settling (see updated §7).

## 7. Sequencing decision (supersedes old §5 numbering)

Given the backfill is substantial and ongoing, and the user was unavailable
to confirm sequencing directly, the reasonable default (stated here so it's
correctable) is:

- **Hold** Phase 0's volume-estimating probes and the Phase 3 overnight
  full build until backfill has substantially settled — right now those
  numbers are a moving target (330k messages today could plausibly be a
  fraction of the eventual steady state).
- **Proceed now** with Phase 1/2 scaffolding (extraction layer, chunker,
  schema) since that code is structural, not volume-dependent, and with the
  parts of Phase 0's preflight checks that are about *capability* rather
  than *size* (SQLite version/FTS5/extension-loading checks, read-only
  `ATTACH` against the live db, a small `attributedBody` decode sample).
- A simple backfill-settled signal to watch for: total message count and
  the adjacent-ROWID inversion rate both flattening across repeated checks
  a day or two apart, **and** the `chat_message_join` coverage ratio
  (`count(distinct message_id) from chat_message_join` / `count(*) from
  message` restricted to `associated_message_type = 0`) approaching ~100%.
