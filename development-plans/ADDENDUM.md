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

## 8. Phase 0 spike resolved: `sqlite-vec` constrained KNN works

**This was flagged as the single highest-risk architectural bet** ("If
constrained KNN is unsupported, the architecture's headline advantage
collapses" — `PLAN.md` §6 Phase 0). Spiked directly against the installed
`sqlite-vec` 0.1.9:

```
rowid IN (SELECT id FROM candidates) AND embedding MATCH vec_int8(?) AND k = 200
```

**Works correctly** (every returned row was inside the candidate set) and is
**fast**:

| Corpus size | Filter selectivity | Constrained KNN | Unconstrained KNN (k=200) |
|---|---|---|---|
| 100,000 | ~14% (1/7) | 8.5 ms | 33.5 ms |
| 500,000 | 2% | ~26 ms | 159 ms |

**One gotcha worth recording**, since it cost some debugging time: a plain
`bytes` blob produced by `sqlite_vec.serialize_int8(...)` is ambiguous to
this version of the extension and gets misinterpreted as float32
(`"expected int8, but a float32 vector was provided"`) unless it is
explicitly wrapped in SQL: `vec_int8(?)` on both insert and query. `PLAN.md`
§5's example SQL doesn't show this wrapper — add it when Phase 3/4 code is
written, e.g.:

```sql
INSERT INTO chunks_vec(rowid, embedding) VALUES (?, vec_int8(?))
...
WHERE embedding MATCH vec_int8(?) AND k = 200
```

**Conclusion:** the "free exact pre-filtering" thesis in `PLAN.md` §3/§4
holds. Full KNN at the plan's ~1.25M-chunk estimate would extrapolate to
roughly 400ms unconstrained (already over the <1s budget on this stage
alone) — constrained KNN is not just an optimization, it's necessary to hit
the latency target at all. Proceeding with the plan's brute-force + exact
pre-filter design as written, with the `vec_int8()` wrapping noted above.

## 9. Decision: proceed with implementation despite ongoing backfill

Per explicit direction (2026-08-12): don't wait for iCloud backfill to
settle. The ingestion flow (Phase 2/3/7) will be built to tolerate a
partial, moving corpus from the start rather than treating that as a
precondition, and a full re-index is an accepted, cheap escape hatch if
early chunking/embedding runs turn out to need redoing once backfill
settles. The historical-insert sync fix in §4 remains necessary work, not
optional, given this.

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

## 10. Phase 0 risk resolved: `mlx-embeddings` does NOT load the cross-encoder reranker directly — fixed with a custom loader

PLAN.md assumes `mlx-embeddings` handles "embeddings AND cross-encoder
reranking" directly (§4 dependency list, §6 Phase 3/4). This was flagged in
the original review as the most significant unvalidated risk and has now
been spiked against the real, installed `mlx-embeddings==0.1.0`.

**The embedding half holds exactly as planned.** `mlx_embeddings.load("BAAI/bge-small-en-v1.5")`
loads cleanly, produces 384-dim L2-normalisable output, and batches fast
(32 texts embedded in ~0.6s on this machine, effectively free per-message
at ingestion scale).

**The reranker half does not load out of the box.**
`mlx_embeddings.load("cross-encoder/ms-marco-MiniLM-L-12-v2")` raises
`ValueError: Received 201 parameters not in model`. Root cause: the
checkpoint is a HF `BertForSequenceClassification` — weights are namespaced
`bert.*` (encoder/pooler) plus a separate top-level `classifier.{weight,bias}`
(a `Linear(384, 1)`) — while `mlx_embeddings.models.bert.Model` is built only
for plain embedding-style BERT checkpoints (unprefixed encoder/pooler
weights, no classifier head). `mlx-embeddings` 0.1.0 has no generic
sequence-classification support and no explicit reranker/cross-encoder API
(`load()` is the only entry point; nothing in `mlx_embeddings.utils.MODEL_REMAPPING`
covers this).

**Fix, spiked and working:** the encoder architecture underneath is
identical to what `mlx_embeddings.models.bert.Model` already implements.
`seaglass/search/rerank.py`'s `CrossEncoderReranker`:
1. Downloads the checkpoint via `mlx_embeddings.utils.get_model_path` (same
   HF Hub cache as `load()`).
2. Reads `model.safetensors` directly (`safetensors.safe_open`, numpy
   framework), strips the `bert.` prefix from encoder/pooler keys, skips
   the non-parameter `bert.embeddings.position_ids` buffer, and separately
   pulls out `classifier.weight`/`classifier.bias`.
3. Loads the encoder/pooler weights into a plain `mlx_embeddings.models.bert.Model`
   (`strict=True` — this now matches exactly).
4. Applies the classifier manually: `logits = pooler_output @ classifier_w.T + classifier_b`.
   The checkpoint's `config.json` declares
   `"sbert_ce_default_activation_function": "torch.nn.modules.linear.Identity"`,
   so raw logits (not a probability) are the correct score — matches
   `sentence-transformers`' own `CrossEncoder` behavior for this checkpoint.

**Validated end to end:** scored a clearly-relevant vs. clearly-irrelevant
(query, candidate) pair — relevant scored −2.39 vs. irrelevant −10.19,
correctly discriminating. Batched 50 (query, candidate) pairs at ~22ms warm
(first call ~180ms, includes one-time compile/dispatch overhead) — inside
PLAN.md's ~60ms reranker latency budget from its "Reranker sizing" table,
on this machine.

**Conclusion:** PLAN.md's "MLX consolidation" architecture (no PyTorch/MPS
dependency anywhere in the query path) holds, but requires this
project-specific loader rather than a bare `mlx_embeddings.load()` call.
This is now implemented as `seaglass/search/rerank.py::CrossEncoderReranker`
and `seaglass/index/embed.py::EmbeddingModel`, both lazy-loading (no network
access at import time), with the quantisation math (`compute_calibration_absmax`,
`quantize_int8`, matching PLAN.md §6 Phase 3's calibrated-absmax scheme
exactly, including the "not `round(v * 127)`" warning) unit-tested, and the
model-loading paths covered by a `pytest -m integration` smoke test (skipped
by default since it needs network + real inference, but green as of this
writing).

**No PyTorch/transformers-model-loading dependency was added.** `transformers`
is present only as `mlx-embeddings`' own tokenizer dependency (`AutoProcessor`/`AutoTokenizer`
usage), not for model weights.

## 11. Phase 3 implemented: `index/build.py`, and a small real-data finding (U+FFFC leakage)

`index/build.py` orchestrates extraction → chunking → both renderings →
embedding → writing `index.db`, per PLAN.md §6 Phase 3: deterministic
chunk ids (stable position counter over ascending `chat_id` then
chronological order, so re-running against an unchanged snapshot
reproduces identical ids), one transaction per batch covering `chunks` +
`chunk_message` + `chunks_vec` then `chunks_fts` last, `meta.build_cursor`
persisted only on commit, and a `int8_absmax`/`embed_version`/format
versions calibration step run once against the first batch's rendered
text. 22 new unit tests (synthetic chat.db fixture + a deterministic
network-free fake embedding model) cover fresh builds, resume-after-crash
(simulated via `limit_chunks`), and idempotent re-runs producing zero
duplicate/colliding chunk ids.

**Validated against real data.** Snapshotted the live (partially-backfilled)
`chat.db` to `/tmp` per the "snapshot first" rule, ran `build_index` capped
to a small `limit_chunks`, and inspected the decompressed `body_semantic`
output directly. Chunking, overlap, role labelling (A/B/C for a group chat,
Me/Them for DMs), URL-to-domain collapsing, and the bare `[attachment]`
placeholder all render exactly as designed on real messages.

**Small real-data finding, fixed:** Apple embeds `U+FFFC` (OBJECT
REPLACEMENT CHARACTER) inline in `message.text` at the position of an
attachment. Before this fix, that raw marker leaked through alongside our
own `[attachment]` placeholder (e.g. `"[REDACTED_MESSAGE_TEXT]\ufffc
[attachment]"`), doubling up meaningless signal in both renderings.
`index/render.py` now strips `\ufffc` (plus surrounding whitespace) from
`message.text` before either rendering runs. Regression-tested in
`test_render.py`.

**Not yet done:** `index/exif.py` (EXIF GPS + reverse geocoding) doesn't
exist yet, so `build_index` always passes an empty `places_by_attachment`
dict to `format_lexical` -- media placeholders currently render bare
(`[attachment]`) with no place name or filename. Wiring EXIF in only
changes `build.py`'s inputs to `_render_chunk`, not its structure.

## 12. Phase 4a implemented: query parsing, baseline retrieval, and the "earliest useful checkpoint" CLI

**`search/parse.py`** implements deterministic query parsing per PLAN.md
§7: `dateparser.search.search_dates` for date ranges, a fixed keyword list
for media intent ("photo", "video", "attachment", etc.), `rapidfuzz` +
`ContactIndex.handle_ids_for_names` for people-name extraction, and the
untouched remainder of the query text as the semantic residual (fail-open
-- if nothing else parses, the whole query still goes to embedding
search).

**Real `dateparser` bug found and worked around:** `search_dates` misreads
the bare word "we" as "Wed[nesday]" (and plausibly other short common
words similarly collide with weekday/month abbreviations). Fixed with
`_looks_like_a_real_date_match()`, which only trusts a matched substring
if it contains a digit or an unambiguous date-vocabulary word (weekday
names, month names, "today"/"tomorrow"/"ago"/"last"/"next"/"week"/
"month"/"year"/season names/time-of-day words). Worth remembering for
any future dateparser use in this codebase.

**`search/retrieve.py`** implements the Phase 4a baseline exactly per
PLAN.md's 4a/4b split -- pre-filter (date/media/people) -> dense int8 KNN
+ sparse BM25 FTS5, independently -> RRF fusion. No reranker yet (that's
4b, gated on `search/rerank.py`'s custom cross-encoder loader from §10).

**Known, intentional simplification:** PLAN.md distinguishes "from X"
(sender-level filter) from "with X" (chat-membership filter), but
`parse.py`'s people extraction doesn't yet distinguish the two
prepositions. `retrieve.py` currently applies the broader "with"
semantics (current `chat_handle_join` membership) to every extracted
name, uniformly. This is over-permissive, never under-inclusive --
acceptable for now, to be revisited once the Phase 3.5 golden-set eval
can actually measure whether it costs precision.

**Validated against real data, end to end.** Built a real (capped, 800
chunk) index from a fresh snapshot of the live, partially-backfilled
`chat.db`, then ran real free-text queries ("what time are we meeting
tonight", "any plans for dinner", "boat") through the full
parse -> retrieve pipeline. Results were semantically on-target in every
case (e.g. "any plans for dinner" surfaced actual dinner-planning threads
ranked above unrelated chat). End-to-end latency including cold MLX model
load (no daemon, per the user's decision) was ~1.1-1.3s for an 800-chunk
index -- this will need to be re-measured against the full-size index
once backfill settles, but confirms the no-daemon starting point is
viable to *start* with, as intended.

**New: `seaglass/cli.py`**, a throwaway (non-production) CLI with `build`
and `search` subcommands, wired to the existing `seaglass` console-script
entry point in `pyproject.toml`. This is PLAN.md's "earliest useful
checkpoint" -- the system is now genuinely queryable end to end against
real data, without the MCP server (a later phase) existing yet. Usage:
`seaglass build <chat_db_snapshot> <index_db>` then
`seaglass search <index_db> "<query>" [--chat-db <chat_db>] [--show N]`.

**Tests:** `test_parse.py` (15 tests) and `test_retrieve.py` (13 tests,
using a new shared `tests/conftest.py` with a reusable
`FakeEmbeddingModel` and a parameterized multi-chat synthetic chat.db
builder) all pass. Full suite: 94 passed (2 integration tests deselected
by default).

## 13. Phase 4b + Phase 5 implemented: rerank, aggregate, expand, hydrate, format — the full pipeline now runs end to end

**`search/rank.py`** implements PLAN.md §6 Phase 4 steps 5-6: batched
cross-encoder rerank of the fused top-K (via `search/rerank.py`'s custom
loader, §10) down to the top 12, then aggregation into `(chat_id, day)`
sessions summing rerank scores (never RRF scores, which are superseded),
capped at 8 sessions, then ±2 time-adjacent context-chunk expansion per
session (zero ranking weight, display-only).

**Judgment call, not specified in PLAN.md:** "day" for `(chat_id, day)`
grouping uses the local system timezone via `datetime.fromtimestamp`, not
UTC. `chat.db` timestamps carry no stored timezone; local wall-clock day
is the closest match to how a person actually remembers "that day we
talked about X". Recorded here as a deviation-by-necessity, not an
oversight.

**Efficient neighbor expansion, exploiting `build.py`'s determinism:**
because `build_index` assigns chunk ids in `(chat_id ascending,
chronological)` order (§11), same-chat chunks are id-contiguous.
`expand_sessions` still verifies this defensively per session (one
extra `chat_id`-scoped query) rather than doing raw id±1 arithmetic
blindly across chats.

**`search/hydrate.py`** implements step 7: pulls raw messages back via
`chunk_message` (never a bare `message.ROWID` range) for a session's hit
and context chunks separately, resolves `is_from_me` to `None` (client
already knows "me") and other senders through `ContactIndex` with a
fallback to the raw handle string.

**`search/format.py`** implements Phase 5: shapes hydrated sessions into
a JSON-able payload with `message_id` citations (never `message://`
links -- that's Mail.app's scheme), a `confidence`/`n_results` signal
(`"none"`/`"low"`/`"high"` by session count), `max_sessions` truncation,
and an opt-in `redact` flag stripping phone numbers/emails from every
message body and sender field via regex.

**`seaglass/cli.py`'s `search` subcommand now runs the full pipeline** by
default (pre-filter → RRF → rerank → aggregate → expand → hydrate →
format, printed as JSON), with `--no-rerank` to drop back to the bare
Phase 4a baseline for comparison, and `--redact` to exercise the
redaction path.

**Validated against real data, end to end, full pipeline.** Built an
800-chunk real index from a fresh live-`chat.db` snapshot and ran
`seaglass search ... "any plans for dinner" --chat-db ...`. Real contact
names resolved correctly (e.g. "[REDACTED_NAME]", "[REDACTED_NAME]");
sessions grouped sensibly by day within a real group chat; one session's
top hit was literally "[REDACTED_MESSAGE_TEXT]
to join" -- a strong true positive for the query. Cold end-to-end latency
(embedding model + reranker model load, no daemon, per the user's
decision to measure this before building one) was **~4.3s** for an
800-chunk index -- higher than Phase 4a's dense+sparse-only ~1.2s,
confirming the reranker's model-load cost is the dominant new latency
term, not its ~22ms-warm scoring cost (§10). This is the number to watch
as the index grows to full size and to weigh against PLAN.md's
"no-daemon-yet, measure and revisit" decision.

**Small known follow-up, not fixed:** hydrated `HydratedMessage.text`
comes straight from raw `chat.db` `message.text`/`attributedBody` (the
display copy), not through `render.py`'s renderings, so it still contains
the raw `U+FFFC` object-replacement marker Apple embeds at attachment
positions (§11 fixed this only in the *indexed* body_semantic/lexical
text, not in this separate raw-display path). Cosmetic only -- worth
stripping in `format.py` or `hydrate.py` before this ships past the
throwaway CLI, not urgent enough to block on now.

**Tests:** `test_rank.py` (7 tests, incl. a deterministic word-overlap
fake reranker so no MLX load is needed in unit tests), `test_hydrate.py`
(3 tests), `test_format.py` (5 tests) -- all passing. Full suite: 109
passed (2 integration tests deselected by default).

**Not yet done:** `index/exif.py` (EXIF GPS + reverse geocoding) still
doesn't exist -- media placeholders in both the indexed lexical body and
the raw hydrated display remain bare. Phase 3.5 (golden-set generation via
`eval/harvest.py`, using GHCP per the user's decision) has also not been
started; it's the next natural milestone, since real ablation/quality
numbers (recall@50/@12/@final per EVALUATION.md §7) require it and can't
be estimated from spot-checking real queries alone.

## 14. Phase 3.5 begun: `eval/harvest.py` (candidate harvesting)

Implements EVALUATION.md §3's four resumable stages against `index.db`
(+ `chat.db` for `chat_handle_join`/`chat.style`): `sql` (regex signals
over decompressed `body_semantic`, plus chunk/chat/temporal-derived
signals), `idf` (mean IDF of the 5 rarest terms per chunk, via FTS5's
own `fts5vocab` shadow table -- no hand-rolled document-frequency
tracking), `nn` (nearest-neighbour cosine distance, §3.2's "prefilter to
~5000 then one batched NumPy pass over the loaded int8 vectors" rather
than 5000 separate KNN queries, excluding self-matches and any chunk
sharing a message with the source), and `score` (composite priority +
automatic category assignment per §3.3/§3.4).

**Validated against real data.** Ran all four stages against an 1000-
chunk real index built from a live-`chat.db` snapshot; completed in
under a second total (the `nn` stage's 644-candidate batched NumPy pass
was the only nontrivial one, still sub-second at this scale -- will need
re-measuring at full corpus size per §3.2's "budget = 5000 × measured
scan latency" warning). Category distribution looked plausible for this
user's real data: `media_geo` dominated (a photo/video-heavy corpus),
followed by `person_filtered` (mostly group chats), a small
`exact_string` slice (URL-bearing chunks), and ~35% left uncategorized
because they fell outside the `nn` stage's token-count prefilter window
(too short to be interesting eval candidates) -- exactly the intended
exclusion.

**Tests:** `tests/test_harvest.py` (9 tests) using the shared
`tests/conftest.py` fixture builder, all passing. Full suite: 118 passed
(2 integration tests deselected by default).

**Not yet done, and the next natural step:** `eval/generate.py`
(question generation via GHCP, per the user's decision to use it
exclusively for LLM calls in this project) and `eval/review.py` (the
human review CLI) haven't been built yet. A quick spike confirmed
`copilot -p "<prompt>" -s --no-color` is usable non-interactively for
scripting, but costs **~4.4s of fixed per-invocation overhead** even for
a one-word answer -- at ~300 candidate chunks this means batching many
chunks into few prompts (not one call per chunk) is essential to keep
golden-set generation from taking tens of minutes in overhead alone.

## 15. Phase 3.5 complete: `eval/generate.py`, `eval/review.py`, `eval/score.py` — the full golden-set + scoring loop now runs end to end

**`eval/ghcp_client.py`** — the sole LLM interface for this project, per
the user's decision. Wraps `copilot -p "<prompt>" -s --no-color` (no
`--allow-all-tools`; generation-only prompts get no shell/file/network
access). `call_ghcp_json` tolerates conversational wrapper text/code
fences around a JSON array by regex-extracting the array before
parsing, and returns `None` (never raises) on a malformed batch, so one
bad batch can't abort a 300-chunk generation run.

**`eval/generate.py`** implements EVALUATION.md §4: stratified selection
over `eval_candidate` (excludes the bottom nn_distance band per §3.3,
quotas 30/40/30 across the remaining three bands, highest composite
score first within each band), batched question generation (10 chunks
per ghcp call, given the ~4.4s fixed per-call overhead measured in §14),
then the §4.2 automatic filters: vocabulary-leakage auto-reject (>40%
content-word overlap between question and source, stopwords excluded),
and a real-retrieval ambiguity/trivially-hard pass (runs every surviving
question through the actual `retrieve()` pipeline, records whether the
source chunk survived to top-200, and flags -- never auto-rejects --
`sparse_rank1_is_source` per the explicit warning not to condition the
eval set on the sparse-only ablation's outcome). Output:
`candidates_for_review.jsonl`.

**`eval/review.py`** — the minimal human review CLI (§4.3):
accept/edit/reject/multi-positive/skip, driven by any callable
`interactive_input` (real `input()` by default, a scripted fake in
tests). Writes accepted/edited entries to `golden.jsonl` in the exact
§4.4 shape (internal `_*` diagnostic fields stripped), rejects to a
`.rejected.jsonl` sidecar for auditability rather than silent deletion,
and skips already-decided entries on rerun so a review session can be
split across sittings.

**`eval/score.py`** implements §7 scoring plus §6's statistics: runs
every golden entry through the *actual* production pipeline
(parse → pre-filter → RRF → rerank → aggregate → expand), computing
recall@50 (post-fusion), recall@12 (post-rerank), and recall@final
(post-aggregation/expansion -- context-expansion chunks count as hits,
per §7's explicit note that being in the context *is* the thing being
measured), plus the two diagnostic gaps and the filter-kill-rate
diagnostic (§8.3: distinguishes "parser wrongly excluded the positive"
from "retrieval ranked it too low", since they demand opposite fixes).
Reports **Wilson intervals** (not Wald -- Wald degenerates to zero width
at p=1.0, per §6.2) and exposes `mcnemar_chi2` for paired A/B config
comparisons (§6.1), plus recall@final broken out by nn_distance band
(Q2 -- the "crowded neighbourhood" band -- is called out as "the honest
number" per §3.3, since the top band flatters the system on isolated,
distractor-free conversations).

**Validated against real data, the entire loop.** Built a 500-chunk real
index, ran `harvest.py` (340 candidates scored), `generate.py` with
`--target 6` (real `copilot -p` calls, 2 batches of 3, ~9-12s each,
producing 4 genuinely vague, non-leaking recall-style questions about
real past conversations), `review.py` interactively (accept/reject/skip
all worked correctly via piped stdin), and `score.py` against the 2
accepted entries (both scored perfect recall@50/@12/@final -- expected
at n=2 against their own source chat, not a meaningful number, but
confirms the full pipeline plumbing is correct end to end).

**Tests:** `test_generate.py` (7), `test_review.py` (7), `test_score.py`
(10) -- all passing, all network/MLX-free (a scripted fake stdin for
review, a mocked `call_ghcp_json` for generate, the same deterministic
word-overlap `FakeReranker` pattern from `test_rank.py` for score). Full
suite: 142 passed (2 integration tests deselected by default).

**Not yet done:** the golden set itself hasn't actually been built at
scale (§4's ~300-candidate / ~200-accepted target) -- that's a ~30-45
minute human sitting per §4.3's estimate, appropriately left for the
user to actually do rather than simulated. `eval/harvest.py`'s `nn`
stage loads the *entire* `chunks_vec` table into a dense NumPy array for
the batched cosine-distance pass (§3.2); this is fine at the corpus
sizes tested so far but should be re-measured for memory/time headroom
once the real corpus (currently still mid-iCloud-backfill) reaches its
final size. §5 (validating the eval set itself: controls, reviewer
self-consistency, coverage audit) and §8/§9 (the ablation matrix and
parameter sweeps) are still unimplemented -- they're straightforward
compositions of `score.py`'s existing per-config recall numbers plus
the "independently switchable" retrieval-stage config flags PLAN.md's
Phase 4 calls for (not yet built as explicit flags; currently swept by
calling the pipeline functions directly with different arguments).

## 16. Phase 6: `seaglass/mcp_server.py` — the production interface, single-process (daemon deferred)

**The main deviation from PLAN.md §6 Phase 6, and the reason it's worth
calling out on its own**: PLAN.md specifies a daemon (`imsearchd`) behind
a Unix socket plus a tiny per-session shim, so that models/page-cache
stay warm *across* concurrent ghcp sessions. Per the user's earlier
explicit decision ("we probably don't need the daemon/shim to start...
eat the startup latency cost at first and see truly how much latency
that is"), this phase ships as a **single Python process, no daemon, no
shim** -- `seaglass/mcp_server.py` speaks MCP directly over stdio and is
what ghcp spawns.

This gets a meaningful fraction of the daemon's benefit for free and
without the extra moving part: an MCP server process is itself
long-lived for an entire client session, so `EmbeddingModel` and
`CrossEncoderReranker` are loaded lazily on first tool call and then
held warm in module-level globals for every subsequent call in that
session -- the "cold MLX load on every call" failure mode is already
avoided. What's genuinely given up versus the daemon design: warm-model
sharing *across* concurrent ghcp sessions (each spawns its own process
and pays its own model-load cost once), the shared/hot SQLite page
cache, and the daemon's serialised-GPU-work guarantee under concurrent
load. These are precisely the "how much does this actually cost"
questions the user asked to measure empirically before building the
extra infrastructure -- if real usage shows concurrent-session cold
starts are a recurring pain, that's the trigger to revisit the daemon.

**Tools implemented**, matching PLAN.md's table minus `sync_index`
(there is no `index/sync.py` yet -- Phase 7 incremental sync is
unbuilt, so re-indexing via `seaglass build` remains the only update
path, which is exactly what the user already agreed to accept given the
in-progress iCloud backfill):
- `search_messages(query, max_sessions=8, redact=False)` -- runs the
  full pipeline (parse -> pre-filter -> RRF -> rerank -> aggregate ->
  expand -> hydrate -> format). Degrades gracefully to un-hydrated chunk
  previews if `SEAGLASS_CHAT_DB` isn't configured, rather than failing
  the call.
- `get_conversation(chat_id, around_ts?, limit?)` -- follow-up
  drill-down into a specific chat. Deliberately does the `around_ts`
  proximity sort/filter in **Python**, in already-`apple_to_unix`-
  converted unix seconds, rather than trying to invert the seconds-vs-
  nanoseconds ambiguity back into a raw SQL `date` predicate (that
  ambiguity is `apple_to_unix`'s own documented caveat, from Phase 1;
  inverting it in SQL would silently misorder rows once the corpus
  spans both eras).
- `index_status()` -- chunk/vector counts, most recent chunk's
  `end_ts`, and whether hydration/contacts are actually available, so
  a caller (model or human) can sanity-check the index before trusting
  a search.

**Configuration is by environment variable** (`SEAGLASS_INDEX_DB`
required, `SEAGLASS_CHAT_DB` optional) since there's no daemon config
file / launchd plist in this design yet -- these get set in ghcp's own
MCP server config for seaglass, not baked into the code.

**Tool docstrings are the tool descriptions** (the `mcp` SDK's
`@server.tool()` uses the function's docstring as the tool description
sent to the client model) -- written per PLAN.md's explicit warning that
this text is "effectively the system prompt for this capability": states
plainly what it's good at (topical recall), bad at (exhaustive
enumeration, counting), and that `person`/participant matching is not
the same as text-mention matching.

**Validated against real data**: built a 500-chunk real index from a
`chat.db` snapshot, called `index_status`, `search_messages("any plans
for dinner")`, and `get_conversation` on the resulting top hit's
`chat_id` directly through the registered tool functions (not just the
underlying pipeline) -- all three returned correct, real results.
`search_messages` end-to-end latency was ~2.9s cold (model loads
included), consistent with the ~4.3s figure from §13's CLI-level
measurement of the same reranking pipeline.

**Tests**: `tests/test_mcp_server.py` (8 tests) -- synthetic chat.db +
index.db via the same `conftest.py` fixtures used everywhere else,
monkeypatching the module's lazy `_get_*` getters to fakes so no
test needs MLX or network. Covers: successful hydrated search, the
zero-fused-results branch, degrading gracefully without a configured
chat.db, `get_conversation`'s ordering and `around_ts` proximity
behaviour, its required-chat.db error, and both branches of
`index_status`. Full suite: **150 passed** (2 integration tests
deselected by default).

**Package changes**: added a `seaglass-mcp` console-script entry point
(`pyproject.toml`) pointing at `seaglass.mcp_server:main`, alongside the
existing `seaglass` CLI entry point. `mcp` was already a declared
dependency from Phase 0's environment probe.

**Not yet done, left for a deliberate follow-up**: no launchd plist (the
daemon design's lifecycle mechanism) exists because there's no daemon to
manage yet; no version-handshake logic (also a daemon-only concern per
PLAN.md item 4); Grogu's own MCP client config still needs to be pointed
at `seaglass-mcp` with the right environment variables set -- that's an
action in Grogu's repo/config, not this one, and hasn't been done as
part of this session.

## 17. Phase 2 (completed retroactively): `index/exif.py` — GPS extraction + offline reverse-geocoding for `attachment_place`

The last core module PLAN.md's Phase 2 called for was still unbuilt when
Phase 6 wrapped up; built and wired in now.

**Deviation from PLAN.md's suggested `exifread`/`pillow` pairing**:
confirmed against real data that iMessage photo attachments are
overwhelmingly HEIC (a scan of 500 recent real image attachments found
essentially all were `.heic`/`.HEIC`), and neither stock Pillow nor
`exifread` can decode HEIC at all -- `exifread` against a real HEIC file
returned only 7 generic tags and no GPS data. Added **`pillow-heif`**
(not in PLAN.md's dependency table) as a new dependency; it registers a
HEIF opener for Pillow, after which `Image.getexif()` /
`get_ifd(ExifTags.IFD.GPSInfo)` handles HEIC, JPEG, and PNG uniformly
through one code path. `index/exif.py` therefore doesn't call `exifread`
at all, despite it remaining an installed (now effectively unused)
dependency -- harmless, cheap to prune in a future cleanup pass, not
worth a disruptive removal now.

**`extract_places_for_attachments()`** distinguishes three outcomes per
attachment, which matters for `attachment_retry` bookkeeping (PLAN.md's
own distinction): (1) opened fine, geotagged -> reverse-geocoded place
string; (2) opened fine, no GPS tag -> silently absent from the result,
*not* a failure (this is the common case: screenshots, memes, or
Location Services simply off at capture time); (3) couldn't open/decode
at all (missing file, corrupt data) -> recorded as a failure, feeding
`attachment_retry` so a later pass can retry once/if the file reappears
(e.g. after iCloud Photos finishes downloading originals -- directly
relevant given this machine's ongoing iCloud backfill).

**Reverse-geocoding uses a small hand-written ISO alpha-2 -> country-name
map** (~15 common countries) rather than a `pycountry` dependency, since
`reverse_geocoder` only returns the alpha-2 code and a full ISO country
database is disproportionate for this narrow a need; an unmapped code
falls back to the raw code string, still a usable (if less pretty)
lexical signal.

**`index/build.py` wiring** (`_resolve_places`): before rendering each
chunk, looks up already-known `attachment_place` rows for that chunk's
attachments (by primary key), and only calls `exif.py` for attachments
not yet resolved -- persisting new rows (and any `attachment_retry`
rows) immediately, independent of the chunk batch's own transaction.
This is deliberately *not* folded into the same transaction as
`_write_batch`: `attachment_place`/`attachment_retry` writes are
idempotent by primary key (`INSERT OR IGNORE`), so a crash between an
exif-resolution commit and the owning chunk batch's commit just means a
restart re-does one cheap SELECT lookup, never a re-decode of an
already-resolved photo -- the expensive part (HEIC decode + reverse
geocode) is never repeated.

**Validated against real data**: a scan of 500 recent real image
attachments found ~17% carried usable GPS EXIF tags; `reverse_geocoder`
correctly resolved real coordinates to real places (e.g. a Catalina
Island coordinate -> "[REDACTED_LOCATION]"). Ran a full
800-chunk real build with the wiring active (17.0s, comparable per-chunk
overhead to the pre-geo 500-chunk baseline from earlier sessions):
found and correctly geocoded 3 real geotagged attachments in that
window (Coronado CA, Leavenworth WA x2, zero failures), and confirmed
via a direct FTS5 `MATCH` query that "Leavenworth" is indexed in the
correct chunks' lexical bodies -- proving the place name reaches
`format_lexical`'s media placeholder and is retrievable, exactly as
PLAN.md's design intends.

**Tests**: `tests/test_exif.py` (9 tests) -- synthetic JPEGs with
hand-written GPS EXIF tags (encoded via the same DMS + hemisphere-ref
scheme real cameras/phones use), covering DMS conversion, round-tripping
known coordinates, the three-outcome distinction above, and empty input.
`tests/test_build.py` gained a new `TestAttachmentPlaceIntegration`
class (2 tests) confirming the full build-time wiring: a geotagged
synthetic attachment ends up in `attachment_place` *and* is retrievable
via FTS5 `MATCH` on its place text, while a missing attachment file ends
up in `attachment_retry` and never in `attachment_place`.

Full suite: **161 passed** (2 integration tests deselected by default).

**Not yet done**: `exifread` is now dead weight in `pyproject.toml`
(kept rather than pruned mid-session, to avoid an unrelated dependency
removal alongside the exif.py work); the small country-name map will
need occasional manual extension if a code outside its ~15 entries
shows up in a real corpus (falls back gracefully to the raw code, not a
correctness bug, just a cosmetic gap).
