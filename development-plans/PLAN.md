# Local Semantic Search over iMessage — Implementation Plan

> **Companion documents in this folder:**
>
> - **`DESIGN-NOTES.md`** — *why* every decision below was made, plus background
>   on the underlying technologies. Read it before deviating from this plan;
>   most "obvious improvements" were already considered and rejected for reasons
>   documented in its §9.
> - **`EVALUATION.md`** — how retrieval quality is measured (**recall@50,
>   recall@12 and recall@final**), and how the golden set is harvested from real
>   content after the build. Phases 0, 3.5, 4 and 8 below depend on it.

---

## 1. Goal

A local semantic + keyword search system over a personal iMessage history,
exposed as an **MCP server**. Retrieval runs entirely on-device; the MCP client's
model reasons over what the tools return.

**Hard requirements:**

- **Indexing, storage and retrieval run entirely on-device.** No message
  content, embeddings, or queries are sent anywhere during index construction or
  search.
- Handles both semantic queries ("what did we decide about the Lisbon trip")
  and structured ones ("messages from Alice in March", "photos from Portugal").
- Does not mutate Apple's system data stores. Read-only throughout; the only
  copy made is a short-lived snapshot for the initial bulk build (Phase 3).
- Runs comfortably on an **M5 MacBook Pro with 16 GB unified memory**.
- **Tool calls block the agent's turn**, so retrieval latency is a UX
  requirement, not a nicety. Target < 1 s.

⚠️ **Privacy boundary — state it plainly, and note where control ends.**
Everything up to and including retrieval is local: the corpus, the index, the
embeddings, the query, and every candidate that does not survive reranking never
leave the machine. **What happens to the tool's return value is the MCP client's
business, not this system's.** A cloud-backed client will send those ~8
conversation sessions upstream. This system is **locally-retrieving**; whether
it is locally-*answering* depends entirely on which client is attached.

Two consequences worth designing around:

- **Retrieval precision is also a privacy control.** Only what the tool returns
  can be forwarded, so a tighter final selection reduces exposure as well as
  token count.
- **Log what each tool returned.** It is the only auditable record of what could
  have left the machine.

**Non-goals (for v1):**

- Multi-user, network-exposed, or cloud deployment of the *index*.
- Non-English message support (corpus is confirmed English-only).
- Image content understanding via a vision model (EXIF metadata only — see Phase 2).
- **Local generation.** No Ollama, no local LLM weights — the MCP client
  supplies the model. See §4.
- Controlling or auditing what the MCP client does with returned data.

---

## 2. Target environment

| | |
|---|---|
| Machine | MacBook Pro, Apple M5, 16 GB unified memory |
| OS | macOS (Apple Silicon) |
| Corpus | ~50 GB total iMessage data — **mostly attachments**; extractable text expected 1–5 GB |
| Language | Python 3.11+ |

> **Note for the implementing agent:** this plan targets macOS APIs (`chat.db`,
> the Contacts framework, MLX). It cannot be executed or tested on Windows or Linux.

---

## 3. Architecture

```
┌─ Apple system stores (read-only; snapshot only for bulk build) ─┐
│  ~/Library/Messages/chat.db          messages, chats, handles,  │
│                                      attachments                │
│  Contacts framework (PyObjC)         name ↔ identifier          │
│  ~/Library/Messages/Attachments/     image files (EXIF)         │
└───────────────────────────┬─────────────────────────────────────┘
                            │  ATTACH ... ?mode=ro  /  framework calls
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│  index.db  — the only store we own                              │
│    chunks             session windows + zstd body_semantic      │
│    chunk_message      msg_id ↔ chunk_id (exact membership)      │
│    attachment_place   EXIF geo, per attachment (NOT rebuildable)│
│    attachment_retry   offloaded files awaiting a retry          │
│    meta               key/value (versions, cursors, sync state) │
│    eval_candidate     harvesting signals (droppable)            │
│    chunks_fts         FTS5 contentless (BM25)                   │
│    chunks_vec         sqlite-vec vec0 (int8[384], cosine)       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
   query → parse → pre-filter → dense+sparse → RRF → hydrate
         → cross-encoder → aggregate → expand → MCP tool result
                                                       │
                                        ═══════════════╪═══════════
                                        privacy boundary — beyond
                                        here is the client's model
```

### Core design principles

1. **Apple's stores are the source of truth.** We store only derived data.
   Anything volatile (display names, chat titles, message text, attachment
   metadata) is resolved live at query time.
2. **Metadata never goes through the embedding.** People, dates and chats are
   `WHERE` clauses. Embeddings handle topics only.
3. **Everything we own is rebuildable — except one thing.** Losing `index.db`
   costs an overnight rebuild, *except* `attachment_place`, which needs the
   original attachment files and may be unrecoverable if they were offloaded to
   iCloud. Back that table up.
4. **Append-only, but convergent.** Chunks are appended, and only the tail chunk
   of a chat can absorb new messages. Chunks whose source messages were edited
   or unsent are rebuilt.

---

## 4. Technology stack

| Layer | Choice | Notes |
|---|---|---|
| Store | **SQLite** (single `index.db` file) | No daemon competing for 16 GB; `chat.db` is already SQLite so cross-DB `ATTACH` joins work |
| Sparse retrieval | **FTS5** (built into SQLite) | Inverted index + `bm25()`; incremental |
| Dense retrieval | **sqlite-vec** (`vec0` virtual table) | Brute-force scan, `int8[384]` |
| Embeddings | **`bge-small-en-v1.5`** (384-d) via **MLX** | 130 MB, English-only, fast on Apple Silicon |
| Fusion | **Reciprocal Rank Fusion**, k = 60 | Parameter-free, scale-invariant |
| Reranking | **`ms-marco-MiniLM-L-12-v2`** (cross-encoder) via **MLX** | 33 M params, ~60 ms batched. English-only corpus; larger rerankers cost 4–30× the compute — see §"Reranker sizing" |
| Query parsing | **Deterministic** — `dateparser` + `rapidfuzz` | No local LLM. See below |
| Generation | **None — the MCP client's model** | Outside this system; see §1 privacy boundary |
| Contacts | **PyObjC** `Contacts` framework | Same API Messages.app uses |
| Geo | `exifread`/`pillow` + `reverse_geocoder` | Fully offline gazetteer |
| Service | **MCP server over stdio** (`mcp` Python SDK) | No HTTP, no port, no auth surface |

### Dependencies

```
sqlite-vec                   # PIN the version — see preflight below
mlx  mlx-embeddings          # embeddings AND cross-encoder reranking
zstandard                    # body_semantic compression
pyobjc-framework-Contacts
phonenumbers                 # E.164 normalisation
rapidfuzz                    # fuzzy contact-name matching
dateparser                   # relative/absolute date extraction from queries
pillow  exifread
reverse_geocoder
typedstream                  # attributedBody decoding
mcp                          # MCP server over stdio
numpy
```

⚠️ **No PyTorch, no `sentence-transformers`, no Ollama, no local LLM weights,
no HTTP server.** `mlx-embeddings` covers both embedding and
sequence-classification reranking; generation belongs to whatever model the MCP
client supplies.

### MLX consolidation — one framework, one memory pool

With generation belonging to the MCP client, **MLX is the only ML runtime in
this system**. That is a meaningful simplification over the earlier design, which
spanned MLX + PyTorch/MPS + Ollama, each with its own import cost, memory pool
and warm-up.

| Stage | Runtime | Notes |
|---|---|---|
| Embed (build + query) | **MLX** | |
| Rerank | **MLX** | was PyTorch+MPS — the slowest stage on the slowest backend |
| Vector scan | CPU NEON SIMD | MLX fallback available, see below |
| BM25 | CPU scalar | inherently not GPU work |
| Query parsing | pure Python | deterministic, no model |
| Generation | the MCP client's model | outside this process |

**Three optimisations, in expected-impact order:**

1. **Batch the reranker.** Scoring 50 candidates is **one batched forward pass**,
   not 50 sequential ones. Sequential inference wastes nearly all GPU
   parallelism; MLX throughput scales close to linearly with batch size, and
   unified memory means no host↔device copies.
2. **`@mx.compile`** the embed and rerank forward passes. MLX evaluates lazily;
   compiling fuses the graph into fewer, larger Metal kernel launches. Build the
   whole batch graph, then a single `mx.eval`.
3. **Right-size the reranker** — see below. This is worth more than quantising
   the wrong model.

### Reranker sizing — a latency budget, not a leaderboard

Reranking is the only compute-bound stage in the query path, and an MCP tool
call blocks the agent's turn. FLOPs ≈ 2 × encoder-params × tokens, over
50 pairs × 512 tokens:

| Model | Params | Est. batched | BEIR nDCG@10 |
|---|---|---|---|
| `ms-marco-MiniLM-L-6-v2` | 22 M | ~30 ms | ~60–62 |
| **`ms-marco-MiniLM-L-12-v2`** ✅ | 33 M | ~60 ms | ~62–64 |
| `bge-reranker-base` | 277 M | ~240 ms | ~66 |
| `bge-reranker-v2-m3` | 568 M | **~860 ms** | ~71.5 |

The strongest rerankers would consume the **entire** sub-second budget on one
stage. MiniLM-L-12 is English-only by construction (30 k BERT vocabulary, not
250 k multilingual), so none of its capacity is spent on languages this corpus
does not contain.

⚠️ **The quality gap is real** — roughly 10 nDCG points between MiniLM-L-6 and
`bge-reranker-v2-m3` on BEIR — but the compute gap is ~30×. Note also that BEIR
measures document retrieval across diverse corpora; this task is conversational
message windows in one personal domain, so the ranking may not transfer.

**Treat model and depth as one joint decision.** Cost is
`model_compute × candidates`, so a fixed ~250 ms budget buys ~400 candidates
with L-6, ~200 with L-12, ~50 with `bge-reranker-base`, or ~14 with
`bge-reranker-v2-m3`. A strong reranker over few candidates may beat a weak one
over many. **Run it as an iso-latency ablation** (`EVALUATION.md` §8.1), not a
leaderboard lookup. Full candidate field in `DESIGN-NOTES.md` §5.

At 33 M params quantisation is unnecessary — fp16 is 66 MB.

**Consequence: co-residency is no longer a problem.** Embedder (66 MB) +
reranker (66 MB) + DB page cache ≈ **~0.4 GB of weights and cache**, against
~10–11 GB available. The serialised load/unload dance an earlier design required
— driven entirely by holding 8 GB of local LLM weights — is gone. Load both
models once at daemon startup and leave them. Full per-process accounting, in
which framework overhead dominates, is in Phase 6.

**Vector scan.** `sqlite-vec` is CPU NEON SIMD and bandwidth-bound, which is
probably fine at 1.25 M vectors. If the Phase 0 spike shows `vec0` cannot accept
a constrained candidate id set, the fallback — an mmapped int8 array scanned via
`mx.matmul` + `topk` — is *also* the GPU path, and now sits inside the only
framework in the system.

### Query parsing without an LLM

The earlier design used Qwen3-4B in constrained-JSON mode to extract filters.
With no local LLM, parsing is deterministic:

- **Dates** — `dateparser` over the query for absolute and relative forms
  ("last March", "in 2023", "before June") → `start_ts` range, padded ±3 days.
- **People** — `rapidfuzz` against the in-memory contact list → `handle_id` set,
  applied only above a confidence threshold (§"Names").
- **Media** — a small keyword set ("photo", "picture", "video", "screenshot")
  → `has_attachment`.
- **Everything else** stays in the semantic residual.

This is not a downgrade so much as an acceleration of a decision the evaluation
was already going to force. The `fuzzy_only` vs `filter_only` ablation
(`EVALUATION.md` §8.2) exists precisely to test whether an LLM parser earns its
keep; the deterministic path is faster, unit-testable, and has no failure mode
where the model invents a filter.

If parsing proves insufficient, the **client model can pass structured arguments
directly** — the `search_messages` signature already accepts `date_from`,
`date_to` and `person`. That is strictly better than calling out to a model from
inside the tool, because the client is already mid-turn and has the query in
context. Measure the deterministic path first.

### Environment preflight — run before anything else

Python's `sqlite3` links whatever SQLite it was **built against**, which on
macOS is frequently older than the system library. Several load-bearing features
are version-gated, and discovering that in Phase 3 is expensive:

```python
assert sqlite3.sqlite_version_info >= (3, 43, 0)   # contentless_delete=1
con.execute("pragma compile_options")               # must include ENABLE_FTS5
con.enable_load_extension(True)                     # often disabled in stock builds
sqlite_vec.load(con)                                # then assert vec_version()
```

Check, and fail loudly with a remediation message, on: SQLite ≥ 3.43, FTS5
present, extension loading permitted, `contentless_delete=1` accepted, the
pinned `sqlite-vec` version, and whether `vec0` supports constrained KNN
(Phase 0 spike). Pin a Python distribution known to satisfy these — a Homebrew
or `python.org` build, not necessarily the system Python.

---

## 5. Schema

```sql
-- ── owned tables ───────────────────────────────────────────────

CREATE TABLE chunks (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,  -- see "rowid coupling"
  chat_id        INTEGER NOT NULL,   -- → chat.db chat.ROWID
  start_ts       INTEGER NOT NULL,   -- unix seconds; the ONLY date mechanism
  end_ts         INTEGER NOT NULL,
  has_attachment INTEGER,            -- media pre-filter; see note
  body_semantic  BLOB                -- zstd; format_semantic() output. The
                                     -- lexical rendering is NOT stored — it is
                                     -- regenerated at write time. See §"Two
                                     -- body renderings".
);
CREATE INDEX idx_chunks_time   ON chunks(start_ts);
CREATE INDEX idx_chunks_chat   ON chunks(chat_id, start_ts);

-- DELIBERATELY ABSENT from this table, each derivable and each previously here:
--   first_msg_id / last_msg_id  → chunk_message. Their presence invited a
--                                 range-join correctness bug; see below.
--   has_from_me                 → useless selectivity: ~95 % of conversational
--                                 chunks contain an outgoing message. The
--                                 useful version is message-level.
--   has_geo                     → EXISTS in attachment_place for the chunk.
--   is_group                    → a property of the CHAT, not the chunk;
--                                 im.chat.style already encodes it.
--   msg_count                   → COUNT over chunk_message; eval_candidate
--                                 materialises it when harvesting needs it.
--   sealed                      → the tail is ORDER BY start_ts DESC LIMIT 1
--                                 per chat, and whether to extend it is just
--                                 the 45-min gap rule applied at the boundary.
--                                 A stored flag can disagree with end_ts.
--   embed_version               → a CONSTANT across the whole index (~37 MB of
--                                 identical strings). Vectors from different
--                                 models are not comparable, so a mixed index
--                                 is broken, not degraded — the column encoded
--                                 a state that must never exist. Lives in meta.
--
-- General rule: constants belong in `meta`, not in every row.
--
-- has_attachment stays because it is a genuine hot-path pre-filter with good
-- selectivity (~10-20 % of chunks) and deriving it means joining
-- im.message_attachment_join per candidate.

-- Why start_ts/end_ts are denormalized here rather than derived from
-- im.message.date via chunk_message (~35 MB total, incl. idx_chunks_time):
--   * Date filtering runs BEFORE the vector scan; its job is to shrink 1.25 M
--     chunks to a candidate set. Deriving it per query means joining 15 M
--     chunk_message rows against the attached chat.db — more expensive than the
--     scan the filter exists to make cheap.
--   * start_ts is also the sort key for (chat_id, day) aggregation, ±2
--     neighbour expansion, tail lookup on sync, and temporal_isolation.
--   * idx_chunks_time is NOT redundant with idx_chunks_chat: that index is
--     sorted chat-first, so by the leftmost-prefix rule it cannot serve a bare
--     `WHERE start_ts BETWEEN ? AND ?`.
-- end_ts is the weakest of the three — it sharpens range-boundary semantics and
-- is what the sync boundary rule compares against.

-- THE membership relation. Explicit, exact, and the only record of which
-- messages are in a chunk.
--
-- There is deliberately NO first_msg_id / last_msg_id on `chunks`. An earlier
-- revision kept them as a "range hint" and that is precisely what produced a
-- silent correctness bug: message.ROWID is assigned GLOBALLY and
-- chronologically across all chats, so `BETWEEN first_msg_id AND last_msg_id`
-- spans unrelated conversations. Removing the columns makes the bug
-- unrepresentable rather than merely discouraged.
--
-- msg_id-first ordering gives the reverse lookup for free, which is what eval
-- scoring (label msg_id → chunk) and reseal (edited msg_id → chunk) need.
CREATE TABLE chunk_message (
  msg_id   INTEGER NOT NULL,          -- → chat.db message.ROWID
  chunk_id INTEGER NOT NULL,
  PRIMARY KEY (msg_id, chunk_id)
) WITHOUT ROWID;
CREATE INDEX idx_chunkmsg_chunk ON chunk_message(chunk_id);

-- ~15 M rows at ~16 B ≈ 240 MB. It replaces two tables that were duplicating
-- data Apple already has:
--   speakers     = chunk_message ⋈ im.message.handle_id
--   participants = im.chat_handle_join ⋈ chunks.chat_id   (already in chat.db)
-- Do not reintroduce either. See DESIGN-NOTES.md §9.
--
-- Sync reads it too:
--   "anything new?"      SELECT MAX(msg_id) FROM chunk_message
--   tail resume point    SELECT MIN(msg_id) FROM chunk_message WHERE chunk_id=?

CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);

-- populated only if Phase 0 shows significant iCloud offloading
CREATE TABLE attachment_retry (
  attachment_id INTEGER PRIMARY KEY,
  chunk_id      INTEGER
);

-- Per-attachment reverse-geocoded place names, from the one-off EXIF scan.
-- Only geotagged attachments get a row (~10-20 %, so ~4 MB).
--
-- Per-ATTACHMENT rather than per-chunk because format_lexical() renders places
-- INSIDE the media placeholder at the position the photo occurs:
--     [photo Lisbon Alfama Portugal beach-sunset.jpg]
-- A chunk-level column could not say which photo was taken where, and a trailing
-- blob would sit far from its context in FTS5 token positions. Ordering comes
-- from im.message_attachment_join → message ROWID → position in the chunk.
--
-- ⚠️ This is the one thing in index.db that is NOT freely rebuildable: it needs
-- the attachment files still on disk. Back it up separately.
CREATE TABLE attachment_place (
  attachment_id INTEGER PRIMARY KEY,  -- → chat.db attachment.ROWID
  place         TEXT NOT NULL         -- "Lisbon Alfama Lisboa Portugal"
);

-- eval harvesting; see EVALUATION.md §3. DROP once golden.jsonl exists.
CREATE TABLE eval_candidate (
  chunk_id            INTEGER PRIMARY KEY,
  nn_distance         REAL,      -- cosine distance to nearest OTHER chunk
  nn_chunk_id         INTEGER,
  nn_band             INTEGER,   -- quartile; see EVALUATION.md §3.3
  idf_mean            REAL,
  entity_count        INTEGER,
  has_url             INTEGER,
  has_number          INTEGER,
  has_long_token      INTEGER,
  token_count         INTEGER,
  has_geo             INTEGER,
  has_attachment      INTEGER,
  is_group            INTEGER,
  participant_count   INTEGER,
  temporal_isolation  REAL,
  msg_count           INTEGER,
  category            TEXT,
  score               REAL
);
CREATE INDEX idx_evalcand_score ON eval_candidate(category, score DESC);

-- ── derived indexes (virtual tables) ───────────────────────────

-- CONTENTLESS: stores only the inverted index, no copy of any text.
-- contentless_delete=1 (SQLite 3.43+) is REQUIRED — it lets FTS5 delete a row
-- WITHOUT being handed the original text, which the rebuild paths need.
--
-- ONE column. `body` here is format_lexical() output, which INCLUDES the
-- EXIF-derived place names — deliberately NOT the same string as
-- chunks.body_semantic (no 512-token cap, role labels stripped, URLs verbatim).
-- See §"Two body renderings".
--
-- Everything expressible as a filter IS a filter: dates → start_ts,
-- media → has_attachment, people → fuzzy contact match.
CREATE VIRTUAL TABLE chunks_fts USING fts5(
  body,
  content='', contentless_delete=1,
  tokenize='porter unicode61'
);

CREATE VIRTUAL TABLE chunks_vec USING vec0(
  embedding int8[384] distance_metric=cosine
);
-- ⚠️ vec0 defaults to L2; declare the metric explicitly.
--
-- On the choice: source vectors are L2-normalised, and calibrated int8
-- quantisation preserves norms to within ~0.2 %. Since ||a-b||² = 2 - 2·cos(a,b)
-- for unit vectors, cosine / L2 / dot product give effectively IDENTICAL
-- rankings here. Cosine is declared because it is invariant to the small thing
-- that does vary. What actually matters is CONSISTENCY: the same metric at
-- index time, at query time, and for nn_distance in eval harvesting.
```

### The rowid coupling — the most dangerous invariant in the design

```
chunks.id  ==  chunks_fts.rowid  ==  chunks_vec.rowid
```

Nothing enforces this. No foreign key, no constraint, no error when it breaks —
and a violation returns **real vectors and real text mapped to the wrong
messages**, which looks like mediocre search rather than a bug.

**It is also unavoidable.** `vec0` KNN and FTS5 `MATCH` both *return rowids*;
that is their only join key. Storing a duplicate `chunk_id` as an auxiliary
column would not help — you would still have to trust the rowid to find the row
carrying it. So the question is not whether to rely on the convention but how to
stop it failing silently.

**Four defences, all cheap:**

1. **`AUTOINCREMENT` on `chunks.id`.** Without it, SQLite **reuses** rowids: the
   next insert takes `max(id)+1`, so deleting the highest-id row frees that id
   for reuse. Our delete paths target the *tail* chunk, which is usually the
   newest and therefore the highest id. A crash between delete and reinsert
   would let a fresh chunk claim an id whose stale FTS or vector row survived.
   `AUTOINCREMENT` makes ids monotonic and never reused, converting a silent
   **collision** into a detectable **orphan**. Costs a `sqlite_sequence` row.
2. **One transaction per chunk write or delete**, covering `chunks`,
   `chunk_message`, `chunks_fts` and `chunks_vec`. Never partial.
3. **Explicit rowids on every insert** — never let SQLite assign one:
   ```python
   cur.execute("INSERT INTO chunks_vec(rowid, embedding) VALUES (?, ?)", (cid, vec))
   cur.execute("INSERT INTO chunks_fts(rowid, body) VALUES (?, ?)",       (cid, lex))
   ```
4. **Two checks, run incrementally during the build, not only at the end:**
   ```sql
   -- structural: no orphans in either direction
   SELECT count(*) FROM chunks c
   LEFT JOIN chunks_vec v ON v.rowid = c.id WHERE v.rowid IS NULL;   -- MUST be 0
   ```
   ```python
   # semantic: sample ids, re-embed the stored body_semantic, compare
   assert cosine(embed(zstd_decompress(row.body_semantic)), stored_vec) > 0.99
   ```
   The structural check catches missing rows; only the semantic one catches
   **misalignment**, where every row exists but is shifted. That is the failure
   mode that would otherwise reach production undetected
   (`EVALUATION.md` §5.1).

### What is text, and what is a filter

The rule: **anything expressible as a structured predicate is a filter, never
indexed text.** Filters are exact, cheap, and cannot go stale.

| Signal | Mechanism | Why not FTS |
|---|---|---|
| People | in-memory fuzzy contact match → `handle_id` filter | exact ids beat fuzzy lexical scores; also keeps names out of the index entirely |
| Dates | `start_ts` range | BM25 cannot express ranges, relative dates, or ordering; "march" fires on every March equally and dilutes `body` scoring |
| Media type | `has_attachment` | `"photo"` would appear in ~10⁵ chunks — near-zero IDF, so BM25 weights it ~0 anyway |
| Filenames | inside the media placeholder, lexical only | `IMG_4821.HEIC` is noise, but a descriptive name is not |
| **Places** | **inside the media placeholder in the lexical body** | **the one genuine exception — see below** |

### Why place names must be text

Consider *"what did we do in lisbon"*. Two different things should match:

- conversations **mentioning** Lisbon → already covered by `body`
- photos **taken** in Lisbon with no textual mention → only EXIF knows this

A structured geo filter would be **wrong**: it would restrict results to
geo-tagged chunks and exclude the textual mentions. What is needed is an **OR
that contributes to ranking** — which is exactly what indexed text provides.

Place data is also immutable (a photo's coordinates never change) and derived from
a one-off EXIF scan, so it carries none of the volatility that disqualified names.

### Two body renderings — `format_semantic()` and `format_lexical()`

The same raw message list is rendered **twice**, by two separate functions with
independent format versions. This is deliberate and part of the design, not an
optimisation to add later.

```
messages[] ─┬─► format_semantic()  ─► chunks.body_semantic (zstd, stored)
            │                         ├─► embedder  (bge-small, 512 tok)
            │                         └─► reranker  (MiniLM-L-12, 512 tok)
            │
attachment_place ─► format_lexical() ─► chunks_fts.body  (indexed, not stored)
```

| | `format_semantic` | `format_lexical` |
|---|---|---|
| **Length cap** | **512 tokens** — hard model limit | **none** |
| Role labels (`Me:` / `Them:` / `A:`) | keep — who committed to what is semantic content | strip — in ~every chunk, so IDF ≈ 0; they only inflate `\|D\|` in BM25's length normalisation |
| URLs | `[link:airbnb.com]` — 90 chars of path tokenise badly and dilute a *pooled* 384-d vector | **verbatim** — this is the `exact_string` category ("that airbnb link Sam sent") |
| Media | bare `[photo]` / `[video]` | **the placeholder carries everything**: place name and descriptive filename, inline at the position the attachment occurs |
| Place names | **excluded** — geography is not what the vector should encode | **inside the media placeholder**, not appended |
| Stemming / cleaning | none — the model wants natural language | none — `porter unicode61` normalises at index time |
| Names, dates, timestamps | never | never (§"Names: metadata vs message text") |

Worked example — the same window rendered twice:

```
format_semantic()                     format_lexical()
─────────────────                     ────────────────
Me: look at this                      look at this
[photo]                               [photo Lisbon Alfama Portugal sunset.jpg]
Them: gorgeous, where is that         gorgeous, where is that
Me: alfama, near the castle           alfama, near the castle
```

**Why places go inside the placeholder rather than appended:**

1. **FTS5 stores term positions.** A trailing metadata blob puts `lisbon` far
   from its conversational context in token space, degrading phrase and `NEAR`
   queries and producing incoherent `snippet()` output.
2. **One chunk can hold several locations** — a travel day, or an old photo sent
   beside a recent one. A flat chunk-level list loses which photo was where.
3. The placeholder becomes the **single carrier** for all attachment-derived
   signal, instead of two parallel mechanisms.

**Why the length cap matters most.** The chunker targets ~400 tokens, but role
labels and long messages push some chunks past 512, where the embedder
**silently truncates**. FTS5 has no such limit. Sharing one string would throw
away lexically-indexable content to satisfy a constraint that only applies to
the neural side.

When `format_semantic` exceeds 512 tokens, drop from the **middle** — the
opening establishes the topic and the closing usually carries the resolution.
Count the occurrences; if it fires on more than a few percent of chunks, the
chunk size target is wrong.

**Why separate versions pay for themselves:**

```sql
INSERT INTO meta VALUES ('semantic_format_version', 'roles-v1');
INSERT INTO meta VALUES ('lexical_format_version',  'raw-v1');
```

Changing the **lexical** rendering costs an FTS rebuild — minutes. Changing the
**semantic** rendering costs a full re-embed — hours. Coupled, every tweak to
either costs the expensive one. Decoupled, you can iterate freely on the cheap
side, which is exactly where the sweepable questions live.

**Storage:** only `body_semantic` is stored (~400 MB zstd), because reranking
reads it on the hot path. The lexical rendering is regenerated from `chat.db` at
write time only — FTS matching runs against the inverted index, never the text,
so it is never needed at query time.

### Convergence under source mutation

`chat.db` is the source of truth, and it **mutates**: iOS supports message
**editing** (same `ROWID`, changed text, `date_edited` set) and **unsending**
(`date_retracted`, or the row disappearing). A purely forward-only sync would
never converge — you would retrieve ghosts of deleted messages and stale wording
of edited ones, with no error.

So "append-only" applies to the *history of chunks*, not to correctness under
source mutation.

```
per sync, for chats with activity since the last run:
  find messages with date_edited / date_retracted, or newly missing ROWIDs
  affected := SELECT chunk_id FROM chunk_message WHERE msg_id IN (...)
  rebuild each: delete + re-create that chunk (row, chunk_message, FTS, vector)
```

`chunk_message` makes this a direct index lookup rather than a scan — the
reverse-lookup ordering exists precisely for this and for eval scoring.

Consequence for the immutability claim: the index is immutable **except** the
open tail chunk of each chat *and* any chunk whose source messages changed. Both
paths delete rows, which is why `contentless_delete=1` is required.

### Deleting from a contentless FTS5 table

```sql
-- CORRECT for content='' with contentless_delete=1 — rowid only:
DELETE FROM chunks_fts WHERE rowid = 42;
-- or equivalently
INSERT INTO chunks_fts(chunks_fts, rowid) VALUES('delete', 42);

-- then reinsert
INSERT INTO chunks_fts(rowid, body) VALUES (42, :lexical_body);
```

⚠️ Do **not** use the external-content form
`INSERT INTO chunks_fts(chunks_fts, rowid, body) VALUES('delete', 42, :old_body)`
— supplying column values is the `content='table'` pattern, which this design
rejected. See `DESIGN-NOTES.md` §8.

Deletes must also remove the matching `chunks_vec` row and `chunk_message` rows.
Wrap the four writes (chunks, chunk_message, fts, vec) in **one transaction** —
a partial delete silently misaligns rowids, producing real vectors mapped to the
wrong messages with no error.

### Names: metadata vs message text

**No contact name is stored as structured metadata anywhere in `index.db`.**
Person matching is a *filter*, not a text match:

```python
parsed = llm_parse(query)                        # may fail or miss the person
if not parsed.people:
    hits = fuzzy_match_contacts(query_tokens)    # in-memory, rapidfuzz
    if hits and hits.confidence >= THRESHOLD:    # else leave UNFILTERED
        parsed.people = hits                     # → structured handle_id filter
```

Strictly better than BM25 name matching: an exact `handle_id` filter rather than
a fuzzy lexical score, catches nicknames and misspellings BM25 would miss, is
deterministic and unit-testable (unlike the LLM parse step it backstops), and
costs microseconds.

⚠️ **Apply the filter only above a confidence threshold, and prefer omitting it
to guessing.** A missed filter costs a little precision; a wrong one excludes the
answer entirely. This is exactly what `filter_kill_rate` measures
(`EVALUATION.md` §8.3). Handles absent from Contacts — one-off numbers,
businesses, short codes — will never resolve, and those queries must degrade to
*unfiltered search*, never to an empty result.

⚠️ **This is NOT a claim that no name appears in `index.db`.**
`chunks_fts.body` is the tokenised text of every message, and people type names
constantly ("thanks Alice", "call Bob"). A contentless FTS5 table still stores
those **term strings** in its inverted index. The precise claim is:

> No *contact roster* name, chat title, or resolved display name is stored as
> metadata. Names typed inside conversations are present as FTS index terms.

The distinction matters twice: for the privacy posture (the index is not
name-free), and functionally — it is why *"what did we say about Sarah"* works
even though Sarah-as-metadata is absent. **Third-party mentions are found
through `body`, not through the person filter.** The query parser must therefore
distinguish *"messages from Alice"* (participant → filter) from *"messages about
Alice"* (mention → leave in the semantic residual); see Phase 4.

**Consequence: a contact or chat rename requires zero index work.** No rebuild
path to write, test, or forget to run.

### Text storage — `body_semantic` IS stored, compressed

`chunks.body_semantic` holds the zstd-compressed output of `format_semantic()`.
**~400 MB** for 1.25 M chunks. This reverses an earlier decision; the reason is
that the pipeline changed underneath it.

**Why it is stored:**

1. **Reranking needs text on the hot path.** The cross-encoder scores
   `(query, document)` pairs over the **top 50** candidates on *every query*.
   Regenerating those 50 from `chat.db` means 50 cross-database joins per query
   — roughly **100 ms of avoidable latency**. One indexed local read instead.
   (When hydration happened once for ~8 final chunks, this argument did not
   exist. It does now.)
2. **Byte-identity is unachievable by versioning alone.** Regeneration is broken
   by source edits, changes to the extract filters, `typedstream` decoder drift,
   and group speaker-index reassignment. A stored body is the text that was
   actually embedded, by definition — no hash, no drift class, no
   `body_sha256` column.
3. **Eval and debugging need it repeatedly**, not "twice a year": harvest
   signals, mutation tests, inspecting what was embedded.
4. **The storage argument was weak.** 400 MB of the strings that *define* the
   vectors, on a personal machine, to remove 100 ms/query and an entire class of
   correctness risk.

**The lexical rendering is deliberately NOT stored.** FTS matching runs against
the inverted index, never the text, so it is needed only at write time —
initial build, tail rebuild, and edit-driven rebuild. Regenerating it from
`chat.db` in those paths is cheap and off the hot path. Storing it would add
~450 MB for nothing.

FTS5 remains **contentless** — orthogonal to all of the above. The inverted
index stores no copy of anything, so `body_semantic` is the single home for
text, and it is compressed, which an FTS shadow table would not have been.

`meta.semantic_format_version` and `meta.lexical_format_version` are recorded
separately, so a lexical change costs an FTS rebuild rather than a re-embed.

`attachment_place` is likewise stored: reverse-geocoded from a one-off EXIF scan
over ~50 GB of attachment files. ⚠️ **This is the one thing not freely
rebuildable** — regenerating it needs the attachment files still on disk, and
Phase 0 may show many have been offloaded to iCloud. Back it up separately, or
accept that geo search degrades after a rebuild.

**Resulting size:**

| Component | Size |
|---|---|
| int8 vectors | 480 MB |
| FTS5 inverted index (contentless) | ~0.75–1.2 GB |
| `chunk_message` (~15 M rows) | ~240 MB |
| `chunks` rows incl. zstd `body_semantic` | ~540 MB |
| `attachment_place` | ~4 MB |
| **`index.db` total** | **~2–2.5 GB** |

All figures are estimates until Phase 3 measures them.

**Deliberately absent:** no `person` table, no `chat_ref` table, no
`chunk_speaker`/`chunk_participant` tables, no `attachment` table, no
`body_sha256`. See `DESIGN-NOTES.md` §"Rejected designs" — each was proposed and
eliminated, several after being tried.

---

## 6. Phases

### Phase 0 — Probe

`probe.py`. Everything downstream rests on unmeasured assumptions. Measure them first.

| Question | Why it matters |
|---|---|
| Total messages; extractable characters of text | Sizes chunk count, embedding time, storage |
| % rows where `message.text` IS NULL but `attributedBody` is not | If high, naive extraction loses most recent history |
| Attachment count; % of files actually present on disk | Decides whether `attachment_retry` is needed |
| % of local images retaining EXIF GPS | Decides whether geo search is viable at all |
| Read-only `ATTACH` against the live `chat.db` succeeds | Validates the no-copy design |
| Distribution of inter-message gaps within conversations | Empirically tunes the 45-minute session threshold |
| **Do `date_edited` / `date_retracted` exist, and how many rows are affected?** | Sizes the reseal problem (§"Convergence") — if edits are rare, sync can check lazily |
| **Does `chat.db` have per-chat message counts that make change detection cheap?** | Determines how reseal scoping is implemented |

**Three spikes, all load-bearing:**

1. **Constrained KNN in `sqlite-vec`.** The whole "brute force gives free exact
   pre-filtering" thesis assumes you can restrict a `vec0` KNN to an arbitrary
   id set. Verify the installed version actually supports it, and **pin that
   version**:
   ```sql
   SELECT rowid, distance FROM chunks_vec
   WHERE rowid IN (SELECT id FROM temp.candidates)
     AND embedding MATCH :qvec AND k = 200;
   ```
   Try the alternatives too — `vec0` partition/metadata columns, `carray`, a
   temp table. **If constrained KNN is unsupported, the architecture's headline
   advantage collapses** into "pull vectors into Python and score there", and
   that needs to be known now, not in Phase 4.
2. **Scan latency, measured.** int8, warm and cold, filtered and unfiltered,
   p50/p95. Every latency figure in these documents is an estimate; replace them
   with real numbers here. This also fixes the `nn_distance` harvest budget
   (§Phase 3.5), which is `5 000 × scan_latency`.
3. **Per-stage latency attribution.** Embed / scan / BM25 / rerank, measured
   separately, **batched vs sequential for the reranker**. The reranker is the
   only compute-bound stage; batching 50 pairs into one MLX forward pass should
   take it from 50–200 ms to 10–30 ms (§4, "MLX consolidation"). Confirm the
   gap is real before optimising anything else — and note that a tool call
   blocks the agent's turn, so this budget is user-visible.

**Exit criteria:** chunk count known to ±20%; go/no-go on photo-geo search;
constrained KNN confirmed or the fallback chosen; one measured latency table
with a batched-reranker number in it.

---

### Phase 1 — Extraction layer

`imessage/source.py` — the **only** module that touches Apple's schema.
Everything downstream consumes clean dataclasses.

- Read-only URI connect and `ATTACH`:
  ```python
  con.execute("ATTACH DATABASE 'file:/Users/<you>/Library/Messages/chat.db?mode=ro' AS im")
  ```
  - Use `mode=ro`. **Never `immutable=1`** — that asserts the file cannot change
    and yields silently corrupt reads on a live database.
  - Keep read transactions **short**; long readers block WAL checkpointing and
    inflate `chat.db-wal`.
  - Requires **Full Disk Access** for the terminal/app.
- **`attributedBody` decoding.** ⚠️ Terminology matters here: the blob is a
  legacy **`typedstream`** (`NSArchiver`) serialisation, *not* an
  `NSKeyedArchiver` keyed archive — reaching for `plistlib` or a keyed-archive
  decoder will fail. Use `typedstream`/`pytypedstream`, fall back to
  `message.text`, and keep a regex fallback for blobs neither handles.
  **Probe representative blobs in Phase 0** rather than assuming one encoding
  across a long history. **Highest-risk component** — validate against messages
  you can visually confirm in Messages.app.
- **Apple epoch conversion.** ⚠️ Units are **not uniform across history**:
  pre-High-Sierra rows store *seconds* since 2001-01-01, newer rows store
  *nanoseconds*. Detect per row by magnitude and centralise one tested function:
  ```python
  def apple_to_unix(v):
      secs = v / 1e9 if v > 1e11 else v      # ns vs s
      return secs + 978307200
  ```
  Verify the threshold against your own oldest messages in Phase 0.
- **Apple epoch conversion:** see the centralised converter above.
- Message → chat via `chat_message_join`; sender via `handle_id` + `is_from_me`.
  ⚠️ **Never identify a chunk's messages by `ROWID` range.** `ROWID` is global
  and chronological across all chats, so a range spans other conversations.
  Membership is the explicit `chunk_message` table, written by the chunker;
  there is no range to intersect.
- Filter out tapbacks (`associated_message_type != 0`), stickers, empty bodies.
  **Pin this filter set** — changing it later alters both body renderings and
  forces a full re-embed.
- Read `date_edited` / `date_retracted` where present, and surface them; Phase 7
  needs them for edit-driven rebuilds.
- **Assert the schema** on open (expected columns exist). Fail loudly if a macOS
  update changed it.

`imessage/contacts.py` — Contacts framework via PyObjC, loaded into memory once:

- `identifier (E.164 / email) → CNContact.identifier`
- `CNContact.identifier → display_name, [all identifiers]`
- E.164 normalisation via `phonenumbers` (fiddlier than it looks — test against
  your own numbers in several formats).
- Fuzzy name lookup via `rapidfuzz`.

> A `CNContact` already owns multiple phone numbers and emails, so **identity
> unification is system-maintained**. This is why no `person` table exists.

**Exit criteria:** dump 200 random messages with resolved names and local
timestamps; spot-check against Messages.app.

---

### Phase 2 — Chunking

`index/chunker.py`

- Group by `chat_id`, order by `date`.
- Start a new chunk when **any** of: gap > 45 min (tune from Phase 0),
  ~400 tokens accumulated, or 40 messages.
- Overlap 2–3 messages between adjacent chunks.
- Never span `chat_id` boundaries.
- Render the message list **twice** (§5, "Two body renderings"):
  - `format_semantic()` → store zstd-compressed in `chunks.body_semantic`;
    hard-capped at 512 tokens, middle-dropped if exceeded (count occurrences)
  - `format_lexical()` → inserted into `chunks_fts.body`, not stored; no length
    cap, role labels stripped, URLs verbatim, place names inline in the media
    placeholder (joined from `attachment_place`)
- Write `attachment_place` rows for geotagged attachments — reverse-geocoded
  city, region and country, e.g. `Lisbon Alfama Lisboa Portugal`. Per
  *attachment*, so `format_lexical()` can place each one inline at the position
  its photo occurs. Filenames, media types and dates are **not** separately
  indexed — the media placeholder carries the descriptive filename, and dates
  are the `start_ts` filter.
- Set `has_attachment`. That is the only boolean on `chunks` — geo is an
  `EXISTS` against `attachment_place`, group-ness comes from `im.chat.style`,
  and direction is not representable at chunk granularity (see §5).
- Emit `chunk_message` rows — one per message in the window. **This is the
  membership record, and the only one.** Do not also store a `ROWID` range.
- Do **not** build speaker or participant tables. Both are derivable:
  - speakers = `chunk_message ⋈ im.message.handle_id`
  - participants = `im.chat_handle_join ⋈ chunks.chat_id` — a table Apple
    already maintains
- No `sealed` flag: the tail is `ORDER BY start_ts DESC LIMIT 1` per chat, and
  whether it is still open is the gap rule applied at the boundary (Phase 7).

`index/exif.py` — invoked **during chunk construction**, not as a separate pass:

- `GPSLatitude`/`GPSLongitude`/`DateTimeOriginal` via `exifread`/`pillow`.
- Offline reverse geocode with `reverse_geocoder` → city, admin1, country.
- Extraction failures → `attachment_retry` (only if Phase 0 justifies it).

**Exit criteria:** chunk-size distribution sane; hand-read 20 chunks and confirm
they are coherent exchanges rather than arbitrary cuts.

---

### Phase 3 — Embed & build

`index/embed.py`, `index/build.py`

**Snapshot first.** Copy `chat.db` (or use SQLite's backup API) to a build-time
file and `ATTACH` *that*, not the live database. A multi-hour full-corpus read
against the live file blocks WAL checkpointing and inflates `chat.db-wal`.
Discard the snapshot after the build; live `ATTACH` is for query and sync only.
This is the one sanctioned exception to the no-copy rule
(`DESIGN-NOTES.md` §9).

- `bge-small-en-v1.5` via **MLX** (`mlx-embeddings`). `@mx.compile` the forward
  pass; batch 64–128.
- **int8 only. No fp32 is ever stored.** L2-normalise, then quantise with a
  **calibrated scale**:
  ```python
  # ONCE, over a sample of ~10k vectors, stored in meta:
  absmax = float(np.percentile(np.abs(sample), 99.9))   # ~0.15–0.25 for BGE

  # per vector, and identically for every QUERY vector:
  q = np.clip(np.round(v * 127.0 / absmax), -127, 127).astype(np.int8)
  ```
  ⚠️ **Do not use `round(v * 127)`.** A unit-norm 384-d vector has components
  around `1/√384 ≈ 0.051` and a maximum near 0.2, so `v * 127` lands in roughly
  [−26, +26] — about 20 % of the int8 range, giving ~8 % relative rounding error
  per component and far more than the ~1 % recall loss int8 should cost. The
  clip never fires. Calibrating against `absmax` uses the full range and cuts
  error ~5×.

  ⚠️ **Queries must use the same `absmax`.** A mismatch between document and
  query scales silently degrades every result. Store it in `meta` beside
  `embed_version` and assert it at query time.

  int8 quantisation done this way costs ~1 % recall, and **fp32 is regenerable
  by re-embedding in 20–40 min** — so keeping a 1.9 GB sidecar to avoid a
  half-hour rerun is a bad trade. If retrieval ever looks suspiciously weak,
  re-embed a sample and compare then. If int8 alone proves insufficient, the
  escape hatch is **binary prefilter + int8 rescore**, which reuses the vectors
  already stored and needs no extra disk.
- Batch 64–128. Run plugged in; the machine will thermally throttle otherwise.
- Eval harvest signals are **not** computed here — `chunks.body_semantic` is
  stored, so they are a post-build job (`EVALUATION.md` §3).
- **Assert rowid alignment incrementally**, not only at the end — a
  misalignment discovered after 8 hours is 8 hours lost. Both checks from §5,
  "The rowid coupling": the structural orphan query, and the semantic
  re-embed-and-compare on a sample.
- FTS5: **bulk-insert every `(rowid, body)` explicitly**, batched inside
  transactions.
  ⚠️ `INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')` **does not work on a
  contentless table** — `rebuild` re-reads text from a content table, and
  `content=''` has none. There is no shortcut; the text must be supplied.
- Record `embed_version`, `int8_absmax`, `semantic_format_version` and
  `lexical_format_version` in `meta`.

**Crash recovery — required, not optional.** This is a multi-hour job on a
laptop that will sleep, thermally throttle, or be closed:

- Write in **idempotent batches** keyed by `chunk_id`; one transaction per batch
  covering `chunks` + `chunk_message` + `chunks_vec`
  (FTS5 comes at the end).
- Persist `meta.build_cursor` after each batch; resume from it.
- On restart, verify the cursor against `MAX(id)` in `chunks` and roll back any
  partial batch.

**Timing — two different numbers, do not conflate them:**

| | Estimate |
|---|---|
| Embed-only throughput | ~500–1500 chunks/s → ~20–40 min for 1.25 M |
| **Full build wall time** | **several hours** — adds `attributedBody` decoding, formatting, tokenisation, EXIF over ~50 GB of attachments, index writes, and throttling |

Plan for an overnight run. The 20–40 min figure is the GPU matmul only and is
not an SLA. At the upper end of the Phase 0 text estimate (~5 GB) chunk count
roughly triples.

**Exit criteria:** orphan check returns 0; ten hand-written queries return
plausible results; a deliberate mid-build kill resumes cleanly from the cursor.

---

### Phase 3.5 — Eval harvesting & golden set

> ⚠️ **Ordering dependency.** Steps 4.2–4.3 of `EVALUATION.md` (the ambiguity
> filter and the review UI showing "top retrieval hits") require a **working
> retriever**. Build the baseline retrieval path from Phase 4 — pre-filter →
> dense + sparse → RRF, no reranker, no expansion — *before* this phase, then
> return to Phase 4 for reranking, aggregation and tuning.
>
> The real order is: **Phase 4a (baseline retriever) → Phase 3.5 (golden set) →
> Phase 4b (rerank, expansion, tuning against real numbers)**. Tuning before the
> golden set exists is tuning blind.

> Runs entirely after indexing. Full detail in `EVALUATION.md` §3–5.

`eval/harvest.py` — standalone and resumable. Every signal derives from
`chunks.body_semantic` (stored) or from SQL, so nothing runs inline during the
build:

```
harvest.py --stage sql     # chunks/participant/temporal signals   ~20 s
harvest.py --stage idf     # one FTS5 statistics query             ~30 s
harvest.py --stage nn      # 5 000 × measured scan latency         see below
harvest.py --stage score   # bands, composite score, category      ~5 s
```

- **`sql`** — `has_attachment` (from `chunks`), `has_geo` as an `EXISTS` against
  `attachment_place`, `is_group` and `participant_count` (from
  `im.chat_handle_join` / `im.chat.style` for the chunk's chat), `msg_count`
  (`COUNT` over `chunk_message`), `temporal_isolation` (`start_ts` window
  function), plus the regex signals
  (`entity_count`, `has_url`, `has_number`, `has_long_token`, `token_count`)
  computed over the decompressed `body_semantic`.
- **`idf`** — `idf_mean` of the 5 rarest terms, from FTS5 statistics.
  Necessarily after FTS5 exists.
- **`nn`** — prefilter to ~5 000 candidates, then for each find its nearest
  *other* chunk, excluding self and any chunk sharing messages (adjacent chunks
  overlap by design). ⚠️ **Budget = 5 000 × the Phase 0 measured scan latency.**
  At 50 ms that is ~4 min; at 300 ms it is ~25 min. If it proves slow, compute
  it as one batched NumPy pass over the mmapped int8 array rather than 5 000
  separate SQL round-trips.
- **`score`** — assign `nn_band` quartiles and the composite score.
  ⚠️ **Do not maximise `nn_distance`** — sample across bands. Selecting only
  isolated chunks biases recall upward by removing the hard cases
  (`EVALUATION.md` §3.3).

Then: `generate.py` (**an LLM writes a question per candidate** — any capable
model; this is an offline batch job, so a round-trip is fine here, unlike the
interactive tool path) → `filter.py` (vocabulary-leakage and ambiguity filters) →
`review.py` (30-minute CLI accept / edit / reject) → `golden.jsonl`, ~200 entries.

⚠️ Question generation sends chunk text off-machine. It touches ~300 sampled
chunks, once. If that is unacceptable, write the Tier-0 questions by hand
instead (§4.3 of `EVALUATION.md`) and accept a smaller set.

**Exit criteria:** ~200 reviewed entries; coverage audit passes
(`EVALUATION.md` §5.3); reviewer self-agreement ≥ 80 % on a 30-entry blind
re-review.

---

### Phase 4 — Retrieval

`search/parse.py` — query → structured filter + semantic residual,
**deterministically, with no model**:

```python
{
  "people_participant": ["alice"],   # rapidfuzz over contacts, above threshold
  "people_mentioned":   [],          # stays in the residual — see below
  "chat":               None,
  "date_from":          "2024-03-01",  # dateparser, padded ±3 days
  "date_to":            "2024-03-31",
  "has_media":          False,       # keyword set: photo/picture/video/screenshot
  "semantic":           "lisbon trip plans",
}
```

- **Dates** via `dateparser` — handles absolute ("March 2024") and relative
  ("last spring", "before June") forms → `start_ts` range.
- **People** via `rapidfuzz` against the in-memory contact index → `handle_id`
  set. Applied **only above a confidence threshold**; prefer no filter to a
  wrong one (§5).
- **Distinguish participation from mention.** *"messages from Alice"* → a
  filter. *"what did we say about Alice"* → leave "Alice" in `semantic`; it is
  found through `body` text (§5). Emitting a filter for a mention excludes every
  conversation that merely discusses the person — a guaranteed zero-result
  query. Heuristic: preposition before the name (`from`, `with`) implies
  participant; `about`, `re`, or a bare mention implies the residual. Ambiguous
  cases go to the residual, which fails open.
- **Fail open:** anything unparsed stays in the semantic residual. The parser
  must never be a single point of failure.
- ⚠️ **Direction filtering is not available at chunk granularity.** "Only the
  messages Alice sent" is a message-level predicate; a chunk is a mixed
  conversational window. There is deliberately no `has_from_me` column — it
  would be true for ~95 % of chunks and answer nothing useful. Treat `from_me`
  as a display-time distinction, not a retrieval filter.

> **Why no LLM here.** With generation belonging to the client, a local LLM
> would exist *solely* to parse queries — 2.5 GB resident for a task
> `dateparser` and `rapidfuzz` do deterministically, testably, and in
> microseconds. And an MCP client can pass `date_from` / `date_to` / `person`
> as structured arguments, so the hard cases have an escape hatch that costs
> nothing. The `fuzzy_only` vs `filter_only` ablation (`EVALUATION.md` §8.2)
> was always going to test whether an LLM parser earns its keep; this starts
> from the answer that ablation was likely to give.

`search/retrieve.py`

1. **Pre-filter** in SQL → candidate chunk ids. Materialise into
   `temp.candidates`:
   - `start_ts` range, `chat_id`, `has_attachment` — direct on `chunks`
   - **"from X"** (sender): `im.message` rows with `handle_id IN (...)` →
     `chunk_message` on `msg_id`
   - **"with X"** (participant): `im.chat_handle_join` → `chat_id` set →
     `idx_chunks_chat`. Note this is *current* chat membership, so someone who
     left a group still matches its older chunks. Acceptable, and historical
     membership is not reliably recorded in `chat.db` anyway.

   ⚠️ **Whether this set can be pushed into the vector scan is the Phase 0
   spike.** If `vec0` cannot accept an arbitrary id set, fall back to: full KNN
   with generous oversampling then post-filter, or scoring candidate vectors
   directly in NumPy. Do not assume the exact-prefilter advantage without
   having verified it.
2. **Dense:** int8 KNN over the candidate set → **top 200**.
   Query prefix: `"Represent this sentence for searching relevant passages: "`
   (BGE is asymmetric — queries only, never documents). Omitting it is a
   measurable quality regression, not a categorical failure; ablate it.
3. **Sparse:** `bm25(chunks_fts)` over the same candidate set → **top 200**.
   One column, so no weights to tune — place names live inside the lexical
   body. (Splitting them back out for separate weighting costs an FTS rebuild,
   minutes, if a sweep ever shows it helps.)
   ⚠️ **FTS5 `bm25()` returns negative values**; best matches sort *ascending*.
   Use rank order, which is what RRF wants anyway.
4. **RRF fuse**, k = 60 → **top 50**.
   ⚠️ Retrieve deep, fuse, *then* truncate. Truncating each arm to 50 before
   fusion discards the long tail RRF depends on: a document at rank 51 in one
   arm gets score 0 rather than a small contribution, which is precisely the
   corroboration signal fusion exists to capture.
5. **Rerank** with `ms-marco-MiniLM-L-12-v2` via **MLX** → **top 12**, reading
   `zstd_decompress(chunks.body_semantic)` for the top 50. One local indexed read
   — this is why `body_semantic` is stored (§5). Decompression of 50 × ~1.6 KB
   is sub-millisecond.
   ⚠️ **One batched forward pass over all 50 pairs**, not 50 sequential calls —
   sequential inference wastes nearly all GPU parallelism. Use `@mx.compile` and
   4-bit group-wise quantised weights (§4, "MLX consolidation"). Target
   **10–30 ms**, not the 50–200 ms a naive sequential implementation costs.
6. **Aggregate, then expand** — in that order, and the order matters:
   - dedup by `chunk_id` first, so a chunk cannot contribute twice
   - group the reranked top 12 by `(chat_id, day)`, summing **reranker scores**
     (not RRF — the ranks were superseded at step 5)
   - **then** pull ±2 time-adjacent chunks per surviving session as *context
     only*: they contribute **zero** to ranking and are never counted as hits
   - cap at ~8 sessions for the context window
7. **Hydrate + enrich the final ~8 sessions only** — raw messages for display
   and citation, resolved display names, attachments, lazy EXIF:
   ```sql
   SELECT m.ROWID, m.text, m.attributedBody, m.date, m.is_from_me, m.handle_id
   FROM chunk_message cm
   JOIN im.message m ON m.ROWID = cm.msg_id
   WHERE cm.chunk_id = :chunk_id
   ORDER BY m.date;
   ```
   ⚠️ Join through **`chunk_message`**, never a bare `ROWID` range —
   `message.ROWID` is global and chronological across all chats, so a range
   spans other conversations.

**Evaluate at the real boundary.** `recall@12` is measured after step 6, but
step 7 can still drop the positive. Report **`recall@final`** over the ~8
sessions actually sent to the model (`EVALUATION.md` §7).

**Exit criteria:** retrieval under 1 s end-to-end (no local generation to wait on), measured; a
baseline config runnable before Phase 3.5 needs it.

**Every alteration in this phase must be independently switchable** — `raw`,
`fuzzy_only`, `filter_only`, `residual` — because query processing is the
component most likely to *hurt*, and `EVALUATION.md` §8.2 ablates it. Build the
config flags in from the start rather than retrofitting them.

---

### Phase 5 — Result formatting

`search/format.py`. There is no generation step to build — the MCP client's
model reasons over what the tool returns. This phase decides **what the tool
returns and how it is shaped**, which is the last thing under our control.

- **Payload:** the ~8 hydrated sessions from Phase 4 step 7 — resolved contact
  names, dates, message ids, attachment references. Structured, not prose: the
  consuming model does better with clean fields than with a pre-written summary.
- **Citations are ids, not links.** Return `message_id` on every message and let
  the client render. ⚠️ Do **not** emit `message://` URLs — that is Mail.app's
  scheme and will not open an iMessage. If deep-linking is wanted, verify a
  mechanism separately (AppleScript, or `imessage://` addressed by handle, which
  opens the *conversation*, not a specific message).
- **Return a `confidence` / `n_results` signal** so the client can tell thin
  retrieval from rich retrieval and decline to speculate. The tool cannot force
  that, but it can make it easy.
- **Budget the payload.** ~8 sessions × ~400 tokens ≈ 3–4 k tokens per call,
  inside one agent turn. Expose `max_sessions` so the client can trade recall
  against context pressure.
- **Optional redaction flag** to strip phone numbers and email addresses. Cheap,
  and meaningful given the payload may be forwarded upstream (§1).
- **Log every returned payload.** It is the only auditable record of what could
  have left the machine.

⚠️ Generation latency and quality are outside this system's control and outside
its measurements. `EVALUATION.md` scopes to retrieval only — already true, now
structurally enforced.

---

### Phase 6 — MCP server

**Architecture: one long-lived daemon, many thin shims.**

```
ghcp session A ──► imsearch-mcp (shim) ──┐
ghcp session B ──► imsearch-mcp (shim) ──┼──► unix socket ──► imsearchd
ghcp session C ──► imsearch-mcp (shim) ──┘                    (models + index)
                        │                                          │
                   stdio/MCP                              loaded once, warm
```

`ghcp` spawns the **shim** — a small executable that speaks MCP on stdio and
forwards to a daemon over a Unix domain socket. The daemon holds the models,
the database handles, and the page cache. Every session shares one copy.

**Why this is worth the extra moving part:**

| | one process per session | shared daemon |
|---|---|---|
| 3 sessions, idle | ~2.3 GB | **~0.9 GB** |
| Model load | once per session | once, ever |
| First-call latency | seconds (MLX init) | none — already warm |
| SQLite page cache | cold per process | **shared and hot** |
| `mx.compile` graph cache | rebuilt per process | shared |
| Sync ownership | ambiguous, needs coordination | exactly one writer |

The warm page cache matters more than it looks: the vector scan reads ~480 MB,
and having that resident across sessions rather than faulted in per process is
a direct latency win on the metric that is user-visible.

**Tools** (identical surface either way — the shim is transparent):

| Tool | Purpose |
|---|---|
| `search_messages(query, max_sessions?, date_from?, date_to?, person?)` | The main tool. Returns ranked, hydrated sessions |
| `get_conversation(chat_id, around_ts?, limit?)` | Follow-up drill-down when the model wants more of a thread it already saw |
| `sync_index()` | Incremental update; returns counts and timings |
| `index_status()` | Chunk/vector counts, last sync, schema and model versions |

**Tool descriptions are load-bearing.** The client model chooses tools and
arguments from the description text alone — it is effectively the system prompt
for this capability. State plainly what `search_messages` is good at (topical
recall over conversation history), what it is bad at (exhaustive enumeration,
counting), and that `person` matches *participants*, not mentions.

#### Daemon concerns the single-process design did not have

1. **Serialise GPU work.** Concurrent sessions will call `search_messages`
   simultaneously. MLX work must go through one lock or queue — parallel Metal
   submissions from multiple request handlers contend and can be slower than
   serialising. SQLite *reads* are safely concurrent under WAL; only the
   embed/rerank path needs the lock.
2. **Do not let `sync_index` block searches.** It is the one long operation.
   Run it on a separate worker with its own connection, and have `search`
   proceed against the pre-sync state rather than waiting.
3. **Lifecycle.** A **launchd user agent** is the macOS-correct owner:
   `KeepAlive` for restarts, `RunAtLoad` so it is warm before the first session.
   If the shim instead auto-spawns on a missing socket, guard the race — two
   shims starting at once must not both spawn a daemon (atomic bind or lockfile).
4. **Version handshake.** After a code change, an old daemon may still be
   running. The shim sends its build id on connect; on mismatch the daemon
   should refuse and exit so launchd restarts it. Silent version skew is the
   classic daemon failure.
5. **Fail gracefully.** If the socket is missing or the daemon is down, the shim
   returns a clear MCP error, not a hang. A tool call blocks the agent's turn.
6. **Socket permissions.** The daemon holds Full Disk Access and can read every
   message, permanently, behind a socket. Put it at
   `~/Library/Application Support/imsearch/imsearchd.sock` with mode `0600`.
   A Unix socket is the right transport precisely because filesystem permissions
   are the access control — no port, no network surface, no auth to implement.
7. **The shim should be tiny.** A Go or Rust static binary is a few MB and
   starts instantly; a Python shim drags ~40 MB and interpreter startup into
   every session for no benefit. The daemon is where Python belongs.

⚠️ **Never write to stdout in the shim** — stdio is the MCP transport, and a
stray `print()` corrupts the protocol stream. Daemon logs go to a file; shim
logs go to stderr.

#### Memory

Framework overhead dominates, which is exactly why sharing it pays:

| | |
|---|---|
| Python interpreter + stdlib | 40 MB |
| **MLX import + Metal device/context** | **~300 MB** |
| numpy / dateparser / rapidfuzz / mcp | 80 MB |
| embedder `bge-small` fp16 | 66 MB |
| reranker `MiniLM-L-12` fp16 | 66 MB |
| contacts in memory | 5 MB |
| SQLite page cache (`PRAGMA cache_size`) | 256 MB |
| **daemon idle RSS** | **~810 MB** |
| + rerank activations (50 × 512) | ~120 MB |
| **daemon peak during a search** | **~930 MB** |
| each shim | ~2–5 MB (Go/Rust) or ~40 MB (Python) |

**~0.9 GB total regardless of session count**, against ~10–11 GB available.
`cache_size` can be raised now that the cost is paid once — a larger shared page
cache is the cheapest remaining latency win.
- ⚠️ **stdio is the transport — never write to stdout.** Logs go to stderr or a
  file; a stray `print()` corrupts the protocol stream. This is the single most
  common way MCP servers fail confusingly.
- **A tool call blocks the agent's turn.** Enforce a hard timeout and return
  partial results rather than hanging.
- No web UI. If a human-facing surface is wanted later, add a thin CLI over the
  same `search/` modules — but the MCP client *is* the interface for v1.

---

### Phase 7 — Incremental sync

`index/sync.py`. Two distinct jobs — **appending new messages** and
**converging on mutated ones**. Only the first is forward-only.

**1. Append.**

⚠️ **A global frontier is not a resume point.** `MAX(msg_id)` answers "is there
anything new?", but if the tail chunk is still open it must be rebuilt from
*its own first message*, which is below that maximum. Reading only
`ROWID > frontier` rebuilds the tail without the messages that belong in it,
destroying the overlap and preventing the tail from ever accreting.

```
frontier := SELECT MAX(msg_id) FROM chunk_message     -- cheap: leading PK column

for each chat with messages having ROWID > frontier:
    tail := SELECT id, end_ts FROM chunks
            WHERE chat_id = ? ORDER BY start_ts DESC LIMIT 1   -- idx_chunks_chat

    if first_new_msg.ts - tail.end_ts < GAP_THRESHOLD:
        -- the tail is still open: it must absorb the new messages
        resume_from := SELECT MIN(msg_id) FROM chunk_message
                       WHERE chunk_id = tail.id                -- idx_chunkmsg_chunk
        delete tail (chunks, chunk_message, fts, vec — one transaction)
    else:
        -- the conversation had closed; start a fresh chunk
        resume_from := first_new_msg.ROWID

    re-chunk that chat from resume_from forward
```

No `sealed` flag: whether the tail is open is just the 45-minute gap rule
applied at the boundary, which is the same rule the chunker uses everywhere
else. A stored flag would be a second source of truth that can disagree with
`end_ts`.

Chats with no new messages are untouched.

**2. Converge** (see §"Convergence under source mutation"). For chats with recent
activity, find messages carrying `date_edited` / `date_retracted` or newly
missing `ROWID`s, map them straight to chunks via `chunk_message`, and rebuild
those chunks — including old ones far behind the frontier. This is how edits and
unsends are handled; without it the index never converges on its own source of
truth.

**Other:**

- FTS5 and vec0 inserts are **incremental** — never rebuild (and `'rebuild'` is
  not available on a contentless table anyway).
- Compact occasionally with the correct syntax — the parameter goes in the
  hidden `rank` column:
  ```sql
  INSERT INTO chunks_fts(chunks_fts, rank) VALUES('merge', 500);
  ```
- Retry any `attachment_retry` rows (files that arrived from iCloud since) and
  insert into `attachment_place` and rebuild the owning chunk's FTS row.
- Record `meta.last_sync_ok` and `meta.last_sync_error`.
- Trigger: `launchd` agent hourly, or on demand via the `sync_index` MCP tool.

---

### Phase 8 — Evaluation harness

> Full detail in **`EVALUATION.md`**. Summary below.
> The golden set itself is built in **Phase 3.5**; this phase is the harness
> that consumes it.

**Scope: the search stack only. Three recall points — recall@50, recall@12,
recall@final.** No answer grading, no LLM-as-judge, no citation checking.
Generation quality is explicitly out of scope for v1.

- **recall@50** — positive present in the fused, pre-rerank candidate set.
  **The ceiling on the entire system:** if the answer is not here, nothing
  downstream can recover it.
- **recall@12** — positive survives reranking. The **gap** to recall@50
  isolates reranker damage.
- **recall@final** — positive survives aggregation, expansion and the ~8-session
  cap, i.e. it is actually in the LLM's context. Without this the last
  selection stage is unmeasured, and it can silently drop the answer.

⚠️ Label by **message id**, never chunk id — chunk boundaries move whenever
chunking parameters are swept, which would invalidate the very labels needed to
run those experiments.

**Validate the eval before trusting it** (`EVALUATION.md` §5). A broken
evaluation returns plausible numbers: an off-by-one in the message-id range check
yields a confident 0.83 that means nothing. Four controls:

| Control | Expected |
|---|---|
| Scoring unit tests on a 20-chunk fixture | exact known values |
| Positive control (query = chunk's own text) | source in top 5 |
| Oracle / ceiling (grep the positive's literal text) | recall = 1.00 |
| **Mutation test** (shuffle `vec0` rowids, disable dense, truncate chunks) | **recall must drop** |

**The mutation test is the most important.** If you sabotage retrieval and the
score does not move, the evaluation is measuring nothing.

**Statistics** (`EVALUATION.md` §6). At n ≈ 200, recall 0.86 has a 95 % Wilson
interval of ±0.05 — a 2 % change is noise. Per-category at n ≈ 20 it is ±0.19,
so those columns are **directional only**. All A/B comparisons run on identical
inputs, so analyse them **paired** (McNemar); unpaired analysis of the int8
go/no-go would need thousands of questions to resolve what ~200 paired ones can.

**Ablations run on two crossed axes** (`EVALUATION.md` §8):

- **Retrieval:** dense-only → sparse-only → hybrid → +rerank → +expansion. If
  sparse-only lands within ~3 % of hybrid, the embeddings
  are contributing nothing — suspect a missing BGE query prefix.
- **Query alteration:** `raw` → `fuzzy_only` → `filter_only` → `residual`
  (→ `+expansion`). This axis is the one most likely to reveal a *regression*:
  an over-eager parser that extracts the wrong date range yields recall = 0, and
  no retrieval tuning recovers it. It also answers two open questions — whether
  filtering helps recall (fewer distractors) rather than only latency, and
  whether stripping filter terms before embedding helps or hurts.

**Filter kill rate** — fraction of queries where the pre-filter *excluded* the
positive — is a required diagnostic counter. It separates parser errors from
retrieval errors, which recall@50 conflates and which demand opposite fixes.
Target < 3 %; above that, make the parser more conservative.

**Release gate:** no category below 0.65 recall@50. A uniformly mediocre system
is usable; one with a silently broken category is not, because the user cannot
tell which answers to trust.

---

## 7. Repository layout

```
imsearch/
  probe.py
  imessage/
    source.py            # ONLY module aware of Apple's schema
    contacts.py          # PyObjC Contacts resolution
    attributedbody.py    # NSKeyedArchiver decoding
  index/
    schema.sql
    chunker.py
    render.py            # format_semantic() + format_lexical()
    exif.py
    embed.py
    build.py
    sync.py
  search/
    parse.py             # query → structured filter + semantic residual
    retrieve.py          # pre-filter → hybrid → RRF
    rerank.py            # cross-encoder (MLX, batched)
    format.py            # tool result shaping + optional redaction
  eval/
    harvest.py           # sql / idf / nn / score stages — post-build, resumable
    generate.py          # question generation from candidates
    filter.py            # vocabulary-leakage + ambiguity filters
    review.py            # CLI accept / edit / reject
    golden.jsonl         # ~200 entries, labelled by MESSAGE id
    run_eval.py          # recall@50, recall@12, recall@final
    ablate.py            # config matrix
    sweep.py             # parameter grid
    controls.py          # unit / positive / oracle / mutation tests
    stats.py             # Wilson intervals, McNemar, paired bootstrap
    report.py
    results.db           # every run, forever
    fixtures/            # 20-chunk corpus for scoring unit tests
  mcp_server.py          # MCP tools over stdio — the only entry point
```

---

## 8. Build order

⚠️ **Phase numbers are not the build order.** Two dependencies force a
reordering around retrieval:

```
Phase 0   probe + three spikes
Phase 1   extraction
Phase 2   chunking + EXIF
Phase 3   embed + build            ← overnight run
Phase 4a  baseline retriever       ← pre-filter → dense+sparse → RRF only
Phase 8   eval harness + controls  ← fixtures, mutation test
Phase 3.5 harvest + golden set
Phase 4b  rerank, aggregation, tuning
Phase 5   result formatting
Phase 6   MCP server
Phase 7   incremental sync + convergence
```

**Why retrieval splits in two:**

- The golden set's ambiguity filter and review UI (Phase 3.5) need a **working
  retriever**, so a baseline must exist first.
- The harness must exist **before** the golden set, or you cannot distinguish
  "bad question" from "bad scoring" during review.
- Tuning before the golden set exists is tuning blind.

See `EVALUATION.md` §12 for the staged gates within that sequence.

**Earliest useful checkpoint.** After Phase 4a the system is queryable from a
throwaway CLI — enough to judge whether chunking and retrieval are producing
coherent results, which is the question that decides whether the rest is worth
building. Wiring `search_messages` into the MCP server early is also reasonable:
the tool is thin, and using the thing from an agent is a better quality signal
than reading CLI output.

**Phase 0 carries the most risk per unit of work.** `attributedBody` decoding,
the `vec0` constrained-KNN spike, and the latency baseline all land there, and
any of the three can change the design.

---

## 9. Risks

| Risk | Early signal | Mitigation |
|---|---|---|
| **`vec0` cannot constrain KNN to a candidate id set** | **Phase 0 spike** | Full KNN + oversampled post-filter, or score candidates in NumPy. This invalidates the "free exact pre-filtering" thesis, so test it first |
| `attributedBody` decoding fails | Phase 0 shows high NULL-`text` share | Budget extra time; `typedstream` + regex fallback |
| Legacy rows use seconds not nanoseconds | Phase 0 date sanity check | Magnitude-based converter |
| EXIF GPS stripped in iMessage transport | Phase 0 | Drop geo search, or add a VLM captioning pass |
| Attachments offloaded to iCloud | Phase 0 | `attachment_retry` list; retry on sync |
| SQLite too old / extensions blocked | Preflight (§4) | Pin a Python distribution that satisfies the checks |
| 16 GB memory pressure | Phase 6 RSS measurement | Largely resolved: no local LLM weights, and one shared daemon (~0.8 GB) rather than one process per session |
| Tool call stalls the agent turn | Phase 6 p95 latency | Hard timeout with partial results; models loaded at startup, not on first call |
| Client model misuses the tools | Manual observation of tool traces | Tool descriptions are the only steering available — state capabilities and limits explicitly |
| stdout pollution corrupts MCP stream | Client reports protocol errors | Route all logging to stderr/file; assert no stdout writes in CI |
| int8 quantisation hurts recall | Phase 8 vs a re-embedded sample | Binary prefilter + int8 rescore; no extra disk |
| Edits / unsends silently desync the index | Retrieving text that no longer exists | `chunk_message` reverse lookup + reseal (Phase 7) |
| Multi-hour build interrupted | Laptop sleep, thermal, disk full | `meta.build_cursor`, idempotent batches |
| macOS update changes `chat.db` schema | Sync fails | Schema assertions in `source.py`; fail loudly, never silently |
| Brute-force scan too slow as corpus grows | Phase 4 latency | int8 → binary prefilter, or mmap + MLX GPU scan; HNSW only as a last resort (breaks exact pre-filtering) |
| Reranker dominates query latency | Phase 0 per-stage attribution | Batch the 50 pairs into one forward pass; `@mx.compile`; 4-bit quantise; reduce rerank depth |

---

## 10. Operations

Small, but absent from earlier drafts and each capable of costing a rebuild.

**Backup.** `index.db` is *mostly* rebuildable — except `attachment_place`, which
needs the original attachment files (§5). Back up `index.db` itself, or at
minimum export `attachment_place`. Everything else rebuilds from `chat.db`.

**Observability.** Persist in `meta`: `last_sync_ok`, `last_sync_error`,
`build_cursor`, `embed_version`, `int8_absmax`, `semantic_format_version`,
`lexical_format_version`, `schema_probe_version`, chunk and vector counts. Log
query traces (parsed filter, candidate-set size, per-stage latency) — the parsed
filter in particular is what makes a `filter_kill_rate` regression diagnosable.

⚠️ **Logs must not go to stdout.** stdio is the MCP transport; a stray `print()`
corrupts the protocol stream. Use stderr or a file.

**Payload audit log.** Record what each `search_messages` call returned —
message ids and a hash of the payload, not necessarily the full text. It is the
only record of what the client *could* have forwarded upstream (§1), and the
only way to answer "what did this thing expose?" after the fact.

**Schema-drift testing.** `source.py` asserts `chat.db`'s shape on open. Keep
small anonymised fixture databases from each macOS version encountered and run
the extractor against them in CI, so an OS upgrade surfaces as a failing test
rather than a silent extraction gap.

**Multi-Mac — per-device index by design.** With Messages in iCloud each Mac's
`chat.db` holds the full message history, so this is not a coverage problem. The
actual blocker to sharing one `index.db` is that **`message.ROWID` is assigned
per device** — the same message has different ids on different Macs, so
`chunk_message` would not transfer. `message.guid` *is* stable across devices,
so keying on guid would make the index portable at the cost of 36-byte keys
throughout.

Not worth it: a rebuild on a second Mac is one overnight run, fully automatic,
and needs no coordination protocol. **Rebuilding is the sync mechanism.** The
one genuine gap is `attachment_place`, since attachment availability differs per
machine — copy that column across if geo search matters on both.

**Locked or mid-write database.** Retry with backoff on `SQLITE_BUSY`; handle a
missing `-shm`/`-wal` (can happen if Messages.app has not run since boot) by
retrying or prompting to launch Messages.

---

## 11. Open decisions

1. **Group-chat chunking** — current plan keeps one chunk per conversational
   burst regardless of participant count. Splitting into speaker sub-threads was
   considered and rejected: group dynamics *are* the content.
2. **Coarse summary layer** — an optional second index of LLM-written
   `(chat_id, day)` summaries, embedded separately and fused at query time.
   Improves whole-conversation questions. Adds ~30–60 k LLM calls (one overnight
   run). **Add only if Phase 8 shows recall gaps on "what did we decide" queries.**
3. **Attachment content understanding** — a VLM captioning pass would make photo
   *content* searchable rather than just location. Expensive; deferred, and a
   hard requirement only if EXIF GPS turns out to be stripped.
4. **Divergence between the two body renderings** — `format_semantic` and
   `format_lexical` start with role-label stripping and URL handling as their
   only differences (plus the 512-token cap, which is not optional). Further
   divergences are cheap to try on the lexical side (FTS rebuild, minutes) and
   expensive on the semantic side (re-embed, hours). Sweep them
   (`EVALUATION.md` §9).
5. **Person-filter aggressiveness** — hard SQL exclusion vs a soft scoring
   boost. Hard filtering is faster and exact, but fails closed when contact
   resolution is wrong. `filter_kill_rate` (`EVALUATION.md` §8.3) is the
   deciding measurement.
