-- seaglass index.db schema. See development-plans/PLAN.md §5 for the full
-- rationale behind every column (and, as importantly, every column
-- deliberately absent). Kept in one file so build.py and sync.py share a
-- single source of truth.

PRAGMA journal_mode = WAL;

-- ── owned tables ───────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS chunks (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,  -- rowid coupling: see PLAN.md §5
  chat_id        INTEGER NOT NULL,   -- -> chat.db chat.ROWID
  start_ts       INTEGER NOT NULL,   -- unix seconds; the ONLY date mechanism
  end_ts         INTEGER NOT NULL,
  has_attachment INTEGER,            -- media pre-filter
  body_semantic  BLOB                -- zstd; format_semantic() output
);
CREATE INDEX IF NOT EXISTS idx_chunks_time ON chunks(start_ts);
CREATE INDEX IF NOT EXISTS idx_chunks_chat ON chunks(chat_id, start_ts);

-- THE membership relation -- explicit, exact, and the only record of which
-- messages are in a chunk. Never derive membership from a ROWID range;
-- message.ROWID is global and chronological across ALL chats.
CREATE TABLE IF NOT EXISTS chunk_message (
  msg_id   INTEGER NOT NULL,          -- -> chat.db message.ROWID
  chunk_id INTEGER NOT NULL,
  PRIMARY KEY (msg_id, chunk_id)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_chunkmsg_chunk ON chunk_message(chunk_id);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

-- populated only if Phase 0 shows significant iCloud offloading
CREATE TABLE IF NOT EXISTS attachment_retry (
  attachment_id INTEGER PRIMARY KEY,
  chunk_id      INTEGER
);

-- Per-attachment reverse-geocoded place names. NOT freely rebuildable --
-- needs the original attachment files still on disk. Back up separately.
CREATE TABLE IF NOT EXISTS attachment_place (
  attachment_id INTEGER PRIMARY KEY,  -- -> chat.db attachment.ROWID
  place         TEXT NOT NULL         -- "Lisbon Alfama Lisboa Portugal"
);

-- eval harvesting; see EVALUATION.md §3. Droppable once golden.jsonl exists.
CREATE TABLE IF NOT EXISTS eval_candidate (
  chunk_id            INTEGER PRIMARY KEY,
  nn_distance         REAL,
  nn_chunk_id         INTEGER,
  nn_band             INTEGER,
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
CREATE INDEX IF NOT EXISTS idx_evalcand_score ON eval_candidate(category, score DESC);

-- ── derived indexes (virtual tables) ───────────────────────────

-- CONTENTLESS: no copy of any text lives here, only the inverted index.
-- contentless_delete=1 (SQLite >= 3.43) lets FTS5 delete a row without
-- being handed the original text -- required for rebuild paths.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  body,
  content='', contentless_delete=1,
  tokenize='porter unicode61'
);

-- vec0 defaults to L2; cosine is declared explicitly (see PLAN.md §5 for
-- why cosine specifically, though L2/dot/cosine are ~equivalent here).
-- ⚠️ Always wrap inserted/queried vectors with vec_int8(...) in SQL --
-- a bare blob parameter is ambiguously interpreted as float32 by the
-- installed sqlite-vec build (ADDENDUM.md §8).
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
  embedding int8[384] distance_metric=cosine
);
