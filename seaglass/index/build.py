"""`index/build.py` — the Phase 3 orchestrator: chat.db → index.db.

Ties together extraction (imessage/source.py), chunking
(index/chunker.py), the two body renderings (index/render.py), and
embedding (index/embed.py) into a crash-recoverable build of `index.db`.
See development-plans/PLAN.md §6 Phase 3.

**Crash recovery.** Chunks are assigned deterministic, stable ids: a
simple 1-indexed position counter over a fixed global order (ascending
`chat_id`, chronological within each chat). Regenerating from an
unchanged chat.db snapshot always assigns the same id to the same
chunk. Each batch of `BATCH_SIZE` chunks is written in one transaction
covering `chunks`, `chunk_message`, `chunks_vec`, then `chunks_fts` last
(PLAN.md §6 Phase 3 ordering). `meta.build_cursor` is persisted only
after a batch's transaction commits, so a crash mid-batch loses at most
one in-flight, uncommitted batch -- never a partially-written one. On
restart, `build_index` reconciles `build_cursor` against
`MAX(id) FROM chunks` and trusts the data over the recorded cursor
(PLAN.md's "verify the cursor ... and roll back any partial batch"),
then re-walks the same deterministic order, skipping any chunk whose id
is already `<= build_cursor` -- no re-embedding, no re-render, of
already-durable work.

**Snapshot first.** Per PLAN.md §6 Phase 3, `chat_db_path` here should be
a build-time copy, not the live database -- this module doesn't make the
copy itself (that's an operational/CLI concern), it just reads whatever
path it's given read-only.

**Known follow-up, not yet implemented:** `attachment_place` lookups
always pass an empty dict (media placeholders render bare) -- Phase 2's
EXIF/reverse-geocoding module (index/exif.py) hasn't been built yet.
Wiring it in only changes `format_lexical`'s inputs, not this module's
structure.
"""

from __future__ import annotations

import dataclasses
import sqlite3
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import zstandard

from seaglass.imessage import source
from seaglass.index import render
from seaglass.index.chunker import Chunk, chunk_messages
from seaglass.index.embed import EmbeddingModel, compute_calibration_absmax, quantize_int8

BATCH_SIZE = 200
CALIBRATION_SAMPLE_SIZE = 2_000
EMBED_VERSION = "bge-small-en-v1.5-1"
SEMANTIC_FORMAT_VERSION = "1"
LEXICAL_FORMAT_VERSION = "1"

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


@dataclasses.dataclass
class RenderedChunk:
    chunk: Chunk
    id: int
    body_semantic: str
    body_lexical: str


def open_index_db(index_db_path: Path) -> sqlite3.Connection:
    """Open (creating if needed) index.db with the schema applied and
    sqlite-vec loaded. WAL mode is set by schema.sql.
    """
    import sqlite_vec

    con = sqlite3.connect(index_db_path)
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    con.executescript(SCHEMA_PATH.read_text())
    return con


def _get_meta(con: sqlite3.Connection, key: str) -> Optional[str]:
    row = con.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def _set_meta(con: sqlite3.Connection, key: str, value: str) -> None:
    con.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))


def _reconcile_build_cursor(con: sqlite3.Connection) -> int:
    """Trust the data over the recorded cursor: if `chunks` contains rows
    beyond the recorded `build_cursor` (which atomic per-batch commits
    should prevent, but a hand-edited/corrupted index.db could produce),
    advance the cursor to match rather than risk colliding ids.
    """
    recorded = int(_get_meta(con, "build_cursor") or 0)
    max_id_row = con.execute("SELECT MAX(id) FROM chunks").fetchone()
    max_id = max_id_row[0] or 0
    cursor = max(recorded, max_id)
    if cursor != recorded:
        _set_meta(con, "build_cursor", str(cursor))
        con.commit()
    return cursor


def iter_chunks_by_chat(
    chat_con: sqlite3.Connection,
    *,
    chunker_kwargs: Optional[dict] = None,
) -> Iterator[Tuple[Chunk, Dict[int, "source.Message"]]]:
    """Yield `(Chunk, messages_by_id)` across every chat, in a stable,
    deterministic order (ascending `chat_id`, chronological within each
    chat). `messages_by_id` covers only that chat's messages, fetched once
    per chat rather than once per chunk.
    """
    chunker_kwargs = chunker_kwargs or {}
    chat_ids = [
        row[0]
        for row in chat_con.execute(
            "SELECT DISTINCT chat_id FROM im.chat_message_join ORDER BY chat_id"
        )
    ]
    for chat_id in chat_ids:
        messages = list(source.iter_messages(chat_con, chat_id=chat_id))
        if not messages:
            continue
        messages_by_id = {m.rowid: m for m in messages}
        for chunk in chunk_messages(messages, **chunker_kwargs):
            yield chunk, messages_by_id


def _render_chunk(
    chunk_id: int,
    chunk: Chunk,
    messages_by_id: Dict[int, "source.Message"],
    attachments_by_msg: Dict[int, List["source.AttachmentRow"]],
    places_by_attachment: Dict[int, str],
) -> RenderedChunk:
    ordered_messages = [messages_by_id[mid] for mid in chunk.msg_ids]
    return RenderedChunk(
        chunk=chunk,
        id=chunk_id,
        body_semantic=render.format_semantic(ordered_messages),
        body_lexical=render.format_lexical(ordered_messages, attachments_by_msg, places_by_attachment),
    )


def _write_batch(
    index_con: sqlite3.Connection,
    rendered_batch: Sequence[RenderedChunk],
    embeddings_int8,
    compressor: "zstandard.ZstdCompressor",
) -> None:
    """Write one batch inside a single transaction: chunks, chunk_message,
    chunks_vec, then chunks_fts last (PLAN.md §6 Phase 3 ordering).
    """
    index_con.execute("BEGIN")
    try:
        for rendered, vector in zip(rendered_batch, embeddings_int8):
            compressed = compressor.compress(rendered.body_semantic.encode("utf-8"))
            index_con.execute(
                "INSERT INTO chunks (id, chat_id, start_ts, end_ts, has_attachment, body_semantic) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    rendered.id,
                    rendered.chunk.chat_id,
                    rendered.chunk.start_ts,
                    rendered.chunk.end_ts,
                    int(rendered.chunk.has_attachment),
                    compressed,
                ),
            )
            index_con.executemany(
                "INSERT INTO chunk_message (msg_id, chunk_id) VALUES (?, ?)",
                [(msg_id, rendered.id) for msg_id in rendered.chunk.msg_ids],
            )
            index_con.execute(
                "INSERT INTO chunks_vec (rowid, embedding) VALUES (?, vec_int8(?))",
                (rendered.id, vector.tobytes()),
            )
        for rendered in rendered_batch:
            index_con.execute(
                "INSERT INTO chunks_fts (rowid, body) VALUES (?, ?)",
                (rendered.id, rendered.body_lexical),
            )
        last_id = rendered_batch[-1].id
        _set_meta(index_con, "build_cursor", str(last_id))
        index_con.execute("COMMIT")
    except Exception:
        index_con.execute("ROLLBACK")
        raise


def _ensure_calibration(
    index_con: sqlite3.Connection,
    embedding_model: EmbeddingModel,
    sample_texts: Sequence[str],
) -> float:
    """Compute and persist `meta.int8_absmax` once, from a sample of
    rendered `body_semantic` texts, if not already recorded. Every chunk
    embedded in this build (and every query embedded at search time) must
    quantise against this same value (PLAN.md §6 Phase 3).
    """
    existing = _get_meta(index_con, "int8_absmax")
    if existing is not None:
        return float(existing)
    sample = list(sample_texts[:CALIBRATION_SAMPLE_SIZE])
    vectors = embedding_model.embed(sample)
    absmax = compute_calibration_absmax(vectors)
    _set_meta(index_con, "int8_absmax", repr(absmax))
    _set_meta(index_con, "embed_version", EMBED_VERSION)
    _set_meta(index_con, "semantic_format_version", SEMANTIC_FORMAT_VERSION)
    _set_meta(index_con, "lexical_format_version", LEXICAL_FORMAT_VERSION)
    index_con.commit()
    return absmax


def build_index(
    chat_db_path: Path,
    index_db_path: Path,
    *,
    embedding_model: Optional[EmbeddingModel] = None,
    batch_size: int = BATCH_SIZE,
    chunker_kwargs: Optional[dict] = None,
    limit_chunks: Optional[int] = None,
) -> int:
    """Run (or resume) a full build. Returns the number of chunks newly
    written in this invocation (0 if already fully caught up).

    ⚠️ `chat_db_path` should be a build-time snapshot, not the live
    chat.db (PLAN.md §6 Phase 3, "Snapshot first").
    """
    embedding_model = embedding_model or EmbeddingModel()
    chat_con = source.connect_readonly(chat_db_path)
    index_con = open_index_db(index_db_path)
    build_cursor = _reconcile_build_cursor(index_con)
    calibrated = _get_meta(index_con, "int8_absmax") is not None

    compressor = zstandard.ZstdCompressor()
    position = 0  # 1-indexed chunk id, stable across restarts
    chunks_written = 0
    batch: List[RenderedChunk] = []

    for chunk, messages_by_id in iter_chunks_by_chat(chat_con, chunker_kwargs=chunker_kwargs):
        position += 1
        chunk_id = position
        if chunk_id <= build_cursor:
            continue  # already durably written in a prior run

        attachments = source.fetch_attachments_for_messages(chat_con, chunk.msg_ids)
        rendered = _render_chunk(chunk_id, chunk, messages_by_id, attachments, {})

        if not calibrated:
            calibrated = True
            _ensure_calibration(index_con, embedding_model, [rendered.body_semantic])

        batch.append(rendered)

        if len(batch) >= batch_size:
            _flush(batch, embedding_model, index_con, compressor)
            chunks_written += len(batch)
            batch = []

        if limit_chunks is not None and chunks_written + len(batch) >= limit_chunks:
            break

    if batch:
        _flush(batch, embedding_model, index_con, compressor)
        chunks_written += len(batch)

    return chunks_written


def _flush(
    batch: List[RenderedChunk],
    embedding_model: EmbeddingModel,
    index_con: sqlite3.Connection,
    compressor: "zstandard.ZstdCompressor",
) -> None:
    absmax = float(_get_meta(index_con, "int8_absmax"))
    vectors = embedding_model.embed([r.body_semantic for r in batch])
    quantized = quantize_int8(vectors, absmax)
    _write_batch(index_con, batch, quantized, compressor)
