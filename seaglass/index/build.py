"""`index/build.py` — the Phase 3 orchestrator: chat.db → index.db.

Ties together extraction (imessage/source.py), chunking
(index/chunker.py), the two body renderings (index/render.py), and
embedding (index/embed.py) into a crash-recoverable build of `index.db`.
See development-plans/PLAN.md §6 Phase 3.

**Crash recovery & incremental (Phase 7) sync.** Ids are per-chat-content
identity, not a raw position: for each chat, `iter_pending_chunks`
re-derives that chat's *current* chunk list from chat.db and compares it,
chunk-by-chunk, against whatever is already committed in `chunks` for
that `chat_id` (ordered by `id`, which increases monotonically with
`start_ts` within a chat since message ROWIDs are globally chronological
-- see schema.sql). Three outcomes per position:

* identical `msg_ids` to what's already committed -> skip entirely, no
  re-render, no re-embed of already-durable work.
* the existing chunk at that position is the *tail* (last committed
  chunk for that chat) and its `msg_ids` differ -> this is exactly the
  case new messages arriving in an existing conversation produce: the
  chunker merged them into what was previously the last chunk rather
  than starting a new one. Rewritten in place, reusing the SAME `id`
  (delete + reinsert across `chunks`, `chunk_message`, `chunks_vec`,
  `chunks_fts`, atomically in the same batch transaction).
* no existing chunk at that position -> brand new chunk, assigned the
  next available id (`MAX(id) FROM chunks` + 1, monotonically
  increasing within the run).

This replaces an earlier, buggier design that assigned chunk ids from a
single global sequential position counter and skipped any id
`<= build_cursor` unconditionally -- that scheme silently dropped new
messages whenever they landed in an already-chunked position (the
common case for any chat that isn't the very last one touched) instead
of appending as a new trailing chunk. `meta.build_cursor` is still kept
as an informational high-water mark (`MAX(id) FROM chunks`), but is no
longer used to decide what to skip.

Each batch of `BATCH_SIZE` pending chunks is written in one transaction
covering `chunks`, `chunk_message`, `chunks_vec`, then `chunks_fts` last
(PLAN.md §6 Phase 3 ordering) -- rewrites additionally delete the old
rows for that id from all four tables inside the same transaction, so a
crash mid-batch loses at most one in-flight batch, never leaves a chunk
half-deleted/half-rewritten.

**Snapshot first.** Per PLAN.md §6 Phase 3, `chat_db_path` here should be
a build-time copy, not the live database -- this module doesn't make the
copy itself (that's an operational/CLI concern), it just reads whatever
path it's given read-only.

**`attachment_place` (Phase 2 EXIF/geo) is wired in.** Before rendering
each chunk, `_resolve_places` looks up already-known places for that
chunk's attachments, and for any attachment not yet in
`attachment_place`, calls `index/exif.py` to extract GPS + reverse-geocode
it, persisting new rows immediately (idempotent by `attachment_id`
primary key, so a crash/restart mid-build just re-does the cheap SELECT
lookup, never re-decodes an already-resolved photo). Attachments whose
file can't be opened/decoded at all are recorded in `attachment_retry`
for a later retry pass, distinct from "opened fine, just not geotagged".
"""

from __future__ import annotations

import dataclasses
import sqlite3
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import zstandard

from seaglass.imessage import source
from seaglass.index import exif, render
from seaglass.index.chunker import Chunk, chunk_messages
from seaglass.index.embed import EmbeddingModel, compute_calibration_absmax, quantize_int8
from seaglass.mlxmem import release_mlx_cache

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
    is_rewrite: bool = False


def open_index_db(index_db_path: Path, *, check_same_thread: bool = True, create: bool = True) -> sqlite3.Connection:
    """Open (creating if needed) index.db with the schema applied and
    sqlite-vec loaded. WAL mode is set by schema.sql.

    `check_same_thread=False` is needed by callers (e.g. mcp_server.py)
    whose connection is used from a thread pool rather than the thread
    that opened it -- SQLite connections are otherwise thread-affine and
    raise ProgrammingError from any other thread.

    `create=False` fails fast with FileNotFoundError instead of silently
    creating an empty database -- important for read-only callers
    (mcp_server.py, eval/score.py) where a typo'd path should be a loud
    error, not a phantom empty index that then reports 0 chunks.
    """
    import sqlite_vec

    if not create and not index_db_path.exists():
        raise FileNotFoundError(f"index.db not found at {index_db_path} (create=False)")
    con = sqlite3.connect(index_db_path, check_same_thread=check_same_thread)
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
    """Keep `meta.build_cursor` as an informational high-water mark
    (`MAX(id) FROM chunks`). No longer used to decide what to skip during
    a build (see module docstring) -- only for status/observability.
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

    Pure re-derivation of the current chunk list from chat.db -- no
    comparison against what's already in index.db. `build_index` uses
    `iter_pending_chunks` (below) for the actual incremental-write
    decision; this function remains a standalone building block (and is
    exercised directly by tests) for "what would a full re-chunk of this
    chat.db produce right now".
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


@dataclasses.dataclass
class PendingChunk:
    """One chunk that needs (re-)writing this run, plus enough context to
    decide whether it's a brand-new id or a rewrite of an existing one.
    """

    chunk: Chunk
    messages_by_id: Dict[int, "source.Message"]
    id: int
    is_rewrite: bool  # True: reuse an existing id, deleting its old rows first


def iter_pending_chunks(
    chat_con: sqlite3.Connection,
    index_con: sqlite3.Connection,
    *,
    chunker_kwargs: Optional[dict] = None,
    stale_ids_out: Optional[List[int]] = None,
) -> Iterator[PendingChunk]:
    """Per chat, re-derive the chat's current chunk list from chat.db and
    diff it against what's already committed in `chunks` for that
    `chat_id` (ordered by `id` -- monotonically increasing with `start_ts`
    within a chat, since message ROWIDs are globally chronological).

    Only chunks that are new or changed are yielded -- see module
    docstring for the three possible outcomes per position. This is what
    lets a sync correctly pick up new messages that got merged into an
    already-committed chat's tail chunk, instead of silently skipping
    them because their *position* was already <= some global cursor.

    **Id ordering invariant.** Ids are handed out in ascending order
    across positions (existing ids are read `ORDER BY c.id`, and brand-new
    ids come from a counter starting above `MAX(id)`), and positions are
    chronological, so "within a chat, `ORDER BY id` == chronological
    order" holds even after an iCloud backfill of *older* messages
    rewrites every position in a chat. `search/rank.py::attach_context`
    depends on exactly that.

    If `stale_ids_out` is provided, chunk ids that are no longer backed by
    any current chunk position are appended to it (see
    `_prune_stale_chunks`) -- collected here rather than in a second pass
    so re-chunking the whole corpus isn't done twice.
    """
    chunker_kwargs = chunker_kwargs or {}
    next_id = (index_con.execute("SELECT MAX(id) FROM chunks").fetchone()[0] or 0) + 1

    chat_ids = [
        row[0]
        for row in chat_con.execute(
            "SELECT DISTINCT chat_id FROM im.chat_message_join ORDER BY chat_id"
        )
    ]
    if stale_ids_out is not None:
        # Whole conversations deleted from chat.db are never visited by
        # the loop below, so their chunks would otherwise linger forever
        # (and keep surfacing deleted messages in search results).
        placeholders = ",".join("?" for _ in chat_ids) or "NULL"
        stale_ids_out.extend(
            row[0]
            for row in index_con.execute(
                f"SELECT id FROM chunks WHERE chat_id NOT IN ({placeholders})", chat_ids
            )
        )

    for chat_id in chat_ids:
        messages = list(source.iter_messages(chat_con, chat_id=chat_id))
        if not messages:
            continue
        messages_by_id = {m.rowid: m for m in messages}
        fresh_chunks = list(chunk_messages(messages, **chunker_kwargs))

        existing_rows = index_con.execute(
            "SELECT id, group_concat(msg_id) FROM chunks c "
            "JOIN chunk_message cm ON cm.chunk_id = c.id "
            "WHERE c.chat_id = ? GROUP BY c.id ORDER BY c.id",
            (chat_id,),
        ).fetchall()
        # chunk_message stores msg_ids unordered per chunk_id -- and even
        # within a chat, message ROWID order does not always match
        # chronological (ts) order in real chat.db data (edits/resends/
        # sync artifacts can leave ROWIDs out of date order). Compare by
        # *set* membership, not order, to decide "unchanged"; the actual
        # rendered order always comes from `fresh.msg_ids` (chunker's
        # chronological order), never from what's stored, so this is safe
        # either way.
        existing_msg_ids_by_position: List[Tuple[int, frozenset]] = [
            (row_id, frozenset(int(x) for x in blob.split(",")))
            for row_id, blob in existing_rows
        ]

        for position, fresh in enumerate(fresh_chunks):
            if position < len(existing_msg_ids_by_position):
                existing_id, existing_msg_ids = existing_msg_ids_by_position[position]
                if existing_msg_ids == frozenset(fresh.msg_ids):
                    continue  # unchanged, already durable -- skip
                # Content at this position differs from what's committed
                # -- either this is the tail chunk and new messages merged
                # into it, or (rarer) chat.db content mutated (edit/
                # retraction) shifting boundaries earlier than the tail.
                # Either way, rewrite this id in place: safe because every
                # downstream table keys off chunk id, and the delete-then-
                # reinsert happens atomically within the batch transaction.
                yield PendingChunk(fresh, messages_by_id, existing_id, is_rewrite=True)
            else:
                yield PendingChunk(fresh, messages_by_id, next_id, is_rewrite=False)
                next_id += 1

        if stale_ids_out is not None and len(fresh_chunks) < len(existing_msg_ids_by_position):
            # The chat shrank (messages deleted in Messages.app, which
            # iCloud propagates). Positions past the end of the current
            # chunk list are orphans referencing messages that no longer
            # exist -- drop them so deleted content stops being
            # searchable.
            stale_ids_out.extend(
                row_id for row_id, _ in existing_msg_ids_by_position[len(fresh_chunks):]
            )


def _prune_stale_chunks(index_con: sqlite3.Connection, stale_ids: Sequence[int]) -> int:
    """Delete chunks that no longer correspond to any live chunk position
    (deleted conversations, or chats that shrank), across all four tables,
    in one transaction. Returns the number of chunks removed.
    """
    if not stale_ids:
        return 0
    index_con.execute("BEGIN")
    try:
        for chunk_id in stale_ids:
            index_con.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
            index_con.execute("DELETE FROM chunk_message WHERE chunk_id = ?", (chunk_id,))
            index_con.execute("DELETE FROM chunks_vec WHERE rowid = ?", (chunk_id,))
            index_con.execute("DELETE FROM chunks_fts WHERE rowid = ?", (chunk_id,))
        index_con.execute("COMMIT")
    except Exception:
        index_con.execute("ROLLBACK")
        raise
    return len(stale_ids)


def _render_chunk(
    chunk_id: int,
    chunk: Chunk,
    messages_by_id: Dict[int, "source.Message"],
    attachments_by_msg: Dict[int, List["source.AttachmentRow"]],
    places_by_attachment: Dict[int, str],
    *,
    is_rewrite: bool = False,
) -> RenderedChunk:
    ordered_messages = [messages_by_id[mid] for mid in chunk.msg_ids]
    return RenderedChunk(
        chunk=chunk,
        id=chunk_id,
        body_semantic=render.format_semantic(ordered_messages),
        body_lexical=render.format_lexical(ordered_messages, attachments_by_msg, places_by_attachment),
        is_rewrite=is_rewrite,
    )



def _resolve_places(
    index_con: sqlite3.Connection,
    chunk_id: int,
    attachments_by_msg: Dict[int, List["source.AttachmentRow"]],
) -> Dict[int, str]:
    """Resolve place names for every attachment referenced in this chunk:
    reuse already-known `attachment_place` rows, and for anything new,
    call `index/exif.py` (GPS extraction + offline reverse-geocode) and
    persist the result immediately. Idempotent and safe to re-run on
    restart -- already-resolved attachment_ids are never re-decoded.
    """
    attachment_ids = sorted(
        {a.attachment_id for atts in attachments_by_msg.values() for a in atts}
    )
    if not attachment_ids:
        return {}

    placeholders = ",".join("?" for _ in attachment_ids)
    existing = dict(
        index_con.execute(
            f"SELECT attachment_id, place FROM attachment_place WHERE attachment_id IN ({placeholders})",
            attachment_ids,
        ).fetchall()
    )
    missing_ids = [aid for aid in attachment_ids if aid not in existing]
    if not missing_ids:
        return existing

    filename_by_id = {
        a.attachment_id: a.filename for atts in attachments_by_msg.values() for a in atts
    }
    targets = [
        exif.AttachmentTarget(attachment_id=aid, path=Path(filename_by_id[aid]).expanduser())
        for aid in missing_ids
        if filename_by_id.get(aid)
    ]
    new_places, failed_ids = exif.extract_places_for_attachments(targets)

    if new_places:
        index_con.executemany(
            "INSERT OR IGNORE INTO attachment_place (attachment_id, place) VALUES (?, ?)",
            list(new_places.items()),
        )
    if failed_ids:
        index_con.executemany(
            "INSERT OR IGNORE INTO attachment_retry (attachment_id, chunk_id) VALUES (?, ?)",
            [(aid, chunk_id) for aid in failed_ids],
        )
    if new_places or failed_ids:
        index_con.commit()

    existing.update(new_places)
    return existing


def _write_batch(
    index_con: sqlite3.Connection,
    rendered_batch: Sequence[RenderedChunk],
    embeddings_int8,
    compressor: "zstandard.ZstdCompressor",
) -> None:
    """Write one batch inside a single transaction: chunks, chunk_message,
    chunks_vec, then chunks_fts last (PLAN.md §6 Phase 3 ordering).

    For any `rendered.is_rewrite` entry, first deletes that id's existing
    rows from all four tables -- required so a chat's tail chunk can be
    "reopened and extended" (Phase 7) with the same id rather than
    silently accumulating stale duplicate rows or being skipped.
    `chunks_fts` is contentless (`contentless_delete=1`), so a plain
    `DELETE ... WHERE rowid = ?` is sufficient there (no need to hand back
    the original body).
    """
    index_con.execute("BEGIN")
    try:
        rewrite_ids = [r.id for r in rendered_batch if r.is_rewrite]
        for rid in rewrite_ids:
            index_con.execute("DELETE FROM chunks WHERE id = ?", (rid,))
            index_con.execute("DELETE FROM chunk_message WHERE chunk_id = ?", (rid,))
            index_con.execute("DELETE FROM chunks_vec WHERE rowid = ?", (rid,))
            index_con.execute("DELETE FROM chunks_fts WHERE rowid = ?", (rid,))

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
        last_id = max(r.id for r in rendered_batch)
        _set_meta(index_con, "build_cursor", str(last_id))
        index_con.execute("COMMIT")
    except Exception:
        index_con.execute("ROLLBACK")
        raise


def _ensure_calibration(
    index_con: sqlite3.Connection,
    embedding_model: EmbeddingModel,
    sample_texts: Sequence[str],
    *,
    embed_batch_size: int = 64,
) -> float:
    """Compute and persist `meta.int8_absmax` once, from a sample of
    rendered `body_semantic` texts, if not already recorded. Every chunk
    embedded in this build (and every query embedded at search time) must
    quantise against this same value (PLAN.md §6 Phase 3).

    Embeds in `embed_batch_size`-sized pieces rather than one giant call:
    `EmbeddingModel.embed()` has no internal batching, and calling it
    with the full ~2000-text calibration sample in one shot can exceed
    MLX's Metal buffer size limit on a single GPU allocation.
    """
    existing = _get_meta(index_con, "int8_absmax")
    if existing is not None:
        return float(existing)
    sample = list(sample_texts[:CALIBRATION_SAMPLE_SIZE])
    all_vectors = []
    for i in range(0, len(sample), embed_batch_size):
        all_vectors.append(embedding_model.embed(sample[i : i + embed_batch_size]))
    vectors = np.concatenate(all_vectors, axis=0) if all_vectors else np.zeros((0, 0))
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
    chunks_written = 0
    batch: List[RenderedChunk] = []
    # IMPROVEMENT-11 fix: calibration must be computed from a real sample
    # of up to CALIBRATION_SAMPLE_SIZE chunks, not just the very first
    # one -- a single chunk's absmax can be an outlier (unusually short/
    # long text) that miscalibrates int8 quantization for the whole
    # index. Buffer rendered-but-uncalibrated chunks here until either
    # the sample target is reached or the corpus runs out, then
    # calibrate once and flush everything buffered so far.
    pending_calibration: List[RenderedChunk] = []
    stale_ids: List[int] = []

    for pending in iter_pending_chunks(
        chat_con, index_con, chunker_kwargs=chunker_kwargs, stale_ids_out=stale_ids
    ):
        chunk, messages_by_id, chunk_id = pending.chunk, pending.messages_by_id, pending.id

        attachments = source.fetch_attachments_for_messages(chat_con, chunk.msg_ids)
        places_by_attachment = _resolve_places(index_con, chunk_id, attachments)
        rendered = _render_chunk(
            chunk_id, chunk, messages_by_id, attachments, places_by_attachment,
            is_rewrite=pending.is_rewrite,
        )

        if not calibrated:
            pending_calibration.append(rendered)
            reached_sample_target = len(pending_calibration) >= CALIBRATION_SAMPLE_SIZE
            reached_run_limit = limit_chunks is not None and chunks_written + len(pending_calibration) >= limit_chunks
            if not (reached_sample_target or reached_run_limit):
                continue
            calibrated = True
            _ensure_calibration(
                index_con, embedding_model, [r.body_semantic for r in pending_calibration]
            )
            batch.extend(pending_calibration)
            pending_calibration = []
        else:
            batch.append(rendered)

        if len(batch) >= batch_size:
            # Loop (not just one `if`): the calibration extend() above can
            # dump up to CALIBRATION_SAMPLE_SIZE chunks into `batch` at
            # once, so a single flush() here would embed all of them in
            # one MLX call and can exceed the GPU's max buffer size.
            while len(batch) >= batch_size:
                _flush(batch[:batch_size], embedding_model, index_con, compressor)
                chunks_written += batch_size
                batch = batch[batch_size:]

        if limit_chunks is not None and chunks_written + len(batch) >= limit_chunks:
            break

    if not calibrated and pending_calibration:
        # Corpus exhausted before reaching the sample target -- calibrate
        # from whatever we collected (this is the correct, deliberate
        # fallback, not the bug: the bug was calibrating from 1 chunk
        # when thousands were available).
        _ensure_calibration(
            index_con, embedding_model, [r.body_semantic for r in pending_calibration]
        )
        batch.extend(pending_calibration)

    if batch:
        while len(batch) > batch_size:
            _flush(batch[:batch_size], embedding_model, index_con, compressor)
            chunks_written += batch_size
            batch = batch[batch_size:]
        _flush(batch, embedding_model, index_con, compressor)
        chunks_written += len(batch)

    # Only prune after a complete pass -- a run cut short by limit_chunks
    # stopped early, so `stale_ids` is not yet a complete picture of what
    # is genuinely orphaned.
    if limit_chunks is None:
        _prune_stale_chunks(index_con, stale_ids)

    # Fold the WAL back into index.db. The app holds long-lived readers on
    # this file, so checkpoint-on-last-close never fires and the -wal would
    # otherwise only grow across syncs. Best-effort: a reader mid-query
    # just means we retry on the next build.
    try:
        index_con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.Error:
        pass

    # A build embeds far larger batches than any query does, and MLX's
    # buffer cache retains that high-water mark for the life of the
    # process. Hand it back now rather than leaving several GB of unified
    # memory resident behind an idle search app (see seaglass/mlxmem.py).
    release_mlx_cache()

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
