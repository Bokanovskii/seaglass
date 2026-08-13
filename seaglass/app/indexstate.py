"""Background index build/sync orchestration for the desktop app.

Building the index can take a long time on a first run (it embeds and
reranks every message chunk in the user's history), so this must never
happen silently/automatically. Instead the app exposes an explicit
"build/sync now" action (see server.py's /api/index/build) that runs the
build in a background thread while this module tracks progress so the UI
can show live status rather than freezing.
"""

from __future__ import annotations

import dataclasses
import sqlite3
import threading
import time
from pathlib import Path

# Cap the snapshot's WAL so it can't outgrow the database it journals.
_WAL_SIZE_LIMIT_BYTES = 64 * 1024 * 1024


class IndexBuildState:
    """Thread-safe progress tracker for an in-app index build/sync run.

    Mirrors the step/state pattern used by WarmupState (see warmup.py) so
    the frontend can poll a single familiar shape.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.running = False
        self.stage = 'idle'  # idle | snapshotting | building | reloading | done | failed
        self.error: str | None = None
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.chunks_before: int | None = None
        self.chunks_written: int = 0
        self.chunks_now: int | None = None

    def snapshot(self) -> dict:
        with self._lock:
            elapsed = None
            if self.started_at is not None:
                end = self.finished_at or time.time()
                elapsed = round(end - self.started_at, 2)
            return {
                'running': self.running,
                'stage': self.stage,
                'error': self.error,
                'elapsed_s': elapsed,
                'chunks_before': self.chunks_before,
                'chunks_written': self.chunks_written,
                'chunks_now': self.chunks_now,
            }

    def _set(self, **kwargs) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self, key, value)


def _checkpoint_snapshot_wal(dest_con: sqlite3.Connection) -> None:
    """Collapse the snapshot's write-ahead log back into the main db file.

    `sqlite3.backup()` copies the *whole* source db (every page) each time
    we sync, and the snapshot inherits WAL journalling from the live
    chat.db. Meanwhile the running app holds long-lived reader connections
    on the snapshot (search hydration), which prevents SQLite's automatic
    checkpoint-on-last-close from ever firing. The net effect is that each
    sync appends another full copy of the database to `-wal`, which grows
    without bound -- observed at 8.5GB against a 655MB db.

    Readers only block a TRUNCATE checkpoint while a read transaction is
    actually open, so retry briefly, then fall back to a PASSIVE
    checkpoint (which at least lets the existing WAL space be reused
    rather than extended). `journal_size_limit` is a standing backstop so
    the file is trimmed on any future checkpoint too.
    """
    try:
        dest_con.execute(f'PRAGMA journal_size_limit = {_WAL_SIZE_LIMIT_BYTES}')
    except sqlite3.Error:
        return
    for _ in range(5):
        try:
            busy, _, _ = dest_con.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()
        except sqlite3.Error:
            return
        if busy == 0:
            return
        time.sleep(0.5)
    try:
        dest_con.execute('PRAGMA wal_checkpoint(PASSIVE)')
    except sqlite3.Error:
        pass


def _count_chunks(index_db_path: Path) -> int | None:
    if not index_db_path.exists():
        return None
    try:
        con = sqlite3.connect(f'file:{index_db_path}?mode=ro', uri=True)
        try:
            row = con.execute('SELECT COUNT(*) FROM chunks').fetchone()
            return int(row[0]) if row else 0
        finally:
            con.close()
    except sqlite3.Error:
        return None


def run_build(
    state: IndexBuildState,
    *,
    chat_db_source: str,
    chat_db_snapshot: str,
    index_db: str,
    on_complete=None,
) -> None:
    """Run a full snapshot + (re)build synchronously. Intended to be
    invoked on a background thread; safe to call repeatedly (build_index
    is idempotent/resumable — see seaglass/index/build.py)."""
    from seaglass.index.build import build_index

    index_path = Path(index_db)
    snapshot_path = Path(chat_db_snapshot)
    source_path = Path(chat_db_source)

    state._set(
        running=True,
        stage='snapshotting',
        error=None,
        started_at=time.time(),
        finished_at=None,
        chunks_before=_count_chunks(index_path),
        chunks_written=0,
        chunks_now=None,
    )
    try:
        if not source_path.exists():
            raise FileNotFoundError(
                f'chat.db not found at {source_path}. Grant Full Disk Access and confirm the path, then try again.'
            )
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        src_con = sqlite3.connect(f'file:{source_path}?mode=ro', uri=True)
        try:
            dest_con = sqlite3.connect(str(snapshot_path))
            try:
                with dest_con:
                    src_con.backup(dest_con)
                _checkpoint_snapshot_wal(dest_con)
            finally:
                dest_con.close()
        finally:
            src_con.close()

        state._set(stage='building')
        index_path.parent.mkdir(parents=True, exist_ok=True)
        written = build_index(snapshot_path, index_path)
        state._set(chunks_written=written, chunks_now=_count_chunks(index_path))

        state._set(stage='done', finished_at=time.time(), running=False)
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI, not swallowed
        state._set(stage='failed', error=str(exc), finished_at=time.time(), running=False)
    finally:
        if on_complete is not None:
            on_complete()
