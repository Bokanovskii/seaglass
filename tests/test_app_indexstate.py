from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from seaglass.app.indexstate import IndexBuildState, run_build

FIXTURE_DIR = Path(__file__).parent / '_synthetic_app_smoke'


def test_run_build_reports_failure_when_source_missing(tmp_path):
    state = IndexBuildState()
    completed = []
    run_build(
        state,
        chat_db_source=str(tmp_path / 'nonexistent-chat.db'),
        chat_db_snapshot=str(tmp_path / 'snapshot.db'),
        index_db=str(tmp_path / 'index.db'),
        on_complete=lambda: completed.append(True),
    )
    snapshot = state.snapshot()
    assert snapshot['running'] is False
    assert snapshot['stage'] == 'failed'
    assert 'chat.db not found' in snapshot['error']
    assert completed == [True]


def test_run_build_snapshots_and_builds_from_synthetic_fixture(tmp_path):
    source = tmp_path / 'source_chat.db'
    shutil.copy(FIXTURE_DIR / 'chat.db', source)
    state = IndexBuildState()
    snapshot_path = tmp_path / 'snapshot.db'
    index_path = tmp_path / 'index.db'

    run_build(
        state,
        chat_db_source=str(source),
        chat_db_snapshot=str(snapshot_path),
        index_db=str(index_path),
        on_complete=None,
    )

    snapshot = state.snapshot()
    assert snapshot['stage'] == 'done'
    assert snapshot['running'] is False
    assert snapshot['error'] is None
    assert snapshot_path.exists()
    assert index_path.exists()
    assert snapshot['chunks_now'] and snapshot['chunks_now'] > 0


def test_run_build_is_idempotent_on_rerun(tmp_path):
    source = tmp_path / 'source_chat.db'
    shutil.copy(FIXTURE_DIR / 'chat.db', source)
    snapshot_path = tmp_path / 'snapshot.db'
    index_path = tmp_path / 'index.db'

    state1 = IndexBuildState()
    run_build(state1, chat_db_source=str(source), chat_db_snapshot=str(snapshot_path), index_db=str(index_path))
    first_chunks = state1.snapshot()['chunks_now']

    state2 = IndexBuildState()
    run_build(state2, chat_db_source=str(source), chat_db_snapshot=str(snapshot_path), index_db=str(index_path))
    second = state2.snapshot()
    assert second['stage'] == 'done'
    assert second['chunks_written'] == 0  # nothing new to index
    assert second['chunks_now'] == first_chunks
