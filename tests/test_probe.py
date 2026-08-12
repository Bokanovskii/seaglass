"""Unit tests for seaglass.probe -- capability checks that don't need a
live chat.db (those paths are exercised manually against the real machine;
see development-plans/ADDENDUM.md for recorded results).
"""

from __future__ import annotations

from pathlib import Path

from seaglass.probe import (
    CHAT_DB_CHECKS,
    CHECKS,
    check_chat_db_present,
    check_readonly_attach,
    run_all,
)


def test_environment_capability_checks_pass_in_this_test_env():
    # These are pure-capability checks (SQLite version/FTS5/extension
    # loading/sqlite-vec) -- they should pass in any correctly set-up dev
    # environment, independent of whether a chat.db is present.
    results = [check() for check in CHECKS]
    failed = [r for r in results if not r.ok]
    assert not failed, f"capability checks failed: {failed}"


def test_missing_chat_db_reports_failure_not_exception(tmp_path: Path):
    missing = tmp_path / "does-not-exist.db"
    result = check_chat_db_present(missing)
    assert result.ok is False
    assert "not found" in result.detail

    attach_result = check_readonly_attach(missing)
    assert attach_result.ok is False


def test_run_all_returns_a_result_for_every_check(tmp_path: Path):
    missing = tmp_path / "does-not-exist.db"
    results = run_all(chat_db=missing)
    names = {r.name for r in results}
    assert "sqlite_version" in names
    assert "chat_db_present" in names
    assert len(results) == len(CHECKS) + len(CHAT_DB_CHECKS)
