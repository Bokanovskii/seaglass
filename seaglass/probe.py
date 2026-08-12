"""Phase 0 — environment preflight and capability probes.

See development-plans/PLAN.md §"Environment preflight" and §6 Phase 0, and
development-plans/ADDENDUM.md §5 for why this module is split in two:

* Capability checks (this module, run anytime): is the environment even
  capable of running seaglass at all? SQLite version, FTS5, extension
  loading, sqlite-vec, read-only ATTACH against the live chat.db, a small
  attributedBody decode sample. None of these depend on how much of the
  corpus has synced down yet.

* Volume/sizing spikes (deferred per ADDENDUM.md §5 until iCloud backfill
  settles): total message/chunk counts, scan latency, per-stage latency
  attribution. Running those now against a moving-target corpus would
  produce numbers that are stale within days. `probe_volume()` below is
  a stub that intentionally does very little until that's revisited.

Run as a script for a human-readable report:

    python -m seaglass.probe
"""

from __future__ import annotations

import dataclasses
import os
import sqlite3
import sys
from pathlib import Path
from typing import Callable, Optional

MIN_SQLITE_VERSION = (3, 43, 0)  # contentless_delete=1 requires this
DEFAULT_CHAT_DB = Path.home() / "Library" / "Messages" / "chat.db"


@dataclasses.dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    remediation: Optional[str] = None


class PreflightError(RuntimeError):
    """Raised by `run_all(fail_fast=True)` on the first failing check."""


def check_sqlite_version() -> CheckResult:
    version = sqlite3.sqlite_version_info
    ok = version >= MIN_SQLITE_VERSION
    detail = f"sqlite3 linked against SQLite {sqlite3.sqlite_version} (need >= {'.'.join(map(str, MIN_SQLITE_VERSION))})"
    remediation = (
        "Python's sqlite3 module links whatever SQLite it was built against, "
        "which on macOS is frequently older than the system library. Use a "
        "Homebrew or python.org Python build (not /usr/bin/python3), e.g. "
        "`brew install python@3.13` or newer."
    )
    return CheckResult("sqlite_version", ok, detail, None if ok else remediation)


def check_fts5() -> CheckResult:
    con = sqlite3.connect(":memory:")
    try:
        opts = {row[0] for row in con.execute("pragma compile_options")}
        ok = "ENABLE_FTS5" in opts
        detail = "ENABLE_FTS5 present in compile_options" if ok else "ENABLE_FTS5 missing"
        remediation = (
            "This SQLite build lacks FTS5. Use a Homebrew or python.org Python "
            "build, which link a full-featured SQLite."
        )
        return CheckResult("fts5", ok, detail, None if ok else remediation)
    finally:
        con.close()


def check_extension_loading() -> CheckResult:
    con = sqlite3.connect(":memory:")
    try:
        con.enable_load_extension(True)
        con.enable_load_extension(False)
        return CheckResult("extension_loading", True, "sqlite3.enable_load_extension succeeded")
    except AttributeError as error:
        return CheckResult(
            "extension_loading",
            False,
            f"enable_load_extension unavailable: {error}",
            "This Python's sqlite3 module was built without extension-loading "
            "support (common on the macOS system Python). Use a Homebrew or "
            "python.org Python build instead.",
        )
    finally:
        con.close()


def check_contentless_delete() -> CheckResult:
    con = sqlite3.connect(":memory:")
    try:
        con.execute(
            "CREATE VIRTUAL TABLE t USING fts5(body, content='', contentless_delete=1)"
        )
        con.execute("INSERT INTO t(rowid, body) VALUES (1, 'hello world')")
        con.execute("DELETE FROM t WHERE rowid = 1")
        return CheckResult("contentless_delete", True, "contentless_delete=1 accepted and deletable")
    except sqlite3.OperationalError as error:
        return CheckResult(
            "contentless_delete",
            False,
            f"contentless_delete=1 rejected: {error}",
            f"Requires SQLite >= {'.'.join(map(str, MIN_SQLITE_VERSION))}; see the sqlite_version check.",
        )
    finally:
        con.close()


def check_sqlite_vec() -> CheckResult:
    try:
        import sqlite_vec
    except ImportError as error:
        return CheckResult(
            "sqlite_vec_import",
            False,
            f"import sqlite_vec failed: {error}",
            "pip install sqlite-vec (pin the version once Phase 0's constrained-KNN "
            "spike, PLAN.md §6 Phase 0 item 1, has been run).",
        )
    con = sqlite3.connect(":memory:")
    try:
        con.enable_load_extension(True)
        sqlite_vec.load(con)
        con.enable_load_extension(False)
        version = con.execute("select vec_version()").fetchone()[0]
        con.execute("CREATE VIRTUAL TABLE t USING vec0(embedding int8[384] distance_metric=cosine)")
        return CheckResult("sqlite_vec", True, f"sqlite-vec {version} loaded; vec0 int8[384] table created")
    except Exception as error:  # noqa: BLE001 - report whatever the extension raises
        return CheckResult(
            "sqlite_vec",
            False,
            f"sqlite-vec load or vec0 creation failed: {error}",
            "Check the installed sqlite-vec version is compatible with this SQLite build.",
        )
    finally:
        con.close()


def check_chat_db_present(chat_db: Path) -> CheckResult:
    ok = chat_db.is_file()
    detail = f"{chat_db} {'exists' if ok else 'not found'}"
    remediation = (
        "Messages.app must have been launched at least once, or pass an "
        "explicit --chat-db path."
    )
    return CheckResult("chat_db_present", ok, detail, None if ok else remediation)


def check_readonly_attach(chat_db: Path) -> CheckResult:
    if not chat_db.is_file():
        return CheckResult("readonly_attach", False, "skipped: chat_db not present", None)
    con = sqlite3.connect(":memory:")
    try:
        con.execute(f"ATTACH DATABASE 'file:{chat_db}?mode=ro' AS im")
        count = con.execute("SELECT COUNT(*) FROM im.message").fetchone()[0]
        return CheckResult("readonly_attach", True, f"read-only ATTACH succeeded; {count} rows in im.message")
    except sqlite3.OperationalError as error:
        return CheckResult(
            "readonly_attach",
            False,
            f"ATTACH failed: {error}",
            "Grant Full Disk Access to the terminal/interpreter running this "
            "(System Settings > Privacy & Security > Full Disk Access).",
        )
    finally:
        con.close()


def check_schema_shape(chat_db: Path) -> CheckResult:
    """Assert the columns this system depends on still exist.

    Mirrors PLAN.md §6 Phase 1: "Assert the schema on open. Fail loudly if a
    macOS update changed it." Delegates to the single source of truth in
    `seaglass.imessage.source` so probe and extraction never drift apart.
    """
    if not chat_db.is_file():
        return CheckResult("schema_shape", False, "skipped: chat_db not present", None)
    from seaglass.imessage.source import SchemaDriftError, assert_schema

    con = sqlite3.connect(":memory:")
    try:
        con.execute(f"ATTACH DATABASE 'file:{chat_db}?mode=ro' AS im")
        try:
            assert_schema(con)
            return CheckResult("schema_shape", True, "all expected tables/columns present")
        except SchemaDriftError as error:
            return CheckResult(
                "schema_shape",
                False,
                str(error),
                "Update EXPECTED_SCHEMA in seaglass/imessage/source.py to match the new shape.",
            )
    finally:
        con.close()


def check_attributedbody_decode(chat_db: Path, sample_size: int = 20) -> CheckResult:
    """Decode a small sample of attributedBody blobs (PLAN.md §6 Phase 1,
    "highest-risk component"). Deliberately capped and read-only; this is a
    capability check, not a full extraction run.
    """
    if not chat_db.is_file():
        return CheckResult("attributedbody_decode", False, "skipped: chat_db not present", None)
    try:
        from typedstream.stream import TypedStreamReader
    except ImportError as error:
        return CheckResult(
            "attributedbody_decode",
            False,
            f"import pytypedstream failed: {error}",
            "pip install pytypedstream",
        )
    con = sqlite3.connect(":memory:")
    try:
        con.execute(f"ATTACH DATABASE 'file:{chat_db}?mode=ro' AS im")
        rows = con.execute(
            """
            SELECT ROWID, attributedBody FROM im.message
            WHERE text IS NULL AND attributedBody IS NOT NULL
            LIMIT ?
            """,
            (sample_size,),
        ).fetchall()
        if not rows:
            return CheckResult(
                "attributedbody_decode",
                True,
                "no NULL-text/attributedBody-present rows found in sample window "
                "(nothing to decode, not a failure)",
            )
        decoded = 0
        failures = []
        for rowid, blob in rows:
            try:
                events = list(TypedStreamReader.from_data(blob))
                # The decoded message body surfaces as a `bytes` event (the
                # NSString's raw payload) inside the stream; typedstream does
                # not reconstruct Python `str` objects. See PLAN.md §6 Phase 1
                # -- "use typedstream, fall back to message.text, and keep a
                # regex fallback for blobs neither handles."
                text_found = any(
                    isinstance(event, bytes) and len(event) > 0
                    for event in events
                )
                if text_found:
                    decoded += 1
            except Exception as error:  # noqa: BLE001 - collect and report, don't abort the sample
                failures.append((rowid, str(error)))
        ok = decoded > 0
        detail = f"decoded {decoded}/{len(rows)} sampled attributedBody blobs; {len(failures)} raised errors"
        remediation = (
            "typedstream decoding is the highest-risk extraction component "
            "(PLAN.md §6 Phase 1). Inspect failures individually; a regex "
            "fallback is the documented backstop."
        )
        return CheckResult("attributedbody_decode", ok, detail, None if ok else remediation)
    finally:
        con.close()


def check_epoch_threshold(chat_db: Path) -> CheckResult:
    """Sanity-check the seconds-vs-nanoseconds Apple-epoch heuristic
    (PLAN.md §6 Phase 1) against this machine's actual oldest/newest rows.
    """
    if not chat_db.is_file():
        return CheckResult("epoch_threshold", False, "skipped: chat_db not present", None)
    con = sqlite3.connect(":memory:")
    try:
        con.execute(f"ATTACH DATABASE 'file:{chat_db}?mode=ro' AS im")
        row = con.execute("SELECT MIN(date), MAX(date) FROM im.message WHERE date > 0").fetchone()
        if row is None or row[0] is None:
            return CheckResult("epoch_threshold", True, "no dated rows to check")
        min_date, max_date = row

        def apple_to_unix(value: int) -> float:
            secs = value / 1e9 if value > 1e11 else value
            return secs + 978307200

        import datetime as dt

        lo = dt.datetime.fromtimestamp(apple_to_unix(min_date), tz=dt.timezone.utc)
        hi = dt.datetime.fromtimestamp(apple_to_unix(max_date), tz=dt.timezone.utc)
        now = dt.datetime.now(tz=dt.timezone.utc)
        plausible = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc) <= lo <= hi <= now + dt.timedelta(days=1)
        detail = f"oldest={lo.isoformat()} newest={hi.isoformat()} (raw min={min_date}, max={max_date})"
        remediation = (
            "The magnitude-based seconds/nanoseconds threshold (1e11) produced an "
            "implausible date. Inspect raw `date` values and adjust the threshold "
            "in seaglass/imessage/source.py."
        )
        return CheckResult("epoch_threshold", plausible, detail, None if plausible else remediation)
    finally:
        con.close()


def check_backfill_progress(chat_db: Path) -> CheckResult:
    """Report (not gate on) how far iCloud backfill / local sync has
    progressed, per ADDENDUM.md §4 and §6. This is a *signal*, not a
    pass/fail capability check -- it always reports `ok=True` when it can
    run, since an incomplete backfill is expected and not itself a defect.
    Watch this number across repeated runs; approach ~100% before running
    the volume probes or the Phase 3 full build.
    """
    if not chat_db.is_file():
        return CheckResult("backfill_progress", False, "skipped: chat_db not present", None)
    con = sqlite3.connect(":memory:")
    try:
        con.execute(f"ATTACH DATABASE 'file:{chat_db}?mode=ro' AS im")
        total = con.execute(
            "SELECT COUNT(*) FROM im.message WHERE associated_message_type = 0"
        ).fetchone()[0]
        joined = con.execute(
            "SELECT COUNT(DISTINCT message_id) FROM im.chat_message_join"
        ).fetchone()[0]
        inversions = con.execute(
            """
            SELECT COUNT(*) FROM im.message m1
            JOIN im.message m2 ON m2.ROWID = m1.ROWID + 1
            WHERE m2.date < m1.date
            """
        ).fetchone()[0]
        pairs = max(total - 1, 1)
        ratio = joined / total if total else 1.0
        detail = (
            f"chat_message_join covers {joined}/{total} messages ({ratio:.1%}); "
            f"adjacent-ROWID date inversions {inversions}/{pairs} ({inversions / pairs:.1%}) "
            "-- both should approach 0 backfill remaining before running volume "
            "probes or the Phase 3 full build (ADDENDUM.md §4, §6, §7)"
        )
        return CheckResult("backfill_progress", True, detail)
    finally:
        con.close()


CHECKS: list[Callable[..., CheckResult]] = [
    check_sqlite_version,
    check_fts5,
    check_extension_loading,
    check_contentless_delete,
    check_sqlite_vec,
]

CHAT_DB_CHECKS: list[Callable[[Path], CheckResult]] = [
    check_chat_db_present,
    check_readonly_attach,
    check_schema_shape,
    check_attributedbody_decode,
    check_epoch_threshold,
    check_backfill_progress,
]


def run_all(chat_db: Path = DEFAULT_CHAT_DB, fail_fast: bool = False) -> list[CheckResult]:
    results = [check() for check in CHECKS]
    results += [check(chat_db) for check in CHAT_DB_CHECKS]
    if fail_fast:
        for result in results:
            if not result.ok:
                raise PreflightError(f"{result.name}: {result.detail}")
    return results


def probe_volume(chat_db: Path = DEFAULT_CHAT_DB) -> None:
    """Deliberately NOT implemented yet.

    See development-plans/ADDENDUM.md §5: volume/sizing probes (message and
    chunk counts, inter-message gap distribution, scan latency spikes) are
    held until iCloud backfill has settled, because the corpus is currently
    a moving target (measured ~50% ROWID/date inversion on 2026-08-12,
    confirming active backfill). Implement this once that has stabilized.
    """
    raise NotImplementedError(
        "Volume probes are intentionally deferred; see development-plans/ADDENDUM.md §5. "
        "Run seaglass.probe.run_all() for capability checks in the meantime."
    )


def _format_report(results: list[CheckResult]) -> str:
    lines = []
    for result in results:
        status = "OK  " if result.ok else "FAIL"
        lines.append(f"[{status}] {result.name}: {result.detail}")
        if not result.ok and result.remediation:
            lines.append(f"       -> {result.remediation}")
    failed = sum(1 for r in results if not r.ok)
    lines.append("")
    lines.append(f"{len(results) - failed}/{len(results)} checks passed")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="seaglass Phase 0 capability preflight")
    parser.add_argument(
        "--chat-db",
        type=Path,
        default=DEFAULT_CHAT_DB,
        help="Path to chat.db (default: ~/Library/Messages/chat.db)",
    )
    args = parser.parse_args(argv)

    results = run_all(chat_db=args.chat_db)
    print(_format_report(results))
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
