#!/usr/bin/env bash
# seaglass setup — one command to get from a fresh clone to a running app.
#
# Usage:
#   ./setup.sh
#
# What it does, in order:
#   1. Creates a Python venv (.venv) if one doesn't exist yet.
#   2. Installs seaglass + the desktop app extras into it.
#   3. Runs the capability preflight (seaglass.probe) so problems like
#      missing Full Disk Access are caught early with a clear message.
#   4. Snapshots your live chat.db to a local, safe, read-only copy
#      (never touches or locks the live Messages database).
#   5. Builds an index.db from that snapshot (first run only — safe to
#      re-run any time to pick up new messages).
#   6. Launches the desktop app.
#
# Safe to re-run: existing venvs/snapshots/indexes are reused, and the
# index build resumes from where it left off rather than starting over.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

SEAGLASS_HOME="${SEAGLASS_HOME:-$HOME/.seaglass}"
CHAT_DB_SRC="${SEAGLASS_CHAT_DB_SRC:-$HOME/Library/Messages/chat.db}"
SNAPSHOT_DB="$SEAGLASS_HOME/chat_snapshot.db"
INDEX_DB="$SEAGLASS_HOME/index.db"

info()  { printf '\033[1;34m==>\033[0m %s\n' "$1"; }
warn()  { printf '\033[1;33m!!\033[0m %s\n' "$1"; }
fail()  { printf '\033[1;31mERROR:\033[0m %s\n' "$1"; exit 1; }

# --- 1. Python check -------------------------------------------------------
find_python() {
  if [ -n "${PYTHON_BIN:-}" ]; then
    echo "$PYTHON_BIN"
    return
  fi
  # macOS system Python (/usr/bin/python3) is usually too old and links an
  # SQLite build missing FTS5 features seaglass needs, so prefer a
  # Homebrew/python.org 3.11+ build if one is available.
  for candidate in python3.14 python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      version=$("$candidate" -c 'import sys; print(sys.version_info[:2] >= (3, 11))' 2>/dev/null || echo False)
      if [ "$version" = "True" ]; then
        echo "$candidate"
        return
      fi
    fi
  done
}

PYTHON_BIN="$(find_python)"
if [ -z "$PYTHON_BIN" ]; then
  fail "No Python 3.11+ found. Install one (e.g. 'brew install python@3.13') and re-run, or set PYTHON_BIN=/path/to/python3.11 ./setup.sh"
fi
PY_VERSION=$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
info "Using $PYTHON_BIN (Python $PY_VERSION)"

# --- 2. Venv + install -------------------------------------------------------
if [ ! -d .venv ]; then
  info "Creating virtual environment (.venv)"
  "$PYTHON_BIN" -m venv .venv
else
  info "Reusing existing .venv"
fi

# shellcheck disable=SC1091
source .venv/bin/activate

info "Installing seaglass + desktop app dependencies (this can take a few minutes on first run)"
pip install -q --upgrade pip
pip install -q -e ".[app,dev]"

# --- 3. Preflight -------------------------------------------------------
info "Running capability preflight"
if ! python -m seaglass.probe --chat-db "$CHAT_DB_SRC"; then
  warn "Preflight reported problems above. Most commonly this means Full Disk Access"
  warn "hasn't been granted yet: System Settings > Privacy & Security > Full Disk Access,"
  warn "then add/enable Terminal (or your terminal app / Python interpreter)."
  warn "Fix the issue above and re-run ./setup.sh."
  exit 1
fi

mkdir -p "$SEAGLASS_HOME"

# --- 4. Snapshot chat.db -------------------------------------------------------
if [ ! -f "$CHAT_DB_SRC" ]; then
  fail "chat.db not found at $CHAT_DB_SRC. Set SEAGLASS_CHAT_DB_SRC to its location and re-run."
fi

info "Snapshotting chat.db (safe, read-only copy — never touches the live database)"
python - "$CHAT_DB_SRC" "$SNAPSHOT_DB" <<'PYEOF'
import sqlite3
import sys

src, dst = sys.argv[1], sys.argv[2]
source = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
dest = sqlite3.connect(dst)
with dest:
    source.backup(dest)
source.close()
dest.close()
PYEOF

# --- 5. Build / resume index -------------------------------------------------------
if [ -f "$INDEX_DB" ]; then
  info "Updating existing index at $INDEX_DB (resumes from where it left off)"
else
  info "Building a new index at $INDEX_DB (first run — indexes your full message history, may take a while)"
fi
python -m seaglass.cli build "$SNAPSHOT_DB" "$INDEX_DB"

# --- 6. Launch -------------------------------------------------------
info "Launching seaglass"
exec seaglass-app --index-db "$INDEX_DB" --chat-db "$SNAPSHOT_DB" "$@"
