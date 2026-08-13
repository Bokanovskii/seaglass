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
#   4. Launches the desktop app.
#
# Building the search index (which reads and embeds your full message
# history) is NOT done by this script — it can take a long time on a
# large history, so it's something you kick off yourself, from inside the
# app, with a visible "Build index now" button and progress indicator.
# The app also shows how in-sync the index is (e.g. "N new messages since
# last build") and lets you re-sync any time.
#
# Safe to re-run: an existing venv is reused.
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

# --- 4. Launch -------------------------------------------------------
if [ -f "$INDEX_DB" ]; then
  info "Launching seaglass (existing index found — app will show if it needs a resync)"
else
  info "Launching seaglass — no index yet. Use the 'Build index now' button in the app when you're ready"
  info "(building reads and embeds your full message history and can take a while for large histories)."
fi
exec seaglass-app --index-db "$INDEX_DB" --chat-db "$SNAPSHOT_DB" --chat-db-source "$CHAT_DB_SRC" "$@"
