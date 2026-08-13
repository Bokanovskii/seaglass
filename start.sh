#!/usr/bin/env bash
# Launch seaglass. Use this once ./setup.sh has been run at least once.
#
# Usage:
#   ./start.sh                 # launch the desktop app
#   ./start.sh --browser       # open in a browser tab instead (debugging)
#   ./start.sh --install-app   # also install ~/Applications/Seaglass.app, then launch
#
# Why this exists as a separate script from setup.sh: setup.sh reinstalls
# dependencies and runs the capability preflight every time, which takes
# long enough to be annoying for something you do several times a day.
# This is the fast path -- it only activates the venv and starts the app.
#
# Note there is deliberately no plain `./seaglass` launcher: `seaglass/`
# is the Python package directory, so a file of that name can't exist
# beside it (the shell resolves `./seaglass` to the directory and refuses
# to execute it).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

SEAGLASS_HOME="${SEAGLASS_HOME:-$HOME/.seaglass}"
CHAT_DB_SRC="${SEAGLASS_CHAT_DB_SRC:-$HOME/Library/Messages/chat.db}"
SNAPSHOT_DB="$SEAGLASS_HOME/chat_snapshot.db"
INDEX_DB="$SEAGLASS_HOME/index.db"

if [ ! -x .venv/bin/seaglass-app ]; then
  echo "seaglass isn't installed yet (no .venv/bin/seaglass-app)." >&2
  echo "Run ./setup.sh first -- it creates the venv, installs seaglass and launches the app." >&2
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

if [ "${1:-}" = "--install-app" ]; then
  shift
  python -m seaglass.app.installapp
fi

mkdir -p "$SEAGLASS_HOME"
exec seaglass-app \
  --index-db "$INDEX_DB" \
  --chat-db "$SNAPSHOT_DB" \
  --chat-db-source "$CHAT_DB_SRC" \
  "$@"
