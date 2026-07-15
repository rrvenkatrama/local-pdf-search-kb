#!/bin/zsh
# ============================================================================
# Local PDF Search KB — one-shot setup script. Idempotent (safe to re-run).
#
#   1. Creates the Python venv and installs dependencies (if missing)
#   2. Installs the launchd agent that runs the indexer daily at 08:00
#
# After this, start the search service manually with:  ./venv/bin/python server.py
# ============================================================================
set -e
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
PLIST_NAME="com.rajesh.pdfkb.indexer"
PLIST_DEST="$HOME/Library/LaunchAgents/$PLIST_NAME.plist"

echo "Project root: $PROJECT_ROOT"

# --- 1. Python environment --------------------------------------------------
if [ ! -x "$PROJECT_ROOT/venv/bin/python" ]; then
    echo "Creating venv and installing dependencies…"
    python3 -m venv "$PROJECT_ROOT/venv"
    "$PROJECT_ROOT/venv/bin/pip" install --quiet --upgrade pip
    "$PROJECT_ROOT/venv/bin/pip" install --quiet -r "$PROJECT_ROOT/requirements.txt"
else
    echo "venv already present — skipping dependency install."
fi

# --- 2. launchd daily schedule ----------------------------------------------
echo "Installing launchd agent → $PLIST_DEST"
mkdir -p "$HOME/Library/LaunchAgents"
# Fill in the real project path (plist template uses __PROJECT_ROOT__).
sed "s|__PROJECT_ROOT__|$PROJECT_ROOT|g" \
    "$PROJECT_ROOT/launchd/$PLIST_NAME.plist" > "$PLIST_DEST"

# Reload cleanly: unload any previous version first (ignore if not loaded).
launchctl bootout "gui/$(id -u)/$PLIST_NAME" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST_DEST"

echo
echo "Done. Daily index runs at 08:00. Verify with:"
echo "  launchctl list | grep pdfkb"
echo "Start the search UI with:"
echo "  cd \"$PROJECT_ROOT\" && ./venv/bin/python server.py"
echo "  → http://localhost:8130/"
