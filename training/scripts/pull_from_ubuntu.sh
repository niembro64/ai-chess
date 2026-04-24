#!/usr/bin/env bash
#
# Pull trained checkpoint(s) from the Ubuntu training box to this Mac.
#
# Usage:
#   scripts/pull_from_ubuntu.sh              # default: --weights (safe, small)
#   scripts/pull_from_ubuntu.sh --weights    # champion.pt + latest.json + eval.csv → runs/ubuntu-import/
#   scripts/pull_from_ubuntu.sh --browser    # pull latest.json + deploy to browser preset
#   scripts/pull_from_ubuntu.sh --full       # rsync entire runs/latest/ (includes 70MB stats.csv)
#   scripts/pull_from_ubuntu.sh --check      # just list remote files (sanity test SSH)
#
# Env overrides (if your setup differs):
#   REMOTE=gpus@192.168.86.34
#   REMOTE_PATH=~/ai-chess/training/runs/latest
#   LOCAL_PATH=~/code/ai-chess/training/runs/ubuntu-import
#
# Destination defaults to runs/ubuntu-import/ so the local runs/latest/
# (if you ever train on the Mac) is not clobbered.

set -euo pipefail

REMOTE="${REMOTE:-gpus@192.168.86.34}"
REMOTE_PATH="${REMOTE_PATH:-~/ai-chess/training/runs/latest}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$TRAINING_DIR/.." && pwd)"
LOCAL_PATH="${LOCAL_PATH:-$TRAINING_DIR/runs/ubuntu-import}"

mode="${1:---weights}"

case "$mode" in
  --check)
    echo "→ ssh $REMOTE 'ls -lh $REMOTE_PATH/'"
    ssh "$REMOTE" "ls -lh $REMOTE_PATH/"
    ;;

  --weights)
    # champion.pt = last *promoted* weights (updates only when the
    # challenger beats the champion + threshold). latest.json = current
    # training-state weights in browser format (rewrites every checkpoint
    # save, so may be a few thousand gens past champion). Pulling both
    # gives you one for analysis (champion.pt) and one for deployment
    # (latest.json). eval.csv is the promotion history.
    mkdir -p "$LOCAL_PATH"
    echo "→ pulling champion.pt, latest.json, eval.csv to $LOCAL_PATH"
    rsync -avzP \
      "$REMOTE:$REMOTE_PATH/champion.pt" \
      "$REMOTE:$REMOTE_PATH/latest.json" \
      "$REMOTE:$REMOTE_PATH/eval.csv" \
      "$LOCAL_PATH/"
    echo
    echo "Done. Files in: $LOCAL_PATH"
    ;;

  --browser)
    # latest.json is the auto-written browser-format weights, NOT
    # champion.pt (that's only in the PyTorch native format). If you
    # want to play the *champion* weights specifically rather than
    # the current training state, you'd need to regenerate JSON from
    # champion.pt — which this script doesn't do. In practice latest
    # is usually close enough to champion for a playthrough.
    mkdir -p "$LOCAL_PATH"
    echo "→ pulling latest.json to $LOCAL_PATH"
    rsync -avzP \
      "$REMOTE:$REMOTE_PATH/latest.json" \
      "$LOCAL_PATH/"
    echo
    echo "→ deploying to browser preset"
    python "$SCRIPT_DIR/deploy_to_browser.py" "$LOCAL_PATH/latest.json"
    echo
    echo "Done. Rebuild or hot-reload the browser app to pick up the new weights."
    ;;

  --full)
    mkdir -p "$LOCAL_PATH"
    echo "→ mirroring entire $REMOTE_PATH/ to $LOCAL_PATH/"
    echo "  (includes stats.csv — ~70 MB telemetry, can take a moment)"
    rsync -avzP "$REMOTE:$REMOTE_PATH/" "$LOCAL_PATH/"
    echo
    echo "Done. Full run mirror in: $LOCAL_PATH"
    ;;

  -h|--help)
    # Docstring comment block (lines 3-18). Strip leading '# '.
    sed -n '3,18p' "$0" | sed 's/^# \{0,1\}//'
    ;;

  *)
    echo "unknown mode: $mode" >&2
    echo "run with --help for usage" >&2
    exit 2
    ;;
esac
