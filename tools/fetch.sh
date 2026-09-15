#!/usr/bin/env bash
# The intended workflow for an agent (or a CI job) with a tiny workspace:
#   pull the tool, use it, throw it away.  Nothing is kept locally except
#   the finished song, and the finished song goes straight back to the repo.
#
#   ./tools/fetch.sh make_song.py -- --style trap-dark --seed 42 --out /tmp/s.wav
set -euo pipefail
REPO="${SONORA_REPO:-https://github.com/aaronprater146-max/sonora}"
TOOL="$1"; shift
[ "${1:-}" = "--" ] && shift
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
git clone --depth 1 -q "$REPO" "$WORK/sonora"
python3 "$WORK/sonora/tools/$TOOL" "$@"
echo "workspace cleaned: $WORK removed"
