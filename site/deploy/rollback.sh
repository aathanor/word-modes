#!/usr/bin/env bash
# Point the live site at the previous release (or at the one named as $1).
set -euo pipefail
LIVE="$HOME/sites/wordmodes"
cur="$(readlink "$LIVE/current")"
if [ $# -ge 1 ]; then target="$LIVE/releases/$1"; else
  target="$(ls -d "$LIVE"/releases/2* | sort | grep -v -x "$cur" | tail -1)"; fi
[ -d "$target" ] || { echo "no release $target"; exit 1; }
ln -sfn "$target" "$LIVE/current.tmp" && mv -Tf "$LIVE/current.tmp" "$LIVE/current"
echo "live: $target"
