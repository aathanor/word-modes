#!/usr/bin/env bash
# Render from the real (gated) results and switch the live site to the new release.
# Refuses smoke/quick runs (mode_probe.site_export checks MODE_PROBE_PUBLISH).
set -euo pipefail
SITE="$(cd "$(dirname "$0")/.." && pwd)"
LIVE="$HOME/sites/wordmodes"
cd "$SITE"
unset MODE_PROBE_SITE_SOURCE
MODE_PROBE_PUBLISH=1 ./render.sh
rel="$LIVE/releases/$(date +%Y%m%d-%H%M%S)-$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
cp -a _site "$rel"
ln -sfn "$rel" "$LIVE/current.tmp" && mv -Tf "$LIVE/current.tmp" "$LIVE/current"
echo "live: $(readlink "$LIVE/current")"
