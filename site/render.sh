#!/usr/bin/env bash
# Render the site. MODE_PROBE_SITE_SOURCE=smoke renders from the smoke run (bannered).
set -euo pipefail
cd "$(dirname "$0")"
export QUARTO_PYTHON="$HOME/miniconda3/envs/mode-probe/bin/python"
export HF_HUB_DISABLE_XET=1
"$HOME/.local/bin/quarto" render "$@"
