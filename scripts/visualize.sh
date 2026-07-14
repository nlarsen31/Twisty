#!/usr/bin/env bash
# Render a handful of cube states (solved, one move, scrambles) to a PNG.
# Usage: scripts/visualize.sh [output.png] [seed]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUTPUT="${1:-cube_states.png}"
SEED="${2:-0}"

source .venv/bin/activate
python -m twisty.scripts.render_states --output "$OUTPUT" --seed "$SEED"
