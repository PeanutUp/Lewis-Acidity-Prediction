#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LATEX_COMPILER="${TECTONIC:-tectonic}"
if ! command -v "$LATEX_COMPILER" >/dev/null 2>&1; then
  LATEX_COMPILER="$TASK_ROOT/work/tex/bin/tectonic"
fi
if [[ ! -x "$LATEX_COMPILER" ]] && ! command -v "$LATEX_COMPILER" >/dev/null 2>&1; then
  echo 'Install Tectonic or set TECTONIC to its executable path.' >&2
  exit 1
fi
mkdir -p "$TASK_ROOT/manuscript/compiled" "$TASK_ROOT/work/tex/cache"
cd "$TASK_ROOT/manuscript"
for manuscript in digital_discovery jcim supporting_information; do
  XDG_CACHE_HOME="$TASK_ROOT/work/tex/cache" "$LATEX_COMPILER" --keep-logs --keep-intermediates \
    --outdir compiled "$manuscript.tex"
done
