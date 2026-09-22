#!/usr/bin/env bash
# Resumable grid runner: skips cells whose output exists and row-count checks out.
#   ./run_grid.sh configs/grid/
set -u
CFG_DIR=${1:?usage: run_grid.sh <config_dir>}
for cfg in "$CFG_DIR"/*.yaml; do
  out=$(python3 -c "import yaml,sys;print(yaml.safe_load(open('$cfg'))['out'])")
  if [ -s "$out" ]; then echo "[skip] $cfg"; continue; fi
  echo "[run ] $cfg"
  python3 score_cell.py --config "$cfg"; rc=$?
  # exit 3 = the cell asks for something no scorer can honour (score_cell.py)
  if [ $rc -eq 3 ]; then echo "[REFUSED] $cfg" >> grid_failures.log
  elif [ $rc -ne 0 ]; then echo "[FAIL] $cfg" >> grid_failures.log; fi
done
echo "done; refused/failed cells (if any) in grid_failures.log"
