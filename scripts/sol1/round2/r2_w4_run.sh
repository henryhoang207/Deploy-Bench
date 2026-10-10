#!/usr/bin/env bash
# W4 on sol1 (after every training run is scored): refresh the gates with all eleven runs, run the registered tests
# once (R2-H1..H4, R2-RL; they use no MLAAD-5 score), then the exploratory analyses (E5 completes on a rerun after
# Pascu's MLAAD-5, DEV-9 amendment 5). Ends with W4_RUN_DONE <tag>; a failed step prints W4_STEP_FAILED.
#   setsid nohup bash ../xfer/r2_w4_run.sh interim > ../logs/r2_w4.log 2>&1 &      (from the clone)
set -u
TAG=${1:?interim or final}
H=/mnt/data8tb/henry; cd $H/deploybench
export DATA_ROOT=$H/data/deepfake OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY="nice $H/envs/db310/bin/python -B"
step() { echo "[$1] start $(date -Is)"; shift; if "$@"; then echo "[step] ok $(date -Is)"; else echo "W4_STEP_FAILED: $* $(date -Is)"; fi; }
if [ "$TAG" = interim ]; then
  step gates $PY analysis/round2/r2_gates.py --xlsr-mamba-ref vendor/XLSR-Mamba/Scores/DF/Bmamba3_LA_WCE_1e-06_ES144_NE12.txt --runs-dir $H/r2_runs
  step registered $PY analysis/round2/r2_registered.py
fi
step explore $PY analysis/round2/r2_explore.py --b 1000
echo "W4_RUN_DONE $TAG $(date -Is)"
