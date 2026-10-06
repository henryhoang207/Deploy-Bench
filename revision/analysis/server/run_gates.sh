#!/bin/bash
# DEV-8 F2 gate: no-trim scoring of the first 200 utterances of each set must reproduce the committed fp32 scores
# (|d| <= 1e-4) for every model before any trimmed run. Read-only use of the deploybench clone; one GPU; nice.
set -uo pipefail
B=${SCRATCH_BASE:?set SCRATCH_BASE to the server base directory}
D=$B/dfv_rev
source $B/env.sh >/dev/null 2>&1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 DATA_ROOT=$B/data/deepfake
G=${1:-1}
cd $D
for m in aasist aasist_l rawnet2 tssdnet xlsr_aasist; do
  CUDA_VISIBLE_DEVICES=$G nice -n 10 python -B f_vad_score.py --model $m --gate > $D/logs/gate_$m.log 2>&1
  echo "$m exit $?  $(grep -E 'GATE (PASSED|FAILED)|GATE FAIL' $D/logs/gate_$m.log | tail -1)"
done
echo "GATES_DONE $(date -Is)"
