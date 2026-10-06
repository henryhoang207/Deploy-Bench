#!/bin/bash
# DEV-8 F2: VAD-trimmed fp32 scoring of LA19 dev, LA19 eval and ITW for the five models, T = 35 dB.
# Three lanes on ONE GPU (argument 1, default 1), models sequential inside a lane; nice; read-only use of the clone.
# f_vad_score.py refuses to overwrite an existing output, so a re-launch cannot silently replace a result.
set -uo pipefail
B=${SCRATCH_BASE:?set SCRATCH_BASE to the server base directory}
D=$B/dfv_rev
source $B/env.sh >/dev/null 2>&1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 DATA_ROOT=$B/data/deepfake
G=${1:-1}
cd $D
[ -s $D/vad/vad_stats.csv ] || { echo "vad_stats.csv missing"; exit 1; }
lane() {
  for m in "$@"; do
    CUDA_VISIBLE_DEVICES=$G nice -n 10 python -B f_vad_score.py --model $m --bounds $D/vad/vad_stats.csv \
        --out-dir $D/scores --progress 10000 > $D/logs/vad_$m.log 2>&1
    echo "$m exit $?  $(date -Is)"
  done
}
echo "start $(date -Is) on GPU $G"
lane aasist aasist_l &
lane rawnet2 tssdnet &
lane xlsr_aasist &
wait
echo "VAD_SCORING_DONE $(date -Is)"
