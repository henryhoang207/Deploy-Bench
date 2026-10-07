#!/usr/bin/env bash
# W1 completion after DEV-9 amendment 3 and the stop of queue A (2026-10-07). envs/r2, the MD5-verified decoder.
#   GPU 0: wait for the orphaned xlsr_mamba mlaad5 scorer (PID $1) to exit; accept its .partial only with all 494,853 rows;
#          then re-score xlsr_mamba on DF21 verified (the first run was quarantined: libsndfile 1.2.0 sign flip).
#   GPU 1: wait for queue B (pascu_lr ... df21_verified) to finish; then score pascu_lr on mlaad5 straight away, so the
#          training scheduler cannot take the GPU first (one training run + Pascu do not fit in 24 GB).
# Ends with W1_FINISH_DONE.
set -u
H=/mnt/data8tb/henry
export DATA_ROOT=$H/data/deepfake CC=$H/envs/cc/bin/gcc PATH=$H/tools/bin:$PATH OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
source $H/envs/r2/bin/activate
cd $H/deploybench
git merge-base --is-ancestor round2-freeze HEAD || { echo "REFUSED: no round2-freeze"; exit 3; }
grep -q "_flac_read_verified" score.py || { echo "REFUSED: score.py lacks the verified FLAC decoder"; exit 3; }
PID_XM=${1:?pid of the xlsr_mamba mlaad5 scorer}

gpu0() {
  while kill -0 "$PID_XM" 2>/dev/null; do sleep 60; done
  local f=results/round2/scores/xlsr_mamba/s0/mlaad5.csv
  local n=$(( $(wc -l < $f.partial) - 1 ))
  if [ "$n" = 494853 ]; then mv $f.partial $f; echo "xlsr_mamba mlaad5 accepted ($n rows) $(date -Is)"
  else echo "xlsr_mamba mlaad5 REJECTED: $n rows $(date -Is)"; fi
  local out=results/round2/scores/xlsr_mamba/s0/df21_verified.csv
  if CUDA_VISIBLE_DEVICES=0 python -B score.py --model xlsr_mamba --dataset asvspoof21_df_eval --device cuda --batch 16 \
       --progress 50000 --out $out.partial > $H/logs/r2_xlsr_mamba_df21_rescore.log 2>&1; then
    mv $out.partial $out; echo "xlsr_mamba df21 re-scored $(date -Is): $(tail -3 $H/logs/r2_xlsr_mamba_df21_rescore.log | tr '\n' ' ')"
  else echo "xlsr_mamba df21 re-score FAILED $(date -Is)"; fi
}

gpu1() {
  until grep -q QUEUE_B_DONE $H/logs/r2_queueB.log; do sleep 30; done
  local out=results/round2/scores/pascu_lr/s0/mlaad5.csv
  [ -e $out ] && { echo "pascu mlaad5 exists"; return; }
  if CUDA_VISIBLE_DEVICES=1 python -B score.py --model pascu_lr --checkpoint $H/r2_runs/pascu/head_train.npz --dataset mlaad \
       --device cuda --batch 1 --progress 50000 --out $out.partial > $H/logs/r2_pascu_lr_mlaad5.log 2>&1; then
    mv $out.partial $out; echo "pascu_lr mlaad5 done $(date -Is)"
  else echo "pascu_lr mlaad5 FAILED $(date -Is)"; fi
}

gpu0 & gpu1 & wait
echo "W1_FINISH_DONE $(date -Is)"
