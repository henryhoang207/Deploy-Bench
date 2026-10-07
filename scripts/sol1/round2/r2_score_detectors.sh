#!/usr/bin/env bash
# W1 (PREREGISTRATION_ROUND2.md Sec. 5-6): xlsr_mamba and pascu_lr, AFTER the round2-freeze tag, in envs/r2.
#   queue A (GPU 0): xlsr_mamba on la19_dev, la19_eval, itw, df21_verified, mlaad5; then pascu_lr on mlaad5
#   queue B (GPU 1): pascu fit (train) -> pipeline-gate fit (train + dev) -> pascu_lr_gate on la19_eval ->
#                    pascu_lr on la19_dev, la19_eval, itw, df21_verified
# Outputs: results/round2/scores/<model>/s0/<surface>.csv in this clone (untracked here; copied to the laptop and
# committed there). Every output refuses to be overwritten. Ends with DETECTORS_DONE fails=<n>.
set -u
H=/mnt/data8tb/henry
export DATA_ROOT=$H/data/deepfake CC=$H/envs/cc/bin/gcc PATH=$H/tools/bin:$PATH OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
source $H/envs/r2/bin/activate
cd $H/deploybench
git merge-base --is-ancestor round2-freeze HEAD || { echo "REFUSED: the clone does not contain tag round2-freeze"; exit 3; }
P=$H/r2_runs/pascu
mkdir -p $P
declare -A DS=([la19_dev]=asvspoof19_dev [la19_eval]=asvspoof19_eval [itw]=itw [df21_verified]=asvspoof21_df_eval [mlaad5]=mlaad)

sc() {  # gpu model surface checkpoint batch
  local gpu=$1 m=$2 s=$3 ck=$4 b=$5
  local out=results/round2/scores/$m/s0/$s.csv
  if [ -e "$out" ]; then echo "SKIP exists $out"; return 0; fi
  echo "[$m/$s] start $(date -Is)"
  local ckarg=(); [ -n "$ck" ] && ckarg=(--checkpoint "$ck")
  if CUDA_VISIBLE_DEVICES=$gpu python -B score.py --model "$m" "${ckarg[@]}" --dataset "${DS[$s]}" --device cuda \
       --batch "$b" --progress 20000 --out "$out.partial" > "$H/logs/r2_${m}_${s}.log" 2>&1; then
    mv "$out.partial" "$out"; echo "[$m/$s] done $(date -Is) $(tail -2 $H/logs/r2_${m}_${s}.log | tr '\n' ' ')"
  else
    echo "[$m/$s] FAILED $(date -Is) (see logs/r2_${m}_${s}.log)"; return 1
  fi
}

queueA() {
  local f=0
  for s in la19_dev la19_eval itw df21_verified mlaad5; do sc 0 xlsr_mamba $s "" 16 || f=$((f + 1)); done
  until [ -e $P/head_train.npz ] || [ -e $P/FIT_FAILED ]; do sleep 60; done
  [ -e $P/head_train.npz ] && { sc 0 pascu_lr mlaad5 $P/head_train.npz 1 || f=$((f + 1)); } || f=$((f + 1))
  echo "QUEUE_A_DONE fails=$f"; return $f
}

queueB() {
  local f=0
  if CUDA_VISIBLE_DEVICES=1 python -B train/pascu_fit.py --splits train --cache $P/feats --out $P/head_train.npz \
       > $H/logs/r2_pascu_fit_train.log 2>&1; then echo "pascu fit (train) done $(date -Is)"; else touch $P/FIT_FAILED; echo "pascu fit FAILED"; return 1; fi
  if CUDA_VISIBLE_DEVICES=1 python -B train/pascu_fit.py --splits train dev --cache $P/feats --out $P/head_traindev.npz \
       > $H/logs/r2_pascu_fit_traindev.log 2>&1; then echo "pascu gate fit (train+dev) done $(date -Is)"; else echo "pascu gate fit FAILED"; f=$((f + 1)); fi
  [ -e $P/head_traindev.npz ] && { sc 1 pascu_lr_gate la19_eval $P/head_traindev.npz 1 || f=$((f + 1)); }
  for s in la19_dev la19_eval itw df21_verified; do sc 1 pascu_lr $s $P/head_train.npz 1 || f=$((f + 1)); done
  echo "QUEUE_B_DONE fails=$f"; return $f
}

queueA > $H/logs/r2_queueA.log 2>&1 & pa=$!
queueB > $H/logs/r2_queueB.log 2>&1 & pb=$!
fa=0; fb=0; wait $pa || fa=$?; wait $pb || fb=$?
echo "DETECTORS_DONE $(date -Is) fails=$((fa + fb))"
