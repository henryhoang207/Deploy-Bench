#!/usr/bin/env bash
# W2 (PREREGISTRATION_ROUND2.md Sec. 4): the eleven AASIST-L runs, launched AFTER the round2-freeze tag, in parallel
# (the recipe's single-process data loading makes one run ~15 h; parallel runs keep the recipe unchanged).
# Each run refuses an existing directory; nothing is restarted. Ends with TRAIN_RUNS_DONE fails=<n>.
set -u
H=/mnt/data8tb/henry
export DATA_ROOT=$H/data/deepfake CC=$H/envs/cc/bin/gcc PATH=$H/tools/bin:$PATH OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
source $H/envs/r2/bin/activate
cd $H/deploybench
git describe --tags --exact-match HEAD 2>/dev/null | grep -q round2-freeze || git merge-base --is-ancestor round2-freeze HEAD \
  || { echo "REFUSED: the clone does not contain tag round2-freeze"; exit 3; }
OUT=$H/r2_runs
mkdir -p $OUT/logs
i=0; pids=(); names=()
for spec in "r2 1" "trim 1" "r2 2" "trim 2" "r2 3" "trim 3" "r2 4" "trim 4" "r2 5" "trim 5" "r2 1234"; do
  set -- $spec; arm=$1; seed=$2
  name=aasist_l_${arm}_s${seed}
  trim=none; [ "$arm" = trim ] && trim=vad35
  gpu=$((i % 2)); i=$((i + 1))
  CUDA_VISIBLE_DEVICES=$gpu python -B train/aasist_r2_train.py --seed "$seed" --trim "$trim" \
      --database-path "$DATA_ROOT/LA" --out-dir "$OUT/$name" > "$OUT/logs/$name.log" 2>&1 &
  pids+=($!); names+=("$name")
  echo "started $name on GPU $gpu (pid $!) $(date -Is)"
done
fails=0
for k in "${!pids[@]}"; do
  if wait "${pids[$k]}"; then echo "finished ${names[$k]} $(date -Is)"; else echo "FAILED ${names[$k]} $(date -Is)"; fails=$((fails + 1)); fi
done
echo "TRAIN_RUNS_DONE $(date -Is) fails=$fails"
