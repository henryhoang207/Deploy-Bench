#!/bin/bash
# D47 / DEVIATIONS.md DEV-6: score the frozen MLAAD-5 protocol ($DATA_ROOT/mlaad5/protocol.csv) with
# the main-matrix artifacts, at the frozen taus (nothing re-calibrated):
#   fp32 / pytorch / cuda (TF32 off via load_model): all 5 models
#   fp16 / ort: aasist, aasist_l, tssdnet (exports/<m>_fp16.onnx) + rawnet2 = the D46 repaired artifact
#   int8 / ort: rawnet2 D39 canonical (r09 seed 2), ledger-verified
#   fp16 / executorch: tssdnet (deploybench.artifacts.et_artifact)
# Each cell: K manifest shards, joined in protocol order; refuses unless every trial appears
# exactly once; refuses to overwrite. Artifacts are verified before scoring.
set -uo pipefail
B=/mnt/data8tb/henry
R=$B/deploybench
PY=$B/envs/db310/bin/python
PE=$B/envs/et/bin/python
V=$R/vendor
export DATA_ROOT=$B/data/deepfake OMP_NUM_THREADS=4
OUT=$R/results/phase_2_conversion_app/mlaad5
CS=$R/results/phase_2_conversion_app/conversion_sensitivity
DS=mlaad
cd $R
echo "HEAD $(git rev-parse --short HEAD)  $(date -Is)"
P=$DATA_ROOT/mlaad5/protocol.csv
$PY - <<EOF || exit 1
import json, sys
sys.path.insert(0, "$R")
from deploybench.manifests import file_sha256
s = json.load(open("$DATA_ROOT/mlaad5/protocol_summary.json"))
got = file_sha256("$P")
if got != s["protocol_sha256"]:
    sys.exit(f"protocol.csv sha256 {got} != frozen {s['protocol_sha256']}")
print("protocol verified", got[:16], s["rows"], "rows")
EOF

ck() {
  case $1 in
    aasist)      echo $V/aasist/models/weights/AASIST.pth ;;
    aasist_l)    echo $V/aasist/models/weights/AASIST-L.pth ;;
    rawnet2)     echo $V/asvspoof2021_baselines/LA/Baseline-RawNet2/pre_trained_DF_RawNet2.pth ;;
    tssdnet)     ls $V/end-to-end-synthetic-speech-detection/pretrained/Res_TSSDNet*.pth ;;
    xlsr_aasist) echo $V/SSL_Anti-spoofing/Best_LA_model_for_DF.pth ;;
  esac
}

cell() {   # tag model outname K python script args...   (script gets --manifest/--out appended)
  local tag=$1 m=$2 name=$3 K=$4; shift 4
  local O=$OUT/$m/$name S=$B/xfer/mlaad_$tag
  if [ -e "$O" ]; then echo "[$tag] REFUSING to overwrite $O"; return 1; fi
  mkdir -p $OUT/$m $S
  $PY - <<EOF || return 1
from deploybench.data import protocol_iter
ids = [u for u, _, _ in protocol_iter("$DS")]
for i in range($K):
    open(f"$S/shard{i}.txt", "w").write("\n".join(ids[i::$K]) + "\n")
open("$S/order.txt", "w").write("\n".join(ids) + "\n")
print("[$tag] trials", len(ids))
EOF
  echo "[$tag] start $(date -Is)"
  for i in $(seq 0 $((K-1))); do
    "$@" --manifest $S/shard$i.txt --out $S/scores_shard$i.csv > $B/logs/mlaad_${tag}_shard$i.log 2>&1 &
  done
  wait
  for i in $(seq 0 $((K-1))); do grep -h "wrote" $B/logs/mlaad_${tag}_shard$i.log || echo "[$tag] shard $i FAILED"; done
  $PY - <<EOF || return 1
import csv, sys
order = open("$S/order.txt").read().split()
rows, header = {}, None
for i in range($K):
    with open(f"$S/scores_shard{i}.csv", newline="") as f:
        rd = csv.reader(f); h = next(rd)
        assert header in (None, h), "header mismatch"; header = h
        for r in rd:
            if r[0] in rows: sys.exit(f"duplicate {r[0]}")
            rows[r[0]] = r
missing = [u for u in order if u not in rows]
if missing or len(rows) != len(order):
    sys.exit(f"join incomplete: missing {len(missing)}, rows {len(rows)} vs {len(order)}")
with open("$O", "w", newline="") as f:
    w = csv.writer(f); w.writerow(header)
    for u in order: w.writerow(rows[u])
print("[$tag] joined", len(order), "->", "$O")
EOF
  echo "[$tag] done $(date -Is)"
}

fp32() {   # model gpu K
  local m=$1 g=$2 K=$3
  ( export CUDA_VISIBLE_DEVICES=$g          # a subshell, so the GPU pin cannot leak
    cell fp32_$m $m scores_fp32_pytorch_mlaad.csv $K \
      $PY score.py --model $m --checkpoint "$(ck $m)" --dataset $DS --device cuda --batch 16 \
      --precision fp32 --runtime pytorch --progress 20000 )
}

ort() {    # tag model onnx precision outname
  cell $1 $2 $5 6 $PY score_onnx.py --onnx $3 --model $2 --dataset $DS --precision $4 \
    --runtime ort --intra-threads 2 --progress 20000
}

verify_ort() {
  $PY - <<EOF || { echo "ARTIFACT CHECK FAILED"; return 1; }
import json, os, sys
sys.path.insert(0, "$R")
from deploybench.manifests import file_sha256
for m in ("aasist", "aasist_l", "tssdnet"):
    log = json.load(open(f"results/phase_2_conversion_app/main_matrix/{m}/conversion_log_{m}_ort_fp16.json"))
    p = f"exports/{m}_fp16.onnx"
    if not log.get("converts") or os.path.getsize(p) != log["shipped_bytes"]:
        sys.exit(f"{p}: size/converts disagree with its conversion log")
    print(m, "fp16", file_sha256(p)[:16])
a = json.load(open("results/phase_2_conversion_app/main_matrix/rawnet2/fp16_repaired_d46/artifact.json"))
if file_sha256("exports/repair_d46/rawnet2_fp16.onnx") != a["sha256"]:
    sys.exit("repaired rawnet2 fp16 sha256 differs from artifact.json")
led = [json.loads(l) for l in open("$CS/ledger_rawnet2.jsonl") if l.strip()]
e = [x for x in led if x["recipe_id"] == "r09" and x["calib_seed"] == 2]
if len(e) != 1 or e[0]["dst_sha256"].get("rawnet2_int8.onnx") != file_sha256("$CS/rawnet2/r09_seed2/rawnet2_int8.onnx"):
    sys.exit("rawnet2 int8 canonical differs from the ledger")
print("all ORT artifacts verified")
EOF
}

# GPU: four small models two per GPU, then XLS-R alone on the GPU with more free memory.
( fp32 aasist 0 3; fp32 tssdnet 0 3 ) &
( fp32 aasist_l 1 3; fp32 rawnet2 1 3 ) &
wait
G=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)
fp32 xlsr_aasist $G 1

# CPU: artifacts two at a time (12 shards x 2 threads).
verify_ort || exit 1
ort fp16_aasist   aasist   exports/aasist_fp16.onnx   fp16 scores_fp16_ort_mlaad.csv &
ort fp16_aasist_l aasist_l exports/aasist_l_fp16.onnx fp16 scores_fp16_ort_mlaad.csv &
wait
ort fp16_tssdnet  tssdnet  exports/tssdnet_fp16.onnx  fp16 scores_fp16_ort_mlaad.csv &
ort fp16r_rawnet2 rawnet2  exports/repair_d46/rawnet2_fp16.onnx fp16 scores_fp16repaired_ort_mlaad.csv &
wait
ort int8_rawnet2  rawnet2  $CS/rawnet2/r09_seed2/rawnet2_int8.onnx int8 scores_int8_ort_mlaad.csv &
PTE=$($PE -m deploybench.artifacts et tssdnet fp16) && [ -n "$PTE" ] && \
  cell et_fp16_tssdnet tssdnet scores_fp16_executorch_mlaad.csv 6 \
    $PE score_pte.py --pte $PTE --model tssdnet --dataset $DS --precision fp16 --progress 20000 &
wait
echo "MLAAD_SCORING_DONE $(date -Is)"
