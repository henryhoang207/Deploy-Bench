#!/usr/bin/env bash
# Res-TSSDNet vertical-slice driver for the sol1 Ubuntu box (bash twin of
# pilot_tssdnet.ps1). Same stages, same artifacts, same reproduction anchor.
#
#   bash scripts/pilot_tssdnet.sh --data-root /data/DF_Dataset --stage check
#   bash scripts/pilot_tssdnet.sh --data-root /data/DF_Dataset --stage all --device cuda
#
# Stages: check | smoke | repro | report | convert | parity | all
# NOTE the author checkpoint filename contains '%' characters — every path
# expansion below is quoted for that reason.
set -euo pipefail

DATA_ROOT_ARG=""
STAGE="all"
DEVICE="cpu"
PYTHON="${PYTHON:-python}"
CKPT=""
REFREEZE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)  DATA_ROOT_ARG="$2"; shift 2 ;;
    --stage)      STAGE="$2";         shift 2 ;;
    --device)     DEVICE="$2";        shift 2 ;;
    --python)     PYTHON="$2";        shift 2 ;;
    --checkpoint) CKPT="$2";          shift 2 ;;
    --refreeze)   REFREEZE="--refreeze"; shift ;;
    -h|--help)    sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$REPO_ROOT")"
# Results moved under the repo on 2026-07-27 so they are version-controlled with
# the code that produced them.
P1="$REPO_ROOT/results/phase_1_reproduction"
P2="$REPO_ROOT/results/phase_2_conversion_app"
EXPORTS="$REPO_ROOT/exports"
VENDOR_DIR="$REPO_ROOT/vendor/end-to-end-synthetic-speech-detection"
mkdir -p "$P1" "$P2" "$EXPORTS"

if [[ -n "$DATA_ROOT_ARG" ]]; then export DATA_ROOT="$DATA_ROOT_ARG"; fi
: "${DATA_ROOT:?set --data-root or export DATA_ROOT}"
export PYTHONPATH="$REPO_ROOT"
cd "$REPO_ROOT"

# Reproduction anchor: the author's README reports 1.6590% on Linux (1.64% on
# Windows, 2021 soundfile FLAC decoding). This box IS Linux, so 1.6590% is the
# like-for-like target here.
PUBLISHED="1.6590"
PUB_NOTE="Linux pretrained Res-TSSDNet, ghua-ac README @36c9705"

say() { printf '\n=== %s\n' "$*"; }

run() {  # run <label> <python-args...>
  local label="$1"; shift
  say "$label"
  printf '%s %s\n' "$PYTHON" "$*"
  if ! "$PYTHON" "$@"; then
    echo "$label FAILED — stopping; later stages would build on a broken result." >&2
    exit 1
  fi
}

resolve_ckpt() {
  if [[ -n "$CKPT" ]]; then echo "$CKPT"; return; fi
  local pre="$VENDOR_DIR/pretrained"
  if [[ ! -d "$pre" ]]; then
    echo "no $pre — clone the vendor repo first:" >&2
    echo "  git clone https://github.com/ghua-ac/end-to-end-synthetic-speech-detection \"$VENDOR_DIR\"" >&2
    exit 1
  fi
  local hit
  hit="$(find "$pre" -maxdepth 1 -name 'Res_TSSDNet*.pth' -print -quit)"
  if [[ -z "$hit" ]]; then echo "no Res_TSSDNet*.pth under $pre" >&2; exit 1; fi
  echo "$hit"
}

all=0
if [[ "$STAGE" == "all" ]]; then all=1; fi

# ------------------------------------------------------------------ 0 check
if [[ $all -eq 1 || "$STAGE" == "check" ]]; then
  say "0. DATA_ROOT layout check (DATA_ROOT=$DATA_ROOT)"
  "$PYTHON" -m deploybench.data --check --datasets asvspoof19_dev,asvspoof19_eval || true
  echo
  echo "If any line says MISS, PASTE THIS OUTPUT BACK to Claude before running later stages."
  echo "REGISTRY still holds layout guesses; do not hand-edit it."
  if [[ "$STAGE" == "check" ]]; then exit 0; fi
fi

CKPT_PATH="$(resolve_ckpt)"
echo "checkpoint: $CKPT_PATH"

# ------------------------------------------------------------------ 1 smoke
if [[ $all -eq 1 || "$STAGE" == "smoke" ]]; then
  run "1. smoke test (200 dev clips)" \
      scripts/smoke_tssdnet.py --device "$DEVICE" --checkpoint "$CKPT_PATH" \
      --n 200 --out "$P1/smoke_tssdnet.json"
  if [[ "$STAGE" == "smoke" ]]; then exit 0; fi
fi

DEV_CSV="$P1/scores_dev_tssdnet.csv"
EVAL_CSV="$P1/scores_repro_tssdnet.csv"
TAU_JSON="$P1/tau_tssdnet.json"

# ------------------------------------------------------------------ 2 repro
if [[ $all -eq 1 || "$STAGE" == "repro" ]]; then
  run "2a. score 19LA DEV (tau calibration source, ~24.8k clips)" \
      score.py --model tssdnet --checkpoint "$CKPT_PATH" \
      --dataset asvspoof19_dev --device "$DEVICE" --precision fp32 \
      --runtime pytorch --condition clean --batch 32 --progress 2000 \
      --out "$DEV_CSV"

  run "2b. score 19LA EVAL (reproduction gate, ~71.2k clips)" \
      score.py --model tssdnet --checkpoint "$CKPT_PATH" \
      --dataset asvspoof19_eval --device "$DEVICE" --precision fp32 \
      --runtime pytorch --condition clean --batch 32 --progress 5000 \
      --out "$EVAL_CSV"
  if [[ "$STAGE" == "repro" ]]; then exit 0; fi
fi

# ----------------------------------------------------------------- 3 report
if [[ $all -eq 1 || "$STAGE" == "report" ]]; then
  run "3. EER + gate verdict + tau freeze" \
      analysis/repro_report.py --model tssdnet \
      --eval-scores "$EVAL_CSV" --dev-scores "$DEV_CSV" \
      --published "$PUBLISHED" --published-note "$PUB_NOTE" --gate 0.15 \
      --out-report "$P1/repro_report_tssdnet.md" --out-tau "$TAU_JSON" \
      ${REFREEZE:+$REFREEZE}
  if [[ "$STAGE" == "report" ]]; then exit 0; fi
fi

# ---------------------------------------------------------------- 4 convert
if [[ $all -eq 1 || "$STAGE" == "convert" ]]; then
  CALIB_LIST="$REPO_ROOT/configs/calibration_500_tssdnet.txt"
  run "4a. calibration manifest (250 bona + 250 spoof, 19LA DEV)" \
      scripts/make_parity_manifest.py --dataset asvspoof19_dev \
      --n-per-class 250 --seed 20260725 --out "$CALIB_LIST"

  run "4b. ONNX fp32 export (single-score adapter, D13)" \
      convert/to_onnx.py --model tssdnet --precision fp32 \
      --checkpoint "$CKPT_PATH" --out "$EXPORTS"

  say "4c. int8 PTQ (S8S8, per-channel, QDQ; D17/D27)"
  echo "OPEN ITEM: --keep-io-fp32 is wired to extra_options{QuantizeBias:False},"
  echo "which is about bias, not graph I/O. Read graph_input_type in the ptq log"
  echo "(1=FLOAT) before calling the input fp32. See PILOT_tssdnet.md step 5."
  CALIB_DIR="$DATA_ROOT/LA/ASVspoof2019_LA_dev/flac"
  if [[ -d "$CALIB_DIR" ]]; then
    run "4c. int8 PTQ" \
        convert/ptq.py --onnx "$EXPORTS/tssdnet_fp32.onnx" \
        --calib-dir "$CALIB_DIR" --calib-manifest "$CALIB_LIST" \
        --input-samples 96000 --calib-method minmax --keep-io-fp32
  else
    echo "SKIPPED: $CALIB_DIR not found — settle the layout first (stage check)." >&2
  fi

  run "4d. ExecuTorch export (XNNPACK)" \
      convert/to_executorch.py --model tssdnet --checkpoint "$CKPT_PATH" \
      --out "$EXPORTS"

  cp -f "$EXPORTS"/conversion_log_*.json "$P2"/ 2>/dev/null || true
  cp -f "$EXPORTS"/*_ptq_log.json        "$P2"/ 2>/dev/null || true
  if [[ "$STAGE" == "convert" ]]; then exit 0; fi
fi

# ----------------------------------------------------------------- 5 parity
if [[ $all -eq 1 || "$STAGE" == "parity" ]]; then
  MANIFEST="$REPO_ROOT/configs/parity_manifest_tssdnet.txt"
  if [[ ! -f "$MANIFEST" ]]; then
    run "5a. parity manifest (500 bona + 500 spoof, 19LA EVAL)" \
        scripts/make_parity_manifest.py --dataset asvspoof19_eval \
        --n-per-class 500 --seed 20260725 --out "$MANIFEST"
  else
    say "5a. parity manifest already exists — REUSING (it is frozen)"
  fi

  REF="$P2/parity_ref_tssdnet_fp32_pytorch.csv"
  run "5b. reference scores (PyTorch fp32) over the manifest" \
      score.py --model tssdnet --checkpoint "$CKPT_PATH" \
      --dataset asvspoof19_eval --manifest "$MANIFEST" --device cpu \
      --precision fp32 --runtime pytorch --out "$REF"

  TAU_FLAT="$P2/tau_tssdnet_flat.json"
  "$PYTHON" -c "import json,sys; json.dump(json.load(open(sys.argv[1]))['taus'], open(sys.argv[2],'w'))" \
      "$TAU_JSON" "$TAU_FLAT"

  for prec in fp32 int8; do
    ONNX="$EXPORTS/tssdnet_${prec}.onnx"
    if [[ ! -f "$ONNX" ]]; then echo "skip $prec — $ONNX not present"; continue; fi
    TEST="$P2/parity_test_tssdnet_${prec}_onnx.csv"
    run "5c. ONNX $prec scores over the manifest" \
        score_onnx.py --onnx "$ONNX" --model tssdnet \
        --dataset asvspoof19_eval --manifest "$MANIFEST" \
        --precision "$prec" --runtime onnx --out "$TEST"

    say "5d. parity gate: ONNX $prec vs PyTorch fp32"
    if "$PYTHON" tests/parity.py --ref "$REF" --test "$TEST" --taus "$TAU_FLAT" \
         | tee "$P2/parity_report_tssdnet_${prec}.json"; then
      echo "PARITY PASS for $prec (zero decision flips)"
    else
      echo "PARITY FAILED for $prec — decision flips are non-zero." >&2
      echo "TSSDNet is the clean-converter control: this indicts OUR pipeline, not the ops." >&2
    fi
  done
fi

say "done. Artifacts:"
echo "  $P1"
echo "  $P2"
echo "  $EXPORTS"
