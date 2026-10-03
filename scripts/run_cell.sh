#!/usr/bin/env bash
# One on-device measurement cell: push batch -> run app batch mode -> pull results.
#   ./run_cell.sh <adb-serial> <model> <runtime> <precision> <wav_dir>
#
# Exit codes are load-bearing -- campaign.sh retries ONLY on 2:
#   0 ok | 2 thermal gate (cool down and retry) | 3 config/artifact error | 4 timeout
set -eu
SERIAL=$1; MODEL=$2; RUNTIME=$3; PRECISION=$4; WAVS=$5
PKG=org.deploybench.app
DEV=/sdcard/deploybench

# --- input_samples, per model. NEVER defaulted: 64,600 vs 96,000 silently
# changes what audio the model sees and a fixed-shape graph still runs.
# BatchActivity refuses to start without it.
case "$MODEL" in
  tssdnet|res_tssdnet)                 INPUT_SAMPLES=96000 ;;
  aasist|aasist_l|rawnet2|xlsr_aasist) INPUT_SAMPLES=64600 ;;
  *) echo "unknown model '$MODEL': refusing to guess input_samples" >&2; exit 3 ;;
esac

# --- runtime must be one the APK can actually honour. The app picks its runner
# from config.runtime (OrtScorer / EtScorer) and refuses anything else; an
# unresolvable executorch artifact fails here in seconds instead of after the
# 60-min poll (review B2: never run one runtime under another's label).
case "$RUNTIME" in
  ort)        MODEL_FILE=exports/${MODEL}_${PRECISION}.onnx ;;
  # DEV-7: the .pte is resolved by deploybench.artifacts.et_artifact (passed desktop
  # fp32 gate; int8 = the canonical seed only; sha256 = sidecar), or exit 3.
  executorch) MODEL_FILE=$(${PYTHON:-python} -m deploybench.artifacts et "$MODEL" "$PRECISION") || exit 3 ;;
  *) echo "unknown runtime '$RUNTIME'" >&2; exit 3 ;;
esac

# --- D23 MLPerf single-stream run rules. Supersedes the retired 200/20 scheme.
# "warmup" is deliberately absent: D23 records that it is not an MLPerf concept.
MIN_QUERIES=${MIN_QUERIES:-1024}
MIN_SECONDS=${MIN_SECONDS:-60}

# --- artifacts must exist BEFORE anything is pushed
if [ ! -f "$MODEL_FILE" ]; then
  echo "missing $MODEL_FILE (exports/ is gitignored -- copy or regenerate it)" >&2; exit 3
fi
if [ ! -d "$WAVS" ]; then echo "missing wav dir '$WAVS'" >&2; exit 3; fi

# --- host-side grants. Both are idempotent and both are required:
# MANAGE_EXTERNAL_STORAGE because $DEV is written by adb (the shell user), not by
# the app; stayon because a sleeping screen drops clocks and the latency
# distribution would then be measuring the screen state.
adb -s "$SERIAL" shell appops set --uid $PKG MANAGE_EXTERNAL_STORAGE allow >/dev/null 2>&1 || true
adb -s "$SERIAL" shell svc power stayon usb >/dev/null 2>&1 || true

# --- clear stale device state. A DONE marker left by a crashed cell would make
# the poll below return instantly and pull the PREVIOUS cell's results.
adb -s "$SERIAL" shell rm -rf $DEV/out $DEV/in
adb -s "$SERIAL" shell mkdir -p $DEV/in $DEV/out $DEV/models

# --- thermal gate (methodology C3/P4): refuse to start above 35 C
# tr -d '\r': adb on Windows returns CRLF, and awk leaves the CR attached to $2.
# "250\r" then fails the -gt below with "integer expression expected", which
# under set -e aborts the cell with exit 1 -- NOT the 2 that campaign.sh waits
# on, so a hot phone would be logged as a hard failure instead of cooling.
TEMP_START=$(adb -s "$SERIAL" shell dumpsys battery | awk '/temperature/{print $2}' | tr -d '\r')
if [ "${TEMP_START:-0}" -gt 350 ]; then
  echo "battery ${TEMP_START}dC > 35.0C - cooling gate, aborting"; exit 2
fi

adb -s "$SERIAL" push "$WAVS"/. $DEV/in/ > /dev/null
# The app opens models/<model>_<precision>.<onnx|pte>; an int8 .pte is named by
# its calibration seed on the host, so it is pushed under the app's name.
if [ "$RUNTIME" = executorch ]; then
  adb -s "$SERIAL" push "$MODEL_FILE" $DEV/models/${MODEL}_${PRECISION}.pte > /dev/null
else
  adb -s "$SERIAL" push "$MODEL_FILE" $DEV/models/ > /dev/null
fi
# ONNX external-data sidecar. fp32 exports keep their weights in a separate
# .onnx.data file; pushing only the .onnx lands a WEIGHTLESS model on the phone
# and the run produces plausible garbage.
if [ -f "${MODEL_FILE}.data" ]; then
  adb -s "$SERIAL" push "${MODEL_FILE}.data" $DEV/models/ > /dev/null
fi

printf '{"model":"%s","runtime":"%s","precision":"%s","input_samples":%s,"min_queries":%s,"min_seconds":%s}' \
  "$MODEL" "$RUNTIME" "$PRECISION" "$INPUT_SAMPLES" "$MIN_QUERIES" "$MIN_SECONDS" > /tmp/db_config.json
adb -s "$SERIAL" push /tmp/db_config.json $DEV/config.json > /dev/null

# --- force-stop before start. BatchActivity is launchMode=singleTask and does
# all its work in onCreate; starting it against a live instance delivers
# onNewIntent instead and the cell silently does not run.
adb -s "$SERIAL" shell am force-stop $PKG
adb -s "$SERIAL" shell am start -n $PKG/.BatchActivity --ez batch true > /dev/null

# --- poll for completion, and FAIL if it never arrives. The previous version
# fell out of the loop after 720 iterations, pulled an absent out/, and printed
# success -- a cell that reported fine and produced nothing.
DONE=0
for _ in $(seq 1 720); do          # 720 x 5 s = 60 min ceiling
  if adb -s "$SERIAL" shell test -f $DEV/out/DONE; then DONE=1; break; fi
  sleep 5
done
if [ "$DONE" -ne 1 ]; then
  echo "TIMEOUT: no $DEV/out/DONE after 60 min ($SERIAL $MODEL $RUNTIME $PRECISION)" >&2
  adb -s "$SERIAL" shell am force-stop $PKG || true
  exit 4
fi

TEMP_END=$(adb -s "$SERIAL" shell dumpsys battery | awk '/temperature/{print $2}' | tr -d '\r')

OUT=results/device/${SERIAL}_${MODEL}_${RUNTIME}_${PRECISION}
mkdir -p "$OUT"
adb -s "$SERIAL" pull $DEV/out/ "$OUT" > /dev/null
# Start/end temperature makes the thermal confound visible per cell rather than
# only gating the start (P4 open item 3).
printf '{"temp_start_dC":%s,"temp_end_dC":%s}' "${TEMP_START:-null}" "${TEMP_END:-null}" \
  > "$OUT/thermal.json"
adb -s "$SERIAL" shell rm -rf $DEV/out $DEV/in
echo "cell -> $OUT"
