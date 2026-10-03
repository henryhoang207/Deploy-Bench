#!/usr/bin/env bash
# 30-min continuous inference soak: latency drift + battery temperature curve.
#   ./soak.sh <adb-serial> <model> <runtime> <precision> <wav_dir>
set -eu
SERIAL=$1; MODEL=$2; RUNTIME=$3; PRECISION=$4; WAVS=$5
PKG=org.deploybench.app
DEV=/sdcard/deploybench

# Same no-default rule as run_cell.sh: BatchActivity refuses to start without
# input_samples, and guessing it silently changes what audio the model sees.
case "$MODEL" in
  tssdnet|res_tssdnet)                 INPUT_SAMPLES=96000 ;;
  aasist|aasist_l|rawnet2|xlsr_aasist) INPUT_SAMPLES=64600 ;;
  *) echo "unknown model '$MODEL': refusing to guess input_samples" >&2; exit 3 ;;
esac

# Same artifact resolution as run_cell.sh (DEV-7 for executorch).
case "$RUNTIME" in
  ort)        MODEL_FILE=exports/${MODEL}_${PRECISION}.onnx ;;
  # DEV-7: the .pte is resolved by deploybench.artifacts.et_artifact (passed desktop
  # fp32 gate; int8 = the canonical seed only; sha256 = sidecar), or exit 3.
  executorch) MODEL_FILE=$(${PYTHON:-python} -m deploybench.artifacts et "$MODEL" "$PRECISION") || exit 3 ;;
  *) echo "unknown runtime '$RUNTIME'" >&2; exit 3 ;;
esac

if [ ! -f "$MODEL_FILE" ]; then echo "missing $MODEL_FILE" >&2; exit 3; fi

OUT=results/device/soak_${SERIAL}_${MODEL}_${RUNTIME}_${PRECISION}
mkdir -p "$OUT"

adb -s "$SERIAL" shell appops set --uid $PKG MANAGE_EXTERNAL_STORAGE allow >/dev/null 2>&1 || true
adb -s "$SERIAL" shell svc power stayon usb >/dev/null 2>&1 || true
adb -s "$SERIAL" shell rm -rf $DEV/out $DEV/in
adb -s "$SERIAL" shell mkdir -p $DEV/in $DEV/out $DEV/models

adb -s "$SERIAL" push "$WAVS"/. $DEV/in/ > /dev/null
if [ "$RUNTIME" = executorch ]; then
  adb -s "$SERIAL" push "$MODEL_FILE" $DEV/models/${MODEL}_${PRECISION}.pte > /dev/null
else
  adb -s "$SERIAL" push "$MODEL_FILE" $DEV/models/ > /dev/null
fi
if [ -f "${MODEL_FILE}.data" ]; then
  adb -s "$SERIAL" push "${MODEL_FILE}.data" $DEV/models/ > /dev/null
fi

# soak_minutes > 0 selects the soak leg; the MLPerf latency leg is skipped, so
# min_queries/min_seconds are not required here.
printf '{"model":"%s","runtime":"%s","precision":"%s","input_samples":%s,"soak_minutes":30}' \
  "$MODEL" "$RUNTIME" "$PRECISION" "$INPUT_SAMPLES" > /tmp/db_config.json
adb -s "$SERIAL" push /tmp/db_config.json $DEV/config.json > /dev/null

adb -s "$SERIAL" shell am force-stop $PKG
adb -s "$SERIAL" shell am start -n $PKG/.BatchActivity --ez soak true > /dev/null

for _ in $(seq 1 180); do   # 30 min, 10 s cadence
  # tr -d '\r': adb on Windows returns CRLF and the CR stays attached to $2,
  # which would embed a stray carriage return in every row of the temp curve.
  T=$(adb -s "$SERIAL" shell dumpsys battery | awk '/temperature/{print $2}' | tr -d '\r')
  echo "$(date +%s),$T" >> "$OUT/battery_temp.csv"; sleep 10
done

DONE=0
for _ in $(seq 1 120); do   # up to 10 more min for the app to finish and flush
  if adb -s "$SERIAL" shell test -f $DEV/out/DONE; then DONE=1; break; fi
  sleep 5
done
if [ "$DONE" -ne 1 ]; then
  echo "TIMEOUT: no $DEV/out/DONE after the soak window ($SERIAL $MODEL)" >&2
  adb -s "$SERIAL" shell am force-stop $PKG || true
  exit 4
fi

adb -s "$SERIAL" pull $DEV/out/ "$OUT" > /dev/null
adb -s "$SERIAL" shell rm -rf $DEV/out $DEV/in
echo "soak -> $OUT"
