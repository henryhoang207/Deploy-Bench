#!/usr/bin/env bash
# One on-device measurement cell: push batch -> run app batch mode -> pull results.
#   ./run_cell.sh <adb-serial> <model> <runtime> <precision> <wav_dir>
set -eu
SERIAL=$1; MODEL=$2; RUNTIME=$3; PRECISION=$4; WAVS=$5
PKG=org.deploybench.app
DEV=/sdcard/deploybench
adb -s "$SERIAL" shell mkdir -p $DEV/in $DEV/out
# thermal gate (methodology C3/P4): refuse to start above 35 C
TEMP=$(adb -s "$SERIAL" shell dumpsys battery | awk '/temperature/{print $2}')
if [ "${TEMP:-0}" -gt 350 ]; then echo "battery ${TEMP}dC > 35.0C — cooling gate, aborting"; exit 2; fi
adb -s "$SERIAL" push "$WAVS"/. $DEV/in/ > /dev/null
# push exported model if present locally (idempotent)
MODEL_FILE=exports/${MODEL}_${PRECISION}.onnx
if [ -f "$MODEL_FILE" ]; then
  adb -s "$SERIAL" shell mkdir -p $DEV/models
  adb -s "$SERIAL" push "$MODEL_FILE" $DEV/models/ > /dev/null
fi
printf '{"model":"%s","runtime":"%s","precision":"%s","warmup":20,"reps":200}' \
  "$MODEL" "$RUNTIME" "$PRECISION" > /tmp/db_config.json
adb -s "$SERIAL" push /tmp/db_config.json $DEV/config.json > /dev/null
adb -s "$SERIAL" shell am start -n $PKG/.BatchActivity --ez batch true
# poll for completion marker
for i in $(seq 1 720); do
  adb -s "$SERIAL" shell test -f $DEV/out/DONE && break; sleep 5
done
OUT=results/device/${SERIAL}_${MODEL}_${RUNTIME}_${PRECISION}
mkdir -p "$OUT"
adb -s "$SERIAL" pull $DEV/out/ "$OUT" > /dev/null
adb -s "$SERIAL" shell rm -rf $DEV/out $DEV/in
echo "cell -> $OUT"
