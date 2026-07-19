#!/usr/bin/env bash
# 30-min continuous inference soak: latency drift + battery temperature curve.
#   ./soak.sh <adb-serial> <model> <runtime> <precision> <wav_dir>
set -eu
SERIAL=$1; MODEL=$2
OUT=results/device/soak_${SERIAL}_${MODEL}_$3_$4
mkdir -p "$OUT"
printf '{"model":"%s","runtime":"%s","precision":"%s","soak_minutes":30}' \
  "$2" "$3" "$4" > /tmp/db_config.json
adb -s "$SERIAL" push /tmp/db_config.json /sdcard/deploybench/config.json > /dev/null
adb -s "$SERIAL" push "$5"/. /sdcard/deploybench/in/ > /dev/null
adb -s "$SERIAL" shell am start -n org.deploybench.app/.BatchActivity --ez soak true
for i in $(seq 1 180); do   # 30 min, 10 s cadence
  T=$(adb -s "$SERIAL" shell dumpsys battery | awk '/temperature/{print $2}')
  echo "$(date +%s),$T" >> "$OUT/battery_temp.csv"; sleep 10
done
adb -s "$SERIAL" pull /sdcard/deploybench/out/ "$OUT" > /dev/null
echo "soak -> $OUT"
