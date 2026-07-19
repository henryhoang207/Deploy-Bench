#!/usr/bin/env bash
# Energy per inference via batterystats deltas over a 500-inference batch.
#   ./energy_batch.sh <adb-serial> <model> <runtime> <precision> <wav_dir>
set -eu
SERIAL=$1
adb -s "$SERIAL" shell dumpsys batterystats --reset
adb -s "$SERIAL" shell dumpsys battery unplug   # simulate unplugged while on USB
./scripts/run_cell.sh "$@"
adb -s "$SERIAL" shell dumpsys batterystats --charged > \
  results/device/${SERIAL}_$2_$3_$4/batterystats.txt
adb -s "$SERIAL" shell dumpsys battery reset
echo "NOTE: parse Estimated power use section; ODPM variant (rooted): dumpsys android.hardware.power.stats"
