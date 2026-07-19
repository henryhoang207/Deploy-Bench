#!/usr/bin/env bash
# Full nested campaign with cooldown gates. Edit the arrays, run overnight.
set -u
PHONES=(SERIAL1 SERIAL2 SERIAL3)
MODELS=(tssdnet lcnn_lfcc aasist rawnet2 wavlm_trunc)
RUNTIMES=(ort executorch)
PRECISIONS=(fp32 fp16 int8)
WAVS=${1:?wav batch dir}
for s in "${PHONES[@]}"; do for m in "${MODELS[@]}"; do
  for r in "${RUNTIMES[@]}"; do for p in "${PRECISIONS[@]}"; do
    until ./scripts/run_cell.sh "$s" "$m" "$r" "$p" "$WAVS"; do
      echo "cooling $s ..."; sleep 120
    done
  done; done
done; done
