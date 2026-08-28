#!/usr/bin/env bash
# Full nested campaign with cooldown gates. Edit the arrays, run overnight.
set -u
PHONES=(SERIAL1 SERIAL2 SERIAL3)
# Roster frozen at 5 (D30). xlsr_aasist is NOT here: at 315M it is expected to
# fail to fit on every tier, and that "N/A -- does not fit" row is produced by
# the conversion/fit attempt, not by a latency campaign.
MODELS=(tssdnet aasist aasist_l rawnet2)
RUNTIMES=(ort executorch)
PRECISIONS=(fp32 fp16 int8)
WAVS=${1:?wav batch dir}
for s in "${PHONES[@]}"; do for m in "${MODELS[@]}"; do
  for r in "${RUNTIMES[@]}"; do for p in "${PRECISIONS[@]}"; do
    # run_cell.sh exits 2 ONLY for the >35 C thermal gate; that is the one
    # condition worth waiting out. Retrying on every non-zero exit turned a
    # crashed cell into an infinite "cooling" loop overnight.
    while true; do
      ./scripts/run_cell.sh "$s" "$m" "$r" "$p" "$WAVS" && break
      rc=$?
      if [ "$rc" -ne 2 ]; then
        echo "FAILED (exit $rc): $s $m $r $p -- recording and moving on" \
          | tee -a campaign_failures.log
        break
      fi
      echo "cooling $s ..."; sleep 120
    done
  done; done
done; done
