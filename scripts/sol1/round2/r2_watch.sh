#!/usr/bin/env bash
# Laptop side of the round-2 run -> verify -> next loop: poll sol1's r2_events.py every 5 minutes and exit (which wakes
# the Claude session that launched it in the background) on the first event, after 12 SSH or script errors in a row, or
# after a heartbeat period with no event (default 6 h; first argument in seconds).
#     R2_HOST=user@host bash scripts/sol1/round2/r2_watch.sh [heartbeat_s]
K="-i $HOME/.ssh/id_ed25519_sol1 -o IdentitiesOnly=yes -o ConnectTimeout=15 -o BatchMode=yes"
S=${R2_HOST:?set R2_HOST to the user@host SSH login of the scoring server}
HEARTBEAT_S=${1:-21600}
t0=$(date +%s); errs=0; last=""
while :; do
  out=$(timeout 120 ssh $K -p 8000 $S 'python3 /mnt/data8tb/henry/xfer/r2_events.py' 2>&1); rc=$?
  case $rc in
    10) echo "$out"; exit 0 ;;
    0) errs=0 ;;
    *) errs=$((errs + 1)); last="rc=$rc $out"
       if [ $errs -ge 12 ]; then echo "WATCHER_ERROR (12 in a row): $last"; exit 0; fi ;;
  esac
  if [ $(( $(date +%s) - t0 )) -ge "$HEARTBEAT_S" ]; then
    echo "HEARTBEAT: no event for $(( HEARTBEAT_S / 3600 )) h $(date -Is)"; exit 0
  fi
  sleep 300
done
