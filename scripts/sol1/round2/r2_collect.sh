#!/usr/bin/env bash
# Collect the round-2 artefacts from sol1 into this clone (run on the laptop, from C:/dev/deploybench). Read-only on sol1
# apart from the staging directory xfer/r2_collect. Every file is listed with its SHA-256 on sol1 and verified here.
#   results/round2/scores/<model>/s<seed>/<surface>.csv(.gz)  large surfaces gzipped (gzip -n: reproducible bytes)
#   results/round2/taus/*.json, gates.json, analysis/*, as5_silence.csv.gz (when present)
#   results/round2/runs/<run>/{run_meta.json,run_final.json,epochs.jsonl,swa.pth,last.pth}
#   results/round2/pascu/head_{train,traindev}.{npz,json}
#   results/round2/stress/{manifest.csv,manifest.json,IR_POOL.txt}   (the r2_parity set's provenance)
set -euo pipefail
K="-i $HOME/.ssh/id_ed25519_sol1 -o IdentitiesOnly=yes -o ConnectTimeout=15 -o BatchMode=yes"
S=${R2_HOST:?set R2_HOST to the user@host SSH login of the scoring server}
ssh $K -p 8000 $S 'bash -s' <<'REMOTE'
set -euo pipefail
H=/mnt/data8tb/henry
ST=$H/xfer/r2_collect
rm -rf $ST && mkdir -p $ST/results/round2
cd $H/deploybench/results/round2
for f in $(find scores -type f -name "*.csv" | sort); do
  mkdir -p "$ST/results/round2/$(dirname $f)"
  case "$(basename $f .csv)" in
    df21_verified|mlaad5|as5_dev|as5_eval) gzip -n -c "$f" > "$ST/results/round2/$f.gz" ;;
    *) cp "$f" "$ST/results/round2/$f" ;;
  esac
done
for d in taus analysis; do [ -d $d ] && cp -r $d $ST/results/round2/; done
for f in gates.json; do [ -e $f ] && cp $f $ST/results/round2/; done
[ -e as5_silence.csv ] && gzip -n -c as5_silence.csv > $ST/results/round2/as5_silence.csv.gz
for r in $H/r2_runs/aasist_l_*_s*; do
  [ -e $r/run_final.json ] || continue
  o=$ST/results/round2/runs/$(basename $r); mkdir -p $o
  cp $r/run_meta.json $r/run_final.json $r/epochs.jsonl $r/weights/swa.pth $r/weights/last.pth $o/
done
mkdir -p $ST/results/round2/pascu $ST/results/round2/stress
cp $H/r2_runs/pascu/head_*.npz $H/r2_runs/pascu/head_*.json $ST/results/round2/pascu/ 2>/dev/null || true
P=$H/data/deepfake/stress/r2_parity
cp $P/manifest.csv $P/manifest.json $P/IR_POOL.txt $ST/results/round2/stress/
cd $ST && find results -type f | sort | xargs sha256sum > SHA256SUMS && tar -cf ../r2_collect.tar SHA256SUMS results
echo "staged $(wc -l < SHA256SUMS) files, $(du -sh ../r2_collect.tar | cut -f1)"
REMOTE
TMP=$(mktemp -d)
scp $K -P 8000 $S:/mnt/data8tb/henry/xfer/r2_collect.tar "$TMP/"
tar -xf "$TMP/r2_collect.tar" -C "$TMP"
(cd "$TMP" && sha256sum -c --quiet SHA256SUMS) && echo "SHA-256 verified"
cp -r "$TMP/results/round2/." results/round2/
cp "$TMP/SHA256SUMS" results/round2/SHA256SUMS_sol1.txt
rm -rf "$TMP"
echo "R2_COLLECT_DONE"
