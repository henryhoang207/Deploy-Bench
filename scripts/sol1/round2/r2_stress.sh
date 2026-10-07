#!/usr/bin/env bash
# W3 (PREREGISTRATION_ROUND2.md Sec. 5): generate the round-2 stress set on the frozen 1,000-clip H5 manifest, AFTER the
# round2-freeze tag, then score the five round-1 models on it (envs/db310, their round-1 checkpoints).
#   conditions: clean (byte copy) + opus6k opus24k amr4.75 mp3_32 snr20 snr10 snr5 mic_ir
#   MUSAN noise/ (openslr 17); IRs: the AIR mock-up-phone responses in RIRS_NOISES (openslr 28), linked into ir_air_phone/
# Round-2 detectors and runs are scored on the same set by their own launchers. Ends with STRESS_DONE fails=<n>.
set -u
H=/mnt/data8tb/henry
source $H/env.sh > /dev/null 2>&1          # db310, DATA_ROOT, cd to the clone
git merge-base --is-ancestor round2-freeze HEAD || { echo "REFUSED: the clone does not contain tag round2-freeze"; exit 3; }
SRC=$H/data/stress_sources
SET=$DATA_ROOT/stress/r2_parity
V=$H/deploybench/vendor
[ -e $SRC/.extracted_musan_noise ] && [ -e $SRC/.extracted_rirs_noises ] || { echo "REFUSED: MUSAN / RIRS_NOISES not ready"; exit 4; }

# --- the IR pool: AIR mock-up phone responses only, each checked to be 16 kHz mono-or-multi wav ---
IRD=$SRC/ir_air_phone
if [ ! -d $IRD ]; then
  mkdir -p $IRD
  for f in $(find $SRC/RIRS_NOISES -name 'air_type1_air_phone_*.wav' | sort); do ln -s "$f" "$IRD/$(basename "$f")"; done
fi
python - <<PY || { echo "REFUSED: IR pool check failed"; exit 5; }
import hashlib, soundfile as sf
from pathlib import Path
irs = sorted(Path("$IRD").glob("*.wav"))
assert irs, "no AIR phone IRs found"
rows = []
for p in irs:
    info = sf.info(str(p))
    assert info.samplerate == 16000, f"{p.name}: {info.samplerate} Hz"
    rows.append(f"{hashlib.sha256(p.resolve().read_bytes()).hexdigest()}  {p.name}  {info.channels}ch  {info.frames}")
Path("$IRD/IR_POOL.txt").write_text("\n".join(rows) + "\n")
print(f"IR pool: {len(irs)} AIR phone IRs, all 16 kHz")
PY

# --- generate (stress.py refuses an existing manifest) ---
if [ ! -e $SET/manifest.csv ]; then
  python -B -m deploybench.stress --dataset asvspoof19_eval --ids configs/parity_manifest_tssdnet.txt --out $SET \
     --conditions clean opus6k opus24k amr4.75 mp3_32 snr20 snr10 snr5 mic_ir --musan $SRC/musan --ir $IRD \
     > $H/logs/r2_stress_gen.log 2>&1 || { echo "STRESS GENERATION FAILED"; exit 6; }
  cp $IRD/IR_POOL.txt $SET/IR_POOL.txt
fi
echo "stress set: $(wc -l < $SET/manifest.csv) manifest lines $(date -Is)"

ck() {
  case $1 in
    aasist)      echo $V/aasist/models/weights/AASIST.pth ;;
    aasist_l)    echo $V/aasist/models/weights/AASIST-L.pth ;;
    rawnet2)     echo $V/asvspoof2021_baselines/LA/Baseline-RawNet2/pre_trained_DF_RawNet2.pth ;;
    tssdnet)     ls $V/end-to-end-synthetic-speech-detection/pretrained/Res_TSSDNet*.pth ;;
    xlsr_aasist) echo $V/SSL_Anti-spoofing/Best_LA_model_for_DF.pth ;;
  esac
}
fails=0
for m in aasist aasist_l rawnet2 tssdnet xlsr_aasist; do
  for c in clean opus6k opus24k amr4.75 mp3_32 snr20 snr10 snr5 mic_ir; do
    out=results/round2/scores/$m/s0/stress_$c.csv
    [ -e $out ] && continue
    mkdir -p $(dirname $out)
    if CUDA_VISIBLE_DEVICES=${STRESS_GPU:-0} python -B score.py --model $m --checkpoint "$(ck $m)" --dataset asvspoof19_eval \
         --stress-manifest $SET/manifest.csv --condition $c --device cuda --batch 16 --out $out.partial \
         > $H/logs/r2_stress_${m}_$c.log 2>&1; then mv $out.partial $out
    else echo "FAILED $m/$c"; fails=$((fails + 1)); fi
  done
  echo "scored $m $(date -Is)"
done
echo "STRESS_DONE $(date -Is) fails=$fails"
