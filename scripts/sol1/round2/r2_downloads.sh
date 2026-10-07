#!/usr/bin/env bash
# Round-2 downloads (allowed before the round2-freeze tag by D54: data only, no scoring).
#   ASVspoof 5 (Zenodo 14498691): dev + eval + protocols + README/LICENSE. Train is not needed (nothing trains on AS5).
#   MUSAN (openslr 17): noise source for snr20/10/5.  RIRS_NOISES (openslr 28): AIR mock-up-phone IRs for mic_ir.
# Resumable (wget -c); every Zenodo file is checked against its published md5; openslr archives get a recorded sha256.
# Ends with R2_DL_DONE (and a count of failures). Re-run the same command if it died.
set -u
ROOT=/mnt/data8tb/henry/data
AS5=$ROOT/deepfake/asvspoof5
SRC=$ROOT/stress_sources
mkdir -p "$AS5/zenodo_14498691" "$SRC"
FAILS=0

fetch() {  # url dest
  local url=$1 dest=$2 a
  for a in 1 2 3 4 5; do
    wget -q -c -O "$dest" "$url" && return 0
    echo "retry $a: $dest"; sleep 60
  done
  return 1
}

echo "start $(date -Is)"
cd "$AS5/zenodo_14498691" || exit 2
LIST="README.txt 784002b8449ba1eeecc042bfad285de7
LICENSE.txt 9cbefc63251e8cbc8eb22e62cd6fe584
ASVspoof5_protocols.tar.gz 865d0e894ea9f686f0f37e5ae3ae3616
flac_D_aa.tar df0be44957623991028cce59792beb17
flac_D_ab.tar 1e8cd685d89b64502692f1bcf1a13db3
flac_D_ac.tar 5e0031f08c30e4bdbf0c59f91b2d662b
flac_E_aa.tar a8c800766f3d4ef87971e2b4f29663e2
flac_E_ab.tar c35064188f54f07c87aba58de22534a0
flac_E_ac.tar ed7cbcd1b2847998b72472ddf6b445e3
flac_E_ad.tar 626a080e994b05df49c577d7b3dede8d
flac_E_ae.tar 05f19a5e64fa556714ae1b38eb2ea70b
flac_E_af.tar 2ca2f52a3bbf827f7ec155ecb47e85d6
flac_E_ag.tar 671200b5de2fc1e74a563b07b057af8f
flac_E_ah.tar eaf885b1299a61eb07f71e82e96dba37
flac_E_ai.tar b2625683bf440abae2af901a45542ad9
flac_E_aj.tar c50281c8233af6f3b0899604d21ade44"
echo "$LIST" | awk '{print $2"  "$1}' > MD5SUMS_dev_eval.txt

# three parallel streams
echo "$LIST" | awk '{print $1}' | xargs -P 3 -I{} bash -c '
  f={}; for a in 1 2 3 4 5; do
    wget -q -c -O "$f" "https://zenodo.org/api/records/14498691/files/$f/content" && exit 0
    echo "retry $a: $f"; sleep 60
  done; echo "DL_FAIL $f"'
if md5sum -c MD5SUMS_dev_eval.txt > md5_check.log 2>&1; then
  echo "AS5 md5 OK ($(grep -c ': OK' md5_check.log) files)"
else
  echo "AS5 MD5_FAIL:"; grep -v ': OK' md5_check.log; FAILS=$((FAILS+1))
fi

# extract (idempotent: skip a tar whose marker exists)
cd "$AS5" || exit 2
for t in zenodo_14498691/ASVspoof5_protocols.tar.gz zenodo_14498691/flac_D_*.tar zenodo_14498691/flac_E_*.tar; do
  m="$AS5/.extracted_$(basename "$t")"
  [ -e "$m" ] && continue
  if tar -xf "$t" -C "$AS5"; then touch "$m"; echo "extracted $(basename "$t")"; else echo "EXTRACT_FAIL $t"; FAILS=$((FAILS+1)); fi
done
# the eval tars unpack to flac_E_eval/; the loader (deploybench/data.py REGISTRY) reads asvspoof5/flac_E
[ -e "$AS5/flac_E" ] || ln -s flac_E_eval "$AS5/flac_E"
echo "AS5 flac counts: D=$(find "$AS5" -path '*flac_D*' -name '*.flac' | wc -l) E=$(find "$AS5" -path '*flac_E*' -name '*.flac' | wc -l)"

# MUSAN (noise subset only is extracted)
cd "$SRC" || exit 2
if [ ! -e .extracted_musan_noise ]; then
  fetch https://www.openslr.org/resources/17/musan.tar.gz musan.tar.gz || fetch https://us.openslr.org/resources/17/musan.tar.gz musan.tar.gz || { echo "DL_FAIL musan"; FAILS=$((FAILS+1)); }
  if tar -xzf musan.tar.gz musan/noise; then touch .extracted_musan_noise; echo "extracted musan/noise"; else echo "EXTRACT_FAIL musan"; FAILS=$((FAILS+1)); fi
fi
# RIRS_NOISES
if [ ! -e .extracted_rirs_noises ]; then
  fetch https://www.openslr.org/resources/28/rirs_noises.zip rirs_noises.zip || fetch https://us.openslr.org/resources/28/rirs_noises.zip rirs_noises.zip || { echo "DL_FAIL rirs"; FAILS=$((FAILS+1)); }
  if unzip -q -o rirs_noises.zip; then touch .extracted_rirs_noises; echo "extracted RIRS_NOISES"; else echo "EXTRACT_FAIL rirs"; FAILS=$((FAILS+1)); fi
fi
sha256sum musan.tar.gz rirs_noises.zip > SHA256SUMS.txt 2>/dev/null
echo "musan noise wavs: $(find musan/noise -name '*.wav' | wc -l); RIRS wavs: $(find RIRS_NOISES -name '*.wav' | wc -l)"
echo "R2_DL_DONE $(date -Is) fails=$FAILS"
