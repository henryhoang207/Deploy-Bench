#!/usr/bin/env bash
# MUSAN (openslr 17) + RIRS_NOISES (openslr 28) for the round-2 stress suite. Pre-tag download (data only).
# Writes the same markers r2_downloads.sh checks, so that script skips them later. Ends with NOISE_DL_DONE.
set -u
SRC=/mnt/data8tb/henry/data/stress_sources
mkdir -p "$SRC" && cd "$SRC" || exit 2
fetch() { local url=$1 dest=$2 a; for a in 1 2 3 4 5; do wget -q -c -O "$dest" "$url" && return 0; echo "retry $a $dest"; sleep 30; done; return 1; }
if [ ! -e .extracted_rirs_noises ]; then
  fetch https://www.openslr.org/resources/28/rirs_noises.zip rirs_noises.zip || fetch https://us.openslr.org/resources/28/rirs_noises.zip rirs_noises.zip || echo DL_FAIL_rirs
  unzip -q -o rirs_noises.zip && touch .extracted_rirs_noises && echo "extracted RIRS_NOISES"
fi
if [ ! -e .extracted_musan_noise ]; then
  fetch https://www.openslr.org/resources/17/musan.tar.gz musan.tar.gz || fetch https://us.openslr.org/resources/17/musan.tar.gz musan.tar.gz || echo DL_FAIL_musan
  tar -xzf musan.tar.gz musan/noise && touch .extracted_musan_noise && echo "extracted musan/noise"
fi
sha256sum musan.tar.gz rirs_noises.zip > SHA256SUMS.txt
echo "musan noise wavs: $(find musan/noise -name '*.wav' | wc -l); RIRS wavs: $(find RIRS_NOISES -name '*.wav' | wc -l); AIR phone IRs: $(find RIRS_NOISES -name 'air_type1_air_phone_*.wav' | wc -l)"
echo "NOISE_DL_DONE $(date -Is)"
