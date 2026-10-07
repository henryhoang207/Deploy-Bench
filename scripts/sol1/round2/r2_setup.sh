#!/usr/bin/env bash
# Round-2 setup on sol1 (pre-tag, D54: software and weights only, no scoring).
#   envs/r2  : CPython 3.10.21, torch 2.2.1+cu118 (XLSR-Mamba's pin), mamba-ssm 1.1.4 + causal-conv1d 1.1.3.post1
#              prebuilt wheels, transformers (XLS-R 2B for Pascu et al.), scikit-learn, fairseq a5402130 on sys.path
#              (not built, exactly as in envs/db310), torchcontrib (SWA in the clovaai AASIST recipe).
#   envs/cc  : conda-forge gcc (micromamba), only so Triton can compile its launcher stubs (sol1 has no gcc).
#   vendor/  : asvspoof5 @ fe23d30, XLSR-Mamba @ 10ec738 (+ HF weights 8aed3da), wav2vec2-xls-r-2b @ 3b6d89d.
# Ends with R2_SETUP_DONE. Idempotent where cheap; refuses to overwrite an existing envs/r2.
set -euo pipefail
H=/mnt/data8tb/henry
export UV_CACHE_DIR=$H/cache/uv UV_PYTHON_INSTALL_DIR=$H/tools/python PATH=$H/tools/bin:$PATH
V=$H/deploybench/vendor
E=$H/envs/r2
echo "start $(date -Is)"

# ---------------------------------------------------------------- vendor code (pinned)
if [ ! -d "$V/asvspoof5/.git" ]; then git clone -q https://github.com/asvspoof-challenge/asvspoof5 "$V/asvspoof5"; fi
git -C "$V/asvspoof5" checkout -q fe23d3053a0889b305d98c9e4cadf418c213d972
if [ ! -d "$V/XLSR-Mamba/.git" ]; then git clone -q https://github.com/swagshaw/XLSR-Mamba "$V/XLSR-Mamba"; fi
git -C "$V/XLSR-Mamba" checkout -q 10ec73810d24091028a2aa5454c8dfd1a239e441
# XLSR-Mamba's SSLModel loads ./xlsr2_300m.pt relative to the CWD to build the architecture (weights are then
# overwritten by the fine-tuned checkpoint): reuse the file the round-1 reference already uses.
ln -sfn "$V/SSL_Anti-spoofing/xlsr2_300m.pt" "$V/XLSR-Mamba/xlsr2_300m.pt"
sha256sum "$V/asvspoof5/Baseline-AASIST/models/weights/AASIST/best.pth"

# ---------------------------------------------------------------- envs/r2
if [ -e "$E" ]; then echo "envs/r2 exists: not rebuilding"; else
  uv venv "$E" --python 3.10.21
  # shellcheck disable=SC1091
  source "$E/bin/activate"
  uv pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.2.1+cu118 torchaudio==2.2.1+cu118
  uv pip install \
    "https://github.com/state-spaces/mamba/releases/download/v1.1.4/mamba_ssm-1.1.4%2Bcu118torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" \
    "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.1.3.post1/causal_conv1d-1.1.3.post1%2Bcu118torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" \
    numpy==1.23.5 scipy==1.11.4 soundfile==0.12.1 pandas==2.0.3 scikit-learn==1.3.2 einops==0.7.0 \
    transformers==4.44.2 safetensors huggingface_hub==0.24.6 hydra-core==1.0.7 omegaconf==2.0.6 \
    antlr4-python3-runtime==4.8 bitarray sacrebleu regex cffi tqdm imageio-ffmpeg==0.6.0 pyyaml ninja packaging \n    "setuptools<70"      # triton 2.2 imports setuptools; uv venvs have none (added 2026-10-07 after the kernel smoke)
  uv pip install --no-deps torchcontrib==0.0.2
  echo "$V/SSL_Anti-spoofing/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1" > "$E/lib/python3.10/site-packages/fairseq_a5402130.pth"
  uv pip freeze > "$H/envs/r2_freeze.txt"
  deactivate
fi

# ---------------------------------------------------------------- C compiler for Triton only
if [ ! -x "$H/envs/cc/bin/gcc" ]; then
  micromamba create -y -q -r "$H/tools/mamba_root" -p "$H/envs/cc" -c conda-forge gcc=12 sysroot_linux-64=2.17
fi
"$H/envs/cc/bin/gcc" --version | head -1

# ---------------------------------------------------------------- weights (pinned revisions)
source "$E/bin/activate"
python - <<'PY'
from huggingface_hub import snapshot_download
V = "/mnt/data8tb/henry/deploybench/vendor"
p = snapshot_download("AustinXiao/XLSR-Mamba-DF", revision="8aed3da9242b161f63647b099f67b3c648acbd00",
                      local_dir=f"{V}/XLSR-Mamba/hf_DF")
print("XLSR-Mamba-DF ->", p)
p = snapshot_download("facebook/wav2vec2-xls-r-2b", revision="3b6d89d0fabead7da552eaaa07549c7c9c36d303",
                      local_dir=f"{V}/wav2vec2-xls-r-2b", allow_patterns=["*.json", "*.bin", "README.md"])
print("xls-r-2b ->", p)
PY
sha256sum "$V/XLSR-Mamba/hf_DF/model.safetensors" "$V/wav2vec2-xls-r-2b/pytorch_model.bin"
echo "R2_SETUP_DONE $(date -Is)"
