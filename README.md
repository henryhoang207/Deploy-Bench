# DeployBench — On-Device Speech Deepfake Detection Benchmark

Companion code for the paper (see project docs). **Roster frozen at 5 (decision D30, 2026-07-29):** AASIST · AASIST-L · RawNet2 · Res-TSSDNet · XLSR+AASIST (unranked server reference). All five are Cohort-19, so every threshold calibrates on ASVspoof19 LA dev. Wrappers for RawGAT-ST, LFCC-LCNN and the WavLM variants are **deliberately retained** — audited against author source, guarded by contract tests, and harness generality is part of the released artifact — but those models are not studied. Don't add them to a grid without a new decision-log entry.

Status: Res-TSSDNet complete through reproduction + desktop conversion/parity (EER 1.6179% vs the 1.6590% published anchor; PyTorch→ONNX fp32 = 0 decision flips over all 71,237 eval clips). Stage P3 device parity not started — needs an APK and a handset.

## Layout
- `deploybench/` — metrics, tau calibration, data loaders, stress suite (core library)
- `models/` — roster model wrappers (one file per model, added Phase 1)
- `convert/` — ONNX / ExecuTorch export + PTQ (Phase 2)
- `android/` — benchmark app source (Phase 2)
- `scripts/` — adb automation (Phase 2)
- `analysis/` — figures + notebooks (Phase 3)
- `tests/` — unit tests; `pytest tests/` must pass before any handoff
- `configs/` — one YAML per experiment cell

## Quickstart
    pip install -r requirements.txt
    pytest tests/ -v          # validates all metrics against analytic ground truth
    export DATA_ROOT=~/data/deepfake

## Score-file schema (all tools read/write this)
CSV/parquet columns: `utt_id,label,score,model,dataset,precision,runtime,condition,seed` *(9 columns — `dataset` was missing from this line)*
- `label`: 1 = bona fide, 0 = spoof
- `score`: higher = more bona-fide-like (decide SPOOF when score < tau)
