# DeployBench — On-Device Speech Deepfake Detection Benchmark

Companion code for the paper (see project docs). Status: Phase 0 — core metrics validated, model wrappers pending checkpoints.

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
CSV/parquet columns: `utt_id,label,score,model,precision,runtime,condition,seed`
- `label`: 1 = bona fide, 0 = spoof
- `score`: higher = more bona-fide-like (decide SPOOF when score < tau)
