# Step 0.3 — Datasets  · Owner: [U] · Week 1 (downloads can run in background)

| Dataset | Source | Size | Notes |
|---|---|---|---|
| ASVspoof 2019 LA | datashare.ed.ac.uk | ~23 GB | train/dev/eval | (done)
| ASVspoof 2021 DF | zenodo.org | ~30 GB | EVAL partition + keys | (done)
| ASVspoof 5 | zenodo (registration) | ~large | train/dev/eval |
| In-the-Wild | deepfake-total.com | ~8 GB | Müller et al. |
| MLAAD | deepfake-total.com | subset ok | en, de, it, fr, pl |
| MUSAN | openslr.org/17 | 11 GB | noise aug |
| RIR/mic IRs | openslr.org/28 + chosen mic-IR set | small | stress suite |

**Actions:** download all under one root, e.g. `~/data/deepfake/`; keep original folder names; `export DATA_ROOT=~/data/deepfake` in your shell profile.
**Done when:** `python repo/deploybench/data.py --check` (Claude provides) prints file counts matching expected per dataset.
**Pitfall:** ASVspoof 5 registration can take days — do it FIRST, today.
