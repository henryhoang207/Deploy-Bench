# Threshold transfer audit: review snapshot

Anonymised snapshot of the code, frozen thresholds and score files behind the paper *Thresholded Once, Deployed Elsewhere: A
Pre-Registered Audit of Threshold Transfer in Speech Deepfake Detection*. It holds what the paper's numbers need. It holds nothing
from the separate study of model conversion, quantization and on-device behaviour.

## Contents

| Path | What |
|---|---|
| `PREREGISTRATION.md` | The registration exactly as frozen in version control on 28 September 2026 (SHA-256 `8ae91d1f1f33ef251ecbc1a4998ee898d24c5aca23648976024b41af21e57758`). |
| `DEVIATIONS.md` | Every departure from the registration, each logged before the analysis it governs. Deviation 8 and its two amendments cover the analyses added in revision. |
| `deploybench/`, `models/`, `score.py`, `requirements.txt` | The scoring harness and the five model wrappers (AASIST, AASIST-L, RawNet2, Res-TSSDNet, XLS-R+AASIST). |
| `configs/` | The frozen 1,000-clip manifest used by H5, and the DF21 FLAC integrity lists (class A, class B), each with a recorded SHA-256 of its trial list. |
| `results/phase_1_reproduction/` | LA19 development and evaluation scores, the frozen thresholds (`tau_<model>.json`, with the SHA-256 of the dev scores they came from) and the reproduction reports. |
| `results/phase_2_conversion_app/main_matrix/<model>/scores_fp32_pytorch_{itw,df21_verified}.csv` | fp32 scores on In-the-Wild and on the verified DF21 subset (the folder name is historical). |
| `results/phase_3_stress/h5/` | Scores of the clean and EnCodec-resynthesised copies (H5). Manifest paths are relative to the data root. |
| `results/analysis/` | The registered analyses: operating points and the cost check, H1/H4, H2, H5. |
| `analysis/` | The scripts that wrote `results/analysis/`. |
| `revision/` | The analyses added in revision, with all outputs (see `revision/README.md`). |
| `tests/` | Unit tests: `python -m pytest tests`. Three tests of components outside this snapshot are skipped. |

## Reproducing the numbers from the committed scores

No audio and no model is needed: every number in the paper is recomputed from the score files in this snapshot.

```bash
pip install -r requirements.txt
python analysis/operating_points.py     # operating points, TTG, registered cost check
python analysis/h2_test.py              # H2 (registered, clip-level)
python analysis/h5_test.py              # H5 (registered, clip-level)
python analysis/h1_h4.py                # H1, H4 (exploratory)
```

The revision's analyses also need four public metadata files (speaker labels and protocols). Put them in `revision/meta/` (list and
SHA-256 values in `revision/meta/PROVENANCE.md`) and follow `revision/README.md`. Every revision script first recomputes the
registered numbers from the score files and stops unless they agree to 1e-12.

## Re-scoring audio (optional)

1. **Data.** Under one root, by default `~/data/deepfake` (set `DATA_ROOT`):
   - ASVspoof 2019 LA (Edinburgh DataShare);
   - In-the-Wild (`release_in_the_wild`);
   - the ASVspoof 2021 DF evaluation partition (Zenodo 10.5281/zenodo.4835108) and its keys.

   Check the layout with `python -m deploybench.data --check`.
2. **Model code and checkpoints**, cloned into `vendor/` (third-party, not included):

   | Folder | Upstream repository | Notes |
   |---|---|---|
   | `vendor/aasist` | `clovaai/aasist` | AASIST and AASIST-L weights |
   | `vendor/asvspoof2021_baselines` | `asvspoof-challenge/2021` | RawNet2 DF baseline, `LA/Baseline-RawNet2/pre_trained_DF_RawNet2.pth` |
   | `vendor/end-to-end-synthetic-speech-detection` | `ghua-ac/end-to-end-synthetic-speech-detection` | Res-TSSDNet |
   | `vendor/SSL_Anti-spoofing` | `TakHemlata/SSL_Anti-spoofing` | `Best_LA_model_for_DF.pth`; also needs fairseq and the XLS-R 300M checkpoint `xlsr2_300m.pt` in that folder |

   The exact checkpoint paths are in `models/wrap_*.py`.
3. **Score**, for example `python score.py --model aasist --dataset asvspoof19_eval --out results/scores/`. Scores are fp32 with TF32
   disabled. On the same platform, re-scoring reproduced the committed scores within 1e-4. Across platforms, float32 differences of
   up to 3.7e-4 were seen for Res-TSSDNet (Deviation 8, amendment 1).
4. **Stress set (H5).** `python -m deploybench.stress --dataset asvspoof19_eval --ids <ids> --out <dir> --conditions clean encodec6k`,
   then `score.py --stress-manifest <dir>/manifest.csv --condition <cond>`.
