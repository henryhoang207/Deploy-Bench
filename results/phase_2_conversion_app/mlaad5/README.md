# MLAAD-5 at the frozen 19LA-dev taus (D47 / DEVIATIONS.md DEV-6), sol1, 2026-10-04/05

**Descriptive only.** MLAAD enters no registered hypothesis test. Protocol: `configs/mlaad5/` (494,853 trials:
284,000 MLAAD spoof clips from snapshot `30c3dec`, plus 210,853 count-matched M-AILABS bona fide clips;
committed before any scoring, sha256 `14bd7b48...`). Launcher: `scripts/sol1/mlaad_scoring.sh` (verifies the
protocol and every artifact first). Report: `python analysis/mlaad_report.py` → `mlaad_report.{txt,json}`.
Score files are committed gzipped (`scores_<p>_<runtime>_mlaad.csv.gz`, ~7 MB each).

## Confound: state it with every MLAAD number
`sample_rate_census.json` (file headers, all 494,853 files): **every spoof clip is 22.05 kHz**, while
**206,853 of 210,853 bona fide clips (98.1%) are native 16 kHz**. The exceptions are all German (1,947 at
22.05 kHz, 1,853 at 44.1 kHz). The harness resamples everything to 16 kHz (`score.py`, `resample_poly`), so
spoof clips carry resampling traces that almost no bona fide clip has. A detector can separate the classes
partly by that, not by the speech, so MLAAD EER and TTG are **not a clean language effect**. Mean durations
are similar across classes (6.1-7.6 s), so length is not an obvious second confound.
English is imbalanced (69,853 bona fide, i.e. all of M-AILABS en_US + en_UK, vs 143,000 spoof). HTER, FRR and
FAR are per class, so the imbalance changes no rate. It only weights the pooled set towards English.

## fp32 (PyTorch, all 5 models)
| model | EER | tau_eer: FRR | FAR | TTG [95% CI] | tau_frr1: FRR (budget 1%) | tau_frr5: FRR (budget 5%) |
|---|---|---|---|---|---|---|
| AASIST | 27.21 | 6.49 | 60.06 | 7.11 [7.00, 7.22] | 10.79 | 78.75 |
| AASIST-L | 30.95 | 9.88 | 58.54 | 3.58 [3.48, 3.69] | 18.33 | 86.18 |
| RawNet2 | 27.58 | 4.52 | 60.22 | 5.03 [4.93, 5.15] | 7.18 | 30.99 |
| Res-TSSDNet | 37.69 | 60.78 | 18.42 | 2.02 [1.92, 2.13] | 66.61 | 90.77 |
| XLS-R+AASIST (ref) | 24.80 | 18.75 | 31.49 | 0.38 [0.32, 0.45] | 56.72 | 82.97 |

Per-language rows are in `mlaad_report.txt` (e.g. AASIST TTG ranges from 0.22 pp (it) to 12.66 pp (en)).

**Readings (exploratory, confound applies):**
- **The direction of the fixed-threshold error reverses for 4 of 5 models.** At tau_eer, AASIST, AASIST-L,
  RawNet2 and XLS-R accept spoofs (FAR 31-60%) far more than they reject bona fide speech (FRR 5-19%). Only
  Res-TSSDNet keeps the ITW/DF21 pattern (FRR 60.8% vs FAR 18.4%). On ITW (H2) FRR > FAR held for 3 of 4 ranked
  models. Which way a transferred threshold fails therefore depends on the target, not only on the model.
- **XLS-R, the model with the largest TTG on ITW (14.96 pp) and DF21 (5.40 pp), has the smallest here (0.38
  pp).** The model with the best EER is not consistently the most fragile; the gap ordering changes with the
  target.
- **FRR budgets do not transfer:** the tau that held FRR at 1% on 19LA dev gives 7.2-66.6% on MLAAD-5.

## Converted artifacts (vs the same model's fp32 on identical trials)
| artifact | dEER | flips at tau_eer | state |
|---|---|---|---|
| AASIST-L fp16 / ORT | +0.00 pp | 160 | done |
| AASIST fp16 / ORT | | | running (ORT fp16 on x86 CPU: ~1.6 clips/s per shard) |
| Res-TSSDNet fp16 / ORT, RawNet2 fp16-repaired(D46) / ORT | | | queued |
| RawNet2 int8-mixed (K=16) / ORT, Res-TSSDNet fp16 / ExecuTorch | | | queued |
