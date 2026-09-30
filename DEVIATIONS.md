# Deviations from PREREGISTRATION.md

PREREGISTRATION.md froze at tag `roster-freeze` (`97e9753`, 2026-09-28). Every departure
from it after that tag is logged here **before** it runs, with its rationale. Registered
results are never replaced; a deviation adds a labelled, exploratory result beside them.

## DEV-1 (2026-09-29): percentile-calibration arm of the conversion-sensitivity study

**Status:** post-hoc, **exploratory**. Declared after the registered (minmax) arm's
results were known (`results/phase_2_conversion_app/conversion_sensitivity/REPORT.md`,
`768714a`). Approved by the user on 2026-09-29 ("Add the percentile arm").

**Why.** D39 allowed a pre-declared calibration-method change in the recipe search.
None was declared, because the search tooling (`rank_sensitivity.py --calib-method`)
defaulted every exclusion attempt to minmax. Claude chose that default and did not
surface it as a design choice. The Res-TSSDNet pilot (non-blind) had already shown that
percentile calibration was needed to make int8 usable for a raw-waveform model. Under
the registered arm, three of four ranked models have no qualifying int8 artifact.
Without this arm, that result cannot be told apart from an artefact of the calibration
method.

**Design, fixed before any percentile run:**
- Same fp32 exports (their conversion-parity gate already passed), same search
  calibration manifest (`calib_search_seed100.txt`), same 2,000-clip diagnostic
  subset, same D36 budget (int8 EER - fp32 EER <= 1.0 pp), same selection rule
  (smallest K, ties to the earlier attempt).
- **Same exclusion lists in the same order as the registered plans.** The ranking is
  NOT recomputed (it came from the minmax r01 artifacts). The only change is
  `calib_method = percentile` (ORT's default percentile). Every other `ptq.py` knob is
  as in the registered arm.
- Attempts that keep every ranked node in fp32 are dropped before running, because
  they cannot produce an int8 graph (tssdnet r12 left no Q/DQ op). So Res-TSSDNet has
  11 attempts and the other models have 12.
- Attempt ids p01..pNN. Files carry a `_percentile` suffix, and runs live in
  `conversion_sensitivity/<model>/percentile/`. Nothing in the registered arm is
  touched.
- If a recipe locks: seeds 1-5 plus the seed-1 rebuild, scored on the diagnostic subset
  and on ASVspoof19 LA eval (as D41), and the canonical rule of D39.
- Failures (including memory failures of histogram calibration) are results. Nothing
  is retried, replaced or re-run.

**What it may and may not be used for.** It is reported as exploratory, next to the
registered arm and never instead of it. **A percentile-arm artifact does not become a
device artifact** unless a further, logged decision says so. Under D39, the registered
device-artifact rule gives RawNet2 seed 2 and no int8 artifact for AASIST, AASIST-L or
Res-TSSDNet.

### DEV-1 amendment (2026-09-29, before any percentile result existed)
- **The first launch (08:29, 4 models in 2 lanes) ran sol1 out of memory.** ORT 1.23's
  `HistogramCalibrater.collect_data` keeps every intermediate output of all 500
  calibration clips in RAM. AASIST p01 died with no ledger entry, most likely killed by the
  kernel OOM killer (the kernel log is not readable without root). AASIST-L p01 passed
  105 GB (machine 125 GB, shared), so Claude stopped every percentile process, including
  tssdnet p01, which was still running. **No percentile attempt had a ledger entry or an
  artifact** (all p01 directories were empty), so nothing was produced and nothing is
  being retried.
- **Fix, part of the arm's recipe from here:** `ptq.py --calib-chunk 20` feeds the
  histogram calibrator 20 clips at a time. HistogramCollector merges repeated calls by
  extending its bins at the first chunk's stride, so the percentile ranges differ slightly
  from a single 500-clip pass. The chunk size is recorded in every sidecar and ledger line.
- The already-derived `search_plan_<model>_percentile.json` files are unchanged: the chunk
  is not a plan field. The arm now runs **one model at a time**, under a watchdog that
  stops the job if sol1's available memory falls below 15 GB.

## DEV-2 (2026-09-30): H5 source set and test details, specified before any stressed audio exists

**Status:** fills details PREREGISTRATION.md leaves open for the confirmatory H5 ("EnCodec-6k
stress increases FRR more than FAR relative to clean, for a majority of the 4 ranked
models"). Decided by the user on 2026-09-30 ("use the parity manifest for H5"). Fixed and
committed **before** any EnCodec audio was generated or scored, so no H5 result informed it.

- **Source set:** the frozen 1,000-clip 19LA-eval parity manifest
  (`configs/parity_manifest_tssdnet.txt`, 500 bona fide + 500 spoof, seed 20260725, ids sha256
  `2a3c8964...`). Conditions: `clean` (a byte copy through the same stress path) and
  `encodec6k` (EnCodec 24 kHz model at 6 kbps, resampled to 16 kHz;
  `deploybench/stress.py`). The versions are recorded in the stress set's `manifest.json`.
- **Scoring:** fp32 PyTorch on CPU, each model's reproduction checkpoint, through
  `score.py --stress-manifest` for both conditions. The precision axis is not part of H5.
- **Test (per ranked model):** at the frozen tau_eer, dFRR = FRR(encodec6k) - FRR(clean) and
  dFAR = FAR(encodec6k) - FAR(clean). The statistic is dFRR - dFAR. Clips are resampled with a
  class-stratified paired bootstrap: the same resampled clip ids feed both conditions.
  B = 1,000, numpy default_rng(0). One-sided p = fraction of resamples with
  dFRR - dFAR <= 0. Holm-Bonferroni within the H5 family (4 models), alpha 0.05. H5 is
  supported if >= 3 of the 4 ranked models reject. Script: `analysis/h5_test.py`.
- **Scale caveat, to be stated with the result:** 500 clips per class give coarse FRR/FAR
  steps (0.2 pp), so small effects will not reach significance at this size.

## DEV-3 (2026-09-30): H3 when a ranked model has no int8 artifact

**Status:** a counting rule PREREGISTRATION.md does not state. Decided by the user on
2026-09-30 ("count models without int8 as not meeting QTS").

- H3: "QTS > 0.01 (1 HTER point) for at least half of (ranked model x int8) cells in-domain;
  per-cell bootstrap CI excludes 0." In-domain = 19LA eval at 19LA-dev taus (D37).
- **Rule:** the denominator is all 4 ranked models. A model with no int8 artifact under the
  registered D39 procedure (AASIST, AASIST-L, Res-TSSDNet) counts as a cell that does **not**
  meet the criterion. The percentile arm (DEV-1) supplies no H3 cells.
- **Disclosed consequence:** the rule was chosen after the conversion results were known.
  With 3 of 4 cells fixed as not meeting it, H3 **cannot be supported** ("at least half" needs
  2 of 4), whatever RawNet2 shows. It is the conservative reading. The alternative (excluding
  missing cells) would have let H3 rest on one model.
- **RawNet2's cell is still computed and reported.** Artifact: the D39 canonical int8-mixed
  (K=16), r09 seed 2. tau_q = the EER tau of that artifact's scores on the full 19LA dev set
  (24,844 trials, scored for this purpose). QTS = HTER(eval, tau_fp32 eer) - HTER(eval, tau_q).
  CI: class-stratified bootstrap of the 71,237 eval trials, taus held fixed, B = 1000, seed 0,
  2.5/97.5 percentiles. The cell meets the criterion if QTS > 0.01 and the CI lower bound
  is > 0. Script: `analysis/h3_test.py`, committed before the int8 dev scores existed.
