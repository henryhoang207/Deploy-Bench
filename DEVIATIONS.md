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
