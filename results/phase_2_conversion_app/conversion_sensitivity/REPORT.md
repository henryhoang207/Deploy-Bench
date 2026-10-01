# Conversion sensitivity (D35/D39/D41): first complete run, 2026-09-28/29

**Exploratory** (PREREGISTRATION, conversion-sensitivity section). Effect sizes only.
Run on sol1 at `c493ed2` (+ `0f2ede7` for the tssdnet resume). ORT 1.23.2, onnx 1.22.0,
numpy 1.23.5, Python 3.10.21. Every number here comes from the files in this directory.

## fp32 export gate (before any PTQ)
PyTorch fp32 (CPU) vs ONNX fp32 export, frozen 1,000-clip 19LA-eval parity manifest, zero
decision flips required at all three frozen taus. **All four exports pass**
(`<model>/parity_report_fp32_conversion.json`); max |score diff| rawnet2 3.7e-5, tssdnet 1.7e-4.

## Recipe search (seed-100 calibration, 2,000-clip diagnostic subset, D36 budget +1.0 pp)

| model | fp32 diag EER | r01 default dEER | best attempt (K) | best dEER | locked recipe |
|---|---|---|---|---|---|
| aasist | 0.30% | +5.89 pp | r12 (48 of 327 ranked) | +1.40 pp | **none** |
| aasist_l | 0.40% | +12.11 pp | r12 (48 of 313) | +1.70 pp | **none** |
| rawnet2 | 0.60% | +3.17 pp | r09 (16 of 82) | +0.90 pp | **r09, K=16** |
| tssdnet | 0.70% | +12.03 pp | r11 (32 of 37) | +2.90 pp | **none** |

Per attempt: `selection_<model>.json`. Frozen plans: `search_plan_<model>.json`. Every
attempt converted (`ledger_<model>.jsonl`, no failures), except that **tssdnet r12 (K=37, every
ranked node) left no Q/DQ op at all**. That is an fp32 graph: `score_onnx.py` refused to label it
int8, and it is recorded as `no_int8_ops` (`0f2ede7`).

**Under the frozen design, AASIST, AASIST-L and Res-TSSDNet have no qualifying int8 artifact.**
Per D39, no calibration repeats run for them and there is no int8 device artifact.

## RawNet2 calibration repeats (locked r09, K=16)

| seed | diag dEER | qualifies | 19LA eval EER | eval dEER | HTER @ tau_eer | FRR | FAR | flips vs fp32 |
|---|---|---|---|---|---|---|---|---|
| fp32 | – | – | 4.5955% | – | 5.989% | | | – |
| 1 | +1.9 pp | no | 5.471% | +0.876 | 6.483% | 0.802% | 12.163% | 1,819 |
| **2** | **+1.0 pp** | **yes** | 5.573% | +0.977 | 7.023% | 0.503% | 13.544% | 2,083 |
| 3 | +2.8 pp | no | 6.910% | +2.315 | 7.002% | 1.659% | 12.346% | 2,743 |
| 4 | +1.1 pp | no | 5.219% | +0.624 | 5.374% | 1.618% | 9.129% | 2,383 |
| 5 | +0.7 pp | yes | 5.778% | +1.182 | 7.170% | 0.517% | 13.822% | 1,910 |

- 5/5 seeds converted; 2/5 meet the budget on the diagnostic subset.
- **Canonical device artifact = seed 2**, the lowest qualifying seed (D39). Its diagnostic
  dEER sits exactly on the +1.0 pp boundary (EER on 1,000 + 1,000 clips moves in 0.05-0.1 pp
  steps).
- **Across-calibration spread on 19LA eval: dEER +0.62 to +2.32 pp** at a fixed recipe.
  Diagnostic-subset order does not predict eval order (seed 1 is worst-but-one on diag and
  second-best on eval).
- Repeatability control (seed 1 rebuilt from the identical manifest): **bit-identical artifact,
  max |score diff| 0.0** on both surfaces (`repeatability_rawnet2.json`).
- Converter fallback list for r01 (H6 secondary DV): GRU, Gather, MaxPool, Reshape, Squeeze,
  Transpose stay fp32.

## Effort (conversion wall-clock from the ledgers; ranking from the plans)
aasist 12 attempts / 42 min + ranking 3.7 h · aasist_l 12 / 39 min + 3.5 h ·
rawnet2 18 / 14 min + 16 min · tssdnet 12 / 10 min + 13 min.

## Caveats a reader must see
1. **The search never varied the calibration method.** Every exclusion attempt used minmax (the
   `rank_sensitivity.py --calib-method` default Claude chose when building the tooling;
   D39 allowed a pre-declared method change, and none was declared). The Res-TSSDNet pilot
   (non-blind) needed *percentile* calibration plus 8 early convs in fp32 to reach +0.94 pp on
   eval. So "no qualifying artifact" is conditional on a minmax-only search. Any percentile arm
   now would be a post-hoc addition: log it in DEVIATIONS.md and report it as exploratory.
2. **tssdnet's 12th attempt was wasted.** The k schedule clips to "all ranked nodes" when the list
   overshoots (37 < 48), and excluding all of them leaves no int8. That is a tooling defect
   (Claude's). It did not change any outcome here, but that attempt told us nothing.
3. The ranking runs simulated quantization (ORT optimizations off). It orders candidates;
   it is never an accuracy number.
4. aasist/aasist_l rank 327/313 nodes and the schedule stops at K=48, so most of the graph
   was never tried in fp32. The curve was still falling at K=48 (+1.4 / +1.7 pp).

## Percentile arm (DEVIATIONS.md DEV-1, EXPLORATORY), completed 2026-10-01

The same frozen exclusion lists as the registered arm, with `calib_method = percentile` and
histogram calibration fed 20 clips at a time (DEV-1 amendment). Files carry a `_percentile`
suffix; runs are in `<model>/percentile/`. It never replaces the registered result above.

| model | registered (minmax) | percentile: locked recipe | seeds 1-5 (diag dEER) | 19LA-eval dEER per seed |
|---|---|---|---|---|
| aasist | none (best +1.40, K=48) | **p06, K=6** (best tested p09 K=16 +0.10) | **all 5 + rebuild FAILED** | – |
| aasist_l | none (best +1.70, K=48) | **p05, K=4** | +0.40, +0.90, +7.55, +0.80, +0.63; canonical seed 1 | +0.90, +1.29, **+8.82**, +2.48, +1.55 |
| rawnet2 | r09, K=16 | **p07, K=8** | +0.30, +0.40, +0.30, +0.30, +0.10; canonical seed 1 | +0.15, +0.06, +0.37, +0.01, -0.12 |
| tssdnet | none (best +2.90, K=32) | none (best p05 K=4: +1.50) | – | – |

**Readings (exploratory):**
- Percentile calibration changes the picture for every model: AASIST and AASIST-L find recipes
  with 4-6 layers in fp32, against none at 48 under minmax; RawNet2 needs half the layers, and
  its seed spread collapses (eval dEER -0.12..+0.37, against +0.62..+2.32). Res-TSSDNet gets
  closer but still misses the budget. The registered "3 of 4 have no int8 artifact" result is
  therefore a property of minmax PTQ, not of the models.
- Calibration sensitivity is still large where a recipe exists: AASIST-L seed 3 costs +8.8 pp on
  19LA eval under a recipe whose other seeds cost +0.9 to +2.5.
- The percentile default (K=0) is not uniformly better: AASIST-L p01 +22.65 pp vs minmax r01
  +12.11.

**AASIST repeat failures are a TOOLING artifact, not a model property.** All six AASIST
calibration repeats (p06 seeds 1-5 + rebuild) died in ORT 1.23
`HistogramCollector.collect_absolute_value` (`np.arange(...)`: "Maximum allowed size
exceeded"). Under chunked calibration the first 20-clip chunk fixes each histogram's bin
width; when a later chunk's maximum is far larger, ORT extends the histogram at that width and
needs more bins than numpy allows. An unchunked pass would not extend, but it ran sol1 out of
memory (DEV-1 amendment). The search attempt p06 itself (seed-100 manifest) succeeded; whether
the extension blows up depends on the order of chunk maxima. Under the frozen rules these are
logged failures, not retried, so the AASIST percentile arm has **no canonical artifact**.
A fix would need a declared deviation (e.g. a first pass that fixes each tensor's range
before binning), with new attempt ids.
