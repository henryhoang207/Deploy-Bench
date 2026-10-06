# Reproduction report — rawnet2

Generated 2026-09-19 02:55Z by `analysis/repro_report.py`. Score contract: label 1 = bona fide, higher score = bona fide, SPOOF if score < tau.

## Inputs

| role | file | n | sha256[:16] | model | dataset | precision | runtime | condition | dup ids |
|---|---|---|---|---|---|---|---|---|---|
| eval | `scores_df21clean_rawnet2.csv` | 305525 | `995e3fac5ebcbbd2` | rawnet2 | asvspoof21_df_eval_clean | fp32 | pytorch | high_m4a,high_mp3,high_ogg,low_m4a,low_mp3,low_ogg,mp3m4a,nocodec,oggm4a | 0 |

## A/B — EER computed two independent ways

| method | EER % | tau at EER |
|---|---|---|
| `deploybench.metrics.eer` | 15.0636 | -10.721274 |
| independent numpy sort-and-sweep | 15.0636 | -10.721274 |
| **absolute difference** | **0.000000** | 0.000000 |

Agreement below 1e-6 — the metrics module is not the source of this number.

## Reproduction gate

- Published anchor: **15.0640%** (official B04 score.txt pooled over the D34 clean surface (305,525 trials); D33 gate)
- Ours: **15.0636%**  (95% bootstrap CI 14.6186–15.6573%, 1000 resamples)
- Relative difference: **-0.00%**  (gate |rel| <= 15%)

### VERDICT: PASS

## Score-sign check

| class | n | mean | median |
|---|---|---|---|
| bona fide (label 1) | 6991 | -2.9937 | -0.0000 |
| spoof (label 0) | 298534 | -11.3161 | -11.9642 |

Bona fide scores higher on both mean and median — convention holds.

## Frozen operating points (tau)

**Not computed — no `--dev-scores` supplied.** tau must be calibrated on ASVspoof19 LA dev (D16) before any conversion or device run.

## Verification summary

| check | result |
|---|---|
| score sign: bona fide scores above spoof | PASS |
| EER: metrics.py agrees with independent sweep | PASS |
| reproduction gate |rel| <= 15% | PASS |

**Overall: PASS**

---

_Section below appended by hand after generation (xfer/df21clean_compare.py), not by repro_report.py._

## Per-trial agreement with the organisers' B04 scores (D33)

Official scores: `keys/DF/CM/RawNet2/score.txt` (DF-keys-full.tar.gz, sha256[:16] `a9ac140df41ec063`).
Rows: 305525 (DF21 clean surface, D34), bona fide 6991.

| statistic | value |
|---|---|
| median abs diff | 9.54e-07 |
| p99 abs diff | 6.68e-06 |
| p99.99 abs diff | 2.28e-05 |
| max abs diff | 3.72e-05 |
| trials with abs diff > 1e-4 | 0 |
| trials with abs diff > 1e-3 | 0 |
| Pearson r | 1 |

| EER on the same rows | % |
|---|---|
| ours | 15.0636 |
| official B04 scores | 15.0636 |

| codec | n | ours EER % | official EER % |
|---|---|---|---|
| high_m4a | 30848 | 19.355 | 19.355 |
| high_mp3 | 30907 | 19.332 | 19.332 |
| high_ogg | 37849 | 12.282 | 12.282 |
| low_m4a | 30909 | 18.924 | 18.924 |
| low_mp3 | 30797 | 19.628 | 19.628 |
| low_ogg | 37738 | 12.413 | 12.413 |
| mp3m4a | 37900 | 12.018 | 12.018 |
| nocodec | 30876 | 19.184 | 19.184 |
| oggm4a | 37701 | 12.685 | 12.685 |
