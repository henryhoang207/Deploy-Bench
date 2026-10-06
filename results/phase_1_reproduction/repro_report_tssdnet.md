# Reproduction report — tssdnet

Generated 2026-07-25 17:01Z by `analysis/repro_report.py`. Score contract: label 1 = bona fide, higher score = bona fide, SPOOF if score < tau.

## Inputs

| role | file | n | sha256[:16] | model | dataset | precision | runtime | condition | dup ids |
|---|---|---|---|---|---|---|---|---|---|
| eval | `scores_repro_tssdnet.csv` | 71237 | `1a4846d8e2481313` | tssdnet | asvspoof19_eval | fp32 | pytorch | clean | 0 |
| dev | `scores_dev_tssdnet.csv` | 24844 | `b9404618fb57de31` | tssdnet | asvspoof19_dev | fp32 | pytorch | clean | 0 |

## A/B — EER computed two independent ways

| method | EER % | tau at EER |
|---|---|---|
| `deploybench.metrics.eer` | 1.6179 | -0.399932 |
| independent numpy sort-and-sweep | 1.6179 | -0.399932 |
| **absolute difference** | **0.000000** | 0.000000 |

Agreement below 1e-6 — the metrics module is not the source of this number.

## Reproduction gate

- Published anchor: **1.6590%** (Linux pretrained Res-TSSDNet, ghua-ac README @36c9705)
- Ours: **1.6179%**  (95% bootstrap CI 1.4887–1.7485%, 1000 resamples)
- Relative difference: **-2.47%**  (gate |rel| <= 15%)

### VERDICT: PASS

## Score-sign check

| class | n | mean | median |
|---|---|---|---|
| bona fide (label 1) | 7355 | +14.0628 | +14.9241 |
| spoof (label 0) | 63882 | -17.8257 | -18.0426 |

Bona fide scores higher on both mean and median — convention holds.

## Frozen operating points (tau)

Calibrated on `asvspoof19_dev` (n=24844) — D16: tau follows the model's FINE-TUNING data; TSSDNet is Cohort-19, so ASVspoof19 LA dev.
Artifact `<project>\phase_1_reproduction\results\tau_tssdnet.json` — frozen now

| point | tau | dev target | dev FRR | dev FAR | eval FRR | eval FAR | eval HTER |
|---|---|---|---|---|---|---|---|
| eer | -8.244781 | FRR = FAR | 0.746% | 0.722% | 0.258% | 6.080% | 3.169% |
| frr1 | -7.054642 | FRR <= 1% | 0.981% | 0.556% | 0.340% | 4.857% | 2.599% |
| frr5 | +0.654632 | FRR <= 5% | 4.984% | 0.112% | 2.053% | 1.371% | 1.712% |

All dev FRR/FAR above are recomputed by brute force (`np.mean(bona < tau)`), not by the module that produced tau.

- FRR-1% point achieves 0.9812% dev FRR (meets the 1% target)
- FRR-5% point achieves 4.9843% dev FRR (meets the 5% target)
- Ordering tau(frr1) <= tau(frr5) — correct: FRR rises with tau, so the stricter 1% rejection budget sits at the lower threshold

These three taus are now the ONLY thresholds this model may use downstream (conversion parity, device runs, stress conditions). Nothing re-tunes them.

## Threshold transfer dev -> eval (in-domain sanity)

| point | HTER at frozen tau | oracle HTER on eval | TTG |
|---|---|---|---|
| eer | 3.169% | 1.587% | +1.583 pp |
| frr1 | 2.599% | 1.587% | +1.012 pp |
| frr5 | 1.712% | 1.587% | +0.126 pp |

In-domain TTG should be small. A large gap here means dev and eval score distributions already differ, which would contaminate every out-of-domain TTG in the paper.

## Run environment (added by hand — required by PILOT_tssdnet.md)

The published anchors differ by OS (Linux 1.6590% vs Windows 1.64%), and the
author's OS note traces to a 2021 soundfile/libsndfile FLAC-decoding issue, so
the audio stack is part of this result's provenance.

| item | value |
|---|---|
| OS | Windows-11-10.0.26200-SP0 |
| CPU | Intel64 Family 6 Model 154 Stepping 4 (12 logical cores, torch using 10) |
| Python | 3.14.3 |
| torch | 2.12.0+cpu (no CUDA — CPU-only build) |
| **soundfile** | **0.13.1** |
| **libsndfile** | **1.2.2** |
| numpy / scipy | 2.5.1 / 1.17.1 |
| vendor commit | `36c9705a5bdbb68d01fae484f1b758dd673d9f43` (2021-10-27) |
| checkpoint | `Res_TSSDNet_..._eEER_1.64%.pth`, 4,251,839 bytes |
| DATA_ROOT | `C:\DF_Dataset` (19LA dev+eval extracted from `LA.zip`) |

**Reading:** our 1.6179% sits *below* both published anchors, and the 95%
bootstrap CI (1.4887–1.7485%) **contains both 1.64% and 1.6590%** — so this run
is statistically indistinguishable from either, and the 2021 Windows/Linux FLAC
discrepancy does not reproduce on libsndfile 1.2.2. No OS-specific correction is
warranted.

**Caveat on comparability:** this row was produced on Windows. Per
`PILOT_tssdnet.md`, every Cohort-19 model in the paper's reproduction table
should be reproduced on the same OS; if that ends up being Linux, re-run this
row on the GPU server (~2 CPU-hours) once its disk blocker clears.

## Verification summary

| check | result |
|---|---|
| score sign: bona fide scores above spoof | PASS |
| EER: metrics.py agrees with independent sweep | PASS |
| reproduction gate |rel| <= 15% | PASS |
| frr1 tau achieves dev FRR <= 1% | PASS |
| frr5 tau achieves dev FRR <= 5% | PASS |
| tau(frr1) <= tau(frr5) | PASS |

**Overall: PASS**
