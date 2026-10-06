# Reproduction report — aasist_l

Generated 2026-09-18 04:15Z by `analysis/repro_report.py`. Score contract: label 1 = bona fide, higher score = bona fide, SPOOF if score < tau.

## Inputs

| role | file | n | sha256[:16] | model | dataset | precision | runtime | condition | dup ids |
|---|---|---|---|---|---|---|---|---|---|
| eval | `scores_repro_aasist_l.csv` | 71237 | `39c11eb51d47a13e` | aasist_l | asvspoof19_eval | fp32 | pytorch | clean | 0 |
| dev | `scores_dev_aasist_l.csv` | 24844 | `d0a7ff003cc22fe9` | aasist_l | asvspoof19_dev | fp32 | pytorch | clean | 0 |

## A/B — EER computed two independent ways

| method | EER % | tau at EER |
|---|---|---|
| `deploybench.metrics.eer` | 0.9909 | +1.763541 |
| independent numpy sort-and-sweep | 0.9909 | +1.763541 |
| **absolute difference** | **0.000000** | 0.000000 |

Agreement below 1e-6 — the metrics module is not the source of this number.

## Reproduction gate

- Published anchor: **0.9900%** (clovaai/aasist README @a04c9863, released AASIST-L.pth)
- Ours: **0.9909%**  (95% bootstrap CI 0.8702–1.1149%, 1000 resamples)
- Relative difference: **+0.09%**  (gate |rel| <= 15%)

### VERDICT: PASS

## Score-sign check

| class | n | mean | median |
|---|---|---|---|
| bona fide (label 1) | 7355 | +6.2953 | +6.5500 |
| spoof (label 0) | 63882 | -4.6649 | -4.3791 |

Bona fide scores higher on both mean and median — convention holds.

## Frozen operating points (tau)

Calibrated on `asvspoof19_dev` (n=24844) — D16: tau follows the model's FINE-TUNING data; TSSDNet is Cohort-19, so ASVspoof19 LA dev.
Artifact `results/phase_1_reproduction/tau_aasist_l.json` — frozen now

| point | tau | dev target | dev FRR | dev FAR | eval FRR | eval FAR | eval HTER |
|---|---|---|---|---|---|---|---|
| eer | -0.944946 | FRR = FAR | 0.510% | 0.507% | 0.095% | 11.138% | 5.616% |
| frr1 | +0.006131 | FRR <= 1% | 0.981% | 0.197% | 0.231% | 5.138% | 2.684% |
| frr5 | +3.455527 | FRR <= 5% | 4.984% | 0.000% | 4.514% | 0.229% | 2.371% |

All dev FRR/FAR above are recomputed by brute force (`np.mean(bona < tau)`), not by the module that produced tau.

- FRR-1% point achieves 0.9812% dev FRR (meets the 1% target)
- FRR-5% point achieves 4.9843% dev FRR (meets the 5% target)
- Ordering tau(frr1) <= tau(frr5) — correct: FRR rises with tau, so the stricter 1% rejection budget sits at the lower threshold

These three taus are now the ONLY thresholds this model may use downstream (conversion parity, device runs, stress conditions). Nothing re-tunes them.

## Threshold transfer dev -> eval (in-domain sanity)

| point | HTER at frozen tau | oracle HTER on eval | TTG |
|---|---|---|---|
| eer | 5.616% | 0.973% | +4.643 pp |
| frr1 | 2.684% | 0.973% | +1.711 pp |
| frr5 | 2.371% | 0.973% | +1.398 pp |

In-domain TTG should be small. A large gap here means dev and eval score distributions already differ, which would contaminate every out-of-domain TTG in the paper.

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
