# Reproduction report — aasist

Generated 2026-09-18 04:14Z by `analysis/repro_report.py`. Score contract: label 1 = bona fide, higher score = bona fide, SPOOF if score < tau.

## Inputs

| role | file | n | sha256[:16] | model | dataset | precision | runtime | condition | dup ids |
|---|---|---|---|---|---|---|---|---|---|
| eval | `scores_repro_aasist.csv` | 71237 | `fa7705f5b4c1fe63` | aasist | asvspoof19_eval | fp32 | pytorch | clean | 0 |
| dev | `scores_dev_aasist.csv` | 24844 | `b0f6724ab34a6474` | aasist | asvspoof19_dev | fp32 | pytorch | clean | 0 |

## A/B — EER computed two independent ways

| method | EER % | tau at EER |
|---|---|---|
| `deploybench.metrics.eer` | 0.8294 | +1.497178 |
| independent numpy sort-and-sweep | 0.8294 | +1.497178 |
| **absolute difference** | **0.000000** | 0.000000 |

Agreement below 1e-6 — the metrics module is not the source of this number.

## Reproduction gate

- Published anchor: **0.8300%** (clovaai/aasist README @a04c9863, released AASIST.pth)
- Ours: **0.8294%**  (95% bootstrap CI 0.7342–0.9706%, 1000 resamples)
- Relative difference: **-0.08%**  (gate |rel| <= 15%)

### VERDICT: PASS

## Score-sign check

| class | n | mean | median |
|---|---|---|---|
| bona fide (label 1) | 7355 | +4.9797 | +5.1177 |
| spoof (label 0) | 63882 | -4.7868 | -4.4557 |

Bona fide scores higher on both mean and median — convention holds.

## Frozen operating points (tau)

Calibrated on `asvspoof19_dev` (n=24844) — D16: tau follows the model's FINE-TUNING data; TSSDNet is Cohort-19, so ASVspoof19 LA dev.
Artifact `results/phase_1_reproduction/tau_aasist.json` — frozen now

| point | tau | dev target | dev FRR | dev FAR | eval FRR | eval FAR | eval HTER |
|---|---|---|---|---|---|---|---|
| eer | -1.093149 | FRR = FAR | 0.510% | 0.507% | 0.163% | 9.827% | 4.995% |
| frr1 | -0.174515 | FRR <= 1% | 0.981% | 0.157% | 0.286% | 4.189% | 2.237% |
| frr5 | +2.908202 | FRR <= 5% | 4.984% | 0.004% | 3.576% | 0.185% | 1.880% |

All dev FRR/FAR above are recomputed by brute force (`np.mean(bona < tau)`), not by the module that produced tau.

- FRR-1% point achieves 0.9812% dev FRR (meets the 1% target)
- FRR-5% point achieves 4.9843% dev FRR (meets the 5% target)
- Ordering tau(frr1) <= tau(frr5) — correct: FRR rises with tau, so the stricter 1% rejection budget sits at the lower threshold

These three taus are now the ONLY thresholds this model may use downstream (conversion parity, device runs, stress conditions). Nothing re-tunes them.

## Threshold transfer dev -> eval (in-domain sanity)

| point | HTER at frozen tau | oracle HTER on eval | TTG |
|---|---|---|---|
| eer | 4.995% | 0.816% | +4.179 pp |
| frr1 | 2.237% | 0.816% | +1.421 pp |
| frr5 | 1.880% | 0.816% | +1.064 pp |

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
