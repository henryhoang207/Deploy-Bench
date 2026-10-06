# Reproduction report — xlsr_aasist

Generated 2026-09-18 04:28Z by `analysis/repro_report.py`. Score contract: label 1 = bona fide, higher score = bona fide, SPOOF if score < tau.

## Inputs

| role | file | n | sha256[:16] | model | dataset | precision | runtime | condition | dup ids |
|---|---|---|---|---|---|---|---|---|---|
| eval | `scores_repro_xlsr_aasist.csv` | 71237 | `9f855908ffdf842f` | xlsr_aasist | asvspoof19_eval | fp32 | pytorch | clean | 0 |
| dev | `scores_dev_xlsr_aasist.csv` | 24844 | `5f8f0f1f48572b11` | xlsr_aasist | asvspoof19_dev | fp32 | pytorch | clean | 0 |

## A/B — EER computed two independent ways

| method | EER % | tau at EER |
|---|---|---|
| `deploybench.metrics.eer` | 0.1496 | +1.711187 |
| independent numpy sort-and-sweep | 0.1496 | +1.711187 |
| **absolute difference** | **0.000000** | 0.000000 |

Agreement below 1e-6 — the metrics module is not the source of this number.

## Reproduction gate

No `--published` anchor supplied; gate not evaluated.

## Score-sign check

| class | n | mean | median |
|---|---|---|---|
| bona fide (label 1) | 7355 | +4.4893 | +4.5171 |
| spoof (label 0) | 63882 | -3.5835 | -3.6319 |

Bona fide scores higher on both mean and median — convention holds.

## Frozen operating points (tau)

Calibrated on `asvspoof19_dev` (n=24844) — D16: tau follows the model's FINE-TUNING data; TSSDNet is Cohort-19, so ASVspoof19 LA dev.
Artifact `results/phase_1_reproduction/tau_xlsr_aasist.json` — frozen now

| point | tau | dev target | dev FRR | dev FAR | eval FRR | eval FAR | eval HTER |
|---|---|---|---|---|---|---|---|
| eer | -1.751550 | FRR = FAR | 0.039% | 0.009% | 0.054% | 0.485% | 0.270% |
| frr1 | +3.576978 | FRR <= 1% | 0.981% | 0.000% | 0.625% | 0.075% | 0.350% |
| frr5 | +4.329467 | FRR <= 5% | 4.984% | 0.000% | 5.901% | 0.022% | 2.961% |

All dev FRR/FAR above are recomputed by brute force (`np.mean(bona < tau)`), not by the module that produced tau.

- FRR-1% point achieves 0.9812% dev FRR (meets the 1% target)
- FRR-5% point achieves 4.9843% dev FRR (meets the 5% target)
- Ordering tau(frr1) <= tau(frr5) — correct: FRR rises with tau, so the stricter 1% rejection budget sits at the lower threshold

These three taus are now the ONLY thresholds this model may use downstream (conversion parity, device runs, stress conditions). Nothing re-tunes them.

## Threshold transfer dev -> eval (in-domain sanity)

| point | HTER at frozen tau | oracle HTER on eval | TTG |
|---|---|---|---|
| eer | 0.270% | 0.144% | +0.125 pp |
| frr1 | 0.350% | 0.144% | +0.206 pp |
| frr5 | 2.961% | 0.144% | +2.817 pp |

In-domain TTG should be small. A large gap here means dev and eval score distributions already differ, which would contaminate every out-of-domain TTG in the paper.

## Verification summary

| check | result |
|---|---|
| score sign: bona fide scores above spoof | PASS |
| EER: metrics.py agrees with independent sweep | PASS |
| frr1 tau achieves dev FRR <= 1% | PASS |
| frr5 tau achieves dev FRR <= 5% | PASS |
| tau(frr1) <= tau(frr5) | PASS |

**Overall: PASS**
