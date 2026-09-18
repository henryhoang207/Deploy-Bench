# Reproduction report — rawnet2

Generated 2026-09-18 08:13Z by `analysis/repro_report.py`. Score contract: label 1 = bona fide, higher score = bona fide, SPOOF if score < tau.

## Inputs

| role | file | n | sha256[:16] | model | dataset | precision | runtime | condition | dup ids |
|---|---|---|---|---|---|---|---|---|---|
| eval | `scores_repro_rawnet2.csv` | 71237 | `5c30b9103642b104` | rawnet2 | asvspoof19_eval | fp32 | pytorch | clean | 0 |
| dev | `scores_dev_rawnet2.csv` | 24844 | `2ddb1b3ff45679ed` | rawnet2 | asvspoof19_dev | fp32 | pytorch | clean | 0 |

## A/B — EER computed two independent ways

| method | EER % | tau at EER |
|---|---|---|
| `deploybench.metrics.eer` | 4.5955 | -0.003135 |
| independent numpy sort-and-sweep | 4.5955 | -0.003135 |
| **absolute difference** | **0.000000** | 0.000000 |

Agreement below 1e-6 — the metrics module is not the source of this number.

## Reproduction gate

- Published anchor: **5.6400%** (Tak et al. ICASSP 2021 Table 2 S1/Mel -- a DIFFERENT training run (train+90% dev) from the released 2021 DF-baseline ckpt; binding gate is DF21 B04 22.38% per D32)
- Ours: **4.5955%**  (95% bootstrap CI 4.3689–4.7557%, 1000 resamples)
- Relative difference: **-18.52%**  (gate |rel| <= 15%)

### VERDICT: FAIL

Diagnose in this order before touching any wrapper:

1. score sign / class-index order (next section)
2. input length — TSSDNet needs exactly 96,000 samples
3. padding — repeat-tile short, truncate long FROM THE START
4. checkpoint identity (byte size, filename EER string)
5. which protocol file was actually iterated (dev vs eval)

## Score-sign check

| class | n | mean | median |
|---|---|---|---|
| bona fide (label 1) | 7355 | -0.0855 | -0.0000 |
| spoof (label 0) | 63882 | -10.3513 | -11.6262 |

Bona fide scores higher on both mean and median — convention holds.

## Frozen operating points (tau)

Calibrated on `asvspoof19_dev` (n=24844) — D16: tau follows the model's FINE-TUNING data; TSSDNet is Cohort-19, so ASVspoof19 LA dev.
Artifact `results/phase_1_reproduction/tau_rawnet2.json` — frozen now

| point | tau | dev target | dev FRR | dev FAR | eval FRR | eval FAR | eval HTER |
|---|---|---|---|---|---|---|---|
| eer | -7.067738 | FRR = FAR | 0.549% | 0.547% | 0.503% | 11.476% | 5.989% |
| frr1 | -5.178793 | FRR <= 1% | 0.981% | 0.265% | 0.707% | 10.097% | 5.402% |
| frr5 | -0.008322 | FRR <= 5% | 4.984% | 0.004% | 3.712% | 5.108% | 4.410% |

All dev FRR/FAR above are recomputed by brute force (`np.mean(bona < tau)`), not by the module that produced tau.

- FRR-1% point achieves 0.9812% dev FRR (meets the 1% target)
- FRR-5% point achieves 4.9843% dev FRR (meets the 5% target)
- Ordering tau(frr1) <= tau(frr5) — correct: FRR rises with tau, so the stricter 1% rejection budget sits at the lower threshold

These three taus are now the ONLY thresholds this model may use downstream (conversion parity, device runs, stress conditions). Nothing re-tunes them.

## Threshold transfer dev -> eval (in-domain sanity)

| point | HTER at frozen tau | oracle HTER on eval | TTG |
|---|---|---|---|
| eer | 5.989% | 4.246% | +1.744 pp |
| frr1 | 5.402% | 4.246% | +1.156 pp |
| frr5 | 4.410% | 4.246% | +0.164 pp |

In-domain TTG should be small. A large gap here means dev and eval score distributions already differ, which would contaminate every out-of-domain TTG in the paper.

## Verification summary

| check | result |
|---|---|
| score sign: bona fide scores above spoof | PASS |
| EER: metrics.py agrees with independent sweep | PASS |
| reproduction gate |rel| <= 15% | FAIL |
| frr1 tau achieves dev FRR <= 1% | PASS |
| frr5 tau achieves dev FRR <= 5% | PASS |
| tau(frr1) <= tau(frr5) | PASS |

**Overall: FAIL**

## Disposition (D32, 2026-09-18) — appended by hand after generation, not by `repro_report.py`

The 19LA verdict above stands as computed: **outside the symmetric 15% gate, on the better side**
(4.5955% vs 5.64%, -18.5%). It is recorded, not re-scored and not re-thresholded.

- **Cause: anchor/run mismatch.** 5.64% is Tak et al. ICASSP 2021 Table 2, S1/Mel, **trained on
  19LA train + 90% of dev**. The released checkpoint (`pre_trained_DF_RawNet2.pth`, md5 `863a3982...`)
  is the **ASVspoof 2021 DF baseline run, trained on 19LA train only** (`asvspoof-challenge/2021`
  `DF/Baseline-RawNet2/README.md` @9b33f5ec). Same recipe, different run and training data.
- **Pipeline verified equal to the author eval path**: `data_utils.pad` (truncate from start /
  repeat-tile to 64,600), `librosa.load(sr=16000)` on 16 kHz FLAC (no resampling), score =
  `batch_out[:, 1]` of `LogSoftmax`. No defect found. A defect would also be expected to raise EER, not lower it.
- **Binding reproduction gate for RawNet2 = ASVspoof 2021 DF eval, B04 = 22.38%**
  (Yamagishi et al. 2021, the published number at the provenance of this checkpoint). The
  15%-relative band is **19.02–25.74%**. Preconditions: fix B4 (21DF loader: `subset=="eval"`,
  emit `compr`) and complete the 21DF eval download.
- **Until that gate runs, RawNet2's gate status is PENDING.** No cross-model claim is finalised
  before it resolves. If it fails, RawNet2 is excluded from cross-model claims per
  PREREGISTRATION's exclusion rule.
- **tau is frozen now**: dev scoring does not depend on the anchor choice, and it is identical to the
  held staging draft.
