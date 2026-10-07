# Pre-registration, round 2 (addendum to PREREGISTRATION.md)

**Status:** frozen at git tag `round2-freeze`. Written 2026-10-07, before any round-2 score existed. Decisions D53-D57
(publication plan and scope), D58 (this freeze); deviation entry DEV-9 in `DEVIATIONS.md`. The registered analysis code
was committed first (`7592468`) and is part of the tagged tree: `analysis/round2/r2_registered.py`, `r2_gates.py`,
`r2_taus.py`, `r2lib.py`, tested by `tests/test_round2.py`.

## 1. Purpose and relation to round 1
Round 1 (`PREREGISTRATION.md`, frozen at tag `roster-freeze` = `97e9753`; deviations DEV-1..8) is unchanged: its roster,
hypotheses H1-H6, tests and outcomes stand as reported. Round 2 asks whether four round-1 findings hold beyond the round-1
checkpoints:
(a) for current detectors (one trained on ASVspoof 5, one calibration-oriented, one 2024-25 SSL model);
(b) for retrained copies of one round-1 recipe (seed variance);
(c) when the silence cue is removed from the training data (a causal test);
(d) under eight more channel conditions.
It registers four confirmatory hypotheses (R2-H1 to R2-H4), one reading rule fixed in advance (R2-RL) and a list of
exploratory analyses (E1-E11). Everything is fp32 desktop scoring; no conversion, int8 or device result (D53).

## 2. What had been seen when this was written
- Every round-1 result: fp32 scores of the five round-1 models on LA19 dev and eval, ITW, DF21 verified and clean, the
  EnCodec 1,000-clip manifest (clean, encodec6k) and MLAAD-5 (where the error reverses to FAR > FRR for 4 of 5 models);
  all registered outcomes (H2 supported 3/4 at the clip level, H5 4/4, H3 and H6 not supported); every DEV-8 analysis
  (speaker-cluster H2: 1 of 4; tau-estimator documentation and dev-equivalent intervals; FAR-budget policies; cost
  sweep; silence statistics of LA19 dev/eval and ITW; test-time VAD-trimmed re-scoring of the five round-1 models on
  LA19 dev/eval and ITW; the three-zone illustration with the dev-equivalent interval as review band).
- Published figures: the ASVspoof 5 Baseline-AASIST README's dev EER 15.2%; XLSR-Mamba's README (LA21 0.93%, DF21 1.88%,
  ITW 6.71%, the last from its LA model); Pascu et al. (arXiv 2309.05384) Table 2, xls-r-2b with logistic regression
  trained on LA19 train + dev: LA19 eval 0.6%, ITW 7.2%; Mueller et al. 2021 (trimming silence before training raised
  EER from 3.6% to 15.5%); the ASVspoof 5 database paper (arXiv 2502.08857, Sec. 4.2): leading/trailing non-speech was
  trimmed and peak amplitude normalised in the **development and evaluation sets but not the training set**.
- Data facts: the ASVspoof 5 track-1 protocol format and row counts; the M-AILABS path layouts in the MLAAD-5 protocol;
  the LA19 train partition on sol1 (25,380 files).
- Pre-tag smoke tests, all on LA19 **training** clips: XLSR-Mamba and as5_aasist each scored 6 training clips and the
  XLS-R 2B path embedded 8 (finite and deterministic; both detectors ranked bona fide above spoof); Mamba / causal-conv / Triton kernel
  checks against PyTorch references; a 30-step AASIST-L training benchmark, untrimmed and trimmed; the trim status counts
  of LA19 train and dev under the VAD rule (train: all 25,380 trimmed, none of exactly 64,600 samples; dev: 24,843
  trimmed, 1 with nothing to cut); a check that the harness VAD reproduces DEV-8's recorded bounds on every
  LA19 dev, LA19 eval and ITW clip. The transformers warning that the XLS-R 2B positional-convolution weights were
  "newly initialized" was checked: the effective weights equal the checkpoint's weight_norm(v, g) exactly.
- **Not seen:** no score of any round-2 model on any dev, evaluation or target set; no stressed audio for the eight new
  conditions; no ASVspoof 5 audio analysed; XLSR-Mamba's released score files sit in the vendor clone unopened and are
  read only by the DF21 gate after the tag.

## 3. Models
| Name | What | Training data | Checkpoint (pinned) | Threshold source | Input | Env |
|---|---|---|---|---|---|---|
| `as5_aasist` | ASVspoof 5 baseline B02 (AASIST; `models/AASIST.py` byte-identical to clovaai's) | ASVspoof 5 train (not post-processed) | asvspoof-challenge/asvspoof5 @ `fe23d30`, `Baseline-AASIST/models/weights/AASIST/best.pth` | ASVspoof 5 dev (D16) | 64,600, pad | db310 |
| `pascu_lr` | Pascu et al. 2024: frozen wav2vec2 XLS-R 2B, mean-pooled last hidden state, logistic regression (C = 1e6, max_iter 5,000) | LA19 **train only** (refit; the authors used train + dev) | backbone HF `facebook/wav2vec2-xls-r-2b` @ `3b6d89d`; head from `train/pascu_fit.py --splits train` | LA19 dev | whole utterance | r2 |
| `pascu_lr_gate` | the same with the authors' fit (train + dev) | LA19 train + dev | `train/pascu_fit.py --splits train dev` | none | whole utterance | r2 |
| `xlsr_mamba` | XLSR-Mamba (Xiao & Das, SPL 2025), DF model | LA19 train, RawBoost algo 3 | swagshaw/XLSR-Mamba @ `10ec738`; HF `AustinXiao/XLSR-Mamba-DF` @ `8aed3da` | LA19 dev | 66,800, pad | r2 |
| `aasist_l_r2` | our runs of the clovaai AASIST-L recipe, seeds 1-5; seed 1234 = pipeline check | LA19 train | `swa.pth` of each run (Sec. 4) | the run's LA19 dev | 64,600, pad | train r2, score db310 |
| `aasist_l_trim` | the same recipe on VAD-trimmed audio, seeds 1-5 | LA19 train, trimmed | `swa.pth` of each run | the run's trimmed LA19 dev | trim, then 64,600 | train r2, score db310 |
| `aasist_l_r2_last`, `aasist_l_trim_last` | the last-epoch weights of the same runs (E3 only) | as above | `last.pth` | as above | as above | db310 |

Verified against author code (2026-10-07): label conventions (bona fide = 1 in both ASVspoof 5's `data_utils.genSpoof_list`
and XLSR-Mamba's), eval scores `batch_out[:, 1]` (both), eval padding (truncate from the start, repeat-tile short clips),
XLSR-Mamba's eval length 66,800 (`Dataset_eval`; its In-the-Wild loader uses 64,600: round 2 uses 66,800 everywhere),
aletheia's feature and classifier recipe (`extract_features.py`, `linear_classifier.py`, `evaluate_feature_type.py`:
`SPLITS = ["train", "dev"]`; the paper's text says max_iter 1,000, the code 5,000: the code is followed). aletheia has no
licence, so `models/wrap_pascu.py` re-implements it. Pascu's score is `-(w.x + b)`, the log-odds of bona fide (their
classifier's label 1 is fake; `predict_proba` saturates at C = 1e6). The round-1 roster (`models.base.ROSTER`) is unchanged.

## 4. Training runs
`train/aasist_r2_train.py` transcribes clovaai `main.py` @ `a04c986` (vendor/aasist) with `config/AASIST-L.conf`
(100 epochs, batch 24, Adam 1e-4 with per-step cosine annealing to 5e-6, weight decay 1e-4, cross-entropy weights
[0.1, 0.9], cudnn deterministic, DataLoader workers 0 as in the recipe). Changes:
1. No evaluation set anywhere (the recipe scores LA19 eval whenever dev EER improves and after SWA, and writes `best.pth`
   by **eval** t-DCF).
2. The run's model is the recipe's final `swa.pth`. The recipe's selection is kept exactly: its initial `best_dev_eer = 1.`
   is compared with a dev EER in percent, so only epochs with dev EER <= 1% are saved and enter SWA; if none does, `swa.pth`
   is the last epoch. Each run's number of SWA snapshots is reported. `last.pth` (last-epoch weights) is saved too (E3).
3. Dev EER = the vendor's `compute_eer` on the dev scores, x 100; the dev t-DCF (logged only) is not computed.
4. Trimmed arm: train and dev audio pass through `deploybench.vad.trim` before `pad_random` / `pad`: the DEV-8 rule
   (energy VAD on the float32 samples, 25 ms frames, 10 ms hop, active if within 35 dB of the clip's loudest frame, both
   ends cut with zero margin, left untrimmed if the result would be shorter than 0.25 s). Scoring applies the same trim to
   every input (`SPECS[...].preproc = "vad35"`).
5. TF32 off; torch 2.2.1+cu118 (envs/r2).
6. Seeds 1, 2, 3, 4, 5 for each arm; seed 1234 (the recipe's default) for the untrimmed pipeline check. Runs may execute in
   parallel; a run is never restarted from a checkpoint.

## 5. Data and targets
| Surface | n | Scored for | Clusters (speaker unless stated) |
|---|---|---|---|
| `la19_dev` | 24,844 | every LA19-trained round-2 model (threshold source) | — |
| `la19_eval` | 71,237 | every round-2 model and run | 67 LA19-eval speakers |
| `itw` | 31,779 | every round-2 model and run | 54 ITW speaker labels |
| `df21_verified` | 455,508 (D34) | `as5_aasist`, `pascu_lr`, `xlsr_mamba` | DF21 speaker |
| `mlaad5` | 494,853 (DEV-6) | `as5_aasist`, `pascu_lr`, `xlsr_mamba` | bona fide: language + M-AILABS speaker (language + book for `mix/` folders); spoof: TTS system |
| `as5_dev` | 140,950 | `as5_aasist` (threshold source) | — |
| `as5_eval` | 680,774 | `as5_aasist` only; reported by condition C00-C11, C00 (no codec) primary (D24) | ASVspoof 5 speaker |
| stress manifest | 1,000 x 9 | every model and run, round 1 included (not the `*_last` models) | 67 LA19-eval speakers |

The training runs are scored on `la19_dev`, `la19_eval`, `itw` and the stress manifest only. The stress manifest is the
frozen 1,000-clip H5 set (`configs/parity_manifest_tssdnet.txt`, DEV-2) generated by `deploybench/stress.py` into a new
set with conditions `clean` (byte copy) and the eight new ones; `encodec6k` is the existing H5 audio. Conditions as coded:
`opus6k`, `opus24k` (libopus), `amr4.75` (libopencore_amrnb at 8 kHz), `mp3_32` (libmp3lame), each decoded back to 16 kHz
PCM16 by ffmpeg 7.0.2 (imageio-ffmpeg 0.6.0); `snr20`, `snr10`, `snr5`: MUSAN `noise/` (openslr 17), one file and offset
per clip from the hash of its utt_id, SNR over the whole clip; `mic_ir`: the AIR mock-up-phone impulse responses in
RIRS_NOISES (openslr 28, `real_rirs_isotropic_noises/air_type1_air_phone_*`: 23 files, 16 kHz, two channels, checked
2026-10-07), one per clip from the hash, the two channels averaged to mono as coded (an IR at another rate is refused),
peak-normalised, output cut to the input length. The chosen IR files and their SHA-256 are
written to the stress set's provenance.

## 6. Scoring, gates and thresholds
- `score.py`, PyTorch fp32, TF32 off, the project schema; the `seed` column carries the training seed (0 for released
  checkpoints). Files: `results/round2/scores/<model>/s<seed>/<surface>.csv[.gz]`; round-1 stress scores under
  `results/round2/scores/<model>/s0/stress_<condition>.csv`.
- **Gates** (`r2_gates.py`; round 1's rule, a reproduced EER within 15% of the published one): `as5_aasist` ASVspoof 5
  dev EER in [12.92, 17.48]%; `xlsr_mamba`, on the DF21 verified trials also in the authors' released DF score file
  (`Scores/DF/Bmamba3_LA_WCE_1e-06_ES144_NE12.txt`, same input length), our EER within 15% of theirs on the same trials;
  `pascu_lr` through `pascu_lr_gate`, LA19-eval EER in [0.51, 0.69]% (their 0.6%). The training runs have no gate.
- **Thresholds** (`r2_taus.py`, through `deploybench.tau`, frozen once written): tau_eer (interpolated EER crossing),
  tau_frr1 and tau_frr5 (strictest threshold with dev FRR <= 1% / 5%), tau_far1 and tau_far5 (most lenient with dev FAR
  <= 1% / 5%), all five registered. Source dev per Sec. 3.

## 7. Confirmatory hypotheses
Common rules: speaker-cluster bootstrap (clusters resampled with replacement with all their clips of both classes),
B = 10,000, seed 0, one-sided p = (1 + #{statistic <= 0}) / (B + 1), Holm at alpha = 0.05 within each family.

**R2-H1, the policy sets the direction.** On ITW, for each round-2 detector (`as5_aasist`, `pascu_lr`, `xlsr_mamba`):
(a) FRR - FAR > 0 at tau_frr1 and (b) FAR - FRR > 0 at tau_far5. Holm over the six tests. A detector supports R2-H1 if
both (a) and (b) are rejected; R2-H1 is supported if at least 2 of 3 detectors support it. A detector that fails its gate or
cannot be scored counts as not supporting. *Reading:* not supported means that the direction of the error under transfer
depends on the detector and not only on the policy.

**R2-H2, the LA19-eval transfer gap survives retraining.** For each `aasist_l_r2` run (seeds 1-5), TTG(tau_eer) on LA19
eval > 1 pp (H0: TTG <= 1 pp), the oracle recomputed in every replicate. Holm over five; supported if at least 4 of 5
reject; a missing run counts as not rejecting. *Reading:* supported means that the round-1 finding (a development
threshold costs AASIST-L HTER on unseen attacks: 4.6 pp for the released checkpoint) holds across retraining at a 1-pp
margin, i.e. it is a property of the recipe, not of one checkpoint.

**R2-H3, removing the silence cue from training.** `aasist_l_trim` vs `aasist_l_r2`, seeds 1-5 each; exact one-sided
permutation tests of the difference of run means over the 252 relabellings: (a) manipulation check, outside the family:
LA19-dev EER higher when trimmed; (b) ITW FRR at tau_eer lower when trimmed; (c) ITW EER lower when trimmed. Holm over
(b) and (c). With fewer than three finished runs in either arm, R2-H3 is not evaluable. *Reading:* (b) without (c) means
that the cue moves the operating point but not discrimination; both mean that it also limits discrimination on ITW;
neither means that the cue does not explain the ITW behaviour of this recipe.

**R2-H4, H5 generalised to eight channel conditions.** For each condition in {opus6k, opus24k, amr4.75, mp3_32, snr20,
snr10, snr5, mic_ir} and each of the four primary round-1 models (AASIST, AASIST-L, RawNet2, Res-TSSDNet): dFRR - dFAR > 0
at the frozen round-1 tau_eer, relative to the clean copy of the same clips (paired). Holm over the four models within each
condition; a condition supports R2-H4 if at least 3 of 4 reject. The components dFRR > 0 and dFAR < 0 are tested and
reported the same way. The result is reported per condition, with the number of supporting conditions.

## 8. Reading rule R2-RL (fixed in advance, descriptive)
Limitation (a) of the ICME revision predicted that, for calibration-oriented or ASVspoof 5-trained detectors, narrower
dev-equivalent intervals and smaller ITW transfer shares would show that the round-1 pattern does not carry over. Per
round-2 detector on ITW at tau_eer: ITW EER >= 20% -> "discrimination-limited" (the round-1 primary-model pattern);
otherwise "transfer-limited" (the round-1 reference pattern carries over) if the transfer share TTG / HTER >= 0.5 or the
dev-equivalent interval (DEV-8 B3 rule on the model's source dev) spans >= 10 pp of ITW FRR; "well-transferred" (it does
not carry over) if the share is < 0.2 and the span < 5 pp; "intermediate" otherwise.

## 9. Exploratory analyses (effect sizes and intervals only; no confirm/refute language)
- **E1** Operating behaviour of every round-2 model on every registered surface at the five policies: FRR, FAR, HTER,
  oracle HTER, TTG, transfer share, dev-equivalent interval and its target FRR/FAR ends, rank-based miss and band (DEV-8 D),
  speaker-cluster 95% intervals (B = 1,000).
- **E2** Seed spread: E1 quantities across the `aasist_l_r2` runs (and the 1234 run), with the released AASIST-L's round-1
  values placed in that spread; the same for the trimmed runs.
- **E3** Selection-free comparison: `last.pth` of both arms (models `*_last`) on LA19 dev, LA19 eval and ITW.
- **E4** Channel effects: dFRR and dFAR per condition (EnCodec included) for every model and run.
- **E5** MLAAD-5 for the three round-2 detectors, by language, with the sample-rate confound (every spoof 22.05 kHz,
  98.1% of bona fide 16 kHz) stated with every number; the round-1 models' MLAAD results are descriptive (seen, D52).
- **E6** ASVspoof 5 eval for `as5_aasist` by condition (C00 primary); silence statistics of ASVspoof 5 dev and eval with
  the DEV-8 VAD.
- **E7** Three-zone (accept / review / reject) policies, designed on source dev only. Primary: a review budget of 10% of
  dev traffic centred in rank on tau_eer (band edges at the pooled-dev score quantiles F(tau_eer) - 0.05 and + 0.05); score
  below the band -> spoof, above -> bona fide, inside -> review. On every target: review share (pooled and per class),
  bona fide auto-rejected, spoof auto-accepted. Compared with the same rule on the target's own labelled scores at its
  oracle threshold, and with a label-free variant whose band edges are re-anchored on the target's unlabelled pooled
  scores (F_target(tau_eer) -/+ 0.05). Secondary: budgets of 5% and 20%; and the error-budget rule (auto-reject below
  tau_frr1, auto-accept from tau_far1; if tau_frr1 >= tau_far1 the band is empty and the policy reduces to tau_eer, which is
  reported as a collapse).
- **E8** ISO/IEC 30107-3 terms: APCER (FAR) and BPCER (FRR) at every policy; BPCER at APCER 1% and 5% with the APCER fixed
  on source dev (tau_far1, tau_far5) and at the target oracle.
- **E9** Actual vs minimum normalised cost over priors: logistic calibration on source dev (`deploybench.metrics.platt_fit`),
  Bayes thresholds for logit(prior) in [-4, 4] in steps of 0.25, actual vs minimum cost on each target.
- **E10** Cost windows at tau_eer (the round-1 cost models, screening and forensic) for every round-2 model.
- **E11** The pipeline-gate model `pascu_lr_gate`, reported only as the gate.

## 10. Failure, missing data and stopping rules
- A model that fails its gate is reported, stays in the exploratory tables with the failure stated, and counts as not
  supporting R2-H1. A model that cannot run in its environment is reported and dropped. No substitute model is added.
- A training run that crashes or produces a NaN loss is reported and not repeated; R2-H2 counts it as not rejecting;
  R2-H3 uses the remaining runs (not evaluable with fewer than three per arm). No additional seeds.
- A trimmed train clip of exactly 64,600 samples would crash the recipe's `pad_random`; the script checks for this before
  training and stops (an amendment would then be logged before any rerun).
- A logistic-regression ConvergenceWarning is recorded, and the head is used as fitted.
- Any loader refusal (unknown label or codec, wrong row count, missing file) stops the job; a fix is logged as a DEV-9
  amendment before the rerun. No hyper-parameter, input-length, policy or test is changed after the tag except through such
  an amendment, which states what had been seen.

## 11. Software
envs/db310 (Python 3.10.21, torch 2.5.1+cu124; round-1 models, `as5_aasist`, the AASIST-L runs at scoring), envs/r2
(Python 3.10.21, torch 2.2.1+cu118, mamba-ssm 1.1.4, causal-conv1d 1.1.3.post1, transformers 4.44.2, scikit-learn 1.3.2,
torchcontrib 0.0.2, fairseq a5402130 on `sys.path`; `envs/r2_freeze.txt`), envs/cc (conda-forge gcc 12.4, used only by
Triton to compile its launchers). Two RTX 4090s; TF32 off everywhere.
