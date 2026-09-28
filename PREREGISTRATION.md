# Pre-registration — DeployBench study
(Commit this file BEFORE any Phase-3 evaluation run; the commit timestamp is
the citable artifact. Drafted from 02_protocol/audit_plan.md + research_methodology.md Part C.)

**Roster frozen 2026-07-29 at 5 models (D30), before commitment — these are amendments,
not deviations.** Ranked cohort: AASIST, AASIST-L, RawNet2, Res-TSSDNet (all Cohort-19,
all tau on ASVspoof19 LA dev). XLSR+AASIST is an **unranked** server reference (D16) and
is excluded from every "majority of models" count below.

## Confirmatory hypotheses (Holm-Bonferroni within family, alpha=0.05)
- **H2** On In-the-Wild at tau* (EER policy), FRR > FAR for the majority of the 4 ranked
  models; family test: per-model paired bootstrap FRR vs FAR.
- **H3** QTS > 0.01 (1 HTER point) for at least half of (ranked model x int8) cells
  in-domain; per-cell bootstrap CI excludes 0.
  **Amendment 2026-09-28 (D37, review S2), before the tag.** "In-domain" means the source
  corpus: ASVspoof19 LA eval scored at taus frozen on ASVspoof19 LA dev. That is **seen corpus,
  unseen attacks**, not in-distribution. Eval attacks A07-A19 are absent from train/dev, except
  A16/A19, which equal A04/A06. The measured dev->eval transfer is not small: at the dev-EER tau,
  AASIST eval HTER is 5.0% vs oracle 0.8% (FAR 9.8%), and Res-TSSDNet's is 3.17% vs 1.59%.
  H3's test, threshold and cells are unchanged. Only the label and its interpretation change:
  a non-zero QTS here is measured under an attack shift, not on matched data.
- **H5** EnCodec-6k stress increases FRR more than FAR relative to clean, for a
  majority of the 4 ranked models (bona-fide corruption asymmetry).
- **H6** **int8 coverage is non-uniform across OP CLASSES, and the fp32-fallback set is
  predictable from the op inventory.** Predicted per mechanism: graph attention
  (ScatterND / topk+gather, AASIST + AASIST-L) and recurrence (GRU, RawNet2) fall back;
  plain conv1d/BN/ReLU (Res-TSSDNet) does not. Test: fallback-op list per artifact vs the
  pre-declared prediction, plus decision flips and dEER per cell.
  **Stated in advance — this is hypothesis-generating, not a predictive rule.** With one
  model per op class, AASIST/AASIST-L is the study's only within-class replication, so a
  confirmed H6 licenses "these op classes behaved this way here", not "GAT models fail".
  **Amendment 2026-09-28 (D36, review S1), before the tag. The primary DV changes. The text above
  is kept as history.**
  - *Primary DV:* per ranked model, the **smallest fp32 exclusion set, among the tested
    configurations, whose int8 artifact keeps EER within 1.0 pp (point estimate) of the same
    model's fp32 EER on the held-out ASVspoof19 LA dev diagnostic subset.** Configurations are
    tried in a bounded, pre-declared order (D35). The diagnostic subset is disjoint from the PTQ
    calibration pool, and its IDs are frozen before any search. Selection never uses eval,
    DF21, ITW or MLAAD scores. This is the smallest set *among those tested*, not a global
    minimum. A model with no qualifying configuration is reported as such, not given a default.
  - *Prediction (same mechanism as above, restated for the new DV):* the minimal exclusion set
    contains graph-attention nodes for AASIST and AASIST-L, and GRU nodes for RawNet2.
  - *Secondary:* the converter's fallback-op list per artifact (the D30 DV above), plus
    decision flips and dEER per cell at the frozen taus.
  - *Scope narrowed to "the behaviour of these five artifacts".* All four ranked models take
    raw waveform input, and three begin with SincConv. So the design cannot attribute an int8
    failure to op class (GAT, GRU) rather than to the shared waveform front-end. The front-end
    is a named, uncontrolled factor.
  - *The Res-TSSDNet arm is NOT blind.* The 2026-07-19 registration (`f3401f8`) predicted that
    plain-conv models would not fail. Default int8 PTQ then gave EER 17.20% (every clip scored
    spoof). The working recipe (percentile calibration, first 8 of 17 convs in fp32) was
    developed with evaluation data. Its full-eval cost is dEER +0.94 pp (CI 0.78-1.07). The
    1.0 pp budget was fixed after that result was known.
  - *Still open before the tag:* OD-6 (how mixed-precision artifacts are labelled), the
    diagnostic subset's size and IDs, the exclusion order and attempt cap, and OD-5
    (calibration size), which are all D35 items.

## Exploratory (effect size + CI only; NO confirm/refute language)
- **H1** Spearman rho between source-oracle EER and ITW TTG over
  model x seed x policy x target points (clustering by model acknowledged).
- **H4** TTG vs parameter count across the roster (85K -> 315M, 3.5 orders of magnitude).
  **Restated from "SSL size buys separability, not calibration" (D30)** — that form needed
  multiple SSL points and the roster has one, itself unranked. XLS-R may enter this scatter
  as an annotated point but not any ranking claim.

## Frozen decisions
- Roster: AASIST, AASIST-L, RawNet2, Res-TSSDNet, XLSR+AASIST (D30). Tag `roster-freeze`.
- tau policies: EER-point, FRR-1%, FRR-5% on **ASVspoof19 LA dev for every model** (D16 as
  collapsed by D30); never re-tuned.
- Primary metric: HTER@tau* with FRR/FAR decomposition; robustness layer:
  screening (p_bona=.95) + forensic (c_far=10) weighted costs.
- Parity gate: zero decision flips on the 1,000-clip manifest at all three policies.
- Reproduction gate: within 15% relative of published EER. **XLS-R+AASIST enters ungated** —
  Tak et al. publish no 19LA-eval number (only 2021 LA 0.82% / DF 2.85%, both with RawBoost),
  so there is no anchor to gate against. Acceptable only because it is unranked; stated in
  the reproduction table rather than silently omitted.
  **RawNet2 is gated on the 2021 DF eval CLEAN surface: the official B04 scores give 15.064% EER
  there** (amendments 2026-09-18 D32 and 2026-09-19 D33). The released checkpoint is the 2021
  DF-baseline run (19LA train only). The 19LA 5.64% (Tak et al. 2021) comes from a different run
  (train + 90% dev), so it is reported but not gated. The pooled B04 22.38% cannot be gated: for
  43% of trials the organisers scored audio that the public files no longer decode to (D34).
  **Gate result 2026-09-19: PASS**, 15.0636% vs 15.064%, with per-trial agreement to max 3.7e-5
  over 305,525 trials.
- Exclusion rule: models failing the reproduction gate are reported but excluded
  from cross-model claims.
- Eval-set versions: frozen at roster freeze. ASVspoof 5 is **deferred** (D30 open call) —
  if it is later added, that is a logged deviation.
- **ASVspoof 2021 DF surface (D34, amendment 2026-09-19):** the eval subset minus FLAC
  integrity class B, which leaves **455,508 trials**. Class B is the 78,420 trials whose only
  available decode fails the file's STREAMINFO MD5, listed in `configs/df21_classB.txt`.
  Class A (149,983 trials that libFLAC >= 1.4 rejects) is decoded by libFLAC-1.3-equivalent
  ffmpeg and accepted only if the decode matches the stored MD5. The clean-only surface
  (305,525 trials) is reported as a sensitivity analysis.
- **Mixed-precision labelling (D38, OD-6, amendment 2026-09-28):** an int8 artifact that keeps
  K > 0 layers in fp32 is reported as **"int8-mixed (K=k)"** in every table, figure and sentence.
  Plain "int8" means K = 0. The score schema's `precision` value stays `int8`, because
  `score_onnx.py` checks it against the graph's QDQ ops. K and the exclusion list travel in the
  artifact's conversion sidecar and must be joined into every reported row. A row whose K is
  unknown is refused, not printed as "int8".
- Deviations from this document are logged in DEVIATIONS.md with rationale.

## Conversion sensitivity and repair effort (D35 design, frozen by D39, amendment 2026-09-28)
**Exploratory.** It supports H6 with effect sizes and spread and uses no confirm/refute
language. It adds no independent models. Scope: ORT static-int8 PTQ for the four ranked
models. XLS-R is outside this budget.
- **Diagnostic subset:** 2,000 ASVspoof19 LA dev clips (1,000 bona fide + 1,000 spoof),
  drawn uniformly within class with seed 0. Its IDs are frozen in a committed manifest
  (with sha256) before any search. It is used for recipe selection and for H6's D36 budget,
  never for tau (tau stays frozen on the full dev set).
- **Calibration pool:** the remaining 22,844 dev clips (1,548 bona fide). **Calibration size (OD-5)
  = 500 clips per subset (250 + 250) for every model.** A memory failure is a logged deviation,
  never a silent reduction. The 100-clip Res-TSSDNet pilot artifact is kept and disclosed.
- **Default recipe (search step 1):** `convert/ptq.py` defaults: QDQ, per-channel weights,
  S8S8, minmax calibration, `quant_pre_process` on, no exclusions.
- **Recipe search:** at most **12 attempts per model**, including the default and any failure,
  within the existing two-week-per-model timebox. Candidate fp32 exclusions are ranked by the
  fp32-vs-int8 intermediate-activation difference on the diagnostic subset. That ordered
  sequence, and any calibration-method change, is frozen before step 2. Every attempt is logged.
  The locked recipe is the first configuration, in frozen order, with the smallest K that
  meets the D36 budget. Ties go to the earlier attempt.
- **Calibration repeats:** N = **5 per model, seeds 1-5**, under the locked recipe, each
  drawn without replacement within the subset from the calibration pool. Subsets may overlap
  across seeds; the overlap is recorded and each manifest is reused across models. That is 20
  planned attempts. Failed seeds are reported, never replaced, and N is never increased. One
  identical-manifest rebuild per locked recipe is the repeatability control.
- **Canonical device artifact:** the **lowest-numbered seed** whose artifact meets the D36
  budget on the diagnostic subset. If none does, the model has no qualifying int8 artifact.
  That is reported, and no other seed is substituted.
- **Res-TSSDNet** goes through the same procedure at 500 clips. Its arm stays non-blind (D36).
- **Evaluation surface (D41):** every seed, plus the repeatability rebuild, is scored on
  **ASVspoof19 LA eval** (71,237 trials) at the frozen taus. Only the canonical device artifact
  is scored on the ASVspoof 2021 DF verified surface and In-the-Wild, as in the main matrix. No
  surface is added in response to observed results.

## Declared limitations of the frozen roster (write into the paper, do not discover later)
- No handcrafted-feature model (LFCC-LCNN excluded), so on-device feature-extraction parity
  is untested for anything but raw-waveform preprocessing.
- One GAT lineage (AASIST/AASIST-L), so family-level generalization about graph-attention
  models is not supported by this study's own data.
- The ranked roster is pre-SSL-era (2021-22); the single SSL model is unranked and expected
  to produce no device rows.
