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
  **RawNet2 is gated on 2021 DF eval, B04 = 22.38%** (Yamagishi et al. 2021; amendment
  2026-09-18, D32). The released checkpoint is the 2021 DF-baseline run (19LA train only). The
  19LA 5.64% (Tak et al. 2021) comes from a different run (train + 90% dev), so it is reported but not gated.
  Until the DF21 gate runs, RawNet2's status is PENDING and no cross-model claim is final.
- Exclusion rule: models failing the reproduction gate are reported but excluded
  from cross-model claims.
- Eval-set versions: frozen at roster freeze. ASVspoof 5 is **deferred** (D30 open call) —
  if it is later added, that is a logged deviation.
- Deviations from this document are logged in DEVIATIONS.md with rationale.

## Declared limitations of the frozen roster (write into the paper, do not discover later)
- No handcrafted-feature model (LFCC-LCNN excluded), so on-device feature-extraction parity
  is untested for anything but raw-waveform preprocessing.
- One GAT lineage (AASIST/AASIST-L), so family-level generalization about graph-attention
  models is not supported by this study's own data.
- The ranked roster is pre-SSL-era (2021-22); the single SSL model is unranked and expected
  to produce no device rows.
