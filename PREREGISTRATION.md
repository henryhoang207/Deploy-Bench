# Pre-registration — DeployBench study
(Commit this file BEFORE any Phase-3 evaluation run; the commit timestamp is
the citable artifact. Drafted from 02_protocol/audit_plan.md + research_methodology.md Part C.)

## Confirmatory hypotheses (Holm-Bonferroni within family, alpha=0.05)
- **H2** On In-the-Wild at cohort-calibrated tau* (EER policy), FRR > FAR for the
  majority of models; family test: per-model paired bootstrap FRR vs FAR.
- **H3** QTS > 0.01 (1 HTER point) for at least half of (model x int8) cells
  in-domain; per-cell bootstrap CI excludes 0.
- **H5** EnCodec-6k stress increases FRR more than FAR relative to clean, for a
  majority of models (bona-fide corruption asymmetry).
- **H6** int8 conversion failure/fallback occurs for BOTH GAT models (AASIST,
  RawGAT-ST) and NEITHER plain-conv model (Res-TSSDNet, LCNN).

## Exploratory (effect size + CI only; NO confirm/refute language)
- **H1** Spearman rho between source-oracle EER and ITW TTG over
  model x seed x policy x target points (clustering by model acknowledged).
- **H4** TTG vs parameter count across cohorts.

## Frozen decisions
- tau policies: EER-point, FRR-1%, FRR-5% on cohort-matched source dev; never re-tuned.
- Primary metric: HTER@tau* with FRR/FAR decomposition; robustness layer:
  screening (p_bona=.95) + forensic (c_far=10) weighted costs.
- Parity gate: zero decision flips on the 1,000-clip manifest at all three policies.
- Reproduction gate: within 15% relative of published EER.
- Exclusion rule: models failing the reproduction gate are reported but excluded
  from cross-model claims.
- Roster + exact eval-set versions: frozen at week-2 triage (tag `roster-freeze`).
- Deviations from this document are logged in DEVIATIONS.md with rationale.
