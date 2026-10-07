# Deviations from PREREGISTRATION.md

PREREGISTRATION.md froze at tag `roster-freeze` (`97e9753`, 2026-09-28). Every departure
from it after that tag is logged here **before** it runs, with its rationale. Registered
results are never replaced; a deviation adds a labelled, exploratory result beside them.

## DEV-1 (2026-09-29): percentile-calibration arm of the conversion-sensitivity study

**Status:** post-hoc, **exploratory**. Declared after the registered (minmax) arm's
results were known (`results/phase_2_conversion_app/conversion_sensitivity/REPORT.md`,
`768714a`). Approved by the user on 2026-09-29 ("Add the percentile arm").

**Why.** D39 allowed a pre-declared calibration-method change in the recipe search.
None was declared, because the search tooling (`rank_sensitivity.py --calib-method`)
defaulted every exclusion attempt to minmax. Claude chose that default and did not
surface it as a design choice. The Res-TSSDNet pilot (non-blind) had already shown that
percentile calibration was needed to make int8 usable for a raw-waveform model. Under
the registered arm, three of four ranked models have no qualifying int8 artifact.
Without this arm, that result cannot be told apart from an artefact of the calibration
method.

**Design, fixed before any percentile run:**
- Same fp32 exports (their conversion-parity gate already passed), same search
  calibration manifest (`calib_search_seed100.txt`), same 2,000-clip diagnostic
  subset, same D36 budget (int8 EER - fp32 EER <= 1.0 pp), same selection rule
  (smallest K, ties to the earlier attempt).
- **Same exclusion lists in the same order as the registered plans.** The ranking is
  NOT recomputed (it came from the minmax r01 artifacts). The only change is
  `calib_method = percentile` (ORT's default percentile). Every other `ptq.py` knob is
  as in the registered arm.
- Attempts that keep every ranked node in fp32 are dropped before running, because
  they cannot produce an int8 graph (tssdnet r12 left no Q/DQ op). So Res-TSSDNet has
  11 attempts and the other models have 12.
- Attempt ids p01..pNN. Files carry a `_percentile` suffix, and runs live in
  `conversion_sensitivity/<model>/percentile/`. Nothing in the registered arm is
  touched.
- If a recipe locks: seeds 1-5 plus the seed-1 rebuild, scored on the diagnostic subset
  and on ASVspoof19 LA eval (as D41), and the canonical rule of D39.
- Failures (including memory failures of histogram calibration) are results. Nothing
  is retried, replaced or re-run.

**What it may and may not be used for.** It is reported as exploratory, next to the
registered arm and never instead of it. **A percentile-arm artifact does not become a
device artifact** unless a further, logged decision says so. Under D39, the registered
device-artifact rule gives RawNet2 seed 2 and no int8 artifact for AASIST, AASIST-L or
Res-TSSDNet.

### DEV-1 amendment (2026-09-29, before any percentile result existed)
- **The first launch (08:29, 4 models in 2 lanes) ran sol1 out of memory.** ORT 1.23's
  `HistogramCalibrater.collect_data` keeps every intermediate output of all 500
  calibration clips in RAM. AASIST p01 died with no ledger entry, most likely killed by the
  kernel OOM killer (the kernel log is not readable without root). AASIST-L p01 passed
  105 GB (machine 125 GB, shared), so Claude stopped every percentile process, including
  tssdnet p01, which was still running. **No percentile attempt had a ledger entry or an
  artifact** (all p01 directories were empty), so nothing was produced and nothing is
  being retried.
- **Fix, part of the arm's recipe from here:** `ptq.py --calib-chunk 20` feeds the
  histogram calibrator 20 clips at a time. HistogramCollector merges repeated calls by
  extending its bins at the first chunk's stride, so the percentile ranges differ slightly
  from a single 500-clip pass. The chunk size is recorded in every sidecar and ledger line.
- The already-derived `search_plan_<model>_percentile.json` files are unchanged: the chunk
  is not a plan field. The arm now runs **one model at a time**, under a watchdog that
  stops the job if sol1's available memory falls below 15 GB.

## DEV-2 (2026-09-30): H5 source set and test details, specified before any stressed audio exists

**Status:** fills details PREREGISTRATION.md leaves open for the confirmatory H5 ("EnCodec-6k
stress increases FRR more than FAR relative to clean, for a majority of the 4 ranked
models"). Decided by the user on 2026-09-30 ("use the parity manifest for H5"). Fixed and
committed **before** any EnCodec audio was generated or scored, so no H5 result informed it.

- **Source set:** the frozen 1,000-clip 19LA-eval parity manifest
  (`configs/parity_manifest_tssdnet.txt`, 500 bona fide + 500 spoof, seed 20260725, ids sha256
  `2a3c8964...`). Conditions: `clean` (a byte copy through the same stress path) and
  `encodec6k` (EnCodec 24 kHz model at 6 kbps, resampled to 16 kHz;
  `deploybench/stress.py`). The versions are recorded in the stress set's `manifest.json`.
- **Scoring:** fp32 PyTorch on CPU, each model's reproduction checkpoint, through
  `score.py --stress-manifest` for both conditions. The precision axis is not part of H5.
- **Test (per ranked model):** at the frozen tau_eer, dFRR = FRR(encodec6k) - FRR(clean) and
  dFAR = FAR(encodec6k) - FAR(clean). The statistic is dFRR - dFAR. Clips are resampled with a
  class-stratified paired bootstrap: the same resampled clip ids feed both conditions.
  B = 1,000, numpy default_rng(0). One-sided p = fraction of resamples with
  dFRR - dFAR <= 0. Holm-Bonferroni within the H5 family (4 models), alpha 0.05. H5 is
  supported if >= 3 of the 4 ranked models reject. Script: `analysis/h5_test.py`.
- **Scale caveat, to be stated with the result:** 500 clips per class give coarse FRR/FAR
  steps (0.2 pp), so small effects will not reach significance at this size.

## DEV-3 (2026-09-30): H3 when a ranked model has no int8 artifact

**Status:** a counting rule PREREGISTRATION.md does not state. Decided by the user on
2026-09-30 ("count models without int8 as not meeting QTS").

- H3: "QTS > 0.01 (1 HTER point) for at least half of (ranked model x int8) cells in-domain;
  per-cell bootstrap CI excludes 0." In-domain = 19LA eval at 19LA-dev taus (D37).
- **Rule:** the denominator is all 4 ranked models. A model with no int8 artifact under the
  registered D39 procedure (AASIST, AASIST-L, Res-TSSDNet) counts as a cell that does **not**
  meet the criterion. The percentile arm (DEV-1) supplies no H3 cells.
- **Disclosed consequence:** the rule was chosen after the conversion results were known.
  With 3 of 4 cells fixed as not meeting it, H3 **cannot be supported** ("at least half" needs
  2 of 4), whatever RawNet2 shows. It is the conservative reading. The alternative (excluding
  missing cells) would have let H3 rest on one model.
- **RawNet2's cell is still computed and reported.** Artifact: the D39 canonical int8-mixed
  (K=16), r09 seed 2. tau_q = the EER tau of that artifact's scores on the full 19LA dev set
  (24,844 trials, scored for this purpose). QTS = HTER(eval, tau_fp32 eer) - HTER(eval, tau_q).
  CI: class-stratified bootstrap of the 71,237 eval trials, taus held fixed, B = 1000, seed 0,
  2.5/97.5 percentiles. The cell meets the criterion if QTS > 0.01 and the CI lower bound
  is > 0. Script: `analysis/h3_test.py`, committed before the int8 dev scores existed.

## DEV-4 (2026-10-02): percentile-arm artifacts on the devices

**Status:** post-hoc, **exploratory**. This is the logged decision DEV-1 requires before a
percentile artifact can become a device artifact (D45). Decided by the user on 2026-10-02 ("go",
on Claude's recommendation). **What was known:** all registered and percentile desktop results
(REPORT.md and its addendum), including AASIST-L seed 3 at +8.8 pp on 19LA eval. **No device
data exist** (no handset has been attached), so no device result informed it.

- **Registered device int8, unchanged:** RawNet2 r09 seed 2, int8-mixed (K=16) (D39). It stays
  the only int8 artifact in any registered device analysis.
- **Added, exploratory, labelled "percentile" everywhere:** the percentile-arm canonical
  artifacts, chosen by the D39 rule applied inside DEV-1 (lowest qualifying seed):
  RawNet2 p07 seed 1, int8-mixed (K=8); AASIST-L p05 seed 1, int8-mixed (K=4).
  They are checked against the ledger sha256 before any device run, like the registered one.
- **Not added:** AASIST (its percentile arm has no canonical artifact; see DEV-5) and
  Res-TSSDNet (no qualifying percentile recipe).
- Each enters the device campaign under the same per-artifact procedure as the registered
  artifact. Results are reported beside the registered ones, never instead of them.
- **Caveat stated with any AASIST-L result:** the device build is one calibration draw. Its
  seeds span +0.9 to +8.8 pp on 19LA eval.

## DEV-5 (2026-10-02): repair effort after the conversion failures

**Status:** post-hoc. Decided by the user on 2026-10-02 (D46). **What was known:** every fp16
result (aasist, aasist_l, tssdnet lossless on the parity manifest; rawnet2 unloadable;
xlsr_aasist export fails), and the AASIST percentile repeat failures.

- **RawNet2 fp16: one repair attempt.** The fp16 pass rewrites `/m/Sinc_conv/Cast`'s declared
  output to float16 but leaves its `to` attribute at float, so ORT refuses the graph
  (`main_matrix/README.md`). The repair keeps that one node in fp32
  (`to_onnx.py --fp16-node-block /m/Sinc_conv/Cast`; the converter wraps it in Casts). The
  mechanism is reproduced on a toy graph in `tests/test_conversion_tooling.py`.
  - Exactly **one attempt**, on the pinned sol1 toolchain. If it does not load, or fails, that
    is reported and nothing else is tried.
  - If it loads, it is evaluated exactly like the other fp16 rows: the 1,000-clip parity
    manifest (diagnostic, never gates), then DF21 verified and ITW at the frozen taus.
  - **Reporting:** the registered outcome stays "fp16 converts for 3 of 4 ranked models out of
    the box". The repaired artifact is labelled "fp16 (repaired: 1 node in fp32)" and reported
    beside it, with the repair effort (1 attempt, 1 node).
- **AASIST percentile repeats: not repaired.** A fix (a first pass that fixes each tensor's
  range before binning) changes the calibrator. The repeats would then not be comparable with
  the locked p06 attempt, so the whole chain would need a re-run as a separate sub-arm. The
  six failures stay as logged tooling failures, and the AASIST percentile arm has no canonical
  artifact.
- **XLS-R export: not repaired.** XLS-R is an unranked server reference and enters no ranking
  claim or hypothesis count. The failure in fairseq's `pad_to_multiple` under the tracer is
  reported as a toolchain finding.

## DEV-6 (2026-10-03): the MLAAD-5 evaluation surface

**Status:** a surface the protocol planned (`02_protocol/audit_plan.md`: "MLAAD v5+, 5 languages
(en, de, it, fr, pl)", the language shift axis) but PREREGISTRATION.md never specified. It is
specified here before any MLAAD or M-AILABS audio has been scored. Decided by the user on
2026-10-03 (D47: "MLAAD: score it"; bona fide = "M-AILABS, matched"). **What was known:** every
19LA-eval, DF21 and ITW result in this study. **Exploratory:** no registered hypothesis test uses
MLAAD. H1 stays the ITW test it was registered as. M-AILABS acquisition started on sol1 before
this entry, which governs only scoring.

- **Spoof side:** Hugging Face `mueller91/MLAAD` (gated, CC BY-NC 4.0), the snapshot current at
  download. Its commit hash and a per-file sha256 list are committed with the protocol. Every
  `fake/<lang>/<system>/*.wav` clip for lang in {en, de, it, fr, pl}, label 0.
- **Bona fide side:** M-AILABS, the corpus MLAAD synthesises from, for the same languages:
  en_US + en_UK, de_DE, it_IT, fr_FR, pl_PL (`ics.tau-ceti.space` mirror; sizes checked
  against Content-Length; sha256 recorded). Per language, a uniform draw without replacement
  of as many bona fide clips as that language has spoof clips, numpy default_rng(20261003),
  label 1. Where M-AILABS has fewer clips, all of them are used and the imbalance is reported.
- **Frozen before scoring:** `scripts/build_mlaad_protocol.py` writes the protocol
  (utt_id, path, label, lang, system), and its sha256 is committed. The loader reads only that
  file and refuses unknown labels.
- **Scoring:** the same artifacts as the main matrix, at the frozen 19LA-dev taus (nothing
  re-calibrated): fp32 PyTorch for all 5 models; ORT fp16 for the 4 ranked models (RawNet2 =
  the D46 repaired artifact, labelled as such); ORT int8 = the D39 canonical RawNet2 artifact;
  and the ExecuTorch artifacts of DEV-7 once they exist. Audio is resampled to 16 kHz by
  the harness's existing `resample_poly` path.
- **Reported:** EER; HTER, FRR and FAR at tau_eer/tau_frr1/tau_frr5; TTG; dEER and flips
  against fp32. Pooled and per language. Descriptive; no confirm/refute language.
- **Confound to state with every MLAAD number:** the bona fide audio is native 16 kHz; the
  spoof audio comes from TTS systems at other rates and is resampled. A detector can separate
  the classes partly by resampling traces, so MLAAD EER is not a clean language effect.

## DEV-7 (2026-10-03): ExecuTorch as a second runtime (OD-4 resolved: report both)

**Status:** post-hoc, **exploratory**. Resolves OD-4. Decided by the user on 2026-10-03 (D48:
"OD-4: report both"; int8 = "Default recipe, 5 seeds"). **What was known:** all ORT conversion
results (REPORT.md and its addendum, main_matrix/README.md). ExecuTorch results never enter a
registered hypothesis test or replace an ORT result.

- **Toolchain:** its own venv on sol1 (`envs/et`, Python 3.12; executorch 1.5.1 needs
  numpy >= 2, which the fairseq-pinned db310 env cannot take). The installed torch/torchao/
  executorch versions are recorded in every conversion log. XNNPACK partitioner.
  `export_module()` is the exported graph (D13), as for ONNX.
- **Models:** the 4 ranked models. XLS-R is not attempted: it is the unranked reference, it
  enters no runtime comparison, and it failed ONNX export in fairseq code.
- **fp32 .pte:** gated like ORT fp32. Zero decision flips at tau_eer/frr1/frr5 against PyTorch
  fp32 on the 1,000-clip parity manifest. A failed gate is reported, not repaired.
- **fp16 .pte:** a half-precision copy of the export module, input cast to fp16 and output to
  fp32 inside the graph, lowered to XNNPACK. Diagnostic against fp32 on the parity manifest.
  A failed export is a result (no repair without a further logged decision).
- **int8 .pte:** PT2E static quantisation with XNNPACKQuantizer and its default symmetric
  config (per-channel weights), with **no fp32 exclusions and no recipe search**. It is
  calibrated on each of the five D39 calibration manifests (seeds 1-5, 500 clips each),
  plus one identical-manifest rebuild as the repeatability control. Each seed is scored on the
  2,000-clip diagnostic subset and on ASVspoof19 LA eval (as D41).
  **Canonical device artifact:** the lowest seed within +1.0 pp diagnostic EER of the
  model's ExecuTorch fp32 (the D36 budget and D39 rule). If none qualifies, the model has no
  ExecuTorch int8 artifact. That is reported, and no seed or recipe is substituted.
- **"Converts":** the executorch Python runtime loads the .pte and returns one finite score
  (same rule as ORT, `to_onnx.ort_smoke`).
- **Surfaces:** parity manifest (all precisions); DF21 verified, ITW and MLAAD-5 (DEV-6) for the
  fp16 and canonical int8 artifacts, as for ORT.
- **Devices:** an ExecuTorch runner in the app. `executorch` cells stay refused (exit 3) until
  the runner exists and the desktop fp32 gate has passed. The device grid then becomes
  4 models x 3 precisions x 2 runtimes per phone.

## DEV-8 (2026-10-04): reviewer-requested sensitivity analyses for the conference paper

**Status:** post-hoc, **exploratory**, declared **before any of it was run**. Every result below is reported
beside the registered ones and never instead of them. Requested by a simulated peer review (ARS
academic-paper-reviewer v1.10.0, mode `full`, 2026-10-04; five reports and an editorial roadmap, venue IEEE ICME;
files in `DFV_Conference/peer_review/`). The user asked for a revision based on it (`/ars-revision`) and approved on
2026-10-04: the read-only copy of four metadata files from sol1, a silence-artefact arm with VAD-trimmed re-scoring,
and this entry (D51).

**Why.** The review found that (i) the registered H2 and H5 tests resample clips, although ITW and LA19 clips come
from a few dozen speakers, and H2's resampling unit was fixed after the descriptive ITW rates had been seen
(DEV-2 left it open); (ii) tau_eer's estimator is not named and its uncertainty is not propagated; (iii) the policy
grid has no lenient (FAR-budget) point, so the direction of the error is partly built into where the thresholds sit;
(iv) TTG and the SD-normalised safe band are not comparable across models; (v) the LA19 silence cue (Mueller et al.
2021) is an untested alternative explanation for the ITW false rejections.

**What was known when this was written:** every registered result and every descriptive table in the conference draft
(`paper/conference/main.tex`, e73c675), including the ITW FRR/FAR at all three policies; the five review reports.
Metadata copied from sol1 on 2026-10-04 (sha256 in `DFV_Conference/revision/meta/PROVENANCE.md`) and inspected for
structure only, using labels and counts and never a threshold or a decision: ITW `meta.csv` has **54** distinct
speaker labels (the dataset paper says 58 public figures; clips per speaker 14-3,636, the top five supply 50.5%),
every speaker has both classes; the 1,000-clip H5 manifest covers all 67 LA19-eval speakers (3-29 clips each) and all
13 eval attacks; DF21 eval has 93 speakers; every label in the metadata equals the label in the committed score files
for all 5 models. **No result of any analysis below exists**: no cluster bootstrap, no new policy, no VAD score.

**Inputs.** The committed fp32 PyTorch score files, the frozen taus (`tau_<model>.json`, never changed), the
metadata above. Every script refuses to run unless the point estimates it recomputes equal the committed
`results/analysis/{operating_points,h1_h4,h2_test,h5_test}.json` to 1e-12 (the harness rule: no plausible wrong number).

**A. Speaker-cluster inference (R1, S1, S2, S4).**
- **A1 H2, cluster version.** Resampling unit = ITW speaker label (54). Resample speakers with replacement, take all
  of each speaker's clips of both classes. Statistic FRR - FAR at the frozen tau_eer, per ranked model. B = 10,000,
  numpy `default_rng(0)`. One-sided p = (1 + #{stat <= 0}) / (B + 1), so p >= 1/(B+1). Holm within the 4 models,
  alpha 0.05, supported if >= 3 of 4 reject (the registered rule, with only the unit changed). The registered
  class-stratified clip-level result stays the registered H2 outcome.
- **A2 H5, cluster version.** Unit = LA19-eval speaker id of the 1,000-clip manifest (67). Same statistic
  (dFRR - dFAR at tau_eer, clean vs encodec6k, same clips in both conditions), B = 10,000, seed 0, same p and Holm
  rule. Plus the two components, each one-sided and Holm-corrected within its own 4-model family: dFRR > 0 and
  dFAR < 0.
- **A3 TTG intervals.** Speaker-cluster bootstrap, B = 1,000, seed 0, percentile 95% interval of TTG at every
  policy, on ITW (54 units), DF21 verified (93 speakers) and LA19 eval (67 speakers). Beside the existing
  class-stratified intervals.
- **A4 Per-speaker view (ITW).** Clips per speaker; per-speaker FRR and FAR at tau_eer; the share of FRR from the top
  five speakers; worst-k speakers (reported as anonymous ranks in the paper).
- **A5 "Cheap check" simulation.** ITW, per ranked model and the reference. For k in {50, 100, 500}, 1,000 random
  splits of the 54 speakers into a calibration half (27) and a held-out half (27). Draw k genuine clips uniformly
  from the calibration half. (i) Estimate FRR at the shipped tau_eer from the sample and compare with the held-out
  genuine FRR. (ii) Set tau_new = the (floor(0.01 k) + 1)-th smallest sample score (the largest threshold whose
  sample FRR is <= 1%, the `tau_at_frr` rule on the sample's bona fide scores); report held-out FRR and FAR (spoofs of
  held-out speakers) at tau_new next to the same at tau_eer. Medians and 5-95% spread over splits. seed 0.

**B. Threshold estimation (R2).**
- **B1 Documentation.** The estimator is `metrics.eer`: the linear interpolation of the dev FRR = FAR crossing over
  the unique dev scores (first grid point with FRR > FAR; interpolation between it and its predecessor). frr1/frr5 are
  dev bona fide quantiles (`tau_at_frr`). Reported per model: dev EER, the three taus, dev margin (smallest dev bona
  fide score minus largest dev spoof score), and the number of dev trials that pin tau_eer (bona fide below it, spoofs
  at or above it).
- **B2 Dev-set bootstrap.** Class-stratified clip resampling of the 24,844 dev trials (2,548 / 22,296), B = 1,000,
  seed 0. In each replicate re-estimate all five policies (eer, frr1, frr5, far1, far5) with the harness functions
  and evaluate FRR, FAR and TTG on the full ITW, DF21-verified and LA19-eval sets. Percentile 95% intervals. This
  carries dev-threshold uncertainty only; the target sampling uncertainty is A3.
- **B3 EER-equivalent interval.** Per model, the connected run of dev grid thresholds around tau_eer whose dev HTER is
  at most HTER_dev(tau_eer) + 0.5 / 2,548 (one dev bona fide trial); target FRR and FAR at both ends and at tau_eer.
- **B4 Reconciliation with Zhou and Wang (arXiv 2606.21584).** FRR and FAR on ITW for every model at a threshold set
  by the same estimator on the LA19 **eval** scores (their convention), against the registered dev threshold. The
  checkpoint and preprocessing differences cannot be tested.

**C. FAR-budget policies (R3, DA M1).** tau_far1 and tau_far5 = `tau_at_far` on dev (the most lenient threshold whose
dev FAR is <= 1% / 5%). FRR, FAR, HTER, TTG on all targets. Descriptive companion: the target-oracle FRR at FAR = 5%
(the smallest target threshold with target FAR <= 5%, a BPCER20-style quantity), to split a lenient-policy FRR into
discrimination and transfer.

**D. Measures (R4, S3).** HTER(tau*) and oracle HTER beside TTG; the transfer share TTG / HTER(tau*) and the
reviewer's TTG / (50 - oracle HTER). Rank-based band: the share of target clips (pooled and per class) whose score
lies in the 1-pp-HTER band around the oracle, and the share whose decision differs between tau* and the oracle
threshold. Both are invariant to monotone score transforms.

**E. Cost sweep and stability (S5, R1).** Normalised cost on an effective-prior axis,
C(p) = [p FRR + (1 - p) FAR] / min(p, 1 - p), p in [0.01, 0.99] on a log-odds grid, at the five frozen thresholds
and at the oracle (minimum over thresholds and the two trivial systems). The registered points are screening
p = 0.95 and forensic p = 1/11 (pi = 0.5, c_far = 10). Break-even for screening: C < 1 iff FRR < (1 - FAR) / 19.
Stability: B = 1,000 (seed 0) speaker-cluster replicates for ITW and DF21 verified; per replicate, the model with the
largest HTER gap, screening cost gap and forensic cost gap at tau_eer; report how often the point-estimate's
argmax is the argmax.

**F. Silence artefact (R7).** On sol1, in a new scratch directory `/mnt/data8tb/henry/dfv_rev/`. The harness is
imported read-only from `/mnt/data8tb/henry/deploybench` with bytecode writing off; nothing in that clone, its
results or the running MLAAD scoring is touched; CPU jobs run under `nice`; scoring uses the less loaded GPU.
- **F1 Statistics (CPU).** For LA19 dev (24,844), LA19 eval (71,237) and ITW (31,779): leading and trailing silence
  at 16 kHz after the harness's `load_wav`. Energy VAD: 25 ms frames, 10 ms hop, only full frames,
  E = 10 log10(mean x^2 + 1e-12) dB, a frame is active if within T dB of the clip's loudest frame, T in {25, 35, 45}.
  Leading silence = start of the first active frame, trailing = samples after the end of the last. Clips shorter than
  one frame have 0 / 0. Relation to the per-clip decisions already held: Spearman(score, leading silence) per class,
  and FRR at tau_eer by leading-silence quintile among ITW genuine clips.
- **F2 VAD-trimmed re-scoring (GPU).** Trim both ends at **T = 35 dB** (fixed here as the primary, with zero margin),
  using the bounds from F1. If the trimmed clip would be shorter than 0.25 s the clip is left untrimmed and counted.
  Sets: LA19 dev, LA19 eval, ITW. All 5 models, fp32 PyTorch on GPU, TF32 off via `load_model`, checkpoints exactly
  as in `scripts/sol1/mlaad_scoring.sh`, path `load_wav -> trim -> fix_length -> model` (only the trim is new).
  **Gate before any trimmed run:** a no-trim run on the first 200 utterances of each set must equal the committed
  scores (|d| <= 1e-4); otherwise stop. DF21 is not re-scored (its eval subset is untrimmed audio, and re-scoring
  455,508 clips would not change the question).
- **F3 Thresholds and reporting.** tau_trim = the five estimators of B/C on the **trimmed dev** scores of each model,
  computed from dev only. Reported, descriptively and with no test: EER on LA19 eval and ITW, trimmed vs untrimmed;
  FRR, FAR, TTG on ITW and LA19 eval at (a) the registered untrimmed taus and (b) tau_trim. Reading rule fixed now: if
  the ITW false-rejection pattern were mainly a silence artefact, trimming both dev and target should shrink ITW
  FRR at tau_trim relative to the untrimmed baseline; if FRR stays above FAR at a similar size the artefact does not
  explain it; mixed outcomes are reported as mixed. Sanity check, descriptive: Mueller et al. report RawNet2's LA19
  EER rising from 3.6% to 15.5% when silence is trimmed; if ours do not degrade the trimming is not biting, and that
  is stated.

**G. Documentation (S6).** Per-condition and per-class counts of DF21 eval (533,928), integrity class A (149,983),
class B (78,420), the verified surface (455,508) and the clean surface (305,525), from `configs/df21_class{A,B}.txt`
joined to `trial_metadata.txt`. H5 manifest composition (attacks, speakers). The ITW speaker table.

**Not done, by decision.** (i) A new or modern detector (S9): the roster is frozen (D1, D30); the paper scopes its
claims instead. (ii) Two-way speaker x recording clustering: ITW `meta.csv` has no recording id. (iii) DF21
`hidden` (18,576 trimmed trials) and `progress` (59,325) subsets: not part of the registered surface, their FLAC
integrity is unclassified, and scoring them would be a further deviation. (iv) Re-running any registered analysis with
other settings.

**Reporting rules, fixed now.** The registered clip-level H2 and H5 remain the registered outcomes. The cluster
versions are reported beside them and labelled as this deviation. If H2 or H5 is supported under one resampling unit
and not under the other, the abstract and the conclusion say so. Every other quantity here is labelled exploratory,
with effect sizes and intervals and no confirm/refute language. Code lives in `DFV_Conference/revision/analysis/`;
it is not committed by Claude, and the user decides what moves into `analysis/`.

### DEV-8 amendment 1 (2026-10-04, after the F2 gate result, before any trimmed run)
- The F2 gate (no-trim scoring of the first 200 utterances per set, |d| <= 1e-4 against the committed scores) passed
  for AASIST (max 4.8e-6), AASIST-L (3.8e-6) and RawNet2 (1.7e-5). It **failed for Res-TSSDNet on LA19 eval: 1.16e-4**
  (LA19 dev 7.4e-5, ITW 0.0).
- **Cause, checked in the committed files:** `scores_repro_tssdnet.csv` was computed on the Windows laptop; the
  committed sol1 copy `scores_repro_tssdnet_linux.csv` differs from it by up to 3.7e-4 (mean 1.2e-5; the scores span -42
  to +26). So the 1e-4 bound was tighter than cross-platform float32 noise for this model. Its ITW reference was
  computed on sol1 and is reproduced exactly (0.0), so the scoring path itself is identical.
- **Amended gate, Res-TSSDNet LA19 eval only:** the platform-matched reference `scores_repro_tssdnet_linux.csv`, same
  1e-4 bound. Every other check and bound is unchanged. If this check also fails, the Res-TSSDNet trimmed run does not
  start (the other models are unaffected).

### DEV-8 amendment 2 (2026-10-05, after the F3 scores were collected; adds no analysis, threshold or test)
- **F2/F3 ran as declared.** The VAD-trimmed fp32 scores of all five models on LA19 dev, LA19 eval and ITW (15 files) were
  copied read-only from `/mnt/data8tb/henry/dfv_rev/scores/` (sha256 equal on both sides) to
  `DFV_Conference/revision/results/vad/`. `f_vad_analysis.py scores` wrote `f_taus_trim.json` (taus from trimmed dev scores
  only, written before any trimmed target was opened) and `f_vad_scores_analysis.{txt,json}`.
- **Sanity check of F3: trimming bites.** LA19-eval EER rises for all five models (0.83 -> 21.31, 0.99 -> 20.43,
  4.60 -> 27.53, 1.62 -> 23.49, 0.15 -> 5.08%).
- **F3 reading rule: mixed, reported as mixed.** At tau_trim on trimmed ITW, FRR falls for all five models (by 9-38 pp)
  but FAR rises (by 3-34 pp); FRR > FAR disappears for AASIST, Res-TSSDNet and XLS-R and persists barely for AASIST-L
  (41.5 vs 39.3%); ITW EER moves by -4.0 to +3.1 pp. A more lenient threshold from any harder dev set would also lower
  FRR, and this single contrast cannot separate that from the silence cue. Descriptive, no test.
- **Correction to the F3 text.** F3 attributes Mueller et al.'s 3.6% -> 15.5% to RawNet2. Their abstract does not name the
  model ("established antispoofing models using signal-based features") and their intervention retrains on trimmed audio.
  The paper describes the result without naming a model and says the two interventions differ.
- **Descriptive quantities added for the text** (no test, no threshold, committed scores only):
  `revision/analysis/facts_extra.py` -> `results/extra_facts.{txt,json}`: ITW AUC per model; score-saturation percentiles
  (XLS-R on LA19 eval and ITW; RawNet2's score ceiling); the share of ITW and DF21 clips, per class, whose score lies inside the
  B3 dev-equivalent interval (the triage illustration for S8). `supp_tables.py` prints the registered cost check at tau_eer
  split into the oracle cost and the remainder, read from the committed `results/analysis/operating_points.json`.
- **Number audit.** `audit_claims.py` rebuilds 77 numerical statements of the revised paper from the result files: 0 failures.
- Nothing in the sol1 clone, its results or the MLAAD scoring was touched.

## DEV-9 (2026-10-07): round 2, registered before any of its scores existed

**Status:** a registered extension, frozen at tag `round2-freeze` together with `PREREGISTRATION_ROUND2.md`, which holds
the full design (models, training runs, data, gates, thresholds, four confirmatory hypotheses R2-H1 to R2-H4, a reading
rule fixed in advance, exploratory analyses E1-E11, failure rules). It departs from `PREREGISTRATION.md` in three ways, all
governed by the addendum, and changes nothing in round 1 (roster, H1-H6, tests and outcomes stand as reported):
1. The frozen five-model roster (D30) is extended, outside every round-1 count, by two released detectors
   (`as5_aasist`, the ASVspoof 5 baseline B02; `xlsr_mamba`, XLSR-Mamba-DF) and one re-implemented calibration-oriented
   detector (`pascu_lr`, refit on LA19 train only). ASVspoof 5 returns as the source corpus of `as5_aasist` (D30 had
   deferred it); its eval set is a target for that model only and is reported by condition (D24).
2. The project trains models again (D55): the clovaai AASIST-L recipe, seeds 1-5 untrimmed and 1-5 on VAD-trimmed audio
   (the DEV-8 rule), plus the default seed 1234 as a pipeline check; and the Pascu logistic-regression head.
3. The stress suite runs eight more conditions on the 1,000-clip H5 manifest (D56).

Requested by the user on 2026-10-06 ("push it to the IEEE T-BIOM", then "Add it to the current plan and next step") and
frozen on 2026-10-07 under "go ahead from end to end with review check between each step": the registration was frozen
after Claude's review check, without a separate approval step. Decisions D53-D58.

**What had been seen:** `PREREGISTRATION_ROUND2.md` Sec. 2: every round-1 and DEV-8 result, the published figures it
names, and pre-tag smoke tests on LA19 training clips only. No score of a round-2 model on any dev, evaluation or target
set existed, and no stressed audio for the eight new conditions.

**Code fixed before the tag** (`7592468` and the registration commit): the ASVspoof 5 loader, the round-2 wrappers,
`score.py`'s whole-utterance and `vad35` paths, the training and fitting scripts, the registered analysis scripts, and one
defect found in review: `deploybench/stress.py` convolved an impulse response read at another sample rate as if it were
16 kHz; it now refuses the mismatch.

### DEV-9 amendment 1 (2026-10-07, after a launch failure, before any relaunch)
**What happened.** The eleven AASIST-L runs were launched together at 09:26 (+07), six on GPU 0 and five on GPU 1.
AASIST-L training needs 3.5-5 GB of GPU memory per run, not the 1-2 GB assumed, so the six runs on GPU 0 failed with CUDA
out-of-memory within ten seconds, before completing a single epoch (empty `epochs.jsonl`, no weights). The five trimmed runs
on GPU 1 were still counting trim statuses (no training step taken) and were stopped by hand to prevent the same failure.
No dev score, checkpoint or training metric of any run exists.

**Amendment.** Sec. 10's rule that a crashed run "is reported and not repeated" targets failures of the recipe (divergence,
NaN loss, a crash during training). A run that fails from resource exhaustion before completing its first epoch, that is
before any dev score exists, is relaunched once from scratch with the same seed, code and data, and the failure is reported.
The eleven runs are relaunched by `scripts/sol1/round2/r2_train_sched.py`, which starts a run only when its GPU has room
(each of our runs counted at 5 GB from launch, other processes at their measured use, 2 GB margin, at most four runs per
GPU) and keeps the arms interleaved (r2 s1, trim s1, r2 s2, ...). The failed directories are kept, renamed, for the record.
Nothing else changes.

**Seen when written:** the out-of-memory tracebacks and the GPU memory figures above; no dev score or training metric.
- *Same day, 09:31-09:33 (+07):* the scheduler's 5 GB per run was wrong too. The first relaunched run
  (`aasist_l_r2_s1`, GPU 0) reserved 15.5 GB once alone, and the scheduler had already started `aasist_l_trim_s1` on the
  same GPU; that run was stopped during its trim-status pre-count (no training step) to keep `aasist_l_r2_s1` safe, and its
  directory was set aside like the first ones. The scheduler now counts each run at 16 GB and skips runs already started,
  which in practice means one run per GPU beside the scoring jobs. Still nothing else changes, and still no dev score or
  training metric had been seen.

### DEV-9 amendment 2 (2026-10-07, while writing E9, before E9 ran on any data)
E9 was registered with "logistic calibration on source dev (`deploybench.metrics.platt_fit`)". A unit test written for E9
showed that `platt_fit` (fixed-step gradient descent, 200 iterations) does not reach the maximum-likelihood fit even on
unit-scale scores (slope 0.889 against 1.001) and settles on a wrong fit at a x20 score scale (slope 0.028 against 0.050)
that RawNet2 and Res-TSSDNet scores have. E9 therefore uses the exact unpenalised logistic fit (scikit-learn), which is
scale-equivariant; the rest of E9 (prior correction, Bayes thresholds over logit(prior) in [-4, 4], actual vs minimum
cost) is unchanged. E9 is exploratory, and no E9 quantity had been computed on any score file.
