# Conference paper (D49): fixed-threshold transfer audit

Venue-neutral IEEE two-column draft (`IEEEtran`, conference mode). **6 pages including references**
(currently ~5.7). fp32 desktop results only. Plan: OneDrive `phase_4_writing/step_05_companion_paper.md`;
split rationale: `02_protocol/publication_split.md`.

**Out of scope, by decision (D49):** conversion, fp16/int8, QTS (H3), H6, calibration-seed spread, the
percentile arm, repairs, ExecuTorch, any device number, the full stress suite, the decision table. They are
the journal paper's new content. One sentence says they are reported separately; nothing more.

## Build
```
python analysis/figures/conference_figures.py      # from the repo root: writes figures/*.pdf
pdflatex main && bibtex main && pdflatex main && pdflatex main
```
Figure PDFs and LaTeX outputs are build products (`paper/.gitignore` ignores `*.pdf`, as for the TIFS
figures); only `figures/conference_figures_data.json` is tracked. MiKTeX now has `ieeetran`.
`\blindtrue` (line ~22) switches to an anonymous author block and placeholder links for double-blind review.

## Where every number comes from
Each paragraph, table and figure carries `% src:` comments. The main sources:

| Element | Source (repo) | Script |
|---|---|---|
| Table I (roster, reproduction) | `results/phase_1_reproduction/repro_report_<m>.md`, `results/analysis/param_counts.json` | `analysis/repro_report.py` |
| Table II (EER, FRR/FAR, TTG, FRR at the 1%/5% budgets) | `results/analysis/operating_points.{txt,json}` | `analysis/operating_points.py` |
| Table II "Band", sensitivity numbers | `paper/figures/figures_data.json` | `analysis/figures/paper_figures.py` |
| Table III (H2, H5) | `results/analysis/h2_test.txt`, `h5_test.txt` | `analysis/h2_test.py`, `h5_test.py` |
| H1, H4, TTG CIs on ITW/DF21 | `results/analysis/h1_h4.{txt,json}` | `analysis/h1_h4.py` |
| Cost-weighted check, clean-only DF21, score percentiles | `results/analysis/operating_points.{txt,json}` | `analysis/operating_points.py` |
| Fig. 1, Fig. 2 | `figures/*.pdf`, `figures/conference_figures_data.json` | `analysis/figures/conference_figures.py` |

`operating_points.py` (new, 2026-10-03) computes the **registered** cost-weighted robustness layer
(PREREGISTRATION "Frozen decisions"; methodology C4) for the first time. It refuses to run unless its HTER, TTG,
FRR and FAR equal `figures_data.json` and `h1_h4.json` to 1e-12 and its clean-only DF21 rows equal
`transfer_matrix.txt`. `conference_figures.py` refuses unless its curves equal `figures_data.json`.

## Open before submission ([U] unless marked)
- Venue (Interspeech 2027 recommended in D49 notes; this draft is IEEE-format at the user's request). Port to
  the venue template and its page/abstract rules once chosen. Abstract is 197 words.
- Title (D30): working title "Calibrated Once, Deployed Elsewhere: A Pre-Registered Audit of Threshold Transfer
  in Speech Deepfake Detection".
- Author block: affiliation, e-mail, co-authors (red placeholders).
- **The harness must be public (or an anonymised mirror) at submission:** the abstract and conclusion say it is
  released; the GitHub repo is private. Repository URL placeholder in `\repo`.
- Whether to disclose H3/H6 outcomes here. The statement says they are reported separately (as the plan
  wrote it); both were not supported, so a reviewer may read the split as selective reporting.
- MLAAD-5: if it lands in time (Track A), add it as a language-shift paragraph with the DEV-6 sample-rate
  confound; the current text lists language shift as a limitation.
- [C] Before submission: re-run `analysis/operating_points.py` and `analysis/figures/conference_figures.py`
  and re-check every `% src:` number; re-check reference venues (refs.bib header).
