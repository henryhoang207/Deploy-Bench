# Analyses added in revision (Deviation 8)

All exploratory except the speaker-cluster versions of H2 and H5, which are reported beside the registered clip-level tests. Each
script imports the harness estimators from the repository root (or from `DEPLOYBENCH`), reads the committed scores and the frozen
thresholds, and stops unless it reproduces the committed registered numbers to 1e-12 (`revlib.gate`). Metadata: put the four files of
`meta/PROVENANCE.md` in `meta/`, or set `DFV_META` to their folder.

Run from `revision/analysis/`, for example `python -B a_cluster.py`.

| Script | Writes (in `results/`) | Paper |
|---|---|---|
| `a_cluster.py` | `a_cluster.{txt,json}`, `per_speaker_itw.csv` | speaker-level H2/H5 (Table IV), TTG intervals (Table III), per-speaker FRR, stability; Tables S8–S11, S19 |
| `b_thresholds.py` | `b_thresholds.{txt,json}` | estimator, dev EER and thresholds (Table II), dev bootstrap, tie interval, eval-set threshold, FAR budgets, transfer share, rank-based band; Tables S3–S7 |
| `a5_cheapcheck.py` | `a5_cheapcheck.{txt,json}` | the genuine-speech check; Table S16 |
| `e_cost.py` | `e_cost.{txt,json}` | cost sweep and break-even windows; Fig. S5, Table S17 |
| `g_docs.py` | `g_docs.{txt,json}`, `g_docs_by_source.txt` | DF21 exclusions by condition, class and source; H5 manifest; ITW speakers; Table S20 |
| `f_vad_stats.py`, `f_vad_score.py`, `server/*.sh` | `vad_stats.csv`, `vad/*.csv` | silence statistics and the VAD-trimmed fp32 re-scoring (run on a GPU server; needs audio and checkpoints) |
| `f_vad_analysis.py stats` / `scores` | `f_vad_stats_analysis.*`, `f_vad_scores_analysis.*`, `f_taus_trim.json` | silence cue; Tables S12–S15 |
| `facts_extra.py` | `extra_facts.{txt,json}` | ITW AUC, score saturation, triage illustration; Table S4 |
| `paper_tables.py`, `supp_tables.py`, `figures_rev.py` | `../paper/gen/*.tex`, `../paper/figures/*.pdf` | the paper's and the supplement's tables and figures |

`per_speaker_itw.csv` and `a_cluster.json` use the dataset's public speaker labels. The paper reports no speaker names.
