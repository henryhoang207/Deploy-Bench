"""R2-H2 with a cross-validated oracle over 10 fold assignments per run (DEVIATIONS.md, DEV-10 F1; specified and
deposited before it ran):

    python analysis/review_r1/run_h2cv_folds.py   # -> results/review_r1/h2cv_folds.json (about 15 min)

A2 (run_h2cv_fast.py) repeated with fold assignment a = 0, ..., 9 of run k drawn by numpy.random.default_rng(k + 10a),
so a = 0 is A2's assignment. Same registered replicates (B = 10,000, seed 0), 1-pp margin and 3-pp window; if the window
refuses for an assignment, that assignment is computed on the full grid (cvlib.Surface.cv_oracle). Reading rule, fixed
in DEV-10: the cross-validated support is robust to the fold assignment if Holm over the five runs, applied to each
run's largest p, rejects for at least 4 of 5. Also reported: for how many assignment indices Holm across the five runs
rejects for at least 4.
"""
from __future__ import annotations

import time

import numpy as np

from rvlib import R, hter, write
from cvlib import Surface, random_folds
from run_h2cv_fast import B, SEED, MARGIN, WIN, K, window, windowed_oracles, windowed_point_cv

N_ASSIGN = 10
META = {"script": "analysis/review_r1/run_h2cv_folds.py", "detector": "aasist_l_r2", "target": "la19_eval",
        "bootstrap": {"B": B, "seed": SEED, "replicates": "r2lib.cluster_counts (the registered R2-H2 replicates)"},
        "margin": MARGIN, "folds": {"K": K, "assignments_per_run": N_ASSIGN,
                                    "rng": "numpy.random.default_rng(k + 10 a) for run k, assignment a = 0..9"},
        "window": f"thresholds whose full-sample HTER is within {WIN} of its minimum; an assignment whose window "
                  "refuses is computed on the full grid",
        "holm": {"alpha": 0.05, "supported_if_rejections_at_least": 4,
                 "robust_if": "Holm over the five runs on each run's largest p rejects for at least 4"}}


def run(k: int) -> dict:
    ids, y, s = R.load("aasist_l_r2", "la19_eval", k)
    tau = R.taus("aasist_l_r2", k)["eer"]
    g, names = R.clusters("la19_eval", ids)
    G = len(names)
    sf = Surface(s, y, g, G)
    cols, _ = window(sf, WIN)
    C = R.cluster_counts(G, B, SEED)
    fr, fa = R.cluster_rates(s, y, g, G, tau, C)
    h_tau = 0.5 * (fr + fa)
    h_pt = hter(s, y, tau)
    rows = []
    for a in range(N_ASSIGN):
        fold_seed = k + 10 * a
        folds = random_folds(G, K, np.random.default_rng(fold_seed))
        try:
            _, h_cv = windowed_oracles(sf, C, folds, cols)
            h_cv_pt = windowed_point_cv(sf, folds, cols)
            method = "window"
        except SystemExit:
            h_cv = sf.cv_oracle(C, folds, K)[0]
            h_cv_pt = float(sf.cv_oracle(np.ones((1, G)), folds, K)[0][0])
            method = "full grid"
        ttg_cv = h_tau - h_cv
        rows.append({"assignment": a, "fold_seed": fold_seed, "method": method, "ttg_cv": float(h_pt - h_cv_pt),
                     "ci_cv": R.pct(ttg_cv), "p_cv": R.pval_le0(ttg_cv - MARGIN)})
    return {"seed": k, "window_cols": int(len(cols)), "assignments": rows,
            "max_p_cv": max(r["p_cv"] for r in rows), "min_ci_lo": min(r["ci_cv"][0] for r in rows)}


def main():
    t0 = time.time()
    cells = {}
    for k in R.RUN_SEEDS:
        c = run(k)
        cells[str(k)] = c
        print(f"seed {k}: CV TTG {', '.join(f'{100 * r['ttg_cv']:.2f}' for r in c['assignments'])} | largest p "
              f"{c['max_p_cv']:.4f}, lowest interval end {100 * c['min_ci_lo']:.2f} [{time.time() - t0:.0f}s]",
              flush=True)
    rej = R.holm({k: c["max_p_cv"] for k, c in cells.items()}, 0.05)
    n = sum(bool(rej.get(str(k), False)) for k in R.RUN_SEEDS)
    per_a = []
    for a in range(N_ASSIGN):
        ra = R.holm({k: c["assignments"][a]["p_cv"] for k, c in cells.items()}, 0.05)
        per_a.append(sum(bool(ra.get(str(k), False)) for k in R.RUN_SEEDS))
    summary = {"holm_on_largest_p": {"rejections": n, "robust": n >= 4, "rejected": {k: bool(v) for k, v in rej.items()}},
               "holm_per_assignment": {"rejections": per_a, "assignments_with_at_least_4": sum(r >= 4 for r in per_a)},
               "full_grid_assignments": sum(r["method"] == "full grid" for c in cells.values() for r in c["assignments"])}
    print(f"Holm on each run's largest p: {n}/5 -> {'robust' if n >= 4 else 'not robust'}; per assignment: {per_a}",
          flush=True)
    write("h2cv_folds", META, cells, summary)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
