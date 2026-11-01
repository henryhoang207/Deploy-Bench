"""R2-H2 with a speaker-disjoint 5-fold cross-validated oracle inside the registered bootstrap (DEV-10; reviewer
question 2):

    python analysis/review_r1/run_h2cv_fast.py [seed ...]   # default all five runs -> results/review_r1/h2cv.json

Same replicates as the registered test (r2lib.cluster_counts(G, 10000, seed 0)), the same one-sided p against the 1-pp
margin and the same Holm rule over the five runs; in every replicate the oracle is a speaker-disjoint 5-fold
cross-validated threshold (one fold assignment per run, numpy.random.default_rng(<run seed>)) instead of the in-sample
minimum. The in-sample p values re-derive the registered test as a check.

The oracle search is restricted to the thresholds whose full-sample HTER is within 3 pp of its minimum. A minimiser on
the edge of that window could hide a lower HTER outside it, so the script refuses if any replicate's minimiser (in-sample
or on a fold's training speakers) lands on the edge. Surface.oracle and Surface.cv_oracle (cvlib.py) are the full-grid
versions of the same quantities.
"""
from __future__ import annotations

import sys
import time

import numpy as np

from rvlib import R, hter, write
from cvlib import Surface, random_folds

B, SEED, MARGIN, WIN, K = 10_000, 0, 0.01, 0.03, 5
META = {"script": "analysis/review_r1/run_h2cv_fast.py", "detector": "aasist_l_r2", "target": "la19_eval",
        "bootstrap": {"B": B, "seed": SEED, "replicates": "r2lib.cluster_counts (the registered R2-H2 replicates)"},
        "margin": MARGIN, "folds": {"K": K, "assignments_per_run": 1, "rng": "numpy.random.default_rng(<run seed>)"},
        "window": f"thresholds whose full-sample HTER is within {WIN} of its minimum; refuses on any edge hit",
        "holm": {"alpha": 0.05, "supported_if_rejections_at_least": 4}}


def window(sf, win):
    """Contiguous grid columns whose full-sample HTER is within `win` of its minimum, and the full-sample HTER curve."""
    frr0 = sf.CB.sum(0) / sf.nb.sum()
    far0 = 1 - sf.CP.sum(0) / sf.ns.sum()
    h0 = 0.5 * (frr0 + far0)
    keep = np.nonzero(h0 <= h0.min() + win)[0]
    return np.arange(keep.min(), keep.max() + 1), h0


def windowed(sf, cols):
    """Cluster counts and HTER curves restricted to the grid columns `cols`, and a check that refuses a minimiser on
    the edge of the window."""
    CB, CP = sf.CB[:, cols], sf.CP[:, cols]

    def curves(W):
        return 0.5 * ((W @ CB) / (W @ sf.nb)[:, None] + 1 - (W @ CP) / (W @ sf.ns)[:, None])

    def inside(j, what):
        if np.any((j == 0) | (j == len(cols) - 1)):
            R.die(f"{what} minimiser lies on the edge of the {len(cols)}-column threshold window; widen the window")

    return CB, CP, curves, inside


def windowed_oracles(sf, C, folds, cols, chunk=250):
    """In-sample and cross-validated oracle HTER per row of C, searching only the grid columns `cols`. Refuses if any
    minimiser lands on the edge of the window; otherwise equals Surface.oracle(C) and Surface.cv_oracle(C, folds, K)."""
    CB, CP, curves, inside = windowed(sf, cols)
    n = C.shape[0]
    or_in = np.empty(n)
    rej = np.zeros(n)
    acc = np.zeros(n)
    for a in range(0, n, chunk):
        W = C[a:a + chunk]
        h = curves(W)
        j = np.argmin(h, axis=1)
        inside(j, "an in-sample")
        or_in[a:a + chunk] = np.minimum(h[np.arange(len(j)), j], 0.5)
        for f in range(K):
            tr = (folds != f).astype(float)
            Wt, We = W * tr, W * (1 - tr)
            jt = np.argmin(curves(Wt), axis=1)
            inside(jt, f"a fold-{f} training")
            rej[a:a + chunk] += np.einsum("rc,cr->r", We, CB[:, jt])
            acc[a:a + chunk] += We @ sf.ns - np.einsum("rc,cr->r", We, CP[:, jt])
    h_cv = 0.5 * (rej / (C @ sf.nb) + acc / (C @ sf.ns))
    return or_in, h_cv


def windowed_point_cv(sf, folds, cols):
    """Cross-validated oracle HTER at the original sample (every cluster once), on the window; refuses as above."""
    CB, CP, curves, inside = windowed(sf, cols)
    one = np.ones((1, sf.G))
    rej_pt = acc_pt = 0.0
    for f in range(K):
        tr = (folds != f).astype(float)
        jt = np.argmin(curves(one * tr), axis=1)
        inside(jt, f"the fold-{f} point-estimate")
        jt = int(jt[0])
        rej_pt += float((1 - tr) @ CB[:, jt])
        acc_pt += float((1 - tr) @ sf.ns - (1 - tr) @ CP[:, jt])
    return 0.5 * (rej_pt / sf.nb.sum() + acc_pt / sf.ns.sum())


def run_seed(k: int, fold_seed: int) -> dict:
    ids, y, s = R.load("aasist_l_r2", "la19_eval", k)
    tau = R.taus("aasist_l_r2", k)["eer"]
    g, names = R.clusters("la19_eval", ids)
    G = len(names)
    sf = Surface(s, y, g, G)
    cols, h0 = window(sf, WIN)
    C = R.cluster_counts(G, B, SEED)
    fr, fa = R.cluster_rates(s, y, g, G, tau, C)
    h_tau = 0.5 * (fr + fa)
    folds = random_folds(G, K, np.random.default_rng(fold_seed))
    or_in, h_cv = windowed_oracles(sf, C, folds, cols)
    ttg_in, ttg_cv = h_tau - or_in, h_tau - h_cv
    h_pt = hter(s, y, tau)                     # point estimates at the original sample (all clusters once)
    pt_cv = h_pt - windowed_point_cv(sf, folds, cols)
    return {"seed": k, "fold_seed": fold_seed, "window_cols": int(len(cols)), "edge_hits": 0,
            "ttg_in": float(h_pt - min(h0.min(), 0.5)), "ci_in": R.pct(ttg_in), "p_in": R.pval_le0(ttg_in - MARGIN),
            "ttg_cv": float(pt_cv), "ci_cv": R.pct(ttg_cv), "p_cv": R.pval_le0(ttg_cv - MARGIN)}


def main():
    seeds = [int(a) for a in sys.argv[1:]] or list(R.RUN_SEEDS)
    t0 = time.time()
    cells = {}
    for k in seeds:
        out = run_seed(k, k)
        cells[str(k)] = out
        print(f"seed {k}: in-sample TTG {100 * out['ttg_in']:.2f} [{100 * out['ci_in'][0]:.2f}, "
              f"{100 * out['ci_in'][1]:.2f}] p={out['p_in']:.4f} | 5-fold-CV TTG {100 * out['ttg_cv']:.2f} "
              f"[{100 * out['ci_cv'][0]:.2f}, {100 * out['ci_cv'][1]:.2f}] p={out['p_cv']:.4f} | window "
              f"{out['window_cols']} cols, no edge hits [{time.time() - t0:.0f}s]", flush=True)
    summary = {}
    if sorted(cells) == sorted(str(k) for k in R.RUN_SEEDS):
        for lab, key in (("in_sample", "p_in"), ("cv", "p_cv")):
            rej = R.holm({k: c[key] for k, c in cells.items()}, 0.05)
            n = sum(bool(rej.get(str(k), False)) for k in R.RUN_SEEDS)
            summary[lab] = {"rejections": n, "supported": n >= 4, "rejected": {k: bool(v) for k, v in rej.items()}}
            print(f"{lab}: Holm rejections {n}/5 -> {'supported' if n >= 4 else 'not supported'}", flush=True)
    write("h2cv", META, cells, {"holm": summary} if summary else None)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
