"""A small labelled audit (DEV-10; reviewer question 4).

    python analysis/review_r1/run_smalllabel.py  # -> results/review_r1/smalllabel.json

For each of 200 random splits of the target's speakers into halves, draw k labelled clips per class from one half, set
the threshold that minimises HTER on them, and score it on the other half against that half's own in-sample oracle; the
static tau_EER is scored on the same held-out half for comparison.
"""
import time

import numpy as np

from rvlib import NAMES, R, ROUND1, DETECTORS_R2, SortedTarget, det_curve, oracle, write

SPLITS, KS = 200, (50, 200, 1000)
META = {"script": "analysis/review_r1/run_smalllabel.py", "splits": SPLITS, "k_per_class": list(KS),
        "rng": "numpy.random.default_rng(0), restarted for every detector and target; draws without replacement",
        "quantiles": [5, 50, 95]}
t0 = time.time()
CELLS = {}
for m in ROUND1 + DETECTORS_R2:
    for surf in ("itw", "la19_eval"):
        ids, y, s = R.load(m, surf)
        tau_e = R.taus(m)["eer"]
        g, names = R.clusters(surf, ids)
        G = len(names)
        rng = np.random.default_rng(0)
        res = {k: [] for k in KS}
        stat = []
        for _ in range(SPLITS):
            half = rng.permutation(G)[: G // 2]
            cal = np.isin(g, half)
            ho = ~cal
            st = SortedTarget(s[ho], y[ho])
            h_or, _ = oracle(s[ho], y[ho])
            stat.append(float(st.hter(tau_e) - h_or))
            ib, ip = np.nonzero(cal & (y == 1))[0], np.nonzero(cal & (y == 0))[0]
            for k in KS:
                sel = np.concatenate([rng.choice(ib, min(k, len(ib)), replace=False),
                                      rng.choice(ip, min(k, len(ip)), replace=False)])
                tg, fr, fa = det_curve(s[sel], y[sel])
                tau_k = tg[int(np.argmin(0.5 * (fr + fa)))]
                res[k].append(float(st.hter(tau_k) - h_or))
        cell = {"static": [float(np.percentile(stat, q)) for q in (5, 50, 95)]}
        for k in KS:
            cell[str(k)] = [float(np.percentile(res[k], q)) for q in (5, 50, 95)]
        CELLS[f"{m}|{surf}"] = cell
        print(f"{NAMES[m]:11s} {surf:9s} held-out TTG, median [5, 95%]: static {100 * cell['static'][1]:5.2f} "
              f"[{100 * cell['static'][0]:5.2f}, {100 * cell['static'][2]:5.2f}]  "
              + "  ".join(f"k={k}: {100 * cell[str(k)][1]:5.2f} [{100 * cell[str(k)][0]:5.2f}, {100 * cell[str(k)][2]:5.2f}]"
                          for k in KS) + f"  [{time.time() - t0:.0f}s]", flush=True)
write("smalllabel", META, CELLS)
print(f"done in {time.time() - t0:.0f}s")
