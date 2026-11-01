"""Sensitivity of the m-out-of-n interval to m (DEVIATIONS.md, DEV-10 F2; specified and deposited before it ran):

    python analysis/review_r1/run_mofn.py      # -> results/review_r1/mofn.json (about 8 min)

E's m-out-of-n speaker bootstrap (run_intervals.py) for m = round(G^(1/2)), round(G^(2/3)) and round(G/2) on the 25
detector-target cells: B = 2,000, numpy.random.default_rng(1) restarted for every cell and every m, so
m = round(G^(2/3)) reproduces E; rescaled basic interval, truncated at 0.
"""
import time

import numpy as np

from rvlib import NAMES, R, ROUND1, DETECTORS_R2, hter, targets_for, write
from cvlib import Surface

RULES = {"G^(1/2)": lambda G: int(round(G ** 0.5)), "G^(2/3)": lambda G: int(round(G ** (2 / 3))),
         "G/2": lambda G: int(round(G / 2))}
META = {"script": "analysis/review_r1/run_mofn.py",
        "m": {k: f"int(round({k.replace('^', '**')}))" for k in RULES},
        "B": 2000, "rng": "numpy.random.default_rng(1), restarted for every cell and every m",
        "interval": "rescaled basic interval of TTG at tau_EER, truncated at 0",
        "reproduces": "m = round(G^(2/3)) is E (results/review_r1/intervals.json, m_of_n)"}
t0 = time.time()


def cells():
    for m in ROUND1:
        for surf in ("la19_eval", "itw", "df21_verified"):
            yield m, surf
    for m in DETECTORS_R2:
        for surf in targets_for(m):
            yield m, surf


CELLS = {}
for m, surf in cells():
    ids, y, s = R.load(m, surf)
    tau = R.taus(m)["eer"]
    g, names = R.clusters(surf, ids)
    G = len(names)
    sf = Surface(s, y, g, G, max_m=min(80_000, int(1.2e7 / G)))
    theta = hter(s, y, tau) - float(sf.oracle(np.ones((1, G)))[0])
    cell = {"G": G, "ttg": theta, "grid_exact": sf.exact}
    for rule, f in RULES.items():
        mm = f(G)
        rng = np.random.default_rng(1)
        Cm = rng.multinomial(mm, np.full(G, 1.0 / G), size=2000).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            frm, fam = R.cluster_rates(s, y, g, G, tau, Cm)
            bm = 0.5 * (frm + fam) - sf.oracle(Cm)
        bm = bm[np.isfinite(bm)]
        q = np.percentile(bm, [2.5, 97.5])
        sc = np.sqrt(mm / G)
        cell[rule] = {"m": mm, "finite_replicates": int(len(bm)),
                      "m_of_n": [max(0.0, theta - sc * (q[1] - theta)), max(0.0, theta - sc * (q[0] - theta))]}
    CELLS[f"{m}|{surf}"] = cell
    print(f"{NAMES[m]:11s} {surf[:10]:10s} {G:4d} {100 * theta:6.2f} "
          + "  ".join(f"m={cell[r]['m']:3d} [{100 * cell[r]['m_of_n'][0]:6.2f},{100 * cell[r]['m_of_n'][1]:6.2f}]"
                      for r in RULES) + f"  [{time.time() - t0:.0f}s]", flush=True)
write("mofn", META, CELLS)
print(f"done in {time.time() - t0:.0f}s")
