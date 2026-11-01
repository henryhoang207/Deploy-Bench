"""Interval methods for TTG near its zero boundary (DEV-10; reviewer's BCa request).

    python analysis/review_r1/run_intervals.py  # -> results/review_r1/intervals.json (about 5 min)

Per cell: the registered-style percentile interval (speaker-cluster bootstrap, B = 1000, seed 0), BCa from the same
replicates with the acceleration from a leave-one-speaker-out jackknife, and an m-out-of-n cluster bootstrap
(m = round(G^(2/3)), B = 2000, rescaled basic interval, truncated at 0), which stays valid at a boundary (Andrews 2000).
"""
import time

import numpy as np
from scipy.special import ndtr, ndtri

from rvlib import NAMES, R, ROUND1, DETECTORS_R2, hter, targets_for, write
from cvlib import Surface

META = {"script": "analysis/review_r1/run_intervals.py",
        "percentile": {"B": 1000, "seed": 0, "replicates": "r2lib.cluster_counts, as registered", "levels": [2.5, 97.5]},
        "bca": {"replicates": "the percentile replicates", "acceleration": "leave-one-speaker-out jackknife"},
        "m_out_of_n": {"m": "round(G^(2/3))", "B": 2000, "rng": "numpy.random.default_rng(1), restarted for every cell",
                       "interval": "rescaled basic interval, truncated at 0"}}
t0 = time.time()


def cells():
    for m in ROUND1:
        for surf in ("la19_eval", "itw", "df21_verified"):
            yield m, surf
    for m in DETECTORS_R2:
        for surf in targets_for(m):
            yield m, surf


CELLS = {}
print(f"{'model':11s} {'tgt':10s} {'G':>4s} {'TTG':>6s} {'percentile':>16s} {'BCa':>16s} {'m-of-n':>16s}  z0     a")
for m, surf in cells():
    ids, y, s = R.load(m, surf)
    tau = R.taus(m)["eer"]
    g, names = R.clusters(surf, ids)
    G = len(names)
    sf = Surface(s, y, g, G, max_m=min(80_000, int(1.2e7 / G)))
    theta = hter(s, y, tau) - float(sf.oracle(np.ones((1, G)))[0])
    # percentile (registered construction)
    C = R.cluster_counts(G, 1000, 0)
    fr, fa = R.cluster_rates(s, y, g, G, tau, C)
    boot = 0.5 * (fr + fa) - sf.oracle(C)
    pc = np.percentile(boot, [2.5, 97.5])
    # BCa
    z0 = ndtri((np.sum(boot < theta) + 0.5 * np.sum(boot == theta)) / len(boot))
    J = np.ones((G, G)) - np.eye(G)
    frj, faj = R.cluster_rates(s, y, g, G, tau, J)
    jack = 0.5 * (frj + faj) - sf.oracle(J)
    d = jack.mean() - jack
    acc = float(np.sum(d ** 3) / (6 * np.sum(d ** 2) ** 1.5))
    al = [ndtr(z0 + (z0 + z) / (1 - acc * (z0 + z))) for z in (ndtri(0.025), ndtri(0.975))]
    bca = np.percentile(boot, [100 * al[0], 100 * al[1]])
    # m-out-of-n
    mm = int(round(G ** (2 / 3)))
    rng = np.random.default_rng(1)
    Cm = rng.multinomial(mm, np.full(G, 1.0 / G), size=2000).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        frm, fam = R.cluster_rates(s, y, g, G, tau, Cm)
        bm = 0.5 * (frm + fam) - sf.oracle(Cm)
    bm = bm[np.isfinite(bm)]
    q = np.percentile(bm, [2.5, 97.5])
    sc = np.sqrt(mm / G)
    mon = [max(0.0, theta - sc * (q[1] - theta)), max(0.0, theta - sc * (q[0] - theta))]
    CELLS[f"{m}|{surf}"] = {"G": G, "ttg": theta, "pct": pc.tolist(), "bca": bca.tolist(), "m": mm, "m_of_n": mon,
                            "z0": float(z0), "a": acc, "grid_exact": sf.exact}
    print(f"{NAMES[m]:11s} {surf[:10]:10s} {G:4d} {100 * theta:6.2f} [{100 * pc[0]:6.2f},{100 * pc[1]:6.2f}] "
          f"[{100 * bca[0]:6.2f},{100 * bca[1]:6.2f}] [{100 * mon[0]:6.2f},{100 * mon[1]:6.2f}]  {z0:+.2f} {acc:+.3f}"
          f"  [{time.time() - t0:.0f}s]", flush=True)
write("intervals", META, CELLS)
print(f"done in {time.time() - t0:.0f}s")
