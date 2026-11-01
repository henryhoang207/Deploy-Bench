"""Is TTG an artifact of how tau_EER is estimated from a near-separable development set? (DEV-10; reviewer question 3)

    python analysis/review_r1/run_dev.py     # -> results/review_r1/dev.json (about 5 min)

For every detector x target: (a) TTG over the dev-equivalent interval (r2lib.tie_interval: every dev threshold within
one bona fide trial of HTER(tau_EER)); its minimum is the transfer loss that no dev-tied threshold avoids. (b) TTG at
tau_EER re-estimated on 1,000 class-stratified resamples of the dev trials. (c) TTG at smoothed dev-only thresholds:
the crossing of Gaussian-KDE FRR and FAR curves (Silverman bandwidth x 0.5, 1, 2), the isotonic (PAV) calibration's
equal-prior Bayes threshold, and the logistic (E9) equal-prior Bayes threshold.
"""
import time

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from rvlib import NAMES, R, ROUND1, DETECTORS_R2, SortedTarget, eer, oracle, source_dev, targets_for, write

B_DEV = 1000
META = {"script": "analysis/review_r1/run_dev.py",
        "dev_bootstrap": {"B": B_DEV, "rng": "numpy.random.default_rng(0), restarted for every detector and run",
                          "resample": "class-stratified, with replacement, tau_EER re-estimated by metrics.eer"},
        "kde": {"bandwidth": "Silverman (0.9 min(sd, IQR / 1.34) n^-0.2) per class", "multipliers": [0.5, 1, 2],
                "threshold": "FRR = FAR of the smoothed curves (Brent, xtol 1e-10)"},
        "isotonic": "PAV on dev, threshold midway between the dev scores where the posterior crosses the dev prior",
        "logistic": "unpenalised (C = 1e12) on dev, equal-prior Bayes threshold (E9)"}
t0 = time.time()


def silverman(x):
    x = np.asarray(x, float)
    sd = x.std(ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    spread = min(sd, iqr / 1.34) if iqr > 0 else sd
    return 0.9 * spread * len(x) ** -0.2


def kde_threshold(sd, yd, mult):
    b, p = sd[yd == 1], sd[yd == 0]
    hb, hp = silverman(b) * mult, silverman(p) * mult

    def f(t):
        return ndtr((t - b) / hb).mean() - (1.0 - ndtr((t - p) / hp).mean())

    lo, hi = sd.min() - 10 * max(hb, hp), sd.max() + 10 * max(hb, hp)
    return brentq(f, lo, hi, xtol=1e-10)


def pav_threshold(sd, yd):
    o = np.argsort(sd, kind="stable")
    xs, ys = sd[o], yd[o]
    post = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(xs, ys).predict(xs)
    pi = ys.mean()
    j = int(np.argmax(post >= pi))            # first sorted dev score whose prior-corrected LLR >= 0
    return 0.5 * (xs[j] + xs[j - 1]) if j > 0 else xs[0]


def logistic_threshold(sd, yd):
    lr = LogisticRegression(C=1e12, max_iter=100_000, tol=1e-10).fit(sd[:, None], yd)
    a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    pi = yd.mean()
    return (np.log(pi / (1 - pi)) - b) / a if a > 0 else float("nan")


def dev_cells():
    for m in ROUND1 + DETECTORS_R2:
        yield m, 0, source_dev(m), targets_for(m)
    for k in R.RUN_SEEDS:
        yield "aasist_l_r2", k, "la19_dev", ("la19_eval", "itw")


CELLS = {}
for m, seed, devsurf, tg in dev_cells():
    _, yd, sd = R.load(m, devsurf, seed)
    t = R.taus(m, seed)
    tau = t["eer"]
    lo, hi = R.tie_interval(sd, yd, tau)
    nb_dev = int((yd == 1).sum())
    pins = (int(np.sum(sd[yd == 1] < tau)), int(np.sum(sd[yd == 0] >= tau)))
    smooth = {"kde0.5": kde_threshold(sd, yd, 0.5), "kde1": kde_threshold(sd, yd, 1.0), "kde2": kde_threshold(sd, yd, 2.0),
              "pav": pav_threshold(sd, yd), "logistic": logistic_threshold(sd, yd)}
    rng = np.random.default_rng(0)
    ib, ip = np.nonzero(yd == 1)[0], np.nonzero(yd == 0)[0]
    taus_b = np.empty(B_DEV)
    for r in range(B_DEV):
        idx = np.concatenate([rng.choice(ib, len(ib)), rng.choice(ip, len(ip))])
        taus_b[r] = eer(sd[idx], yd[idx])[1]
    for surf in tg:
        _, y, s = R.load(m, surf, seed)
        st = SortedTarget(s, y)
        h_or, _ = oracle(s, y)
        inside = np.unique(s[(s > lo) & (s < hi)])
        grid = np.concatenate([[lo], inside, [hi]])
        hh = st.hter(grid)
        ttg_grid = hh - h_or
        jmin, jmax = int(np.argmin(hh)), int(np.argmax(hh))
        ttg_b = st.hter(taus_b) - h_or
        frr_b = st.frr(taus_b)
        ttg_star = float(st.hter(tau) - h_or)
        cell = {"model": m, "seed": seed, "target": surf, "tau": tau, "tie": [lo, hi], "dev_pins": pins, "nb_dev": nb_dev,
                "ttg_star": ttg_star, "frr_star": float(st.frr(tau)), "far_star": float(st.far(tau)),
                "ttg_tie_min": float(ttg_grid[jmin]), "ttg_tie_max": float(ttg_grid[jmax]),
                "tie_min_at": float(grid[jmin]), "frr_far_at_tie_min": [float(st.frr(grid[jmin])), float(st.far(grid[jmin]))],
                "ttg_lo": float(ttg_grid[0]), "ttg_hi": float(ttg_grid[-1]),
                "devboot_ttg_q": [float(np.percentile(ttg_b, q)) for q in (2.5, 50, 97.5)],
                "devboot_frr_q": [float(np.percentile(frr_b, q)) for q in (2.5, 50, 97.5)],
                "devboot_share_below_star": float(np.mean(ttg_b < ttg_star)),
                "smooth": {k: {"tau": float(v), "in_tie": bool(lo <= v <= hi), "ttg": float(st.hter(v) - h_or),
                               "frr": float(st.frr(v)), "far": float(st.far(v))} for k, v in smooth.items()}}
        CELLS[f"{m}|s{seed}|{surf}"] = cell
        lab = NAMES.get(m, f"AASIST-L r{seed}")
        sm = "  ".join(f"{k}:{100 * v['ttg']:.2f}{'' if v['in_tie'] else '*'}" for k, v in cell["smooth"].items())
        print(f"{lab:12s} {surf[:10]:10s} pins {pins[0]}/{pins[1]} TTG* {100 * ttg_star:6.2f} | tie TTG "
              f"[{100 * cell['ttg_tie_min']:5.2f}, {100 * cell['ttg_tie_max']:5.2f}] | devboot "
              f"{100 * cell['devboot_ttg_q'][1]:5.2f} [{100 * cell['devboot_ttg_q'][0]:5.2f}, {100 * cell['devboot_ttg_q'][2]:5.2f}] "
              f"below* {cell['devboot_share_below_star']:.2f} | {sm}  [{time.time() - t0:.0f}s]", flush=True)

write("dev", META, CELLS)
print(f"done in {time.time() - t0:.0f}s   (* = smoothed threshold outside the dev-equivalent interval)")
