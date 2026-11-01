"""Does a deployer who reacts to the shift recover the gap? (DEV-10; reviewer question 4)

    python analysis/review_r1/run_ladder.py  # needs results/review_r1/cv.json -> ladder.json (about 2 min)

Threshold policies, from no reaction to full labels, each scored by HTER and by TTG against the in-sample oracle:
  static     tau_EER fixed on source dev (registered)
  logistic   E9: logistic calibration on source dev, equal-prior Bayes threshold
  znorm      unsupervised score normalisation: target scores standardised by their own pooled mean/SD onto dev's
  ratematch  unsupervised re-anchoring: the target threshold that rejects the same share of traffic as tau_EER on dev
  emquant    EM prior estimate (Saerens et al. 2002) on the dev-calibrated posteriors, then reject that share of traffic
  gmm2       two-component Gaussian mixture on the unlabelled target scores, threshold where the two densities cross
  loso       labelled: leave-one-speaker-out oracle (from run_cv.py)
  oracle     labelled, in-sample
HTER weights the classes equally, so its Bayes threshold does not depend on the class prior: a label-shift correction
(EM, BBSE) cannot move it. Those corrections are scored instead by the total error rate at the target's true class mix.
"""
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture

from rvlib import (NAMES, R, ROUND1, DETECTORS_R2, SortedTarget, det_curve, frr_at, far_at, oracle, read, source_dev,
                   targets_for, write)

META = {"script": "analysis/review_r1/run_ladder.py",
        "logistic": "unpenalised (C = 1e12) on source dev; Bayes threshold of the dev-calibrated log-likelihood ratio",
        "gmm2": {"components": 2, "covariance": "full", "random_state": 0, "n_init": 3, "max_iter": 1000},
        "em": {"start": 0.5, "max_iter": 5000, "tol": 1e-12, "reference": "Saerens et al. 2002"},
        "bbse": "black-box shift estimate from the dev FRR and FAR at tau_EER, clipped to [1e-6, 1 - 1e-6]",
        "loso": "from results/review_r1/cv.json"}
t0 = time.time()
CV = read("cv")["cells"]


def logit(p):
    return np.log(p / (1 - p))


def sigmoid(z):
    return 0.5 * (1 + np.tanh(0.5 * z))


def gmm_threshold(s):
    gm = GaussianMixture(n_components=2, covariance_type="full", random_state=0, n_init=3, max_iter=1000)
    gm.fit(s[:, None])
    mu = gm.means_[:, 0]
    sd = np.sqrt(gm.covariances_[:, 0, 0])
    w = gm.weights_
    lo, hi = np.argsort(mu)
    m1, s1, m2, s2 = mu[lo], sd[lo], mu[hi], sd[hi]
    A = 1 / (2 * s2 ** 2) - 1 / (2 * s1 ** 2)
    Bq = m1 / s1 ** 2 - m2 / s2 ** 2
    Cq = m2 ** 2 / (2 * s2 ** 2) - m1 ** 2 / (2 * s1 ** 2) + np.log(s2 / s1)
    if abs(A) < 1e-14:
        roots = np.array([-Cq / Bq])
    else:
        disc = Bq ** 2 - 4 * A * Cq
        roots = np.array([(-Bq + np.sqrt(disc)) / (2 * A), (-Bq - np.sqrt(disc)) / (2 * A)]) if disc >= 0 else np.array([])
    between = roots[(roots >= m1) & (roots <= m2)] if len(roots) else roots
    tau = float(between[0]) if len(between) else 0.5 * (m1 + m2)
    return tau, float(w[hi]), [float(m1), float(s1), float(m2), float(s2)]


def em_prior(llr, pi0=0.5, iters=5000):
    pi = pi0
    for _ in range(iters):
        new = float(np.mean(sigmoid(llr + logit(pi))))
        new = min(max(new, 1e-9), 1 - 1e-9)
        if abs(new - pi) < 1e-12:
            return new
        pi = new
    return pi


CELLS = {}
for m in ROUND1 + DETECTORS_R2:
    _, yd, sd = R.load(m, source_dev(m))
    tau_e = R.taus(m)["eer"]
    lr = LogisticRegression(C=1e12, max_iter=100_000, tol=1e-10).fit(sd[:, None], yd)
    a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    pi_dev = float(yd.mean())

    def bayes(pi):          # raw-score threshold where the dev-calibrated LLR equals -logit(pi)
        return (-logit(pi) + logit(pi_dev) - b) / a

    mu_d, sd_d = float(sd.mean()), float(sd.std())
    rate_dev = float(np.mean(sd < tau_e))
    frr_dev, far_dev = frr_at(tau_e, sd, yd), far_at(tau_e, sd, yd)
    for surf in targets_for(m):
        ids, y, s = R.load(m, surf)
        st = SortedTarget(s, y)
        h_or, t_or = oracle(s, y)
        pi_true = float(y.mean())
        llr = a * s + b - logit(pi_dev)
        pi_em = em_prior(llr)
        q_bona = float(np.mean(s >= tau_e))
        pi_bbse = float(np.clip((q_bona - far_dev) / (1 - frr_dev - far_dev), 1e-6, 1 - 1e-6))
        tau_gmm, w_gmm, gpar = gmm_threshold(s)
        pol = {"static": tau_e, "logistic": bayes(0.5),
               "znorm": float(s.mean() + (tau_e - mu_d) * s.std() / sd_d),
               "ratematch": float(np.quantile(s, rate_dev)),
               "emquant": float(np.quantile(s, 1 - pi_em)),
               "gmm2": tau_gmm, "oracle": t_or}
        rows = {}
        for k, tau in pol.items():
            fr, fa = float(st.frr(tau)), float(st.far(tau))
            rows[k] = {"tau": float(tau), "frr": fr, "far": fa, "hter": 0.5 * (fr + fa), "ttg": 0.5 * (fr + fa) - h_or}
        c = CV.get(f"{m}|s0|{surf}")
        if c:
            rows["loso"] = {"frr": c["loso_frr"], "far": c["loso_far"], "hter": c["oracle_loso"],
                            "ttg": c["oracle_loso"] - h_or}
        # prior-weighted objective: total error rate at the target's true class mix
        _, frc, fac = det_curve(s, y)
        ter_curve = pi_true * frc + (1 - pi_true) * fac
        ter = {"static": pi_true * st.frr(tau_e) + (1 - pi_true) * st.far(tau_e)}
        for k, pi in (("bayes_pi0.5", 0.5), ("bayes_em", pi_em), ("bayes_bbse", pi_bbse), ("bayes_trueprior", pi_true)):
            tt = bayes(pi)
            ter[k] = float(pi_true * st.frr(tt) + (1 - pi_true) * st.far(tt))
        ter["oracle"] = float(ter_curve.min())
        ter = {k: float(v) for k, v in ter.items()}
        CELLS[f"{m}|{surf}"] = {"model": m, "target": surf, "oracle_hter": h_or, "pi_true": pi_true, "pi_em": pi_em,
                                "pi_bbse": pi_bbse, "pi_gmm": w_gmm, "gmm": gpar, "logistic": [a, b], "policies": rows,
                                "ter": ter}
        lab = NAMES[m]
        line = "  ".join(f"{k}:{100 * v['ttg']:6.2f}" for k, v in rows.items() if k != "oracle")
        print(f"{lab:11s} {surf[:10]:10s} TTG  {line}", flush=True)
        print(f"{'':22s} prior bona true {pi_true:.3f} | EM {pi_em:.3f} BBSE {pi_bbse:.3f} GMM {w_gmm:.3f} || TER% "
              + "  ".join(f"{k}:{100 * v:5.2f}" for k, v in ter.items()), flush=True)

write("ladder", META, CELLS)
print(f"done in {time.time() - t0:.0f}s")
