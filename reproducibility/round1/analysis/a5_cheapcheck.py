"""DEV-8 A5: the "cheap check" simulation (exploratory).

    python a5_cheapcheck.py          # gate first; writes results/a5_cheapcheck.json and results/a5_cheapcheck.txt

ITW, per model. For k in {50, 100, 500}, 1,000 random splits of the 54 speaker labels into a calibration half (27) and a
held-out half (27), numpy default_rng(0). Draw k genuine clips uniformly (without replacement) from the calibration half.
 (i)  estimate FRR at the shipped tau_eer from the sample; compare with the held-out genuine FRR at the same tau;
 (ii) tau_new = the (floor(0.01 k) + 1)-th smallest sample score, i.e. the largest threshold whose sample FRR is <= 1%
      (the harness's tau_at_frr rule restricted to the bona fide scores); report held-out FRR and FAR (the spoofs of the
      held-out speakers) at tau_new, next to the same at the shipped tau_eer.
No attack data of the deployment channel are used to set tau_new: only genuine speech.
"""
from __future__ import annotations

import numpy as np

import revlib as R

SEED, NSPLIT = 0, 1000
KS = (50, 100, 500)


def main():
    R.gate(verbose=False)
    L = ["gate passed: every point estimate recomputed here equals the committed numbers to 1e-12", ""]
    ids0, _, _ = R.load(R.MODELS[0], "itw")
    g0, names = R.clusters("itw", ids0)
    G = len(names)
    rng = np.random.default_rng(SEED)
    splits = [rng.permutation(G) for _ in range(NSPLIT)]
    data = {}
    for m in R.MODELS:
        ids, y, s = R.load(m, "itw")
        g, nm = R.clusters("itw", ids)
        assert nm == names
        data[m] = (y, s, g)
    res = {m: {k: {"est_err_pp": [], "frr_shipped_held": [], "far_shipped_held": [], "frr_new_held": [], "far_new_held": [],
                   "tau_new_minus_shipped": []} for k in KS} for m in R.MODELS}
    draw_rng = np.random.default_rng(SEED + 1)
    for r in range(NSPLIT):
        perm = splits[r]
        cal = np.zeros(G, bool)
        cal[perm[:G // 2]] = True
        for m in R.MODELS:
            y, s, g = data[m]
            tau = R.frozen_taus(m)[0]["eer"]
            in_cal = cal[g]
            bona_cal = np.where((y == 1) & in_cal)[0]
            held_b = np.sort(s[(y == 1) & ~in_cal])
            held_s = np.sort(s[(y == 0) & ~in_cal])
            frr_h = np.searchsorted(held_b, tau, "left") / len(held_b)
            far_h = 1.0 - np.searchsorted(held_s, tau, "left") / len(held_s)
            for k in KS:
                pick = draw_rng.choice(bona_cal, size=k, replace=False)
                samp = np.sort(s[pick])
                est = float(np.mean(samp < tau))
                tau_new = samp[int(np.floor(0.01 * k))]            # (floor(0.01 k) + 1)-th smallest
                frr_n = np.searchsorted(held_b, tau_new, "left") / len(held_b)
                far_n = 1.0 - np.searchsorted(held_s, tau_new, "left") / len(held_s)
                o = res[m][k]
                o["est_err_pp"].append(100 * (est - frr_h))
                o["frr_shipped_held"].append(100 * frr_h)
                o["far_shipped_held"].append(100 * far_h)
                o["frr_new_held"].append(100 * frr_n)
                o["far_new_held"].append(100 * far_n)
                o["tau_new_minus_shipped"].append(float(tau_new - tau))
    out = {}
    L.append(f"== A5  Cheap check on ITW: {NSPLIT} speaker-disjoint splits (27 calibration / 27 held-out speakers), k genuine calibration clips; medians and 5-95% ranges, % (pp for errors)")
    L.append("   est. error = FRR estimated from the k-clip sample at the SHIPPED tau_eer minus the held-out genuine FRR at the same tau")
    L.append("   re-set = tau_new at the 1% sample quantile of the k genuine clips (no attack data); held-out FRR / FAR there")
    L.append(f"{'model':12s} {'k':>4s} | {'FRR est. error (pp)':>26s} | {'held-out FRR/FAR at shipped tau':>34s} | {'held-out FRR / FAR after re-set':>40s}")
    for m in R.MODELS:
        out[m] = {}
        for k in KS:
            o = res[m][k]
            q = lambda a: (np.median(a), np.percentile(a, 5), np.percentile(a, 95))
            e, fs, ns, fn, an = q(o["est_err_pp"]), q(o["frr_shipped_held"]), q(o["far_shipped_held"]), q(o["frr_new_held"]), q(o["far_new_held"])
            share_within5 = float(np.mean(np.abs(o["est_err_pp"]) <= 5))
            out[m][k] = {"est_error_pp_med_p5_p95": e, "share_est_within_5pp": share_within5,
                         "held_frr_shipped": fs, "held_far_shipped": ns, "held_frr_new": fn, "held_far_new": an,
                         "tau_shift_med": float(np.median(o["tau_new_minus_shipped"]))}
            L.append(f"{R.NICE[m]:12s} {k:4d} | {e[0]:+6.1f} [{e[1]:+6.1f},{e[2]:+6.1f}] ({100 * share_within5:3.0f}% <=5) | "
                     f"{fs[0]:5.1f} [{fs[1]:5.1f},{fs[2]:5.1f}] / {ns[0]:5.1f} [{ns[1]:5.1f},{ns[2]:5.1f}] | "
                     f"{fn[0]:5.1f} [{fn[1]:5.1f},{fn[2]:5.1f}] / {an[0]:5.1f} [{an[1]:5.1f},{an[2]:5.1f}]")
    R.write_json("a5_cheapcheck.json", {"nsplit": NSPLIT, "seed": SEED, "results": out})
    (R.RESULTS / "a5_cheapcheck.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
