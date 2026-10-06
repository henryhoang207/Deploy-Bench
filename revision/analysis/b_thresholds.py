"""DEV-8 B, C, D: threshold estimation, FAR-budget policies and the comparability measures.

    python b_thresholds.py          # gate first; writes results/b_thresholds.json and results/b_thresholds.txt

B1  tau_eer estimator documentation per model: dev EER, taus, dev margin, the dev trials that pin tau_eer.
B2  Dev-set bootstrap of all five taus (class-stratified, B = 1,000, seed 0) carried to ITW / DF21 / LA19-eval FRR, FAR, TTG.
B3  EER-equivalent interval: connected run of dev thresholds around tau_eer with dev HTER <= HTER(tau_eer) + 0.5/n_bona.
B4  Zhou-and-Wang-style threshold: the same estimator on the LA19 EVAL scores, applied to ITW / DF21.
C   FAR-budget policies tau_far1 / tau_far5 (harness tau_at_far on dev) on every target + the target-oracle FRR at FAR = 5%.
D   HTER(tau*), oracle HTER, transfer share, TTG / (50 - oracle); rank-based band and miss (invariant to monotone maps).
"""
from __future__ import annotations

from multiprocessing import Pool

import numpy as np

import revlib as R

B2, SEED = 1000, 0


def dev_bootstrap(m):
    ids, y, s = R.load(m, "dev")
    o = np.argsort(ids)                    # utt_id order, so replicate r draws the same clips for every model
    y, s = y[o], s[o]
    ib, isp = np.where(y == 1)[0], np.where(y == 0)[0]
    rng = np.random.default_rng(SEED)
    taus = {p: np.empty(B2) for p in R.POLICIES5}
    for r in range(B2):
        idx = np.concatenate([ib[rng.integers(0, len(ib), len(ib))], isp[rng.integers(0, len(isp), len(isp))]])
        sr, yr = s[idx], y[idx]
        c = R.calibrate_tau(sr, yr)
        for p in ("eer", "frr1", "frr5"):
            taus[p][r] = c[p]
        taus["far1"][r] = R.tau_at_far(sr, yr, 0.01)
        taus["far5"][r] = R.tau_at_far(sr, yr, 0.05)
    out = {"taus": {p: R.pct(v) for p, v in taus.items()}, "targets": {}}
    for tgt in R.TARGETS:
        _, yt, st = R.load(m, tgt)
        bona, spoof = np.sort(st[yt == 1]), np.sort(st[yt == 0])
        h_or = R.min_hter(st, yt)[0]
        out["targets"][tgt] = {}
        for p in R.POLICIES5:
            t = taus[p]
            frr = np.searchsorted(bona, t, side="left") / len(bona)
            far = 1.0 - np.searchsorted(spoof, t, side="left") / len(spoof)
            ttg = 0.5 * (frr + far) - h_or
            out["targets"][tgt][p] = {"frr_ci95": R.pct(frr), "far_ci95": R.pct(far), "ttg_ci95": R.pct(ttg),
                                      "frr_median": float(np.median(frr)), "far_median": float(np.median(far))}
    return m, out


def b1_b3_b4_c_d(m):
    ids, y, s = R.load(m, "dev")
    taus = R.all_taus(m)
    bona, spoof = s[y == 1], s[y == 0]
    e_dev, t_eer = R.eer(s, y)
    res = {"dev": {"n_bona": int(len(bona)), "n_spoof": int(len(spoof)), "eer_pct": 100 * e_dev,
                   "taus": {k: float(v) for k, v in taus.items()},
                   "margin": float(bona.min() - spoof.max()),
                   "bona_min": float(bona.min()), "spoof_max": float(spoof.max()),
                   "n_bona_below_tau_eer": int((bona < t_eer).sum()), "n_spoof_at_or_above_tau_eer": int((spoof >= t_eer).sum()),
                   "dev_frr_at_tau": {k: float(R.frr_at(taus[k], s, y)) for k in R.POLICIES5},
                   "dev_far_at_tau": {k: float(R.far_at(taus[k], s, y)) for k in R.POLICIES5}}}
    # B3: EER-equivalent interval
    grid, frr, far = R.det_curve(s, y)
    h = 0.5 * (frr + far)
    tol = 0.5 / len(bona)
    ok = h <= R.hter_at(t_eer, s, y) + tol
    j = int(np.searchsorted(grid, t_eer))
    cand = [p for p in (j - 1, j) if 0 <= p < len(grid) and ok[p]]
    if cand:
        lo = hi = cand[0]
        while lo - 1 >= 0 and ok[lo - 1]:
            lo -= 1
        while hi + 1 < len(grid) and ok[hi + 1]:
            hi += 1
        tau_lo, tau_hi = float(grid[lo]), float(grid[hi])
    else:
        tau_lo = tau_hi = float(t_eer)
    res["eer_equivalent_interval"] = {"tau_lo": tau_lo, "tau_hi": tau_hi, "tau_eer": float(t_eer),
                                      "width_in_dev_score_percentiles": [
                                          float(100 * np.mean(s < tau_lo)), float(100 * np.mean(s < tau_hi))]}
    # B4: eval-set threshold (the Zhou-and-Wang convention), same estimator
    ide, ye, se = R.load(m, "19la_eval")
    e_ev, t_ev = R.eer(se, ye)
    res["eval_threshold"] = {"eer_eval_pct": 100 * e_ev, "tau": float(t_ev)}
    res["targets"] = {}
    for tgt in R.TARGETS:
        idt, yt, st = R.load(m, tgt)
        sd = float(np.std(st))
        tg, fr, fa = R.det_curve(st, yt)
        hh = 0.5 * (fr + fa)
        hmin = float(hh.min())
        okb = tg[hh <= hmin + 0.01]
        band_lo, band_hi = float(okb.min()), float(okb.max())
        t_or = float(tg[int(np.argmin(hh))])
        isb = yt == 1
        cell = {"n": int(len(yt)), "eer_pct": 100 * R.eer(st, yt)[0], "oracle_hter_pct": 100 * hmin, "tau_oracle": t_or,
                "score_sd": sd, "band_sd": (band_hi - band_lo) / sd,
                "band_rank": {"pooled": float(np.mean((st >= band_lo) & (st < band_hi))),
                              "bona": float(np.mean((st[isb] >= band_lo) & (st[isb] < band_hi))),
                              "spoof": float(np.mean((st[~isb] >= band_lo) & (st[~isb] < band_hi)))},
                "policies": {}}
        for p in R.POLICIES5:
            pt = R.point(m, tgt, taus[p])
            lo_, hi_ = sorted((taus[p], t_or))
            sel = (st >= lo_) & (st < hi_)
            cell["policies"][p] = {
                "tau": float(taus[p]), "frr_pct": 100 * pt["frr"], "far_pct": 100 * pt["far"], "hter_pct": 100 * pt["hter"],
                "ttg_pp": 100 * pt["ttg"], "transfer_share": pt["ttg"] / pt["hter"] if pt["hter"] > 0 else float("nan"),
                "ttg_over_50_minus_oracle": pt["ttg"] / (0.5 - pt["hter_oracle"]) if pt["hter_oracle"] < 0.5 else float("nan"),
                "miss_sd": float((taus[p] - t_or) / sd),
                "miss_rank": {"pooled": float(np.mean(sel)), "bona": float(np.mean(sel[isb])), "spoof": float(np.mean(sel[~isb]))}}
        # C: target-oracle FRR at a target FAR budget (BPCER20-style)
        cell["target_oracle_frr_at_far"] = {}
        for bud in (0.01, 0.05):
            t_b = R.tau_at_far(st, yt, bud)
            cell["target_oracle_frr_at_far"][str(bud)] = {"tau": float(t_b), "frr_pct": 100 * R.frr_at(t_b, st, yt),
                                                          "far_pct": 100 * R.far_at(t_b, st, yt)}
        # B3 / B4 applied to the targets
        for name, t in (("tau_lo", tau_lo), ("tau_hi", tau_hi), ("tau_eval", t_ev)):
            cell[name] = {"frr_pct": 100 * R.frr_at(t, st, yt), "far_pct": 100 * R.far_at(t, st, yt)}
        res["targets"][tgt] = cell
    return m, res


def main():
    R.gate(verbose=False)
    L = ["gate passed: every point estimate recomputed here equals the committed numbers to 1e-12", ""]
    # validate the band definition against the committed figures_data.json before using the rank variants
    import json
    fig = json.loads((R.CLONE / "paper" / "figures" / "figures_data.json").read_text())
    with Pool(5) as pool:
        det = dict(pool.map(b1_b3_b4_c_d, R.MODELS))
        boot = dict(pool.map(dev_bootstrap, R.MODELS))
    names = {"19la_eval": "19LA eval", "itw": "ITW", "df21_verified": "DF21 verified"}
    for m in R.MODELS:
        for tgt in R.TARGETS:
            c = det[m]["targets"][tgt]
            f = fig[f"{m}|{names[tgt]}"]
            R.check(f"{m}/{tgt} safe_band_sd vs figures_data", c["band_sd"], f["safe_band_sd"])
            R.check(f"{m}/{tgt} offset_sd vs figures_data", c["policies"]["eer"]["miss_sd"], f["tau_star_offset_sd"])
    L.append("band / offset definitions equal paper/figures/figures_data.json to 1e-12 (15 model x target cells)\n")

    L.append("== B1  The tau_eer estimator (metrics.eer): linear interpolation of the dev FRR = FAR crossing over the unique dev scores;")
    L.append("   frr budgets = dev bona fide quantiles (tau_at_frr); far budgets = tau_at_far. Dev n = 24,844 (2,548 bona fide / 22,296 spoof).")
    L.append(f"{'model':12s} {'dev EER%':>8s} {'tau_eer':>9s} {'tau_frr1':>9s} {'tau_frr5':>9s} {'tau_far1':>9s} {'tau_far5':>9s} "
             f"{'margin':>8s} {'bona<tau':>8s} {'spoof>=tau':>10s}")
    for m in R.MODELS:
        d = det[m]["dev"]
        t = d["taus"]
        L.append(f"{R.NICE[m]:12s} {d['eer_pct']:8.3f} {t['eer']:+9.3f} {t['frr1']:+9.3f} {t['frr5']:+9.3f} {t['far1']:+9.3f} {t['far5']:+9.3f} "
                 f"{d['margin']:+8.2f} {d['n_bona_below_tau_eer']:8d} {d['n_spoof_at_or_above_tau_eer']:10d}")
    L.append("   margin = smallest dev bona fide score - largest dev spoof score (negative: the classes overlap on dev).")
    L.append("   bona<tau / spoof>=tau = the dev trials on the wrong side of tau_eer, i.e. what pins the crossing.")

    L.append("\n== B3  EER-equivalent interval on dev (dev HTER within one bona fide trial of HTER(tau_eer)) and what its ends do on the targets (FRR/FAR %)")
    L.append(f"{'model':12s} {'tau_lo':>9s} {'tau_eer':>9s} {'tau_hi':>9s} | " + " | ".join(f"{names[t]:^47s}" for t in ("itw", "df21_verified")))
    for m in R.MODELS:
        e = det[m]["eer_equivalent_interval"]
        parts = []
        for t in ("itw", "df21_verified"):
            c = det[m]["targets"][t]
            parts.append(f"lo {c['tau_lo']['frr_pct']:5.1f}/{c['tau_lo']['far_pct']:5.1f}  reg {c['policies']['eer']['frr_pct']:5.1f}/{c['policies']['eer']['far_pct']:5.1f}  hi {c['tau_hi']['frr_pct']:5.1f}/{c['tau_hi']['far_pct']:5.1f}")
        L.append(f"{R.NICE[m]:12s} {e['tau_lo']:+9.3f} {e['tau_eer']:+9.3f} {e['tau_hi']:+9.3f} | " + " | ".join(parts))

    L.append("\n== B2  Dev-set bootstrap (class-stratified, B = 1,000): 95% interval of the target FRR / FAR (%) and TTG (pp), carrying dev-threshold uncertainty only")
    for tgt in ("itw", "df21_verified", "19la_eval"):
        L.append(f"-- {names[tgt]}")
        L.append(f"{'model':12s} " + " ".join(f"{p:>40s}" for p in ("eer", "frr1", "frr5")))
        for m in R.MODELS:
            row = []
            for p in ("eer", "frr1", "frr5"):
                c = boot[m]["targets"][tgt][p]
                row.append(f"FRR [{100*c['frr_ci95'][0]:5.1f},{100*c['frr_ci95'][1]:5.1f}] TTG [{100*c['ttg_ci95'][0]:5.2f},{100*c['ttg_ci95'][1]:5.2f}]")
            L.append(f"{R.NICE[m]:12s} " + " ".join(f"{r:>40s}" for r in row))
    L.append("   tau 95% intervals (dev bootstrap): " + "; ".join(
        f"{R.NICE[m]} eer [{boot[m]['taus']['eer'][0]:+.2f},{boot[m]['taus']['eer'][1]:+.2f}] frr1 [{boot[m]['taus']['frr1'][0]:+.2f},{boot[m]['taus']['frr1'][1]:+.2f}]" for m in R.MODELS))

    L.append("\n== B4  Threshold set on the LA19 EVAL scores (Zhou-and-Wang convention) vs the registered dev threshold: FRR/FAR % on the shifted targets")
    L.append(f"{'model':12s} {'EER_eval%':>9s} {'tau_eval':>9s} {'tau_dev':>9s} | {'ITW eval-tau':>16s} {'ITW dev-tau':>16s} | {'DF21 eval-tau':>16s} {'DF21 dev-tau':>16s}")
    for m in R.MODELS:
        ev = det[m]["eval_threshold"]
        i, dfc = det[m]["targets"]["itw"], det[m]["targets"]["df21_verified"]
        L.append(f"{R.NICE[m]:12s} {ev['eer_eval_pct']:9.3f} {ev['tau']:+9.3f} {det[m]['dev']['taus']['eer']:+9.3f} | "
                 f"{i['tau_eval']['frr_pct']:7.1f}/{i['tau_eval']['far_pct']:6.1f}% {i['policies']['eer']['frr_pct']:7.1f}/{i['policies']['eer']['far_pct']:6.1f}% | "
                 f"{dfc['tau_eval']['frr_pct']:7.1f}/{dfc['tau_eval']['far_pct']:6.1f}% {dfc['policies']['eer']['frr_pct']:7.1f}/{dfc['policies']['eer']['far_pct']:6.1f}%")

    L.append("\n== C  Registered and FAR-budget policies on every target: FRR/FAR % (HTER %, TTG pp) and the target-oracle FRR at FAR <= 5% (discrimination only)")
    for tgt in R.TARGETS:
        L.append(f"-- {names[tgt]}")
        L.append(f"{'model':12s} {'oracleHTER':>10s} {'FRR@FAR5*':>9s} " + " ".join(f"{p:>26s}" for p in R.POLICIES5))
        for m in R.MODELS:
            c = det[m]["targets"][tgt]
            row = [f"{c['policies'][p]['frr_pct']:5.1f}/{c['policies'][p]['far_pct']:5.1f} ({c['policies'][p]['hter_pct']:4.1f},{c['policies'][p]['ttg_pp']:5.2f})" for p in R.POLICIES5]
            L.append(f"{R.NICE[m]:12s} {c['oracle_hter_pct']:10.2f} {c['target_oracle_frr_at_far']['0.05']['frr_pct']:9.1f} " + " ".join(f"{r:>26s}" for r in row))
    L.append("   * FRR at the smallest TARGET threshold whose TARGET FAR <= 5% (BPCER20-style); compare with the FRR column of far5.")

    L.append("\n== D  Transfer share of the deployed error at tau_eer (TTG / HTER) and the rank-based band / miss (shares of target clips, %), vs the SD band")
    L.append(f"{'model':12s} {'target':14s} {'HTER%':>6s} {'oracle%':>7s} {'TTG':>6s} {'share':>6s} {'band SD':>8s} {'band rank%':>10s} {'miss SD':>8s} {'miss rank%':>10s}")
    for tgt in R.TARGETS:
        for m in R.MODELS:
            c = det[m]["targets"][tgt]
            p = c["policies"]["eer"]
            L.append(f"{R.NICE[m]:12s} {names[tgt]:14s} {p['hter_pct']:6.2f} {c['oracle_hter_pct']:7.2f} {p['ttg_pp']:6.2f} {p['transfer_share']:6.2f} "
                     f"{c['band_sd']:8.2f} {100 * c['band_rank']['pooled']:10.2f} {p['miss_sd']:+8.2f} {100 * p['miss_rank']['pooled']:10.2f}")
    R.write_json("b_thresholds.json", {"deterministic": det, "dev_bootstrap": boot, "B2": B2, "seed": SEED})
    (R.RESULTS / "b_thresholds.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
