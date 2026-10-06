"""DEV-8 A1-A4 and E2: speaker-cluster inference.

    python a_cluster.py            # runs the gate first; writes results/a_cluster.json and results/a_cluster.txt

A1  H2, cluster version (ITW speaker labels), B = 10,000, seed 0, p = (1 + #{stat <= 0}) / (B + 1), Holm over 4, >= 3 of 4.
A2  H5, cluster version (LA19-eval speaker ids of the 1,000-clip manifest) + the two component tests.
A3  TTG intervals with a speaker-cluster bootstrap (B = 1,000, seed 0) at all five policies; LA19 eval, ITW, DF21 verified.
E2  Stability of the largest-gap orderings across the same cluster replicates (HTER, screening, forensic).
A4  Per-speaker view of ITW at tau_eer (descriptive).
Registered clip-level numbers are read from the committed JSON and printed beside the cluster ones.
"""
from __future__ import annotations

import json

import numpy as np

import revlib as R

B_TEST, B_CI, SEED = 10000, 1000, 0
OUT = []


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def pval_le0(stat):
    return float((1 + np.sum(stat <= 0)) / (len(stat) + 1))


def a1_h2():
    reg = json.loads((R.COMMITTED / "h2_test.json").read_text())["per_model"]
    res, pv = {}, {}
    for m in R.RANKED:
        ids, y, s = R.load(m, "itw")
        g, names = R.clusters("itw", ids)
        G = len(names)
        C = R.cluster_counts(G, B_TEST, SEED)
        tau = R.frozen_taus(m)[0]["eer"]
        fr, fa = R.cluster_rates(s, y, g, G, tau, C)
        st = fr - fa
        nb, rej, ns, acc = R.cluster_aggregates(s, y, g, G, tau)
        point = float(rej.sum() / nb.sum() - acc.sum() / ns.sum())
        # leave-one-speaker-out influence and the speaker-macro average (descriptive)
        loo = (rej.sum() - rej) / (nb.sum() - nb) - (acc.sum() - acc) / (ns.sum() - ns)
        macro = float(np.mean(rej / nb - acc / ns))
        pv[m] = pval_le0(st)
        res[m] = {"delta_pp": 100 * point, "ci95_pp": [100 * x for x in R.pct(st)], "p_one_sided": pv[m],
                  "registered_ci95_pp": [100 * x for x in reg[m]["ci95"]], "registered_p": reg[m]["p_one_sided"],
                  "loo_min_pp": 100 * float(loo.min()), "loo_max_pp": 100 * float(loo.max()),
                  "loo_argmin": names[int(loo.argmin())], "loo_argmax": names[int(loo.argmax())],
                  "macro_speaker_delta_pp": 100 * macro,
                  "share_replicates_delta_positive": float(np.mean(st > 0))}
    rej = R.holm_bonferroni(pv, 0.05)
    for m in R.RANKED:
        res[m]["holm_reject_h0"] = bool(rej[m])
    n = int(sum(rej.values()))
    out = {"B": B_TEST, "seed": SEED, "clusters": G, "per_model": res, "n_models_frr_gt_far": n,
           "h2_supported_cluster": n >= 3}
    say(f"== A1  H2 with a speaker-cluster bootstrap (ITW, {G} speaker labels, B = {B_TEST}, seed {SEED})")
    say(f"{'model':12s} {'FRR-FAR':>8s}  {'cluster 95% CI':>20s}  {'p (1+k)/(B+1)':>13s}  {'Holm':>5s}  | registered clip-level CI, p")
    for m in R.RANKED:
        r = res[m]
        say(f"{R.NICE[m]:12s} {r['delta_pp']:+8.2f}  [{r['ci95_pp'][0]:+7.2f}, {r['ci95_pp'][1]:+7.2f}]  "
            f"{r['p_one_sided']:13.4f}  {str(r['holm_reject_h0']):>5s}  | [{r['registered_ci95_pp'][0]:+.2f}, "
            f"{r['registered_ci95_pp'][1]:+.2f}], p={r['registered_p']:.3f}")
    say(f"-> FRR > FAR (Holm, alpha .05) for {n} of 4 ranked models under the cluster bootstrap; "
        f"majority needs 3 -> H2 {'SUPPORTED' if n >= 3 else 'NOT SUPPORTED'} at the speaker level.")
    say("   Influence (delete one speaker): " + "; ".join(
        f"{R.NICE[m]} {res[m]['loo_min_pp']:+.2f}..{res[m]['loo_max_pp']:+.2f} pp" for m in R.RANKED))
    say("   Speaker-macro average of FRR-FAR: " + "; ".join(
        f"{R.NICE[m]} {res[m]['macro_speaker_delta_pp']:+.2f}" for m in R.RANKED))
    return out


def a2_h5():
    reg = json.loads((R.COMMITTED / "h5_test.json").read_text())["per_model"]
    proto = R.la19_protocol("eval")
    res, pv = {}, {"stat": {}, "dfrr": {}, "dfar": {}}
    for m in R.RANKED:
        ids, y, sc, se = R.h5_inputs(m)
        names_all = np.array([proto[u][0] for u in ids])
        uniq, g = np.unique(names_all, return_inverse=True)
        G = len(uniq)
        C = R.cluster_counts(G, B_TEST, SEED)
        tau = R.frozen_taus(m)[0]["eer"]
        b, p = y == 1, y == 0
        nb = np.bincount(g[b], minlength=G).astype(float)
        ns = np.bincount(g[p], minlength=G).astype(float)
        rc = np.bincount(g[b & (sc < tau)], minlength=G).astype(float)
        re_ = np.bincount(g[b & (se < tau)], minlength=G).astype(float)
        ac = np.bincount(g[p & (sc >= tau)], minlength=G).astype(float)
        ae = np.bincount(g[p & (se >= tau)], minlength=G).astype(float)
        dfrr = (C @ (re_ - rc)) / (C @ nb)
        dfar = (C @ (ae - ac)) / (C @ ns)
        st = dfrr - dfar
        pv["stat"][m] = pval_le0(st)
        pv["dfrr"][m] = pval_le0(dfrr)                       # H0: dFRR <= 0
        pv["dfar"][m] = pval_le0(-dfar)                      # H0: dFAR >= 0
        res[m] = {"dfrr_pp": 100 * float((re_.sum() - rc.sum()) / nb.sum()),
                  "dfar_pp": 100 * float((ae.sum() - ac.sum()) / ns.sum()),
                  "stat_pp": 100 * float((re_.sum() - rc.sum()) / nb.sum() - (ae.sum() - ac.sum()) / ns.sum()),
                  "stat_ci95_pp": [100 * x for x in R.pct(st)],
                  "dfrr_ci95_pp": [100 * x for x in R.pct(dfrr)], "dfar_ci95_pp": [100 * x for x in R.pct(dfar)],
                  "p_stat": pv["stat"][m], "p_dfrr": pv["dfrr"][m], "p_dfar": pv["dfar"][m],
                  "registered_ci95_pp": [100 * x for x in reg[m]["ci95"]], "registered_p": reg[m]["p_one_sided"],
                  "n_bona": int(nb.sum()), "n_spoof": int(ns.sum()), "dfrr_bona_clips_changed": int(round(re_.sum() - rc.sum()))}
    holm = {k: R.holm_bonferroni(pv[k], 0.05) for k in pv}
    for m in R.RANKED:
        for k in pv:
            res[m][f"holm_{k}"] = bool(holm[k][m])
    ns_ = {k: int(sum(holm[k].values())) for k in pv}
    out = {"B": B_TEST, "seed": SEED, "clusters": G, "per_model": res, "n_rejected": ns_,
           "h5_supported_cluster": ns_["stat"] >= 3}
    say(f"\n== A2  H5 with a speaker-cluster bootstrap (1,000-clip manifest, {G} LA19-eval speakers, B = {B_TEST})")
    say(f"{'model':12s} {'dFRR-dFAR':>9s} {'cluster CI':>18s} {'p':>7s} {'Holm':>5s} | {'dFRR':>6s} {'p':>7s} {'Holm':>5s} | "
        f"{'dFAR':>6s} {'p':>7s} {'Holm':>5s} | registered CI")
    for m in R.RANKED:
        r = res[m]
        say(f"{R.NICE[m]:12s} {r['stat_pp']:+9.1f} [{r['stat_ci95_pp'][0]:+6.1f},{r['stat_ci95_pp'][1]:+6.1f}] "
            f"{r['p_stat']:7.4f} {str(r['holm_stat']):>5s} | {r['dfrr_pp']:+6.1f} {r['p_dfrr']:7.4f} {str(r['holm_dfrr']):>5s} | "
            f"{r['dfar_pp']:+6.1f} {r['p_dfar']:7.4f} {str(r['holm_dfar']):>5s} | "
            f"[{r['registered_ci95_pp'][0]:+.1f}, {r['registered_ci95_pp'][1]:+.1f}]")
    say(f"-> statistic rejected for {ns_['stat']}/4 (cluster); dFRR > 0 for {ns_['dfrr']}/4; dFAR < 0 for {ns_['dfar']}/4 (Holm, alpha .05)")
    return out


def replicates(target):
    ids0, _, _ = R.load(R.MODELS[0], target)
    g0, names = R.clusters(target, ids0)
    G = len(names)
    C = R.cluster_counts(G, B_CI, SEED)
    unit = np.ones((1, G))
    rep, pt = {}, {}
    for m in R.MODELS:
        ids, y, s = R.load(m, target)
        g, nm = R.clusters(target, ids)
        assert nm == names
        taus = R.all_taus(m)
        orc = R.Oracle(s, y, g)
        h_or, c_or = orc.run(C)
        h_pt, c_pt = orc.run(unit)
        rep[m], pt[m] = {}, {}
        for pol in R.POLICIES5:
            fr, fa = R.cluster_rates(s, y, g, G, taus[pol], C)
            f1, a1 = R.cluster_rates(s, y, g, G, taus[pol], unit)
            hter = 0.5 * (fr + fa)
            rep[m][pol] = {"ttg": hter - h_or, "frr": fr, "far": fa,
                           "gap": {k: R.fixed_costs(fr, fa, k) - c_or[k] for k in R.COSTS}}
            pt[m][pol] = {"ttg": float(0.5 * (f1 + a1)[0] - h_pt[0]),
                          "gap": {k: float(R.fixed_costs(f1, a1, k)[0] - c_pt[k][0]) for k in R.COSTS}}
    return rep, pt, names


def a3_e2():
    reg_h14 = {(c["model"], c["target"], c["policy"]): c for c in
               json.loads((R.COMMITTED / "h1_h4.json").read_text())["cells"]}
    reg_op = {(c["model"], c["target"], c["policy"]): c for c in
              json.loads((R.COMMITTED / "operating_points.json").read_text())["cells"]}
    out = {"B": B_CI, "seed": SEED, "ttg": {}, "stability": {}}
    for tgt in R.TARGETS:
        rep, pt, names = replicates(tgt)
        G = len(names)
        out["ttg"][tgt] = {"clusters": G, "cells": {}}
        say(f"\n== A3  TTG with a speaker-cluster bootstrap: {tgt} ({G} clusters, B = {B_CI}), pp; registered clip-level CI in brackets")
        say(f"{'model':12s} " + " ".join(f"{p:>34s}" for p in R.POLICIES5))
        for m in R.MODELS:
            row = []
            for pol in R.POLICIES5:
                ci = [100 * x for x in R.pct(rep[m][pol]["ttg"])]
                p0 = 100 * pt[m][pol]["ttg"]
                regci = None
                if (m, tgt if tgt != "19la_eval" else "19la_eval", pol) in reg_op and tgt == "19la_eval" and pol == "eer":
                    regci = [100 * x for x in reg_op[(m, tgt, pol)]["ttg_ci95"]]
                key = {"itw": "itw", "df21_verified": "df21_verified"}.get(tgt)
                if key and (m, key, pol) in reg_h14 and reg_h14[(m, key, pol)].get("ttg_ci95"):
                    regci = [100 * x for x in reg_h14[(m, key, pol)]["ttg_ci95"]]
                out["ttg"][tgt]["cells"][f"{m}|{pol}"] = {"ttg_pp": p0, "cluster_ci95_pp": ci, "registered_ci95_pp": regci}
                row.append(f"{p0:6.2f} [{ci[0]:6.2f},{ci[1]:6.2f}]" + (f" ({regci[0]:.2f},{regci[1]:.2f})" if regci else "").ljust(14))
            say(f"{R.NICE[m]:12s} " + " ".join(f"{c:>34s}" for c in row))
        # E2: stability of the largest gap (all five models, paired replicates)
        stab = {}
        for pol in R.POLICIES3:
            stab[pol] = {}
            for metric in ("hter", "screening", "forensic"):
                vals = np.stack([rep[m][pol]["ttg"] if metric == "hter" else rep[m][pol]["gap"][metric]
                                 for m in R.MODELS])                       # (5, B)
                pts = np.array([pt[m][pol]["ttg"] if metric == "hter" else pt[m][pol]["gap"][metric] for m in R.MODELS])
                am = vals.argmax(axis=0)
                freq = {R.MODELS[i]: float(np.mean(am == i)) for i in range(5)}
                top = R.MODELS[int(pts.argmax())]
                stab[pol][metric] = {"point_argmax": top, "freq": freq, "share_point_argmax": freq[top]}
        out["stability"][tgt] = stab
        say(f"-- E2  how often is each model the one with the largest gap ({tgt}, paired cluster replicates)")
        for pol in R.POLICIES3:
            for metric in ("hter", "screening", "forensic"):
                s_ = stab[pol][metric]
                say(f"   {pol:4s} {metric:9s} point-estimate argmax {R.NICE[s_['point_argmax']]:12s} in {100 * s_['share_point_argmax']:5.1f}% of replicates;  "
                    + ", ".join(f"{R.NICE[k]} {100 * v:.0f}%" for k, v in s_["freq"].items() if v >= 0.005))
    return out


def a4_speakers():
    out = {}
    say("\n== A4  Per-speaker view of ITW at tau_eer (descriptive)")
    ids0, _, _ = R.load(R.MODELS[0], "itw")
    g0, names = R.clusters("itw", ids0)
    G = len(names)
    rows = []
    for m in R.MODELS:
        ids, y, s = R.load(m, "itw")
        g, _ = R.clusters("itw", ids)
        tau = R.frozen_taus(m)[0]["eer"]
        nb, rej, ns, acc = R.cluster_aggregates(s, y, g, G, tau)
        frr_s = rej / nb
        far_s = acc / ns
        order = np.argsort(-rej)
        top5 = rej[order[:5]].sum() / rej.sum()
        worst = np.argsort(-frr_s)
        out[m] = {"clips_per_speaker_min_med_max": [int((nb + ns).min()), float(np.median(nb + ns)), int((nb + ns).max())],
                  "share_of_all_clips_top5": float(np.sort(nb + ns)[::-1][:5].sum() / (nb + ns).sum()),
                  "share_of_genuine_clips_top5": float(np.sort(nb)[::-1][:5].sum() / nb.sum()),
                  "share_of_rejections_from_top5_speakers": float(top5),
                  "speakers_frr_gt_50": int((frr_s > 0.5).sum()), "speakers_frr_gt_90": int((frr_s > 0.9).sum()),
                  "speaker_frr_quartiles": [float(x) for x in np.percentile(frr_s, [25, 50, 75])],
                  "pooled_frr": float(rej.sum() / nb.sum()), "macro_frr": float(frr_s.mean()),
                  "pooled_far": float(acc.sum() / ns.sum()), "macro_far": float(far_s.mean())}
        for i in range(G):
            rows.append((m, names[i], int(nb[i]), int(ns[i]), float(frr_s[i]), float(far_s[i])))
        o = out[m]
        say(f"{R.NICE[m]:12s} pooled FRR {100 * o['pooled_frr']:5.1f}% / macro (per-speaker mean) {100 * o['macro_frr']:5.1f}% | "
            f"speaker FRR quartiles {[round(100 * x, 1) for x in o['speaker_frr_quartiles']]} | >50%: {o['speakers_frr_gt_50']}/{G}, "
            f">90%: {o['speakers_frr_gt_90']}/{G} | top-5 speakers carry {100 * o['share_of_rejections_from_top5_speakers']:.0f}% of rejections "
            f"(and {100 * o['share_of_genuine_clips_top5']:.0f}% of genuine clips)")
    import csv
    p = R.RESULTS / "per_speaker_itw.csv"
    R.RESULTS.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "speaker", "n_bona", "n_spoof", "frr_at_tau_eer", "far_at_tau_eer"])
        w.writerows(rows)
    say(f"   per-speaker table -> {p.name} (speaker names are the dataset's public labels; the paper uses anonymous ranks)")
    return out


def main():
    R.gate(verbose=False)
    say("gate passed: every point estimate recomputed here equals the committed numbers to 1e-12\n")
    res = {"a1_h2": a1_h2(), "a2_h5": a2_h5(), "a3_e2": a3_e2(), "a4_speakers": a4_speakers()}
    R.write_json("a_cluster.json", res)
    (R.RESULTS / "a_cluster.txt").write_text("\n".join(OUT) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
