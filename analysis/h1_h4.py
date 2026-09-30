"""H1 and H4: EXPLORATORY (PREREGISTRATION; methodology C5). Effect sizes + CIs only;
no confirm/refute language anywhere in the output.

    python analysis/h1_h4.py [--out results/analysis/h1_h4.json]

H1: Spearman rho between source-oracle EER (19LA eval, fp32) and ITW TTG over
    model x seed x policy x target points. Primary set: the 4 ranked models x 3 frozen
    policies (eer, frr1, frr5) on ITW, fp32 (each checkpoint has one seed): 12 points.
    Secondary set: ITW + DF21 verified as two targets: 24 points.
H4: TTG vs parameter count across the roster. Spearman rho over the 4 ranked models
    (log10 params vs TTG at tau_eer), per target. XLS-R+AASIST (unranked, D16/D30) is
    reported as an ANNOTATED point and never enters rho.
TTG = HTER at the frozen 19LA-dev tau minus the target-oracle min-HTER (metrics.ttg,
D19). TTG CIs: class-stratified bootstrap of target trials, B=1000, seed 0. rho CIs:
metrics.spearman_ci (point bootstrap, B=1000, seed 0).
Caveat printed with every rho: H1 has only 4 distinct x values (one oracle EER per
model), so points cluster by model; H4 has n=4. Bootstrap CIs at that n are fragile.
Parameter counts: results/analysis/param_counts.json (counted from the loaded models).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
from deploybench.metrics import eer, hter_at, min_hter, spearman_ci, ttg  # noqa: E402

RANKED = ("aasist", "aasist_l", "rawnet2", "tssdnet")
POLICIES = ("eer", "frr1", "frr5")
TARGETS = {"itw": "scores_fp32_pytorch_itw.csv", "df21_verified": "scores_fp32_pytorch_df21_verified.csv"}
MM = R / "results" / "phase_2_conversion_app" / "main_matrix"
P1 = R / "results" / "phase_1_reproduction"
B, SEED = 1000, 0


def load(p):
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    if {r["precision"] for r in rows} != {"fp32"}:
        raise SystemExit(f"{p}: not all fp32")
    return np.array([float(r["score"]) for r in rows]), np.array([int(r["label"]) for r in rows])


def ttg_ci(s, y, tau, n_boot=B, seed=SEED):
    ib, isp = np.where(y == 1)[0], np.where(y == 0)[0]
    rng = np.random.default_rng(seed)
    v = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([ib[rng.integers(0, len(ib), len(ib))],
                              isp[rng.integers(0, len(isp), len(isp))]])
        v[i] = hter_at(tau, s[idx], y[idx]) - min_hter(s[idx], y[idx])[0]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(R / "results" / "analysis" / "h1_h4.json"))
    a = ap.parse_args()
    params = json.loads((R / "results" / "analysis" / "param_counts.json").read_text())
    models = list(RANKED) + ["xlsr_aasist"]
    oracle, cells = {}, []
    for m in models:
        s, y = load(P1 / f"scores_repro_{m}.csv")
        oracle[m] = eer(s, y)[0]
        taus = json.loads((P1 / f"tau_{m}.json").read_text())["taus"]
        for tgt, fn in TARGETS.items():
            p = MM / m / fn
            if not p.is_file():
                continue
            st, yt = load(p)
            for pol in POLICIES:
                r = ttg(st, yt, taus[pol])
                cell = {"model": m, "ranked": m in RANKED, "target": tgt, "policy": pol,
                        "oracle_eer_19la": oracle[m], "ttg": r["ttg"], "hter_fixed": r["hter_fixed"],
                        "hter_oracle": r["hter_oracle"]}
                if pol == "eer":
                    cell["ttg_ci95"] = ttg_ci(st, yt, taus[pol])
                cells.append(cell)

    def rho(points, xkey):
        x = [c[xkey] for c in points]
        return spearman_ci(x, [c["ttg"] for c in points], n_boot=B, seed=SEED)

    rk = [c for c in cells if c["ranked"]]
    h1 = {"primary_itw": rho([c for c in rk if c["target"] == "itw"], "oracle_eer_19la"),
          "secondary_itw_df21": rho(rk, "oracle_eer_19la"),
          "distinct_x": len({c["oracle_eer_19la"] for c in rk})}
    for c in cells:
        c["log10_params"] = float(np.log10(params[c["model"]]["params"]))
    h4 = {}
    for tgt in TARGETS:
        pts = [c for c in rk if c["target"] == tgt and c["policy"] == "eer"]
        if len(pts) == 4:
            h4[tgt] = rho(pts, "log10_params")
    out = {"exploratory": True, "B": B, "seed": SEED, "h1": h1, "h4": h4,
           "params": {m: params[m]["params"] for m in models}, "cells": cells}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print("EXPLORATORY (effect size + CI only)\n")
    print("per model at tau_eer: 19LA-eval oracle EER | params | TTG on target [95% CI]")
    for m in models:
        for tgt in TARGETS:
            c = next((c for c in cells if c["model"] == m and c["target"] == tgt and c["policy"] == "eer"), None)
            if c:
                tag = "" if c["ranked"] else "   (unranked, annotated only)"
                print(f"  {m:11s} {tgt:13s} EER {oracle[m]*100:5.2f}% | {params[m]['params']:>11,d} | "
                      f"TTG {c['ttg']*100:+6.2f} pp [{c['ttg_ci95'][0]*100:+.2f}, {c['ttg_ci95'][1]*100:+.2f}]{tag}")
    f = lambda r: f"rho {r['rho']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}], n={r['n']}"
    print(f"\nH1 primary (ITW, 4 models x 3 policies): {f(h1['primary_itw'])}")
    print(f"H1 secondary (ITW + DF21):               {f(h1['secondary_itw_df21'])}")
    print(f"   caveat: only {h1['distinct_x']} distinct oracle-EER values; points cluster by model.")
    for tgt, r in h4.items():
        print(f"H4 {tgt:13s} (4 ranked, log10 params vs TTG@eer): {f(r)}   caveat: n=4")


if __name__ == "__main__":
    main()
