"""Operating points at the frozen taus, plus the registered cost-weighted robustness layer.
DESCRIPTIVE: no hypothesis test is computed here.

    python analysis/operating_points.py     # -> results/analysis/operating_points.{json,txt}

Registered in PREREGISTRATION.md ("Frozen decisions"): primary metric HTER@tau* with its
FRR/FAR decomposition at the three frozen 19LA-dev policies (eer, frr1, frr5); robustness
layer = screening (p_bona=.95) and forensic (c_far=10) weighted costs. Methodology C4: the
headline results are recomputed under both cost models, and an instability is reported as a
finding. Never computed before 2026-10-03; written for the conference paper (D49).

Per model x target x policy, fp32 PyTorch scores, frozen taus (nothing re-calibrated):
  FRR, FAR, HTER at tau*; target-oracle EER and min-HTER; TTG (metrics.ttg, D19).
  Cost model k (metrics.weighted_cost): screening p_bona=0.95, c_frr=1, c_far=1;
  forensic p_bona=0.5, c_frr=1, c_far=10. Each cost is divided by the cost of the better
  trivial system, min(p_bona*c_frr, (1-p_bona)*c_far): C_norm > 1 means the detector at tau*
  costs more than not detecting at all (accept all, or reject all). This is a constant
  rescaling per cost model, so no ordering changes. Cost-weighted TTG_k = C_norm,k(tau*) -
  min_tau C_norm,k(tau) (the minimum includes both trivial systems, so it is <= 1).
  TTG CI at tau_eer on 19LA eval only (h1_h4.ttg_ci: class-stratified, B=1000, seed 0); the
  ITW and DF21 CIs already exist in results/analysis/h1_h4.json.
  Per model x target (policy-free): per-class score percentiles 1/10/25/50/75/90/99, to show
  where the frozen and oracle thresholds fall in each class's distribution.
Targets: 19LA eval (seen corpus, unseen attacks; D37), ITW, DF21 verified (D34), and DF21
clean-only (the D34 sensitivity surface; ids = scores_df21clean_rawnet2.csv, as in
transfer_matrix.py).
Refuses (exit) unless: every file is fp32 with labels in {0,1}; the trial counts match the
frozen surfaces; HTER/TTG/FRR/FAR equal paper/figures/figures_data.json and
results/analysis/h1_h4.json to 1e-12; the DF21-clean fp32 rows match transfer_matrix.txt to
its printed precision.
"""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from deploybench.metrics import det_curve, eer, far_at, frr_at, hter_at, weighted_cost  # noqa: E402
from h1_h4 import ttg_ci  # noqa: E402

MODELS = ("aasist", "aasist_l", "rawnet2", "tssdnet", "xlsr_aasist")
RANKED = MODELS[:4]
POLICIES = ("eer", "frr1", "frr5")
P1 = R / "results" / "phase_1_reproduction"
MM = R / "results" / "phase_2_conversion_app" / "main_matrix"
OUT = R / "results" / "analysis"
TARGETS = {  # name -> (score file, frozen trial count, key in figures_data.json, key in h1_h4.json)
    "19la_eval": (lambda m: P1 / f"scores_repro_{m}.csv", 71237, "19LA eval", None),
    "itw": (lambda m: MM / m / "scores_fp32_pytorch_itw.csv", 31779, "ITW", "itw"),
    "df21_verified": (lambda m: MM / m / "scores_fp32_pytorch_df21_verified.csv", 455508,
                      "DF21 verified", "df21_verified"),
    "df21_clean": (lambda m: MM / m / "scores_fp32_pytorch_df21_verified.csv", 305525, None, None),
}
COSTS = {"screening": {"p_bona": 0.95, "c_frr": 1.0, "c_far": 1.0},
         "forensic": {"p_bona": 0.5, "c_frr": 1.0, "c_far": 10.0}}
PCTS = (1, 10, 25, 50, 75, 90, 99)
TOL = 1e-12


def die(msg):
    raise SystemExit(f"REFUSED: {msg}")


def load(p: Path, keep=None):
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    if {r["precision"] for r in rows} != {"fp32"}:
        die(f"{p}: not all fp32")
    if keep is not None:
        rows = [r for r in rows if r["utt_id"] in keep]
    y = np.array([int(r["label"]) for r in rows])
    if set(np.unique(y)) != {0, 1}:
        die(f"{p}: labels are not exactly {{0, 1}}")
    return np.array([float(r["score"]) for r in rows]), y


def cost_norm(k):
    c = COSTS[k]
    return min(c["p_bona"] * c["c_frr"], (1.0 - c["p_bona"]) * c["c_far"])


def cell(s, y, tau, curve):
    taus, frr, far = curve
    h = 0.5 * (frr + far)
    i = int(np.argmin(h))
    out = {"tau": tau, "frr": frr_at(tau, s, y), "far": far_at(tau, s, y),
           "hter": hter_at(tau, s, y), "hter_oracle": float(h[i]), "tau_oracle": float(taus[i])}
    out["ttg"] = out["hter"] - out["hter_oracle"]
    out["costs"] = {}
    for k, c in COSTS.items():
        z = cost_norm(k)
        fixed = weighted_cost(tau, s, y, **c) / z
        curve_c = (c["p_bona"] * c["c_frr"] * frr + (1.0 - c["p_bona"]) * c["c_far"] * far) / z
        out["costs"][k] = {"c_norm_fixed": fixed, "c_norm_min": float(curve_c.min()),
                           "ttg_cost": fixed - float(curve_c.min())}
    return out


def check(name, got, want, tol=TOL):
    if abs(got - want) > tol:
        die(f"{name}: computed {got!r} != committed {want!r}")


def main():
    fig = json.loads((R / "paper" / "figures" / "figures_data.json").read_text(encoding="utf-8"))
    h14 = json.loads((OUT / "h1_h4.json").read_text(encoding="utf-8"))
    h14c = {(c["model"], c["target"], c["policy"]): c for c in h14["cells"]}
    clean_ids = {r["utt_id"] for r in csv.DictReader(open(
        P1 / "scores_df21clean_rawnet2.csv", encoding="utf-8"))}
    tm = (MM / "transfer_matrix.txt").read_text(encoding="utf-8")
    cells, dists = [], []
    for m in MODELS:
        taus = json.loads((P1 / f"tau_{m}.json").read_text(encoding="utf-8"))["taus"]
        for tgt, (path, n_frozen, fig_key, h14_key) in TARGETS.items():
            s, y = load(path(m), clean_ids if tgt == "df21_clean" else None)
            if len(y) != n_frozen:
                die(f"{m}/{tgt}: n={len(y)}, frozen surface has {n_frozen}")
            curve = det_curve(s, y)
            e = eer(s, y)[0]
            dists.append({"model": m, "target": tgt, "tau_eer": taus["eer"],
                          "percentiles": list(PCTS),
                          "bona": [float(v) for v in np.percentile(s[y == 1], PCTS)],
                          "spoof": [float(v) for v in np.percentile(s[y == 0], PCTS)]})
            for pol in POLICIES:
                c = {"model": m, "ranked": m in RANKED, "target": tgt, "policy": pol,
                     "n": int(len(y)), "n_bona": int((y == 1).sum()), "eer": e}
                c.update(cell(s, y, taus[pol], curve))
                if pol == "eer" and tgt == "19la_eval":
                    c["ttg_ci95"] = ttg_ci(s, y, taus[pol])
                if fig_key and pol == "eer":
                    f = fig[f"{m}|{fig_key}"]
                    for a, b in (("hter", "hter_fixed"), ("hter_oracle", "hter_oracle"),
                                 ("frr", "frr_fixed"), ("far", "far_fixed")):
                        check(f"{m}/{tgt}/{a} vs figures_data", c[a], f[b])
                if h14_key:
                    g = h14c[(m, h14_key, pol)]
                    for a, b in (("hter", "hter_fixed"), ("hter_oracle", "hter_oracle"), ("ttg", "ttg")):
                        check(f"{m}/{tgt}/{pol}/{a} vs h1_h4", c[a], g[b])
                if tgt == "df21_clean" and pol == "eer" and m in RANKED:
                    row = re.search(rf"^{m}/df21_clean\s+fp32\s+n=\s*(\d+) EER\s+([\d.]+)% \| tau_eer HTER\s+"
                                    rf"([\d.]+)% FRR\s+([\d.]+)% FAR\s+([\d.]+)%", tm, re.M)
                    if not row:
                        die(f"{m}: no df21_clean fp32 row in transfer_matrix.txt")
                    check(f"{m}/df21_clean n", c["n"], int(row.group(1)), 0)
                    for a, v in zip(("eer", "hter", "frr", "far"), row.groups()[1:]):
                        check(f"{m}/df21_clean/{a} vs transfer_matrix", 100 * c[a], float(v), 0.005 + 1e-9)
                cells.append(c)

    for d in dists:  # tau_oracle is policy-free; take it from the tau_eer cell
        d["tau_oracle"] = next(c["tau_oracle"] for c in cells if c["model"] == d["model"]
                               and c["target"] == d["target"] and c["policy"] == "eer")
    out = {"descriptive": True, "costs": COSTS,
           "cost_normaliser": {k: cost_norm(k) for k in COSTS}, "cells": cells,
           "score_distributions": dists}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "operating_points.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    L = ["DESCRIPTIVE. fp32 PyTorch, frozen 19LA-dev taus. Rates in %. C = cost / cost of the better",
         "trivial system (screening: 0.95 FRR + 0.05 FAR, /0.05; forensic: 0.5 FRR + 5 FAR, /0.5);",
         "C > 1 = worse than not detecting. dC = C(tau*) - min_tau C (cost-weighted TTG).", ""]
    for tgt in TARGETS:
        L.append(f"== {tgt}")
        L.append(f"{'model':11s} {'pol':4s} {'EER':>6s} {'FRR':>6s} {'FAR':>6s} {'HTER':>6s} {'TTG':>6s} "
                 f"| {'C_scr':>6s} {'dC_scr':>6s} | {'C_for':>6s} {'dC_for':>6s}")
        for c in (c for c in cells if c["target"] == tgt):
            k = c["costs"]
            ci = (f"  TTG CI [{100*c['ttg_ci95'][0]:.2f}, {100*c['ttg_ci95'][1]:.2f}]"
                  if "ttg_ci95" in c else "")
            L.append(f"{c['model']:11s} {c['policy']:4s} {100*c['eer']:6.2f} {100*c['frr']:6.2f} "
                     f"{100*c['far']:6.2f} {100*c['hter']:6.2f} {100*c['ttg']:6.2f} | "
                     f"{k['screening']['c_norm_fixed']:6.2f} {k['screening']['ttg_cost']:6.2f} | "
                     f"{k['forensic']['c_norm_fixed']:6.2f} {k['forensic']['ttg_cost']:6.2f}{ci}")
        L.append("")
    L.append("Stability of the headline orderings (methodology C4). Largest gap per target x policy:")
    for tgt in ("19la_eval", "itw", "df21_verified", "df21_clean"):
        for pol in POLICIES:
            cs = [c for c in cells if c["target"] == tgt and c["policy"] == pol]
            top = {"HTER": max(cs, key=lambda c: c["ttg"])["model"]}
            for k in COSTS:
                top[k] = max(cs, key=lambda c: c["costs"][k]["ttg_cost"])["model"]
            worse = {k: sum(c["costs"][k]["c_norm_fixed"] > 1 for c in cs) for k in COSTS}
            L.append(f"  {tgt:13s} {pol:4s} largest TTG: HTER {top['HTER']:11s} screening "
                     f"{top['screening']:11s} forensic {top['forensic']:11s} | models with C>1 at tau*: "
                     f"screening {worse['screening']}/5, forensic {worse['forensic']}/5")
    L.append("\nPer-class score percentiles " + "/".join(map(str, PCTS)) + " (tau* = frozen tau_eer, "
             "tau_or = target-oracle min-HTER threshold):")
    for d in dists:
        L.append(f"  {d['model']:11s} {d['target']:13s} tau* {d['tau_eer']:+7.2f} tau_or {d['tau_oracle']:+7.2f}"
                 f" | bona " + " ".join(f"{v:+6.2f}" for v in d["bona"])
                 + " | spoof " + " ".join(f"{v:+6.2f}" for v in d["spoof"]))
    L.append("\nChecks passed: fp32 only; labels {0,1}; n = 71,237 / 31,779 / 455,508 / 305,525; HTER, TTG,")
    L.append("FRR, FAR equal figures_data.json and h1_h4.json to 1e-12; DF21-clean rows equal transfer_matrix.txt.")
    (OUT / "operating_points.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
