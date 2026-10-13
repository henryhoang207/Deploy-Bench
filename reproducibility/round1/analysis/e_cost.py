"""DEV-8 E1: the cost sweep (S5) and the break-even windows.

    python e_cost.py                 # gate first; writes results/e_cost.json and results/e_cost.txt

Normalised cost on an effective-prior axis, p = P(bona fide) when both errors cost 1:
    C(p) = [p FRR + (1 - p) FAR] / min(p, 1 - p)       (< 1 beats the better trivial system: accept all / reject all)
The registered screening model (pi = 0.95, equal costs) is p = 0.95; the registered forensic model (pi = 0.5, c_far = 10)
is p = 1/11. For one detector at one threshold, C < 1 exactly for
    FAR / (FAR + 1 - FRR)  <  p  <  (1 - FAR) / (1 - FAR + FRR)          (the break-even window of bona fide priors)
and for screening (p = 0.95) this is the condition FRR < (1 - FAR) / 19.
"""
from __future__ import annotations

import numpy as np

import revlib as R

XS = np.linspace(-4.6, 4.6, 185)
P = 1.0 / (1.0 + np.exp(-XS))


def window(frr, far):
    lo = far / (far + 1.0 - frr) if (far + 1.0 - frr) > 0 else 1.0
    hi = (1.0 - far) / (1.0 - far + frr) if (1.0 - far + frr) > 0 else 0.0
    return (float(lo), float(hi)) if hi > lo else None


def main():
    R.gate(verbose=False)
    L = ["gate passed: every point estimate recomputed here equals the committed numbers to 1e-12", ""]
    out = {"p_grid": P.tolist(), "curves": {}, "windows": {}}
    names = {"19la_eval": "LA19 eval", "itw": "ITW", "df21_verified": "DF21 verified"}
    L.append("== E1  Break-even window of bona fide priors (p_lo, p_hi) in which the detector beats the better trivial system, at each threshold")
    L.append("   (screening = p 0.95, forensic = p 0.091; 'none' = never beats a trivial system. Oracle = best threshold for the target: its normalised cost at the")
    L.append("    two registered points and the largest bona fide prior at which it still gains >= 10% over the better trivial system)")
    for tgt in R.TARGETS:
        L.append(f"-- {names[tgt]}")
        L.append(f"{'model':12s} " + " ".join(f"{p:>15s}" for p in ("eer", "frr1", "frr5", "far1", "far5")) + f" {'oracle C scr/for/pmax':>26s}")
        for m in R.MODELS:
            ids, y, s = R.load(m, tgt)
            taus = R.all_taus(m)
            grid, frr, far = R.det_curve(s, y)
            row = []
            out["curves"][f"{m}|{tgt}"] = {}
            out["windows"][f"{m}|{tgt}"] = {}
            for pol in R.POLICIES5:
                fr, fa = R.frr_at(taus[pol], s, y), R.far_at(taus[pol], s, y)
                c = (P * fr + (1 - P) * fa) / np.minimum(P, 1 - P)
                out["curves"][f"{m}|{tgt}"][pol] = [round(float(v), 5) for v in c]
                w = window(fr, fa)
                out["windows"][f"{m}|{tgt}"][pol] = w
                row.append("none" if w is None else f"({w[0]:.2f},{w[1]:.2f})")
            # oracle: best threshold at each p (includes the two trivial systems through the pads of det_curve)
            co = np.array([float(np.min((p * frr + (1 - p) * far) / min(p, 1 - p))) for p in P])
            out["curves"][f"{m}|{tgt}"]["oracle"] = [round(float(v), 5) for v in co]
            c95 = float(np.min((0.95 * frr + 0.05 * far) / 0.05))            # registered screening point
            c09 = float(np.min((1 / 11 * frr + 10 / 11 * far) / (1 / 11)))   # registered forensic point
            good = P[co <= 0.9]                                               # priors where even the oracle gains >= 10% over trivial
            pmax = float(good.max()) if len(good) else float("nan")
            out["windows"][f"{m}|{tgt}"]["oracle"] = {"c_screening": c95, "c_forensic": c09, "max_p_with_gain_10pct": pmax}
            orow = f"{c95:5.2f} / {c09:5.2f} / {pmax:4.2f}"
            L.append(f"{R.NICE[m]:12s} " + " ".join(f"{r:>15s}" for r in row) + f" {orow:>26s}")
    L.append("")
    L.append("== Screening break-even FRR < (1 - FAR)/19 at the transferred tau_eer: FRR needed vs FRR obtained (%)")
    for tgt in ("itw", "df21_verified"):
        parts = []
        for m in R.MODELS:
            ids, y, s = R.load(m, tgt)
            t = R.frozen_taus(m)[0]["eer"]
            fr, fa = R.frr_at(t, s, y), R.far_at(t, s, y)
            parts.append(f"{R.NICE[m]}: need <{100 * (1 - fa) / 19:.1f}, got {100 * fr:.1f}")
        L.append(f"{names[tgt]:14s} " + " | ".join(parts))
    R.write_json("e_cost.json", out)
    (R.RESULTS / "e_cost.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
