"""Descriptive facts quoted in the revised paper that no earlier script wrote down (DEV-8 amendment 2).

    python facts_extra.py       # -> results/extra_facts.{txt,json}

All are simple statistics of the committed fp32 scores (gate-checked through revlib); none is a test.
  1. ITW AUC per model (RawNet2 near 0.5 is why H2 has three informative models).
  2. Score saturation: XLS-R on LA19 eval and ITW, RawNet2's ceiling on LA19 eval (Fig. S1 caption).
  3. Triage illustration (R3 W4, Q4): the share of target clips whose score lies inside the dev-equivalent interval of tau_EER (B3), per
     class. If that interval were the review band of a three-zone policy, these clips would go to a human.
"""
from __future__ import annotations

import json

import numpy as np

import revlib as R

OUT = []
RES = {}


def say(s=""):
    OUT.append(s)
    print(s, flush=True)


def auc(s, y):
    """P(score of a random bona fide clip > score of a random spoof) with ties counted half; rank-based, O(n log n)."""
    order = np.argsort(s, kind="mergesort")
    ss, yy = s[order], y[order]
    # average ranks for ties
    uniq, inv, cnt = np.unique(ss, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    avg_rank = csum - (cnt - 1) / 2.0
    ranks = avg_rank[inv]
    n1, n0 = int((yy == 1).sum()), int((yy == 0).sum())
    return float((ranks[yy == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def main():
    R.gate(verbose=False)
    b3 = json.loads((R.RESULTS / "b_thresholds.json").read_text(encoding="utf-8"))["deterministic"]
    say("== 1  AUC on ITW (bona fide score > spoof score)")
    RES["auc_itw"] = {}
    for m in R.MODELS:
        ids, y, s = R.load(m, "itw")
        a = auc(s, y)
        RES["auc_itw"][m] = a
        say(f"   {R.NICE[m]:12s} AUC {a:.3f}")

    say("\n== 2  Score saturation")
    ids, y, s = R.load("xlsr_aasist", "19la_eval")
    lo_s, hi_s = np.percentile(s[y == 0], [1, 99])
    lo_b, hi_b = np.percentile(s[y == 1], [1, 99])
    tau = R.frozen_taus("xlsr_aasist")[0]["eer"]
    say(f"   XLS-R LA19 eval: 98% of spoof scores in [{lo_s:.2f}, {hi_s:.2f}], 98% of bona fide scores in [{lo_b:.2f}, {hi_b:.2f}]; tau_EER {tau:.2f}")
    RES["xlsr_la19"] = {"spoof_p1_p99": [float(lo_s), float(hi_s)], "bona_p1_p99": [float(lo_b), float(hi_b)], "tau_eer": float(tau)}
    ids, y, s = R.load("xlsr_aasist", "itw")
    q1, q3 = np.percentile(s[y == 0], [25, 75])
    med_b = float(np.median(s[y == 1]))
    taus_all = R.all_taus("xlsr_aasist")
    say(f"   XLS-R ITW: half of the spoof scores in [{q1:.2f}, {q3:.2f}]; genuine median {med_b:.2f}; tau_EER {tau:.2f}")
    RES["xlsr_itw"] = {"spoof_q1_q3": [float(q1), float(q3)], "bona_median": med_b}
    ids, y, s = R.load("rawnet2", "19la_eval")
    ceil = float(np.max(s))
    share = float(np.mean(s[y == 1] >= -1e-3))
    say(f"   RawNet2 LA19 eval: max score {ceil:.4f}; share of bona fide clips with score >= -1e-3: {100 * share:.1f}%")
    RES["rawnet2_ceiling"] = {"max": ceil, "bona_share_within_1e-3": share}

    say("\n== 3  Triage illustration: share of clips (%) whose score lies in [tau_lo, tau_hi) of the dev-equivalent interval of tau_EER")
    say(f"   {'model':12s} | {'ITW genuine':>11s} {'ITW spoof':>10s} | {'DF21 genuine':>12s} {'DF21 spoof':>10s}")
    RES["triage"] = {}
    for m in R.MODELS:
        iv = b3[m]["eer_equivalent_interval"]
        lo, hi = iv["tau_lo"], iv["tau_hi"]
        row = {}
        for t in ("itw", "df21_verified"):
            ids, y, s = R.load(m, t)
            inb = (s >= lo) & (s < hi)
            row[t] = {"genuine": float(np.mean(inb[y == 1])), "spoof": float(np.mean(inb[y == 0]))}
        RES["triage"][m] = row
        say(f"   {R.NICE[m]:12s} | {100 * row['itw']['genuine']:11.1f} {100 * row['itw']['spoof']:10.1f} | "
            f"{100 * row['df21_verified']['genuine']:12.1f} {100 * row['df21_verified']['spoof']:10.1f}")
    R.write_json("extra_facts.json", RES)
    (R.RESULTS / "extra_facts.txt").write_text("\n".join(OUT) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
