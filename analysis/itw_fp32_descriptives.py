"""Descriptive fp32 In-the-Wild numbers per ranked model at its FROZEN 19LA-dev taus.

    python analysis/itw_fp32_descriptives.py

DESCRIPTIVE ONLY. H2 (FRR > FAR on ITW at tau_eer, majority of the four ranked models)
is confirmatory: its registered test (per-model paired bootstrap, Holm-Bonferroni) runs in
analysis/run_analysis.py once all four ranked models have fp32 ITW scores. Nothing here
states a confirm/refute verdict.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
from deploybench.metrics import eer, far_at, frr_at, hter_at, ttg  # noqa: E402

MODELS = ("aasist", "aasist_l", "rawnet2", "tssdnet")


def auc(s, y):
    b, sp = s[y == 1], s[y == 0]
    order = np.argsort(np.concatenate([b, sp]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return (ranks[: len(b)].sum() - len(b) * (len(b) + 1) / 2) / (len(b) * len(sp))


def main():
    for m in MODELS:
        p = R / "results" / "phase_2_conversion_app" / "main_matrix" / m / "scores_fp32_pytorch_itw.csv"
        if not p.is_file():
            print(f"{m:9s} (no fp32 ITW scores yet)")
            continue
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        if {r["precision"] for r in rows} != {"fp32"} or {r["model"] for r in rows} != {m}:
            raise SystemExit(f"{p}: rows are not all {m}/fp32")
        s = np.array([float(r["score"]) for r in rows])
        y = np.array([int(r["label"]) for r in rows])
        taus = json.loads((R / "results" / "phase_1_reproduction" / f"tau_{m}.json").read_text())["taus"]
        t = taus["eer"]
        print(f"{m:9s} n={len(s)} EER {eer(s, y)[0]*100:6.2f}%  AUC {auc(s, y):.3f} | tau_eer: "
              f"HTER {hter_at(t, s, y)*100:6.2f}%  FRR {frr_at(t, s, y)*100:6.2f}%  "
              f"FAR {far_at(t, s, y)*100:6.2f}%  TTG {ttg(s, y, t)['ttg']*100:+6.2f} pp")


if __name__ == "__main__":
    main()
