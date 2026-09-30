"""H6 (confirmatory), as amended by D36, on the REGISTERED (minmax) conversion arm.

    python analysis/h6_analysis.py [--out results/analysis/h6.json]

Primary DV (D36): per ranked model, the smallest fp32 exclusion set among the tested
configurations whose int8 artifact keeps diagnostic-subset EER within 1.0 pp of fp32.
Prediction (D36 restatement): that set contains graph-attention nodes for AASIST and
AASIST-L, and GRU nodes for RawNet2.
Secondary (D30 wording, kept by D36): the converter's fallback-op list per artifact vs
"graph attention (ScatterND / TopK+Gather) and recurrence (GRU) fall back; plain
conv1d/BN/ReLU (Res-TSSDNet) does not"; plus decision flips and dEER per cell at the
frozen taus.

H6 yields categorical outcomes, not p-values. The percentile arm (DEVIATIONS.md DEV-1)
is exploratory and is NOT used here. "Graph attention node" = a node whose name contains
"GAT" (the GAT_layer_* / HtrgGAT_layer_* modules of the AASIST source); graph-pooling nodes
("/pool") are reported separately, not counted as graph attention.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
from deploybench.manifests import read_ids  # noqa: E402
from deploybench.metrics import eer  # noqa: E402

CS = R / "results" / "phase_2_conversion_app" / "conversion_sensitivity"
RANKED = ("aasist", "aasist_l", "rawnet2", "tssdnet")
GA_OPS = {"ScatterND", "TopK", "Gather", "GatherElements"}      # D30's named GAT ops
PREDICTED_FALLBACK = {"aasist": "graph attention", "aasist_l": "graph attention",
                      "rawnet2": "recurrence (GRU)", "tssdnet": "none"}


def module(node: str) -> str:
    return "graph_attention" if "GAT" in node else "graph_pooling" if "/pool" in node else "other"


def scores(p: Path, ids):
    rows = {r["utt_id"]: (int(r["label"]), float(r["score"])) for r in csv.DictReader(open(p))}
    return np.array([rows[u][0] for u in ids]), np.array([rows[u][1] for u in ids])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(R / "results" / "analysis" / "h6.json"))
    a = ap.parse_args()
    diag = read_ids(R / "configs" / "conversion_sensitivity" / "diag_seed0.txt")
    out = {}
    for m in RANKED:
        plan = json.loads((CS / f"search_plan_{m}.json").read_text())
        sel = json.loads((CS / f"selection_{m}.json").read_text())
        side = json.loads((CS / m / "r01" / f"{m}_int8_ptq_log.json").read_text())
        taus = json.loads((R / "results" / "phase_1_reproduction" / f"tau_{m}.json").read_text())["taus"]
        ops = {r["node"]: r["op_type"] for r in plan["ranking"]}
        y, s32 = scores(R / "results" / "phase_1_reproduction" / f"scores_dev_{m}.csv", diag)

        # primary DV
        locked = sel["locked_recipe"]
        if locked is None:
            primary = {"minimal_set": None, "evaluable": False,
                       "best_attempt": min((x for x in sel["attempts"] if x.get("delta_eer_pp") is not None),
                                           key=lambda x: x["delta_eer_pp"])}
            prediction = "not evaluable: no tested configuration meets the budget"
        else:
            t = next(x for x in plan["attempts"] if x["recipe_id"] == locked)
            ex = [{"node": n, "op_type": ops[n], "module": module(n)} for n in t["exclude_nodes"]]
            primary = {"minimal_set": ex, "K": len(ex), "recipe": locked, "evaluable": True}
            if m == "rawnet2":
                has = any(e["op_type"] == "GRU" for e in ex)
                gru_ranked = any(r["op_type"] == "GRU" for r in plan["ranking"])
                prediction = ("met" if has else "NOT met: the set contains no GRU node"
                              + ("" if gru_ranked else "; ORT never quantized the GRU, so no "
                                 "exclusion set could contain one"))
            else:
                prediction = "met" if any(e["module"] == "graph_attention" for e in ex) else "NOT met"

        # secondary DV: converter fallback list of the default artifact (r01)
        fb, q = set(side["fp32_fallback_op_types"]), set(side["quantized_op_types"])
        if PREDICTED_FALLBACK[m] == "graph attention":
            sec = ("NOT met: " + ", ".join(sorted(GA_OPS & q)) + " were quantized"
                   if not (GA_OPS & fb) else "met")
        elif PREDICTED_FALLBACK[m] == "recurrence (GRU)":
            sec = "met" if "GRU" in fb else "NOT met"
        else:
            sec = "met" if not ({"Conv", "BatchNormalization", "Relu"} & fb) else "NOT met"

        # secondary: decision flips and dEER on the diagnostic subset, per tested attempt
        cells = []
        for x in sel["attempts"]:
            p = CS / m / x["recipe_id"] / "scores_diag.csv"
            if not p.is_file():
                cells.append({"recipe_id": x["recipe_id"], "outcome": x["outcome"]})
                continue
            _, sq = scores(p, diag)
            cells.append({"recipe_id": x["recipe_id"], "K": x["n_fp32_excluded"],
                          "delta_eer_pp": round((eer(sq, y)[0] - eer(s32, y)[0]) * 100, 3),
                          "flips": {k: int(((sq >= t) != (s32 >= t)).sum()) for k, t in taus.items()}})
        out[m] = {"primary": primary, "prediction_primary": prediction,
                  "r01_fallback_ops": sorted(fb), "r01_quantized_ops": sorted(q),
                  "predicted_fallback": PREDICTED_FALLBACK[m], "prediction_secondary": sec,
                  "diag_cells": cells}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for m, r in out.items():
        pr = r["primary"]
        head = (f"K={pr['K']} ({pr['recipe']})" if pr["evaluable"] else
                f"no qualifying set (best {pr['best_attempt']['recipe_id']} K="
                f"{pr['best_attempt']['n_fp32_excluded']}, {pr['best_attempt']['delta_eer_pp']:+.2f} pp)")
        print(f"{m:9s} primary: {head}\n          prediction (D36): {r['prediction_primary']}\n"
              f"          r01 fallback: {', '.join(r['r01_fallback_ops'])}\n"
              f"          prediction (D30 secondary, fallback = {r['predicted_fallback']}): {r['prediction_secondary']}")


if __name__ == "__main__":
    main()
