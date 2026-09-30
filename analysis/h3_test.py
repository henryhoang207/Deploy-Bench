"""H3, the confirmatory test, under DEVIATIONS.md DEV-3 (committed before RawNet2's int8 dev scores existed).

    python analysis/h3_test.py [--out results/analysis/h3_test.json]

H3: "QTS > 0.01 (1 HTER point) for at least half of (ranked model x int8) cells in-domain; per-cell
bootstrap CI excludes 0". In-domain = 19LA eval at 19LA-dev taus (D37). DEV-3: the denominator is
the 4 ranked models; a model with no registered (D39) int8 artifact is a cell that does not meet it.
QTS = HTER(s_q, tau_fp32) - HTER(s_q, tau_q) on 19LA eval at the EER point, where tau_q is the EER
tau of the int8 artifact's own full 19LA-dev scores (metrics.qts, tau.calibrate_tau).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
from deploybench.artifacts import report_label  # noqa: E402
from deploybench.metrics import hter_at, qts  # noqa: E402
from deploybench.tau import calibrate_tau  # noqa: E402

CS = R / "results" / "phase_2_conversion_app" / "conversion_sensitivity"
RANKED = ("aasist", "aasist_l", "rawnet2", "tssdnet")
B, SEED, THRESH = 1000, 0, 0.01


def load(p, model, dataset, n):
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    bad = {(r["model"], r["dataset"], r["precision"]) for r in rows} - {(model, dataset, "int8")}
    if bad or len(rows) != n or len({r["utt_id"] for r in rows}) != n:
        raise SystemExit(f"{p}: expected {n} unique {model}/{dataset}/int8 rows ({bad}, {len(rows)})")
    return np.array([float(r["score"]) for r in rows]), np.array([int(r["label"]) for r in rows])


def boot_qts(s, y, t32, tq, n_boot=B, seed=SEED):
    ib, isp = np.where(y == 1)[0], np.where(y == 0)[0]
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([ib[rng.integers(0, len(ib), len(ib))],
                              isp[rng.integers(0, len(isp), len(isp))]])
        out[i] = hter_at(t32, s[idx], y[idx]) - hter_at(tq, s[idx], y[idx])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(R / "results" / "analysis" / "h3_test.json"))
    a = ap.parse_args()
    cells = {}
    for m in RANKED:
        can = CS / f"canonical_{m}.json"
        c = json.loads(can.read_text()) if can.is_file() else {}
        if c.get("canonical_seed") is None:
            cells[m] = {"artifact": None, "meets": False,
                        "reason": "no registered int8 artifact (D39); counts as not meeting (DEV-3)"}
            continue
        run = CS / m / f"{c['locked_recipe']}_seed{c['canonical_seed']}"
        label = report_label("int8", run / f"{m}_int8_ptq_log.json")
        sd, yd = load(run / "scores_19la_dev.csv", m, "asvspoof19_dev", 24844)
        se, ye = load(run / "scores_19la_eval.csv", m, "asvspoof19_eval", 71237)
        t32 = json.loads((R / "results" / "phase_1_reproduction" / f"tau_{m}.json").read_text())["taus"]["eer"]
        tq = calibrate_tau(sd, yd)["eer"]
        r = qts(se, ye, t32, tq)
        d = boot_qts(se, ye, t32, tq)
        lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
        cells[m] = {"artifact": str(run.relative_to(R)), "label": label, "tau_fp32": t32, "tau_q": tq,
                    **{k: float(v) for k, v in r.items()}, "ci95": [lo, hi],
                    "meets": bool(r["qts"] > THRESH and lo > 0)}
    n = sum(c["meets"] for c in cells.values())
    out = {"hypothesis": "H3", "spec": "DEVIATIONS.md DEV-3", "B": B, "seed": SEED,
           "threshold": THRESH, "cells": cells, "n_meeting": n, "required": 2,
           "h3_supported": n >= 2}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for m, c in cells.items():
        if c["artifact"] is None:
            print(f"{m:9s} {c['reason']}")
        else:
            print(f"{m:9s} {c['label']}: QTS {c['qts']*100:+.3f} HTER pts  CI [{c['ci95'][0]*100:+.3f}, "
                  f"{c['ci95'][1]*100:+.3f}]  (HTER {c['hter_tau_fp32']*100:.3f}% at tau_fp32 vs "
                  f"{c['hter_tau_quant']*100:.3f}% at tau_q; tau shift {c['tau_shift']:+.4f})  meets: {c['meets']}")
    print(f"\n{n} of 4 cells meet QTS > 0.01 with CI > 0; at least half (2) required -> "
          f"H3 {'SUPPORTED' if out['h3_supported'] else 'NOT SUPPORTED'}")


if __name__ == "__main__":
    main()
