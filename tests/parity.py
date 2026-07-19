"""Parity suite (phase_2 step_03): converted artifact vs PyTorch fp32 reference.
Decision-grounded gate (methodology C3/P3): ZERO decision flips at all three
operating points on the 1,000-clip manifest; score deltas reported descriptively.

    python tests/parity.py --ref ref_scores.csv --test test_scores.csv --taus taus.json
"""
import argparse
import json
import numpy as np
import pandas as pd


def compare(ref: pd.DataFrame, test: pd.DataFrame, taus: dict) -> dict:
    m = ref.merge(test, on="utt_id", suffixes=("_ref", "_test"))
    d = (m.score_test - m.score_ref).abs()
    out = {"n": len(m), "max_abs_delta": float(d.max()),
           "mean_abs_delta": float(d.mean()),
           "pearson_r": float(np.corrcoef(m.score_ref, m.score_test)[0, 1]),
           "decision_flips": {}}
    for name, tau in taus.items():
        flips = int(((m.score_ref >= tau) != (m.score_test >= tau)).sum())
        out["decision_flips"][name] = flips
    out["pass"] = all(v == 0 for v in out["decision_flips"].values())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--taus", required=True, help="json {'eer':t,'frr1':t,'frr5':t}")
    a = ap.parse_args()
    r = compare(pd.read_csv(a.ref), pd.read_csv(a.test),
                json.loads(open(a.taus).read()))
    print(json.dumps(r, indent=2))
    raise SystemExit(0 if r["pass"] else 1)
