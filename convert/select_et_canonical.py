"""DEV-7 canonical ExecuTorch int8 artifact: the lowest calibration seed (1-5) whose
diagnostic-subset EER is within +1.0 pp (D36 budget) of the same model's ExecuTorch
fp32 EER on that subset (the D39 rule). Seeds that failed to convert or score are
reported, never replaced. No qualifying seed -> no ExecuTorch int8 artifact.

    python convert/select_et_canonical.py --model tssdnet --dir results/.../executorch/tssdnet
reads   <dir>/diag_fp32_executorch.csv, <dir>/int8_seed{1..5}/diag_int8_executorch.csv
writes  <dir>/canonical_int8.json (refuses to overwrite)
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deploybench.metrics import eer  # noqa: E402

BUDGET_PP = 1.0
SEEDS = (1, 2, 3, 4, 5)


def diag_eer(p: Path):
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    if len(rows) != 2000:
        raise SystemExit(f"{p}: {len(rows)} rows, the diagnostic subset has 2,000")
    y = np.array([int(r["label"]) for r in rows])
    s = np.array([float(r["score"]) for r in rows])
    return float(eer(s, y)[0]) * 100


def select(d: Path) -> dict:
    ref = diag_eer(d / "diag_fp32_executorch.csv")
    seeds, canonical = {}, None
    for k in SEEDS:
        p = d / f"int8_seed{k}" / "diag_int8_executorch.csv"
        if not p.is_file():
            seeds[k] = {"status": "missing (failed conversion or scoring)"}
            continue
        e = diag_eer(p)
        ok = e - ref <= BUDGET_PP
        seeds[k] = {"diag_eer_pct": round(e, 4), "deer_pp": round(e - ref, 4), "within_budget": ok}
        if ok and canonical is None:
            canonical = k
    return {"fp32_diag_eer_pct": round(ref, 4), "budget_pp": BUDGET_PP,
            "seeds": seeds, "canonical_seed": canonical}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    d = Path(a.dir)
    out = d / "canonical_int8.json"
    if out.exists():
        raise SystemExit(f"REFUSING to overwrite {out}")
    r = {"model": a.model, "rule": "DEV-7: lowest seed within +1.0 pp of ExecuTorch fp32 diag EER"}
    r.update(select(d))
    out.write_text(json.dumps(r, indent=2))
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
