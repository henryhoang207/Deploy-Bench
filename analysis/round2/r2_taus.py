"""Round-2 thresholds (PREREGISTRATION_ROUND2.md, R2-P): the ONLY place round-2 taus are computed, through
deploybench.tau (calibrate_tau for eer / frr1 / frr5, tau_at_far for far1 / far5), from the model's own source-dev
scores, then frozen: an existing file is never overwritten.

    python analysis/round2/r2_taus.py --model xlsr_mamba --seed 0
    python analysis/round2/r2_taus.py --model aasist_l_r2 --seed 3

Source dev (D16): la19_dev for every cohort19 model; as5_dev for as5_aasist. For aasist_l_trim the dev scores were
produced through the vad35 trim (score.py applies the model's preproc), so the trimmed dev is its source dev.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r2lib as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from deploybench.tau import calibrate_tau, tau_at_far
    from deploybench.metrics import eer
    from models.base import SPECS
    spec = SPECS[a.model]
    if spec.round != 2:
        raise SystemExit(f"{a.model} is not a round-2 model; round-1 taus are frozen in results/phase_1_reproduction")
    surface = "as5_dev" if spec.cohort == "cohort_as5" else "la19_dev"
    out = R.R2_TAUS / f"{a.model}__s{a.seed}.json"
    if out.exists():
        raise SystemExit(f"{out} exists: round-2 taus are frozen once computed")
    p = R._open(R.score_path(a.model, surface, a.seed))
    ids, y, s = R.load(a.model, surface, a.seed)
    t = calibrate_tau(s, y)
    t["far1"] = tau_at_far(s, y, 0.01)
    t["far5"] = tau_at_far(s, y, 0.05)
    rec = {"model": a.model, "seed": a.seed, "source": surface, "dev_scores_file": str(p.relative_to(R.REPO)),
           "dev_scores_sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "n_bona": int((y == 1).sum()),
           "n_spoof": int((y == 0).sum()), "dev_eer": float(eer(s, y)[0]),
           "taus": {k: float(v) for k, v in t.items()}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
