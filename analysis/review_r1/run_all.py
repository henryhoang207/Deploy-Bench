"""Run the analyses prompted by the round-1 review (DEVIATIONS.md, DEV-10) in order, then print their numbers:

    python analysis/review_r1/run_all.py     # -> results/review_r1/*.json and rv_checks.txt (about 50 min)

Exploratory: none of them is a registered analysis. They read only committed score files and frozen thresholds. Each
JSON output records its resampling sizes and RNG seeds under "meta". Part F (F1-F3) was specified in DEV-10 and
deposited on OSF before it first ran.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ("run_cv.py",           # cv.json: in-sample vs cross-validated oracles
         "run_dev.py",          # dev.json: tie interval, dev bootstrap, smoothed dev thresholds
         "run_ladder.py",       # ladder.json: threshold policies after the shift (reads cv.json)
         "run_smalllabel.py",   # smalllabel.json: small labelled audit
         "run_intervals.py",    # intervals.json: percentile, BCa and m-out-of-n intervals
         "run_h2cv_fast.py",    # h2cv.json: R2-H2 with a cross-validated oracle
         "run_h2cv_folds.py",   # h2cv_folds.json: F1, the same over 10 fold assignments per run
         "run_mofn.py",         # mofn.json: F2, the m-out-of-n interval for three rules for m
         "run_atc.py",          # atc.json: F3, label-free accuracy estimation (reads ladder.json)
         "rv_checks.py")        # rv_checks.txt: every number in the form the paper quotes


def main():
    t0 = time.time()
    for step in STEPS:
        print(f"== {step}", flush=True)
        if subprocess.run([sys.executable, str(HERE / step)]).returncode:
            raise SystemExit(f"REFUSED: {step} failed; later steps not run")
    print(f"all steps done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
