"""W2 review check for the AASIST-L runs (PREREGISTRATION_ROUND2.md Sec. 4, DEV-9 amendment 1). Read-only.

For every run directory r2_runs/aasist_l_{r2,trim}_s<k> that has finished (run_final.json):
  - run_meta.json: seed and trim arm match the directory name; not a smoke run;
  - epochs.jsonl: epochs 0..99 in order, every loss and dev EER finite;
  - weights/: swa.pth and last.pth present with the SHA-256 recorded in run_final.json; the number of epoch_*.pth snapshots
    equals n_swa_update and each snapshot's dev EER is <= 1 and non-increasing (the recipe's rule); no best.pth;
  - nothing evaluation-related anywhere in the directory (no file name containing "eval").
Unfinished runs are listed with their last epoch. Prints RUN_REVIEW_PASS / RUN_REVIEW_FAIL.

    python3 scripts/sol1/round2/r2_run_review.py [/mnt/data8tb/henry/r2_runs]
"""
import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/data8tb/henry/r2_runs")
fails, rows = [], []


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


for d in sorted(ROOT.glob("aasist_l_*_s*")):
    m = re.fullmatch(r"aasist_l_(r2|trim)_s(\d+)", d.name)
    if not m:
        continue
    arm, seed = m.group(1), int(m.group(2))
    ep = [json.loads(x) for x in (d / "epochs.jsonl").read_text().splitlines() if x.strip()] \
        if (d / "epochs.jsonl").exists() else []
    if not (d / "run_final.json").exists():
        rows.append(f"{d.name:20s} running: {len(ep)} epochs, last dev EER {ep[-1]['dev_eer']:.3f}%" if ep
                    else f"{d.name:20s} running: no epoch yet")
        continue
    meta = json.loads((d / "run_meta.json").read_text())
    fin = json.loads((d / "run_final.json").read_text())
    bad = []
    if meta["seed"] != seed or meta["trim"] != ("vad35" if arm == "trim" else "none") or meta.get("smoke"):
        bad.append(f"meta {meta['seed']}/{meta['trim']}/smoke={meta.get('smoke')}")
    if [e["epoch"] for e in ep] != list(range(100)):
        bad.append(f"{len(ep)} epochs, not 0..99")
    if not all(math.isfinite(e["loss"]) and math.isfinite(e.get("dev_eer", float("nan"))) for e in ep):
        bad.append("non-finite loss or dev EER")
    w = d / "weights"
    snaps = sorted(w.glob("epoch_*.pth"), key=lambda p: int(p.name.split("_")[1]))
    if len(snaps) != fin["n_swa_update"]:
        bad.append(f"{len(snaps)} snapshots vs n_swa_update {fin['n_swa_update']}")
    eers = [float(p.stem.split("_")[2]) for p in snaps]
    if any(x > 1.0 for x in eers) or any(b > a for a, b in zip(eers, eers[1:])):
        bad.append(f"snapshot dev EERs break the recipe rule: {eers}")
    for f in ("swa", "last"):
        if not (w / f"{f}.pth").exists() or sha(w / f"{f}.pth") != fin[f"{f}_sha256"]:
            bad.append(f"{f}.pth missing or SHA-256 differs")
    if (w / "best.pth").exists():
        bad.append("best.pth exists")
    ev = [p.name for p in d.rglob("*") if "eval" in p.name.lower()]
    if ev:
        bad.append(f"eval-named files: {ev[:3]}")
    best = min(e["dev_eer"] for e in ep) if ep else float("nan")
    rows.append(f"{d.name:20s} {'OK  ' if not bad else 'FAIL'} epochs {len(ep)}  swa snapshots {fin['n_swa_update']:3d}  "
                f"best dev EER {best:.3f}%  last {ep[-1]['dev_eer']:.3f}%" + (f"  <- {'; '.join(bad)}" if bad else ""))
    if bad:
        fails.append(d.name)

print("\n".join(rows))
print("RUN_REVIEW_" + ("PASS" if not fails else "FAIL: " + ", ".join(fails)))
