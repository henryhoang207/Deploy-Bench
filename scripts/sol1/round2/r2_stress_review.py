"""W3 review check (run after r2_stress.sh): the round-2 stress set and its round-1 scores.

1. manifest: 9 conditions x 1,000 clips, every path exists, ids equal the frozen H5 manifest;
2. audio: every stressed file decodes at 16 kHz mono with a length within 1 s of its clean source;
3. scores: each round-1 model's stress_clean scores equal its committed H5 clean scores (same audio bytes), max |d|;
4. the IR pool file exists and lists 23 AIR phone responses.
    python -B scripts/sol1/round2/r2_stress_review.py      (from the clone, envs/db310, DATA_ROOT set)
Prints STRESS_REVIEW_PASS or STRESS_REVIEW_FAIL with the reasons.
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, ".")
from deploybench.manifests import read_ids  # noqa: E402

DATA = Path(os.environ["DATA_ROOT"])
SET = DATA / "stress" / "r2_parity"
CONDS = ["clean", "opus6k", "opus24k", "amr4.75", "mp3_32", "snr20", "snr10", "snr5", "mic_ir"]
fails = []

ids = read_ids(Path("configs/parity_manifest_tssdnet.txt"))
rows = list(csv.DictReader(open(SET / "manifest.csv", newline="", encoding="utf-8")))
by = {}
for r in rows:
    by.setdefault(r["condition"], []).append(r)
for c in CONDS:
    got = [r["utt_id"] for r in by.get(c, [])]
    if sorted(got) != sorted(ids):
        fails.append(f"manifest {c}: {len(got)} rows, ids differ from the H5 manifest")
print(f"manifest: {len(rows)} rows over {len(by)} conditions")

clean_len = {r["utt_id"]: sf.info(r["path"]).frames for r in by["clean"]}
for c in CONDS[1:]:
    bad = 0
    for r in by[c]:
        p = Path(r["path"])
        if not p.is_file():
            bad += 1
            continue
        info = sf.info(str(p))
        if info.samplerate != 16000 or info.channels != 1 or abs(info.frames - clean_len[r["utt_id"]]) > 16000:
            bad += 1
    if bad:
        fails.append(f"audio {c}: {bad} files missing, not 16 kHz mono, or off by more than 1 s")
    print(f"audio {c}: {len(by[c])} files, {bad} problems")

for m in ("aasist", "aasist_l", "rawnet2", "tssdnet"):
    a = pd.read_csv(f"results/round2/scores/{m}/s0/stress_clean.csv", dtype={"utt_id": str}).set_index("utt_id")
    b = pd.read_csv(f"results/phase_3_stress/h5/scores_{m}_clean.csv", dtype={"utt_id": str}).set_index("utt_id")
    sa, sb = a["score"].reindex(b.index).to_numpy(), b["score"].to_numpy()
    d = np.abs(sa - sb)
    import json
    tau = json.loads(Path(f"results/phase_1_reproduction/tau_{m}.json").read_text())["taus"]["eer"]
    flips = int(np.sum((sa < tau) != (sb < tau)))
    print(f"{m}: stress_clean (GPU) vs H5 clean (CPU) max|d| = {d.max():.3e}, decision flips at tau_eer = {flips}")
    if not np.all(np.isfinite(d)) or d.max() > 1e-3 or flips:
        fails.append(f"{m}: stress_clean differs from H5 clean (max {d.max():.3e}, flips {flips})")

pool = SET / "IR_POOL.txt"
n_ir = len(pool.read_text().splitlines()) if pool.exists() else 0
if n_ir != 23:
    fails.append(f"IR pool lists {n_ir} files, expected 23")
print(f"IR pool: {n_ir} files")
print("STRESS_REVIEW_" + ("PASS" if not fails else "FAIL: " + "; ".join(fails)))
