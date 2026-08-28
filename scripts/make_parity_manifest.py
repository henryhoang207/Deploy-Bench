"""Build the FIXED parity manifest (phase_2 step 2.3): 500 bona fide + 500
spoof utt_ids, deterministic given --seed, checked into the repo and never
regenerated afterwards. The reference (PyTorch fp32) and every converted
artifact score exactly these clips, in this order.

The same generator produces the PTQ calibration pool: D27 fixes a 500-clip
BALANCED calibration set, and calibration clips must come from source DEV,
never from eval.

    python scripts/make_parity_manifest.py \
        --dataset asvspoof19_eval --n-per-class 500 \
        --out configs/parity_manifest_tssdnet.txt

    python scripts/make_parity_manifest.py \
        --dataset asvspoof19_dev --n-per-class 250 \
        --out configs/calibration_500.txt
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="asvspoof19_eval")
    ap.add_argument("--n-per-class", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260725)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from deploybench.data import protocol_iter, DATA_ROOT

    bona, spoof = [], []
    for uid, _path, lab in protocol_iter(a.dataset):
        (bona if lab == 1 else spoof).append(uid)
    if len(bona) < a.n_per_class or len(spoof) < a.n_per_class:
        raise SystemExit(
            f"{a.dataset} has {len(bona)} bona fide / {len(spoof)} spoof — "
            f"cannot draw {a.n_per_class} of each")

    rng = np.random.default_rng(a.seed)
    # sort first so the draw does not depend on protocol-file line order
    pick_b = rng.choice(sorted(bona), a.n_per_class, replace=False)
    pick_s = rng.choice(sorted(spoof), a.n_per_class, replace=False)
    ids = sorted(pick_b.tolist()) + sorted(pick_s.tolist())

    digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    header = [
        "# FIXED parity/calibration manifest — do not regenerate once used.",
        f"# generated : {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"# dataset   : {a.dataset}",
        f"# data_root : {DATA_ROOT}",
        f"# seed      : {a.seed}",
        f"# n_bona    : {a.n_per_class}   (the first {a.n_per_class} ids)",
        f"# n_spoof   : {a.n_per_class}",
        f"# pool      : {len(bona)} bona fide / {len(spoof)} spoof",
        f"# sha256    : {digest}",
    ]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(header + ids) + "\n", encoding="utf-8")
    print(f"wrote {len(ids)} utt_ids -> {out}")
    print(f"sha256(ids) = {digest}")


if __name__ == "__main__":
    main()
