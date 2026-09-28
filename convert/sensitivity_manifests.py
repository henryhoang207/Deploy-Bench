"""D39 manifests for the conversion-sensitivity study (PREREGISTRATION, D35/D39).

All drawn from ASVspoof19 LA dev with deploybench.manifests.draw_balanced:

  diag_seed0.txt           2,000 clips (1,000 + 1,000), seed 0. Recipe selection
                           and the D36 budget. Never used for tau or calibration.
  calib_search_seed100.txt   500 clips (250 + 250) from the POOL (dev minus diag),
                           seed 100. Used only by the recipe search, so the
                           calibration-repeat seeds stay separate from search
                           (step_02a checklist).
  calib_seed{1..5}.txt       500 clips each from the POOL, seeds 1-5. Drawn
                           independently, so they may overlap; index.json
                           records the overlap.

index.json records every manifest's ids sha256, the pool, the overlap matrix and
where the ids came from. The files are frozen once committed: this script refuses
to overwrite them. `--verify` redraws in memory and fails loudly on any difference,
which is how the committed files are re-checked against the real protocol on sol1.

    python convert/sensitivity_manifests.py                  # protocol, needs DATA_ROOT
    python convert/sensitivity_manifests.py --pool-csv results/phase_1_reproduction/scores_dev_aasist.csv
    python convert/sensitivity_manifests.py --verify         # on sol1, against the protocol
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from deploybench.manifests import (draw_balanced, ids_digest, read_ids,  # noqa: E402
                                   write_manifest)

OUT_DIR = REPO / "configs" / "conversion_sensitivity"
DATASET = "asvspoof19_dev"
DEV_N = (2548, 22296)                 # bona fide / spoof in the 19LA dev protocol
DIAG = ("diag_seed0.txt", 1000, 0)
SEARCH = ("calib_search_seed100.txt", 250, 100)
REPEATS = tuple((f"calib_seed{s}.txt", 250, s) for s in (1, 2, 3, 4, 5))


def pool_from_protocol():
    from deploybench.data import protocol_iter
    return [(u, lab) for u, _p, lab in protocol_iter(DATASET)], f"protocol:{DATASET}"


def pool_from_csv(path: Path):
    rows = [(r["utt_id"], int(r["label"])) for r in csv.DictReader(open(path))]
    return rows, f"csv:{Path(path).as_posix()}"


def build(rows):
    """Pure: (utt_id, label) rows -> {filename: (ids, n_per_class, seed)} + pool info."""
    ids = [u for u, _ in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate utt_ids in the dev pool")
    bona = [u for u, lab in rows if lab == 1]
    spoof = [u for u, lab in rows if lab == 0]
    if len(bona) + len(spoof) != len(rows):
        raise ValueError("dev pool has labels other than 0/1")
    if (len(bona), len(spoof)) != DEV_N:
        raise ValueError(f"dev pool is {len(bona)} bona fide / {len(spoof)} spoof; "
                         f"ASVspoof19 LA dev is {DEV_N[0]} / {DEV_N[1]}")
    out = {}
    name, n, seed = DIAG
    diag = draw_balanced(bona, spoof, n, seed)
    out[name] = (diag, n, seed)
    held = set(diag)
    pb = [u for u in bona if u not in held]
    ps = [u for u in spoof if u not in held]
    for name, n, seed in (SEARCH,) + REPEATS:
        out[name] = (draw_balanced(pb, ps, n, seed), n, seed)
    for name, (m, _n, _s) in out.items():
        if name != DIAG[0] and held & set(m):
            raise AssertionError(f"{name} overlaps the diagnostic subset")
    pool_digest = hashlib.sha256("\n".join(
        f"{u},{lab}" for u, lab in sorted(rows)).encode()).hexdigest()
    info = {"dev_bona": len(bona), "dev_spoof": len(spoof),
            "pool_bona": len(pb), "pool_spoof": len(ps),
            "dev_rows_sha256": pool_digest}
    return out, info


def overlap(manifests):
    names = [SEARCH[0]] + [r[0] for r in REPEATS]
    sets = {k: set(manifests[k][0]) for k in names}
    return {a: {b: len(sets[a] & sets[b]) for b in names} for a in names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-csv", default=None,
                    help="a 19LA-dev score CSV (utt_id,label,...) instead of the protocol")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--verify", action="store_true",
                    help="redraw and compare with the committed files; write nothing")
    a = ap.parse_args()

    rows, source = pool_from_csv(Path(a.pool_csv)) if a.pool_csv else pool_from_protocol()
    manifests, info = build(rows)
    out_dir = Path(a.out_dir)

    if a.verify:
        index = json.loads((out_dir / "index.json").read_text(encoding="utf-8"))
        bad = []
        for name, (ids, _n, _s) in manifests.items():
            if read_ids(out_dir / name) != ids:
                bad.append(f"{name}: file ids differ from the redraw")
            if index["manifests"][name]["ids_sha256"] != ids_digest(ids):
                bad.append(f"{name}: index.json sha256 differs from the redraw")
        if index["pool"]["dev_rows_sha256"] != info["dev_rows_sha256"]:
            bad.append(f"dev (utt_id,label) set differs from the one the files were "
                       f"drawn from ({index['source']})")
        if bad:
            raise SystemExit("VERIFY FAILED against " + source + ":\n  " + "\n  ".join(bad))
        print(f"verified {len(manifests)} manifests + pool against {source}")
        return

    clash = [n for n in list(manifests) + ["index.json"] if (out_dir / n).exists()]
    if clash:
        raise SystemExit(f"refusing to overwrite frozen D39 manifests in {out_dir}: "
                         f"{', '.join(clash)} (use --verify to re-check them)")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    index = {"design": "PREREGISTRATION.md, conversion sensitivity (D35/D39)",
             "generated": now, "source": source, "dataset": DATASET,
             "pool": info, "manifests": {}, "calibration_overlap": overlap(manifests)}
    for name, (ids, n, seed) in manifests.items():
        role = ("diagnostic" if name == DIAG[0] else
                "search calibration" if name == SEARCH[0] else "calibration repeat")
        drawn_from = ("19LA dev" if name == DIAG[0] else
                      f"19LA dev minus {DIAG[0]}")
        header = [f"# FROZEN D39 {role} manifest. Do not regenerate once used.",
                  f"# generated : {now}",
                  f"# dataset   : {DATASET} ({source})",
                  f"# drawn from: {drawn_from}",
                  f"# seed      : {seed}",
                  f"# n_bona    : {n}   (the first {n} ids)",
                  f"# n_spoof   : {n}"]
        digest = write_manifest(out_dir / name, ids, header)
        index["manifests"][name] = {"role": role, "seed": seed, "n_bona": n,
                                    "n_spoof": n, "ids_sha256": digest}
        print(f"{name:26s} {2 * n:5d} ids  sha256 {digest[:16]}")
    (out_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n",
                                        encoding="utf-8")
    print(f"wrote index.json -> {out_dir}")


if __name__ == "__main__":
    main()
