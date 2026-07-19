"""Merge all score CSVs into the single source-of-truth results database.

    python analysis/build_database.py --scores results/scores/ --out results/database.parquet

Adds nothing, computes nothing — provenance-only merge with schema validation.
"""
import argparse
import hashlib
from pathlib import Path
import pandas as pd

SCHEMA = ["utt_id", "label", "score", "model", "dataset", "precision",
          "runtime", "condition", "seed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    frames = []
    for f in sorted(Path(args.scores).glob("**/*.csv")):
        df = pd.read_csv(f)
        missing = set(SCHEMA) - set(df.columns)
        if missing:
            raise SystemExit(f"{f}: missing columns {missing}")
        df["source_file"] = f.name
        frames.append(df)
    if not frames:
        raise SystemExit("no score files found")
    db = pd.concat(frames, ignore_index=True)
    dup = db.duplicated(subset=["utt_id", "model", "dataset", "precision",
                                "runtime", "condition", "seed"])
    if dup.any():
        raise SystemExit(f"{dup.sum()} duplicate cells — refusing to build")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    db.to_parquet(out)
    h = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
    print(f"{len(db)} rows, {db['model'].nunique()} models -> {out}\nsha256[:16]={h}"
          "\nRecord this hash at grid freeze (phase_3 step_05).")


if __name__ == "__main__":
    main()
