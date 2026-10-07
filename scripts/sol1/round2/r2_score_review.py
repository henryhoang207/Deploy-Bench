"""Review check for the round-2 score files (the verify step after each scoring milestone of the W1-W3 loop).
Read-only. For every registered score file that exists (r2_score_sched.py, r2_score_detectors.sh, r2_w1_finish.sh):

  - columns exactly utt_id,label,score,model,dataset,precision,runtime,condition,seed; precision fp32; labels {0, 1}
    with both present; finite, non-constant scores; no duplicate ids; the model and seed columns match the path;
  - rows: the registered surface size (1,000 for a stress condition);
  - the (utt_id, label) pairs equal those of every other file of the same surface (stress: all models, round 1 too);
  - the scoring log's FLAC report: DF21 needs MD5-verified + verified-fallback = 455,508 and none unverifiable (DEV-9
    amendment 3); for the other FLAC surfaces the counts are printed (LA19 runs before amendment 3 have the old
    one-line report; LA19 was checked unaffected);
  - a tau file for every source-dev file (la19_dev; as5_dev for as5_aasist).
Prints one line per file (rows, EER %, FLAC counts), the registered files still missing per unit, and
SCORE_REVIEW_PASS or SCORE_REVIEW_FAIL with the reasons. Optional arguments restrict the full check to some units
(model or model/s<seed>); the consistency references are still read.

    cd /mnt/data8tb/henry/deploybench && DATA_ROOT=/mnt/data8tb/henry/data/deepfake \
      ../envs/db310/bin/python -B ../xfer/r2_score_review.py [as5_aasist aasist_l_r2/s1 ...]
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from deploybench.metrics import eer  # noqa: E402

H = Path("/mnt/data8tb/henry")
SC, TAUS = Path("results/round2/scores"), Path("results/round2/taus")
COLS = ["utt_id", "label", "score", "model", "dataset", "precision", "runtime", "condition", "seed"]
N = {"la19_dev": 24844, "la19_eval": 71237, "itw": 31779, "df21_verified": 455508, "mlaad5": 494853,
     "as5_dev": 140950, "as5_eval": 680774}
FLAC = {"la19_dev", "la19_eval", "df21_verified", "as5_dev", "as5_eval"}
CONDS = ["clean", "opus6k", "opus24k", "amr4.75", "mp3_32", "snr20", "snr10", "snr5", "mic_ir"]
STRESS = [f"stress_{c}" for c in CONDS + ["encodec6k"]]


def registered():
    r = [("as5_aasist", 0, s) for s in ["as5_dev", "as5_eval", "la19_eval", "itw", "df21_verified", "mlaad5"] + STRESS]
    for m in ("xlsr_mamba", "pascu_lr"):
        r += [(m, 0, s) for s in ["la19_dev", "la19_eval", "itw", "df21_verified", "mlaad5"] + STRESS]
    r.append(("pascu_lr_gate", 0, "la19_eval"))
    for arm, seeds in (("r2", (1, 2, 3, 4, 5, 1234)), ("trim", (1, 2, 3, 4, 5))):
        for k in seeds:
            r += [(f"aasist_l_{arm}", k, s) for s in ["la19_dev", "la19_eval", "itw"] + STRESS]
            r += [(f"aasist_l_{arm}_last", k, s) for s in ["la19_dev", "la19_eval", "itw"]]
    for m in ("aasist", "aasist_l", "rawnet2", "tssdnet", "xlsr_aasist"):   # round 1 on the round-2 stress set
        r += [(m, 0, s) for s in STRESS[:-1]]
    return r


def log_for(m, k, s):
    """The scoring log of one file: scheduler task log, W1-finish re-score log, or queue log (in that order)."""
    cands = [H / "logs/r2_score" / f"{m}_s{k}_{s}.log"]
    if (m, s) == ("xlsr_mamba", "df21_verified"):
        cands.append(H / "logs/r2_xlsr_mamba_df21_rescore.log")   # the queue-A log belongs to the quarantined run
    else:
        cands.append(H / f"logs/r2_{m}_{s}.log")
    for c in cands:
        if c.exists():
            return c
    return None


def flac_counts(log):
    t = log.read_text(errors="replace") if log else ""
    fb = re.findall(r"fallback \(D34\): (\d+)", t)
    ver = re.findall(r"MD5-verified: (\d+); unverifiable \(MD5 unset or not 16-bit mono\): (\d+)", t)
    return (int(fb[-1]) if fb else None, int(ver[-1][0]) if ver else None, int(ver[-1][1]) if ver else None)


def main():
    only = sys.argv[1:]
    sel = lambda m, k: not only or m in only or f"{m}/s{k}" in only  # noqa: E731
    fails, lines, ref = [], [], {}
    reg = registered()
    present = [(m, k, s) for m, k, s in reg if (SC / m / f"s{k}" / f"{s}.csv").exists()]
    # one consistency reference per surface: the first present file in registry order
    for m, k, s in present:
        key = "stress" if s.startswith("stress_") else s
        if key not in ref:
            d = pd.read_csv(SC / m / f"s{k}" / f"{s}.csv", usecols=["utt_id", "label"], dtype={"utt_id": str})
            ref[key] = (f"{m}/s{k}/{s}", d.sort_values("utt_id").reset_index(drop=True))
    for m, k, s in present:
        if not sel(m, k):
            continue
        p = SC / m / f"s{k}" / f"{s}.csv"
        name = f"{m}/s{k}/{s}"
        bad = []
        df = pd.read_csv(p, dtype={"utt_id": str}, float_precision="round_trip")
        if list(df.columns) != COLS:
            bad.append(f"columns {list(df.columns)}")
            fails.append(f"{name}: {'; '.join(bad)}")
            continue
        n_want = 1000 if s.startswith("stress_") else N[s]
        if len(df) != n_want:
            bad.append(f"rows {len(df)} != {n_want}")
        if set(df["precision"]) != {"fp32"}:
            bad.append(f"precision {sorted(set(df['precision']))}")
        if set(df["label"].unique()) != {0, 1}:
            bad.append(f"labels {sorted(set(df['label']))}")
        sc = df["score"].to_numpy(np.float64)
        if not np.all(np.isfinite(sc)) or np.std(sc) == 0:
            bad.append("non-finite or constant scores")
        if df["utt_id"].duplicated().any():
            bad.append("duplicate utt_ids")
        if set(df["model"]) != {m} or set(pd.to_numeric(df["seed"], errors="coerce").fillna(-1).astype(int)) != {k}:
            bad.append(f"model/seed columns {sorted(set(df['model']))}/{sorted(set(df['seed']))}")
        key = "stress" if s.startswith("stress_") else s
        rname, rd = ref[key]
        mine = df[["utt_id", "label"]].sort_values("utt_id").reset_index(drop=True)
        if not mine.equals(rd):
            bad.append(f"(utt_id, label) pairs differ from {rname}")
        fb, ver, unv = flac_counts(log_for(m, k, s)) if s in FLAC else (None, None, None)
        if s == "df21_verified" and (ver is None or fb + ver != N[s] or unv != 0):
            bad.append(f"DF21 FLAC report fallback={fb} verified={ver} unverifiable={unv} (need sum {N[s]}, 0)")
        src = "as5_dev" if m == "as5_aasist" else "la19_dev"
        if s == src and not (TAUS / f"{m}__s{k}.json").exists():
            bad.append("tau file missing")
        e = 100 * eer(sc, df["label"].to_numpy(np.int64))[0] if not bad else float("nan")
        fl = "" if s not in FLAC else (f"  flac fb={fb} ver={ver} unv={unv}" if ver is not None
                                       else f"  flac fb={fb} (pre-amendment-3 report)")
        lines.append(f"{'OK  ' if not bad else 'FAIL'} {name:40s} n={len(df):7d}  EER {e:6.2f}%{fl}"
                     + (f"  <- {'; '.join(bad)}" if bad else ""))
        if bad:
            fails.append(f"{name}: {'; '.join(bad)}")
    print("\n".join(lines))
    missing = {}
    for m, k, s in reg:
        if (m, k, s) not in present:
            missing.setdefault(f"{m}/s{k}", []).append(s)
    print(f"-- {len(present)} of {len(reg)} registered score files present; missing by unit:")
    for u, ss in missing.items():
        print(f"   {u}: {len(ss)} ({', '.join(ss[:4])}{', ...' if len(ss) > 4 else ''})")
    print("SCORE_REVIEW_" + ("PASS" if not fails else "FAIL:\n  " + "\n  ".join(fails)))


if __name__ == "__main__":
    main()
