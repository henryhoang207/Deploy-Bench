"""DEV-8 F2: VAD-trimmed fp32 PyTorch scoring, through the harness's own code path.

Runs on a GPU server from a scratch directory. The harness is imported READ-ONLY from the deploybench clone
(bytecode writing off; nothing in the clone is created or modified).

    # 1. gate (must pass before any trimmed run): no trim, first 200 utterances of each set vs the committed scores
    python -B f_vad_score.py --model aasist --gate
    # 2. the trimmed run
    python -B f_vad_score.py --model aasist --bounds $SCRATCH/dfv_rev/vad/vad_stats.csv \
        --out-dir $SCRATCH/dfv_rev/scores

Path per clip (only the trim is new):  load_wav (16 kHz float32) -> [trim] -> fix_length -> model.score_batch.
Trim rule, fixed in DEVIATIONS.md DEV-8: T = 35 dB energy VAD, both ends, zero margin, using the bounds written by
f_vad_stats.py; if the trimmed clip would be shorter than 0.25 s (4,000 samples) the clip is left untrimmed and counted.
Checkpoints are exactly those of the harness scoring scripts (see CKPT below). TF32 is off (load_model, device cuda).
Output schema = the project score schema, condition = vad35.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = os.environ.get("DEPLOYBENCH", str(Path(__file__).resolve().parents[2]))
START_CWD = os.getcwd()          # relative --bounds / --out-dir paths are resolved against THIS, not the clone
sys.path.insert(0, REPO)
os.chdir(REPO)                   # the harness expects its own root as cwd; this writes nothing

import numpy as np  # noqa: E402

V = f"{REPO}/vendor"
CKPT = {
    "aasist": f"{V}/aasist/models/weights/AASIST.pth",
    "aasist_l": f"{V}/aasist/models/weights/AASIST-L.pth",
    "rawnet2": f"{V}/asvspoof2021_baselines/LA/Baseline-RawNet2/pre_trained_DF_RawNet2.pth",
    "tssdnet": (sorted(glob.glob(f"{V}/end-to-end-synthetic-speech-detection/pretrained/Res_TSSDNet*.pth")) or [None])[0],
    "xlsr_aasist": f"{V}/SSL_Anti-spoofing/Best_LA_model_for_DF.pth",
}
DATASETS = ("asvspoof19_dev", "asvspoof19_eval", "itw")
MM = f"{REPO}/results/phase_2_conversion_app/main_matrix"
P1 = f"{REPO}/results/phase_1_reproduction"


REF_OVERRIDE = {}      # dataset -> gate reference file (DEV-8 amendment 1: platform-matched reference)


def committed_scores(model: str, ds: str) -> str:
    if ds in REF_OVERRIDE:
        return REF_OVERRIDE[ds]
    return {"asvspoof19_dev": f"{P1}/scores_dev_{model}.csv",
            "asvspoof19_eval": f"{P1}/scores_repro_{model}.csv",
            "itw": f"{MM}/{model}/scores_fp32_pytorch_itw.csv"}[ds]


MIN_LEN = 4000          # 0.25 s at 16 kHz
THR = 35


def load_bounds(path: str):
    d = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            d[(r["dataset"], r["utt_id"])] = (int(r["n_samples"]), int(r[f"lead{THR}"]), int(r[f"trail{THR}"]))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(CKPT))
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    ap.add_argument("--bounds", default=None, help="vad_stats.csv from f_vad_stats.py (needed unless --gate)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--gate", action="store_true", help="no trim, first --gate-n utterances, compare with committed scores")
    ap.add_argument("--gate-n", type=int, default=200)
    ap.add_argument("--ref-override", action="append", default=[], metavar="DATASET=FILE",
                    help="gate only: use FILE as the reference for DATASET (DEV-8 amendment 1)")
    ap.add_argument("--smoke", type=int, default=0,
                    help="trimmed-path smoke test on the first N utterances per set; writes smoke_*.csv, never a result")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--progress", type=int, default=5000)
    a = ap.parse_args()
    if not a.gate and not (a.bounds and a.out_dir):
        raise SystemExit("a trimmed run needs --bounds and --out-dir")
    if a.bounds:
        a.bounds = os.path.join(START_CWD, a.bounds)
    if a.out_dir:
        a.out_dir = os.path.join(START_CWD, a.out_dir)
        if os.path.realpath(a.out_dir).startswith(os.path.realpath(REPO)):
            raise SystemExit("refusing to write inside the deploybench clone")
    for kv in a.ref_override:
        k, v = kv.split("=", 1)
        REF_OVERRIDE[k] = v
    ck = CKPT[a.model]
    if not ck or not Path(ck).is_file():
        raise SystemExit(f"checkpoint missing for {a.model}: {ck}")

    from deploybench.data import protocol_iter
    from models.base import SPECS, load_model
    from score import fix_length, load_wav

    model = load_model(a.model, ck, a.device)
    n_in = SPECS[a.model].input_samples
    bounds = None if a.gate else load_bounds(a.bounds)
    worst_all = 0.0

    for ds in a.datasets:
        out = None
        if not a.gate:
            out = Path(a.out_dir) / (("smoke_" if a.smoke else "") + f"scores_{a.model}_{ds}_vad{THR}.csv")
            if out.exists():
                raise SystemExit(f"{out} exists: refusing to overwrite")
            out.parent.mkdir(parents=True, exist_ok=True)
        rows, ids, waves, labels = [], [], [], []
        n_trimmed = n_guard = n_seen = 0
        t0 = time.time()

        def flush():
            if not waves:
                return
            sc = model.score_batch(np.stack(waves))
            rows.extend(zip(ids, labels, (float(s) for s in sc)))
            ids.clear(); waves.clear(); labels.clear()

        for uid, path, label in protocol_iter(ds):
            if a.gate and n_seen >= a.gate_n:
                break
            if a.smoke and n_seen >= a.smoke:
                break
            wave = load_wav(Path(path))
            if not a.gate:
                n0, lead, trail = bounds[(ds, uid)]
                if n0 != len(wave):
                    raise SystemExit(f"{ds}/{uid}: bounds say {n0} samples, loaded {len(wave)}")
                end = n0 - trail
                if lead + trail > 0 and end - lead < MIN_LEN:
                    n_guard += 1                   # would be < 0.25 s: left untrimmed
                elif lead + trail > 0:
                    wave = wave[lead:end]
                    n_trimmed += 1
            ids.append(uid); labels.append(int(label)); waves.append(fix_length(wave, n_in))
            n_seen += 1
            if len(waves) >= a.batch:
                flush()
            if a.progress and n_seen % a.progress == 0:
                el = time.time() - t0
                print(f"  [{a.model}/{ds}] {n_seen} clips  {el:.0f}s  {n_seen / el:.1f}/s", flush=True)
        flush()

        if a.gate:
            ref = {}
            with open(committed_scores(a.model, ds), newline="") as f:
                for r in csv.DictReader(f):
                    ref[r["utt_id"]] = float(r["score"])
            miss = [u for u, _, _ in rows if u not in ref]
            if miss:
                raise SystemExit(f"GATE FAIL {a.model}/{ds}: {len(miss)} ids absent from the committed file")
            worst = max(abs(s - ref[u]) for u, _, s in rows)
            worst_all = max(worst_all, worst)
            print(f"GATE {a.model}/{ds}: n={len(rows)} max|d|={worst:.3e}", flush=True)
        else:
            tmp = out.with_suffix(".partial")
            with open(tmp, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["utt_id", "label", "score", "model", "dataset", "precision", "runtime", "condition", "seed"])
                for u, l, s in rows:
                    w.writerow([u, l, s, a.model, ds, "fp32", "pytorch", f"vad{THR}", 0])
            os.replace(tmp, out)
            print(f"wrote {len(rows)} scores -> {out}  trimmed={n_trimmed} guard_untrimmed={n_guard} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    if a.gate:
        if worst_all > 1e-4:
            print(f"GATE FAILED: max|d| = {worst_all:.3e} > 1e-4", flush=True)
            sys.exit(2)
        print(f"GATE PASSED for {a.model}: max|d| = {worst_all:.3e} <= 1e-4", flush=True)


if __name__ == "__main__":
    main()
