"""Round-2 gates (PREREGISTRATION_ROUND2.md, Sec. 4), computed after the tag from the gate scores; writes
results/round2/gates.json, which r2_registered.py requires. The rule is round 1's: a reproduced EER within 15%
(relative) of the published one. A model that fails is reported and counts as not supporting R2-H1.

  as5_aasist   ASVspoof 5 dev EER vs the Baseline-AASIST README's 15.2%      -> pass if in [12.92, 17.48] %
  xlsr_mamba   EER on the DF21 verified trials that also appear in the authors' released DF score file
               (vendor/XLSR-Mamba/Scores/DF/Bmamba3_LA_WCE_1e-06_ES144_NE12.txt), ours vs theirs on the same trials
               (labels from our file)                                           -> pass if ours within 15% of theirs
  pascu_lr     the pipeline-check head (LA19 train + dev, the authors' recipe; model pascu_lr_gate) on LA19 eval vs
               their Table 2 (xls-r-2b, LogReg C = 1e6): 0.6%                   -> pass if in [0.51, 0.69] %
  aasist_l_*   no accept/reject gate: each run's training record (run_final.json, epochs.jsonl) is summarised.

    python analysis/round2/r2_gates.py --xlsr-mamba-ref <file> --runs-dir <dir with aasist_l_*_s<k>/>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r2lib as R  # noqa: E402


def within(x, ref, rel=0.15):
    return ref * (1 - rel) <= x <= ref * (1 + rel), [ref * (1 - rel), ref * (1 + rel)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsr-mamba-ref", required=True)
    ap.add_argument("--runs-dir", required=True)
    a = ap.parse_args()
    from deploybench.metrics import eer
    out = {}

    _, y, s = R.load("as5_aasist", "as5_dev")
    e = 100 * eer(s, y)[0]
    ok, rng = within(e, 15.2)
    out["as5_aasist"] = {"passed": ok, "eer_pct": e, "published_pct": 15.2, "range_pct": rng,
                         "summary": f"AS5 dev EER {e:.2f}% vs 15.2% -> {'PASS' if ok else 'FAIL'}"}

    ids, y, s = R.load("xlsr_mamba", "df21_verified")
    ref = {}
    for line in open(a.xlsr_mamba_ref, encoding="utf-8"):
        f = line.split()
        if len(f) >= 2:
            ref[f[0]] = float(f[-1])
    keep = np.array([u in ref for u in ids])
    theirs = np.array([ref[u] for u in ids[keep]])
    ours = s[keep]
    yk = y[keep]
    e_o, e_t = 100 * eer(ours, yk)[0], 100 * eer(theirs, yk)[0]
    ok, rng = within(e_o, e_t)
    d = np.abs(ours - theirs)
    out["xlsr_mamba"] = {"passed": ok, "n_overlap": int(keep.sum()), "eer_ours_pct": e_o, "eer_theirs_pct": e_t,
                         "range_pct": rng, "abs_diff_max": float(d.max()), "abs_diff_median": float(np.median(d)),
                         "pearson": float(np.corrcoef(ours, theirs)[0, 1]),
                         "summary": f"DF21 overlap n={int(keep.sum())}: EER ours {e_o:.2f}% vs theirs {e_t:.2f}% "
                                    f"-> {'PASS' if ok else 'FAIL'}"}

    _, y, s = R.load("pascu_lr_gate", "la19_eval")
    e = 100 * eer(s, y)[0]
    ok, rng = within(e, 0.6)
    out["pascu_lr"] = {"passed": ok, "gate_model": "pascu_lr_gate (LA19 train + dev)", "eer_pct": e,
                       "published_pct": 0.6, "range_pct": rng,
                       "summary": f"pipeline head LA19-eval EER {e:.2f}% vs 0.6% -> {'PASS' if ok else 'FAIL'}"}

    runs = {}
    for d_ in sorted(Path(a.runs_dir).glob("aasist_l_*_s*")):
        fin, ep = d_ / "run_final.json", d_ / "epochs.jsonl"
        rec = {"finished": fin.exists()}
        if fin.exists():
            rec.update(json.loads(fin.read_text()))
        if ep.exists():
            ev = [json.loads(x) for x in ep.read_text().splitlines() if x.strip()]
            rec["epochs"] = len(ev)
            rec["dev_eer_last_pct"] = ev[-1].get("dev_eer")
            rec["nan_loss"] = any(not np.isfinite(x["loss"]) for x in ev)
        runs[d_.name] = rec
    out["runs"] = runs
    (R.R2 / "gates.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k in ("as5_aasist", "xlsr_mamba", "pascu_lr"):
        print(out[k]["summary"])
    for k, v in runs.items():
        print(f"{k}: finished={v['finished']} swa_updates={v.get('n_swa_update')} epochs={v.get('epochs')}")


if __name__ == "__main__":
    main()
