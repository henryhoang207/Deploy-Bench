"""Full-eval precision comparison (H6): fp32 vs fp16 vs int8 on all 71,237 clips.

WHY THIS EXISTS: every precision number before this came from the 1,000-clip parity
manifest, whose own fp32 EER is 1.40% against a full-eval 1.6179%. With 500 bona
fide clips the manifest resolves EER to roughly 0.2 pp, so a delta measured there
is barely separable from its own noise floor and is not publishable.

WHICH fp32 IS THE REFERENCE -- this script had this wrong once. Deltas for the
converted precisions are taken against **ONNX fp32**, not PyTorch fp32. Comparing
ONNX int8 to PyTorch fp32 folds conversion error into the quantisation cost and
conflates two separate H6 columns; the project already corrected exactly this
mistake on 2026-07-26 for the manifest run. PyTorch fp32 is still loaded, for two
jobs it alone can do: it gates the run against the frozen reproduction anchor, and
it gives the conversion delta as a reported row rather than a hidden term.

Every EER is computed TWICE -- once through deploybench.metrics and once through an
independent numpy sort-and-sweep sharing no code -- because this project has been
bitten by trusting a single implementation. Decision flips are counted at the
FROZEN taus (D16), not at each precision's own operating point, because that is
what a deployed threshold actually does.

    python analysis/precision_report.py --out-md <md> --out-json <json>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deploybench.metrics import eer as metrics_eer      # noqa: E402

FP32_ANCHOR = 1.6179          # results/phase_1_reproduction/repro_report_tssdnet.md
ANCHOR_TOL = 5e-3             # pp; recomputation must be essentially exact
REF = "onnx_fp32"             # the reference every converted arm is measured against
GATE = "pt_fp32"              # the arm that must reproduce the frozen anchor


def sweep_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """Independent EER%: sort descending, sweep, take the FRR/FAR crossing.
    Deliberately shares no code with metrics.py."""
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    frr = 1.0 - np.cumsum(y) / n_pos
    far = np.cumsum(1 - y) / n_neg
    i = int(np.nanargmin(np.abs(frr - far)))
    return float((frr[i] + far[i]) / 2 * 100)


def boot_ci(scores, labels, n_boot=1000, seed=20260727, alpha=0.05):
    """Percentile bootstrap CI on EER%, resampling clips with replacement."""
    rng = np.random.default_rng(seed)
    n = len(scores)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = sweep_eer(scores[idx], labels[idx])
    return float(np.percentile(vals, 100 * alpha / 2)), \
        float(np.percentile(vals, 100 * (1 - alpha / 2)))


def paired_boot_delta(s_ref, s_test, labels, n_boot=1000, seed=20260727, alpha=0.05):
    """CI on the EER DIFFERENCE with clips resampled jointly, so both arms see the
    same clips in every replicate. Unpaired CIs would overstate the spread."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        lab = labels[idx]
        vals[b] = sweep_eer(s_test[idx], lab) - sweep_eer(s_ref[idx], lab)
    return float(np.percentile(vals, 100 * alpha / 2)), \
        float(np.percentile(vals, 100 * (1 - alpha / 2)))


def main() -> int:
    ap = argparse.ArgumentParser()
    # repo/results/, not the old project-root phase_*/results/ (moved 2026-07-27
    # so the artifacts are version-controlled alongside the code that made them).
    p1 = Path(__file__).resolve().parent.parent / "results" / "phase_1_reproduction"
    ap.add_argument("--pt-fp32", default=str(p1 / "scores_repro_tssdnet.csv"))
    ap.add_argument("--onnx-fp32", default=str(p1 / "scores_eval_tssdnet_fp32_onnx.csv"))
    ap.add_argument("--fp16", default=str(p1 / "scores_eval_tssdnet_fp16_onnx.csv"))
    ap.add_argument("--int8", default=str(p1 / "scores_eval_tssdnet_int8_onnx.csv"))
    ap.add_argument("--taus", default=str(p1 / "tau_tssdnet.json"))
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()

    order = [("pt_fp32", a.pt_fp32), ("onnx_fp32", a.onnx_fp32),
             ("fp16", a.fp16), ("int8", a.int8)]
    arms = {}
    for name, path in order:
        df = pd.read_csv(path).sort_values("utt_id").reset_index(drop=True)
        arms[name] = df
        print(f"{name:10s} {len(df):7d} rows  {path}")

    base = arms[GATE]
    for name, df in arms.items():
        if not base.utt_id.equals(df.utt_id):
            print(f"FATAL: {name} utt_id set differs", file=sys.stderr)
            return 2
        if not base.label.equals(df.label):
            print(f"FATAL: {name} labels differ", file=sys.stderr)
            return 2
    labels = base.label.to_numpy()
    print(f"\nclips {len(labels)}  bona {int(labels.sum())}  "
          f"spoof {int(len(labels) - labels.sum())}")

    taus = json.loads(Path(a.taus).read_text(encoding="utf-8-sig"))
    taus = taus.get("taus", taus)
    print(f"frozen taus: {taus}")

    # ---- gate: PyTorch fp32 must reproduce the frozen reproduction number.
    s_gate = base.score.to_numpy()
    e_gate_m = metrics_eer(s_gate, labels)[0] * 100
    e_gate_s = sweep_eer(s_gate, labels)
    print(f"\n{GATE} recomputed: metrics {e_gate_m:.4f}%  sweep {e_gate_s:.4f}%  "
          f"anchor {FP32_ANCHOR}%")
    if abs(e_gate_s - FP32_ANCHOR) > ANCHOR_TOL:
        print(f"FATAL: {GATE} recomputation {e_gate_s:.4f}% != anchor {FP32_ANCHOR}%. "
              f"This script is wrong; ignore every other number.", file=sys.stderr)
        return 3

    s_ref = arms[REF].score.to_numpy()
    e_ref = sweep_eer(s_ref, labels)

    out = {"n_clips": int(len(labels)), "n_bona": int(labels.sum()), "taus": taus,
           "fp32_anchor_pct": FP32_ANCHOR, "n_boot": a.n_boot,
           "delta_reference": REF, "gate_arm": GATE, "arms": {}}

    for name, df in arms.items():
        s = df.score.to_numpy()
        e_m = metrics_eer(s, labels)[0] * 100
        e_s = sweep_eer(s, labels)
        row = {"eer_metrics_pct": e_m, "eer_sweep_pct": e_s,
               "eer_agreement": abs(e_m - e_s), "score_std": float(s.std()),
               "ci95": list(boot_ci(s, labels, a.n_boot))}
        # pt_fp32 is measured against nothing; onnx_fp32 against pt_fp32 (that
        # delta IS the conversion error); everything else against onnx_fp32.
        cmp_name = None if name == GATE else (GATE if name == REF else REF)
        if cmp_name:
            c = arms[cmp_name].score.to_numpy()
            e_c = e_gate_s if cmp_name == GATE else e_ref
            row["compared_to"] = cmp_name
            row["delta_eer_pp"] = e_s - e_c
            row["delta_ci95"] = list(paired_boot_delta(c, s, labels, a.n_boot))
            row["pearson_r"] = float(np.corrcoef(c, s)[0, 1])
            row["spearman_r"] = float(pd.Series(c).corr(pd.Series(s), method="spearman"))
            row["max_abs_delta"] = float(np.abs(s - c).max())
            row["mean_abs_delta"] = float(np.abs(s - c).mean())
            row["decision_flips"] = {k: int(((c >= t) != (s >= t)).sum())
                                     for k, t in taus.items()}
            row["flip_pct"] = {k: 100.0 * v / len(labels)
                               for k, v in row["decision_flips"].items()}
        out["arms"][name] = row
        print(f"\n{name}: EER {e_s:.4f}% (metrics {e_m:.4f}%)  CI {row['ci95']}")
        if cmp_name:
            print(f"      vs {cmp_name}: dEER {row['delta_eer_pp']:+.4f} pp  "
                  f"CI {row['delta_ci95']}  flips {row['decision_flips']}  "
                  f"max|d| {row['max_abs_delta']:.3e}")

    Path(a.out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")

    md = ["# TSSDNet precision comparison - FULL EVAL (H6)", "",
          f"ASVspoof19 LA eval, **{len(labels):,} clips** ({int(labels.sum()):,} bona "
          f"fide). Frozen taus (D16). Bootstrap {a.n_boot} resamples, paired for deltas.", "",
          "Supersedes the 1,000-clip manifest numbers, which resolved EER only to ~0.2 pp.", "",
          "**Deltas for the converted precisions are against ONNX fp32, not PyTorch fp32** -- "
          "measuring them against PyTorch would fold conversion error into quantisation cost. "
          "The ONNX fp32 row's own delta IS the conversion error, reported rather than hidden.",
          "",
          "| arm | EER % | 95% CI | vs | dEER pp | dEER 95% CI | flips e/f1/f5 | max\\|d\\| | Pearson |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name in ("pt_fp32", "onnx_fp32", "fp16", "int8"):
        r = out["arms"][name]
        ci = f"{r['ci95'][0]:.4f}-{r['ci95'][1]:.4f}"
        if "compared_to" not in r:
            md.append(f"| {name} | {r['eer_sweep_pct']:.4f} | {ci} | -- | -- | -- | -- | -- | -- |")
        else:
            f = r["decision_flips"]
            dci = f"{r['delta_ci95'][0]:+.4f} to {r['delta_ci95'][1]:+.4f}"
            md.append(f"| {name} | {r['eer_sweep_pct']:.4f} | {ci} | {r['compared_to']} | "
                      f"{r['delta_eer_pp']:+.4f} | {dci} | "
                      f"{f['eer']}/{f['frr1']}/{f['frr5']} | {r['max_abs_delta']:.3e} | "
                      f"{r['pearson_r']:.6f} |")
    md += ["", "Every EER computed twice (deploybench.metrics vs an independent numpy sweep); "
           f"max disagreement {max(r['eer_agreement'] for r in out['arms'].values()):.2e} pp.",
           "", f"The run GATES on the PyTorch fp32 arm reproducing the frozen reproduction "
           f"anchor {FP32_ANCHOR}% and aborts otherwise.", ""]
    Path(a.out_md).write_text("\n".join(md), encoding="utf-8")
    print(f"\nwrote {a.out_md}\nwrote {a.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
