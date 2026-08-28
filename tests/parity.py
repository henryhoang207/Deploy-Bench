"""Parity / conversion-comparison suite (phase_2 step_03).

THREE DISTINCT COMPARISONS live here. Conflating them was a real bug: the pilot
ran an int8-vs-fp32 precision comparison, gated it at ZERO decision flips, and
reported the result as "PARITY FAILED". That bar demands lossless quantisation,
which nothing satisfies, and it is not what experiment_protocol.md Stage P3 asks
for. P3 says the 1,000 clips are "scored on desktop runtime AND on-device" —
parity is the SAME artifact in TWO PLACES.

    --mode device       (GATE)  same precision, desktop vs on-device.
                                This is Stage P3 / methodology B5. Any flip is a
                                PORT defect: kernel differences, preprocessing
                                mismatch, resampler drift. Gate: zero flips.

    --mode conversion   (GATE)  same precision, PyTorch vs converted graph
                                (e.g. PyTorch fp32 vs ONNX fp32). Same maths,
                                different runtime, so it must be near-exact.
                                Gate: zero flips.

    --mode quantization (DIAGNOSTIC, never gates) different precisions, e.g.
                                ONNX int8 vs PyTorch fp32. Decision flips here
                                are the COST of quantisation, an H6 result row —
                                not a pass/fail. Always exits 0; read the numbers.

    python tests/parity.py --ref ref.csv --test test.csv --taus taus.json \
                           --mode device
"""
import argparse
import json

import numpy as np
import pandas as pd

GATING_MODES = ("device", "conversion")


def _eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """EER% from the project score contract (higher = bona fide, label 1 = bona
    fide). Only meaningful when the csv carries real labels (not -1)."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    order = np.argsort(-scores)
    y = labels[order]
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    frr = 1.0 - np.cumsum(y) / n_pos
    far = np.cumsum(1 - y) / n_neg
    i = int(np.nanargmin(np.abs(frr - far)))
    return float((frr[i] + far[i]) / 2 * 100)


def compare(ref: pd.DataFrame, test: pd.DataFrame, taus: dict,
            mode: str = "device") -> dict:
    m = ref.merge(test, on="utt_id", suffixes=("_ref", "_test"))
    if m.empty:
        raise SystemExit("no overlapping utt_ids between --ref and --test")
    d = (m.score_test - m.score_ref).abs()
    out = {"mode": mode, "gates": mode in GATING_MODES, "n": len(m),
           "max_abs_delta": float(d.max()),
           "mean_abs_delta": float(d.mean()),
           "pearson_r": float(np.corrcoef(m.score_ref, m.score_test)[0, 1]),
           "decision_flips": {}}

    for name, tau in taus.items():
        out["decision_flips"][name] = int(
            ((m.score_ref >= tau) != (m.score_test >= tau)).sum())

    # A precision comparison is about how much DISCRIMINATION survives, which
    # decision flips against a foreign threshold cannot express: a graph whose
    # scores are merely shifted flips everything while ranking perfectly, and a
    # graph that lost all ranking can still flip nothing. Report EER on both
    # sides so the two failure modes are distinguishable.
    if mode == "quantization" and "label_ref" in m:
        lab = m.label_ref.to_numpy()
        if len(np.unique(lab)) >= 2:
            e_ref = _eer(m.score_ref.to_numpy(), lab)
            e_test = _eer(m.score_test.to_numpy(), lab)
            out["eer_pct"] = {"ref": e_ref, "test": e_test,
                              "delta_pp": e_test - e_ref}
            out["spearman_r"] = float(
                m.score_ref.corr(m.score_test, method="spearman"))
            out["score_std"] = {"ref": float(m.score_ref.std()),
                                "test": float(m.score_test.std())}

    if mode in GATING_MODES:
        out["pass"] = all(v == 0 for v in out["decision_flips"].values())
    else:
        out["pass"] = None            # explicitly not a verdict
        out["note"] = ("DIAGNOSTIC ONLY — decision flips vs a different "
                       "precision are the cost of quantisation (H6), not a "
                       "parity result. Stage P3 parity is --mode device.")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--taus", required=True, help="json {'eer':t,'frr1':t,'frr5':t}")
    ap.add_argument("--mode", default="device",
                    choices=["device", "conversion", "quantization"],
                    help="device/conversion GATE at zero flips; quantization "
                         "is diagnostic and never fails the build")
    a = ap.parse_args()
    # utf-8-sig, not the locale default: pilot_tssdnet.ps1 writes the flat tau
    # file with PowerShell 5.1's `Out-File -Encoding utf8`, which emits a UTF-8
    # BOM. Python's open() then decodes it with the Windows codepage, the BOM
    # becomes 'i>>?' and json dies with "Expecting value: line 1 column 1".
    # utf-8-sig strips a BOM if present and is a no-op if it is not.
    r = compare(pd.read_csv(a.ref), pd.read_csv(a.test),
                json.loads(open(a.taus, encoding="utf-8-sig").read()),
                mode=a.mode)
    print(json.dumps(r, indent=2))
    raise SystemExit(0 if r["pass"] in (True, None) else 1)
