"""Precision x transfer-target matrix at the FROZEN 19LA-dev taus (nothing re-calibrated).

    python analysis/transfer_matrix.py

For every ranked model and every score file that exists:
  fp32  results/phase_2_conversion_app/main_matrix/<m>/scores_fp32_pytorch_{itw,df21_verified}.csv
  fp16  results/phase_2_conversion_app/main_matrix/<m>/scores_fp16_ort_{itw,df21_verified}.csv
  int8  the D39 canonical artifact only (canonical_<m>.json -> run dir scores_{itw,df21_verified}.csv)
  fp16r the D46 one-node repair (rawnet2 only): main_matrix/<m>/fp16_repaired_d46/
        scores_fp16repaired_ort_*.csv, shown as "fp16-repaired(D46)", never as plain fp16
  et-*  the ExecuTorch (XNNPACK) leg, DEV-7: executorch/<m>/scores_<p>_executorch_*.csv
it prints EER, HTER/FRR/FAR at tau_eer, and, for fp16/int8, dEER and decision flips against
the fp32 file on identical trials. DF21 is also shown clean-only (D34 sensitivity surface).
Descriptive: H3 (QTS) and H6 have their own registered analyses.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
from deploybench.artifacts import report_label  # noqa: E402
from deploybench.metrics import eer, far_at, frr_at, hter_at  # noqa: E402

MM = R / "results" / "phase_2_conversion_app" / "main_matrix"
CS = R / "results" / "phase_2_conversion_app" / "conversion_sensitivity"
ET = R / "results" / "phase_2_conversion_app" / "executorch"
RANKED = ("aasist", "aasist_l", "rawnet2", "tssdnet")
DATASETS = {"itw": "itw", "df21_verified": "asvspoof21_df_eval"}


def load(p: Path, model: str, precision: str, dataset: str) -> dict:
    out = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if (r["model"], r["precision"], r["dataset"]) != (model, precision, dataset):
            raise SystemExit(f"{p}: row {r['utt_id']} is {r['model']}/{r['precision']}/"
                             f"{r['dataset']}, expected {model}/{precision}/{dataset}")
        if r["utt_id"] in out:
            raise SystemExit(f"{p}: duplicate {r['utt_id']}")
        out[r["utt_id"]] = (int(r["label"]), float(r["score"]))
    return out


def files(model: str, key: str):
    yield "fp32", MM / model / f"scores_fp32_pytorch_{key}.csv", None
    yield "fp16", MM / model / f"scores_fp16_ort_{key}.csv", None
    yield "fp16r", MM / model / "fp16_repaired_d46" / f"scores_fp16repaired_ort_{key}.csv", None
    for p in ("fp16", "int8"):
        yield f"et-{p}", ET / model / f"scores_{p}_executorch_{key}.csv", None
    can = CS / f"canonical_{model}.json"
    if can.is_file():
        c = json.loads(can.read_text())
        if c.get("canonical_seed") is not None:
            run = CS / model / f"{c['locked_recipe']}_seed{c['canonical_seed']}"
            yield "int8", run / f"scores_{key}.csv", run / f"{model}_int8_ptq_log.json"


def row(tag, ids, ref, sc, tau, label):
    y = np.array([ref[u][0] for u in ids])
    s = np.array([sc[u][1] for u in ids])
    e = eer(s, y)[0]
    return (f"{tag:22s} {label:18s} n={len(ids):6d} EER {e*100:6.2f}% | tau_eer HTER "
            f"{hter_at(tau, s, y)*100:6.2f}% FRR {frr_at(tau, s, y)*100:6.2f}% "
            f"FAR {far_at(tau, s, y)*100:6.2f}%"), e, s


def main():
    clean = set(u for u in (r["utt_id"] for r in csv.DictReader(open(
        R / "results" / "phase_1_reproduction" / "scores_df21clean_rawnet2.csv", encoding="utf-8"))))
    for m in RANKED:
        tau = json.loads((R / "results" / "phase_1_reproduction" / f"tau_{m}.json").read_text())["taus"]["eer"]
        for key, ds in DATASETS.items():
            loaded = {}
            for prec, p, side in files(m, key):
                if p.is_file():
                    label = (report_label(prec, side) if prec == "int8"
                             else "fp16-repaired(D46)" if prec == "fp16r"
                             else f"{prec[3:]} executorch" if prec.startswith("et-") else prec)
                    # the repaired graph is fp16 (score_onnx checks it); only the file says repaired.
                    # ExecuTorch rows carry their own precision; load() checks the CSV's columns.
                    want = "fp16" if prec == "fp16r" else prec[3:] if prec.startswith("et-") else prec
                    loaded[prec] = (load(p, m, want, ds), label)
            if "fp32" not in loaded:
                print(f"{m:9s} {key:14s} (no fp32 scores)")
                continue
            ref = loaded["fp32"][0]
            surfaces = [(key, sorted(ref))]
            if key == "df21_verified":
                surfaces.append(("df21_clean", sorted(u for u in ref if u in clean)))
            for sname, ids in surfaces:
                _, e32, s32 = row("", ids, ref, ref, tau, "fp32")
                for prec in ("fp32", "fp16", "fp16r", "int8", "et-fp16", "et-int8"):
                    if prec not in loaded:
                        continue
                    sc, label = loaded[prec]
                    if set(sc) != set(ref):
                        raise SystemExit(f"{m}/{key}: {prec} and fp32 cover different trials")
                    line, e, s = row(f"{m}/{sname}", ids, ref, sc, tau, label)
                    if prec != "fp32":
                        flips = int(((s >= tau) != (s32 >= tau)).sum())
                        line += f" | dEER {(e-e32)*100:+.2f} pp, flips {flips}"
                    print(line)
            print()


if __name__ == "__main__":
    main()
