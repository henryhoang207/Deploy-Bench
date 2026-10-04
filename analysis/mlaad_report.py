"""MLAAD-5 descriptive report (D47 / DEVIATIONS.md DEV-6). Frozen 19LA-dev taus, nothing re-calibrated, and no
hypothesis test (MLAAD enters no registered test).

    python analysis/mlaad_report.py     # -> results/phase_2_conversion_app/mlaad5/mlaad_report.{txt,json}

Per artifact (`mlaad5/<model>/scores_<p>_<runtime>_mlaad.csv.gz`): EER; HTER / FRR / FAR at tau_eer, tau_frr1, tau_frr5;
TTG at tau_eer = HTER(tau*) - min HTER, with a class-stratified bootstrap 95% CI (B = 1000, seed 0, the h1_h4.py
convention) for fp32; for converted artifacts dEER and decision flips at tau_eer against the same model's fp32
on identical trials. fp32 is also broken down per language. Language comes from the frozen protocol
(configs/mlaad5/protocol.csv.gz), never from the utt_id. The sample-rate census (if present) is reproduced,
because the bona fide / spoof sample-rate difference is a confound for every number here.
"""
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
from deploybench.metrics import eer, far_at, frr_at, hter_at, min_hter  # noqa: E402

D = R / "results" / "phase_2_conversion_app" / "mlaad5"
P1 = R / "results" / "phase_1_reproduction"
MODELS = ("aasist", "aasist_l", "rawnet2", "tssdnet", "xlsr_aasist")
# (file stem, label, schema precision, schema runtime)
ARTIFACTS = (("fp32_pytorch", "fp32", "fp32", "pytorch"),
             ("fp16_ort", "fp16", "fp16", "ort"),
             ("fp16repaired_ort", "fp16-repaired(D46)", "fp16", "ort"),
             ("int8_ort", "int8-mixed (K=16)", "int8", "ort"),
             ("fp16_executorch", "fp16 executorch", "fp16", "executorch"))
LANGS = ("en", "de", "it", "fr", "pl")
B, SEED = 1000, 0


def protocol_langs():
    with gzip.open(R / "configs" / "mlaad5" / "protocol.csv.gz", "rt", encoding="utf-8") as f:
        return {r["utt_id"]: r["lang"] for r in csv.DictReader(f)}


def score_file(m: str, stem: str):
    """Committed as .csv.gz (the 495k-row CSVs are ~63 MB each); a raw .csv is accepted locally."""
    for p in (D / m / f"scores_{stem}_mlaad.csv.gz", D / m / f"scores_{stem}_mlaad.csv"):
        if p.is_file():
            return p
    return None


def load(p: Path, model: str, precision: str, runtime: str) -> dict:
    out = {}
    fh = gzip.open(p, "rt", encoding="utf-8") if p.suffix == ".gz" else open(p, encoding="utf-8")
    for r in csv.DictReader(fh):
        if (r["model"], r["precision"], r["runtime"], r["dataset"]) != (model, precision, runtime, "mlaad"):
            raise SystemExit(f"{p}: row {r['utt_id']} is {r['model']}/{r['precision']}/{r['runtime']}/{r['dataset']}")
        if r["utt_id"] in out:
            raise SystemExit(f"{p}: duplicate {r['utt_id']}")
        out[r["utt_id"]] = (int(r["label"]), float(r["score"]))
    return out


def ttg_ci(s, y, tau):
    ib, isp = np.where(y == 1)[0], np.where(y == 0)[0]
    rng = np.random.default_rng(SEED)
    v = np.empty(B)
    for i in range(B):
        idx = np.concatenate([ib[rng.integers(0, len(ib), len(ib))], isp[rng.integers(0, len(isp), len(isp))]])
        v[i] = hter_at(tau, s[idx], y[idx]) - min_hter(s[idx], y[idx])[0]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def cell(s, y, taus):
    e = eer(s, y)[0]
    h_or = min_hter(s, y)[0]
    c = {"n": int(len(y)), "n_bona": int(y.sum()), "eer": e, "hter_oracle": h_or}
    for pol, t in taus.items():
        c[f"at_{pol}"] = {"hter": hter_at(t, s, y), "frr": frr_at(t, s, y), "far": far_at(t, s, y)}
    c["ttg_eer"] = c["at_eer"]["hter"] - h_or
    return c


def main():
    lang = protocol_langs()
    report, lines = {"models": {}}, []
    for m in MODELS:
        taus = json.loads((P1 / f"tau_{m}.json").read_text())["taus"]
        rows, ref = {}, None
        for stem, label, prec, rt in ARTIFACTS:
            p = score_file(m, stem)
            if p is None:
                continue
            sc = load(p, m, prec, rt)
            if set(sc) != set(lang):
                raise SystemExit(f"{p}: covers {len(sc)} trials, the protocol has {len(lang)}")
            ids = sorted(sc)
            y = np.array([sc[u][0] for u in ids])
            s = np.array([sc[u][1] for u in ids])
            c = cell(s, y, taus)
            if stem == "fp32_pytorch":
                ref = (ids, s)
                c["ttg_eer_ci95"] = ttg_ci(s, y, taus["eer"])
                c["per_language"] = {}
                for lg in LANGS:
                    k = np.array([lang[u] == lg for u in ids])
                    c["per_language"][lg] = cell(s[k], y[k], taus)
            elif ref is not None:
                c["deer_vs_fp32"] = c["eer"] - eer(ref[1], y)[0]
                c["flips_eer_vs_fp32"] = int(((s >= taus["eer"]) != (ref[1] >= taus["eer"])).sum())
            rows[label] = c
        if rows:
            report["models"][m] = rows
    census = R / "results" / "phase_2_conversion_app" / "mlaad5" / "sample_rate_census.json"
    if census.is_file():
        report["sample_rate_census"] = json.loads(census.read_text())

    pct = lambda v: f"{100 * v:6.2f}"   # noqa: E731
    lines.append("MLAAD-5 at the frozen 19LA-dev taus (descriptive; DEV-6). % throughout.\n")
    for m, rows in report["models"].items():
        for label, c in rows.items():
            a = c["at_eer"]
            line = (f"{m:11s} {label:20s} n={c['n']} EER {pct(c['eer'])} | tau_eer HTER {pct(a['hter'])} "
                    f"FRR {pct(a['frr'])} FAR {pct(a['far'])} | TTG {pct(c['ttg_eer'])}")
            if "ttg_eer_ci95" in c:
                lo, hi = c["ttg_eer_ci95"]
                line += f" [{pct(lo).strip()}, {pct(hi).strip()}]"
            if "deer_vs_fp32" in c:
                line += f" | dEER {100 * c['deer_vs_fp32']:+.2f} pp, flips {c['flips_eer_vs_fp32']}"
            lines.append(line)
            for pol in ("frr1", "frr5"):
                a = c[f"at_{pol}"]
                lines.append(f"{'':33s} tau_{pol:4s} HTER {pct(a['hter'])} FRR {pct(a['frr'])} FAR {pct(a['far'])}")
        fp = rows.get("fp32")
        if fp:
            for lg, c in fp["per_language"].items():
                a = c["at_eer"]
                lines.append(f"{'':12s}fp32 {lg}: n={c['n']} (bona {c['n_bona']}) EER {pct(c['eer'])} | tau_eer "
                             f"FRR {pct(a['frr'])} FAR {pct(a['far'])} TTG {pct(c['ttg_eer'])}")
        lines.append("")
    out = D / "mlaad_report"
    out.with_suffix(".json").write_text(json.dumps(report, indent=1) + "\n")
    out.with_suffix(".txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
