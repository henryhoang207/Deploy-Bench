"""Regenerate and verify the Speech Communication tables and figures without model execution.

    python scripts/reproduce_paper.py --verify
    python scripts/reproduce_paper.py --write

Verification writes only to a temporary directory. It recomputes the round-1
estimator gate from utterance scores, regenerates every table from its numeric
inputs, compares table text with the manuscript, and renders all seven figures.
Figure PDFs are checked for successful generation; their bytes vary with PDF
metadata and plotting-library versions and are not claimed to be identical.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "paper" / "speechcomm"
ROUND1 = REPO / "reproducibility" / "round1"

# These are the documented manuscript edits in audit_claims.py. They clarify the
# analysis status without changing any numeric input or generated result.
TABLE_EDITS = {"tab_tests.tex": [(b"Clips (registered)", b"Clips (first run)"),
                                (b"Speakers (this revision)", b"Speakers (exploratory)")]}
FIGURES = ("fig_frr_far.pdf", "figS_sensitivity.pdf", "figS_h5.pdf", "figS_overview.pdf",
           "figS_speakers.pdf", "figS_cost.pdf", "figS_silence.pdf")


def verify_inputs():
    manifest = json.loads((REPO / "reproducibility" / "source_manifest.json").read_text(encoding="utf-8"))
    for record in manifest["files"]:
        path = REPO / record["destination"]
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != record["package_sha256"]:
            raise SystemExit(f"REFUSED: package hash differs for {path.relative_to(REPO)}")
        if record["gzip"] and hashlib.sha256(gzip.decompress(data)).hexdigest() != record["source_sha256"]:
            raise SystemExit(f"REFUSED: decompressed source hash differs for {path.relative_to(REPO)}")
    print(f"Verified {len(manifest['files'])} bundled source artifacts.", flush=True)
    snapshots = json.loads((REPO / "reproducibility" / "registrations" /
                            "manifest.json").read_text(encoding="utf-8"))["tags"]
    for name, snapshot in snapshots.items():
        for record in snapshot["files"]:
            if hashlib.sha256((REPO / record["path"]).read_bytes()).hexdigest() != record["sha256"]:
                raise SystemExit(f"REFUSED: {name} registration snapshot hash differs: {record['path']}")
    print("Verified both original registration snapshots.", flush=True)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def mlaad_table(destination):
    models = json.loads((REPO / "results" / "phase_2_conversion_app" / "mlaad5" /
                         "mlaad_report.json").read_text(encoding="utf-8"))["models"]
    names = (("aasist", "AASIST"), ("aasist_l", "AASIST-L"), ("rawnet2", "RawNet2"),
             ("tssdnet", "Res-TSSDNet"), ("xlsr_aasist", "XLS-R+AASIST (ref.)"))
    lines = ["% src: results/phase_2_conversion_app/mlaad5/mlaad_report.json (models.*.fp32 only).",
             r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
             r"Model & EER & FRR & FAR & HTER & TTG \\", r"\midrule"]
    for model, label in names:
        row = models[model]["fp32"]
        point = row["at_eer"]
        values = [row["eer"], point["frr"], point["far"], point["hter"], row["ttg_eer"]]
        lines.append(label + " & " + " & ".join(f"{100 * value:.2f}" for value in values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (destination / "r1_mlaad.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def regenerate(output):
    gen = output / "gen"
    figures = output / "figures"
    gen.mkdir(parents=True)
    figures.mkdir()
    os.environ["DEPLOYBENCH"] = str(REPO)
    os.environ["DFV_META"] = str(REPO / "reproducibility" / "metadata")
    os.environ["DFV_GEN"] = str(gen)
    os.environ["DFV_FIGURES"] = str(figures)
    os.environ["R2_GEN"] = str(gen)
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(ROUND1 / "analysis"))
    import revlib as R
    R.gate()
    import paper_tables as T
    for generate in (T.tab_roster, T.tab_operating, T.tab_tests):
        generate()
    import supp_tables as S
    for generate in (S.s_thresholds, S.s_interval, S.s_evalthr, S.s_policies, S.s_ttg, S.s_band,
                     S.s_h2, S.s_speakers, S.s_h5, S.s_h14, S.s_df21, S.s_silence, S.s_silence_frr,
                     S.s_trim, S.s_cheap, S.s_cost, S.s_costsplit):
        generate()
    mlaad_table(gen)
    for name, changes in TABLE_EDITS.items():
        path = gen / name
        data = path.read_bytes()
        for old, new in changes:
            if old not in data:
                raise SystemExit(f"REFUSED: documented table edit source missing in {name}")
            data = data.replace(old, new)
        path.write_bytes(data)
    sys.path.insert(0, str(REPO / "analysis" / "round2"))
    module("speechcomm_round2_tables", REPO / "analysis" / "round2" / "r2_paper_tables.py").main()
    import figures_rev as F
    for generate in (F.fig_sensitivity, F.fig_frr_far, F.figS_h5, F.figS_overview,
                     F.figS_speakers, F.figS_cost, F.figS_silence):
        generate()
    for name in FIGURES:
        path = figures / name
        if not path.is_file() or path.stat().st_size < 1000 or not path.read_bytes().startswith(b"%PDF-"):
            raise SystemExit(f"REFUSED: figure was not generated: {name}")
    return gen, figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--verify", action="store_true", help="temporary regeneration and verification (default)")
    action.add_argument("--write", action="store_true", help="replace generated manuscript tables and figures")
    args = parser.parse_args()
    verify_inputs()
    with tempfile.TemporaryDirectory(prefix="deploybench-paper-") as temporary:
        gen, figures = regenerate(Path(temporary))
        expected = {p.name for p in (PAPER / "gen").glob("*.tex")}
        generated = {p.name for p in gen.glob("*.tex")}
        if expected != generated:
            raise SystemExit(f"REFUSED: table set differs; missing={sorted(expected-generated)}, extra={sorted(generated-expected)}")
        if args.write:
            for source in sorted(gen.glob("*.tex")):
                shutil.copyfile(source, PAPER / "gen" / source.name)
            for name in FIGURES:
                shutil.copyfile(figures / name, PAPER / "figures" / name)
            print(f"Regenerated {len(generated)} manuscript tables and {len(FIGURES)} figures.")
        else:
            different = [name for name in sorted(generated) if
                         (gen / name).read_bytes().replace(b"\r\n", b"\n") !=
                         (PAPER / "gen" / name).read_bytes().replace(b"\r\n", b"\n")]
            if different:
                raise SystemExit(f"REFUSED: generated tables differ from manuscript: {', '.join(different)}")
            print(f"VERIFIED: all {len(generated)} manuscript tables match; all {len(FIGURES)} figures regenerate.")


if __name__ == "__main__":
    main()
