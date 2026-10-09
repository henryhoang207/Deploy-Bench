"""Render existing round-1 fp32 MLAAD findings; never score or read round-2 data."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "results/phase_2_conversion_app/mlaad5/mlaad_report.json"
report = json.loads(SOURCE.read_text(encoding="utf-8"))["models"]
names = {"aasist": "AASIST", "aasist_l": "AASIST-L", "rawnet2": "RawNet2",
         "tssdnet": "Res-TSSDNet", "xlsr_aasist": "XLS-R+AASIST (ref.)"}
lines = ["% src: results/phase_2_conversion_app/mlaad5/mlaad_report.json (models.*.fp32 only).",
         r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
         r"Model & EER & FRR & FAR & HTER & TTG \\", r"\midrule"]
for model, name in names.items():
    row = report[model]["fp32"]
    point = row["at_eer"]
    values = [row["eer"], point["frr"], point["far"], point["hter"], row["ttg_eer"]]
    lines.append(name + " & " + " & ".join(f"{100 * x:.2f}" for x in values) + r" \\")
lines.extend([r"\bottomrule", r"\end{tabular}"])
(HERE / "gen/r1_mlaad.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
