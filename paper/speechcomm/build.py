"""Build both local manuscripts; refuse failed tools, unresolved citations or clipped tables."""
from pathlib import Path
import argparse
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--audit", action="store_true", help="also run the source and number audit")
parser.add_argument("--final", action="store_true", help="also require a completed numerical audit")
args = parser.parse_args()

highlights = (HERE / "highlights.txt").read_text(encoding="utf-8").splitlines()
if not 3 <= len(highlights) <= 5 or any(not s or len(s) > 85 for s in highlights):
    sys.exit("REFUSED: highlights require three to five nonempty lines, each at most 85 characters")

for stem in ("main", "supplement"):
    for command in (["pdflatex", "-interaction=nonstopmode", "-halt-on-error", stem + ".tex"],
                    ["bibtex", stem],
                    ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", stem + ".tex"],
                    ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", stem + ".tex"]):
        run = subprocess.run(command, cwd=HERE, capture_output=True, text=True, errors="replace")
        if run.returncode:
            print(run.stdout[-4000:] + run.stderr[-1000:])
            sys.exit(f"REFUSED: {command[0]} failed for {stem}")
    log = (HERE / (stem + ".log")).read_text(encoding="utf-8", errors="replace")
    pages = re.search(r"Output written on .*? \((\d+) pages?", log)
    if not pages or "undefined" in log or "multiply defined" in log:
        sys.exit(f"REFUSED: missing PDF or unresolved references in {stem}.log")
    overfull = re.findall(r"Overfull \\hbox \(([^)]+)\)", log)
    print(f"{stem}.pdf: {pages.group(1)} pages; {len(overfull)} overfull boxes")
    if overfull:
        print("  " + ", ".join(overfull))
        sys.exit(f"REFUSED: content extends beyond the text width in {stem}.log")

print("Highlights: " + ", ".join(str(len(s)) for s in highlights) + " characters")
if args.audit or args.final:
    command = [sys.executable, str(HERE / "audit_claims.py")]
    if args.final:
        command.append("--final")
    sys.exit(subprocess.run(command, cwd=HERE.parents[1]).returncode)
