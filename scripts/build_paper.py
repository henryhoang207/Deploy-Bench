"""Build the article and supplement with pdflatex/bibtex, without requiring Perl."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=REPO / "paper" / "speechcomm")
    parser.add_argument("--output", type=Path, default=REPO / "paper" / "speechcomm" / "output" / "pdf")
    args = parser.parse_args()
    source, output = args.source_dir.resolve(), args.output.resolve()
    tools = {name: shutil.which(name) for name in ("pdflatex", "bibtex")}
    if not all(tools.values()):
        parser.error("Install a LaTeX distribution providing pdflatex and bibtex, including elsarticle.")
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["BIBINPUTS"] = str(source) + os.pathsep + env.get("BIBINPUTS", "")
    for stem in ("main", "supplement"):
        if not (source / f"{stem}.tex").is_file():
            parser.error(f"Missing source: {source / (stem + '.tex')}")
        tex = [tools["pdflatex"], "-interaction=nonstopmode", "-halt-on-error",
               "-output-directory=" + str(output), stem + ".tex"]
        log = output / f"{stem}-build.txt"
        with log.open("w", encoding="utf-8") as stream:
            for command, cwd in ((tex, source), ([tools["bibtex"], stem], output),
                                 (tex, source), (tex, source)):
                result = subprocess.run(command, cwd=cwd, env=env,
                                        stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode:
                    stream.flush()
                    print("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-35:]))
                    raise SystemExit(f"Build failed; details: {log}")
        pdf = output / f"{stem}.pdf"
        if not pdf.is_file() or not pdf.read_bytes().startswith(b"%PDF-"):
            raise SystemExit(f"Expected PDF was not produced: {pdf}")
        print(f"Built {pdf}", flush=True)


if __name__ == "__main__":
    main()
