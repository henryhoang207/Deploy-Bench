"""Build the frozen MLAAD-5 protocol (D47 / DEVIATIONS.md DEV-6).

Spoof side: every MLAAD clip under <mlaad>/fake/<lang>/<system>/*.wav for the five
languages. Bona fide side: per language, a uniform draw without replacement from the
M-AILABS wavs of that language's locales, as many clips as the language has spoof clips
(all of them, and a reported imbalance, if M-AILABS has fewer). One generator per language,
numpy default_rng([seed, language index]), so adding or dropping a language never changes
another language's draw.

    python scripts/build_mlaad_protocol.py --mlaad-dir $DATA_ROOT/mlaad \
        --mailabs-dir $DATA_ROOT/mailabs --mlaad-revision <hf commit>

Writes <DATA_ROOT>/mlaad5/protocol.csv (utt_id,path,label,lang,system; path relative to
DATA_ROOT; label 1 = bona fide) and protocol_summary.json next to it. Refuses to overwrite:
the protocol is frozen before any scoring.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deploybench.manifests import file_sha256  # noqa: E402

LANGS = ("en", "de", "it", "fr", "pl")                       # audit_plan.md, DEV-6
LOCALES = {"en": ("en_US", "en_UK"), "de": ("de_DE",), "it": ("it_IT",),
           "fr": ("fr_FR",), "pl": ("pl_PL",)}
SEED = 20261003


def utt_id(prefix: str, rel: Path) -> str:
    """Unique by construction (the relative path), and safe for the id manifests,
    which split on whitespace and treat '#' as a comment."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", f"{prefix}__{rel.with_suffix('').as_posix()}")


def build(root: Path, mlaad: Path, mailabs: Path, seed: int = SEED, langs=LANGS):
    rows, summary = [], {"seed": seed, "languages": {}}
    for lang in langs:
        li = LANGS.index(lang)          # fixed per language, whatever subset runs
        fake = sorted((mlaad / "fake" / lang).glob("*/*.wav"))
        if not fake:
            raise SystemExit(f"no MLAAD spoof clips under {mlaad / 'fake' / lang}")
        pool = sorted(p for loc in LOCALES[lang] for p in (mailabs / loc).rglob("*.wav"))
        if not pool:
            raise SystemExit(f"no M-AILABS wavs for {lang} under {mailabs} {LOCALES[lang]}")
        n = min(len(fake), len(pool))
        pick = np.random.default_rng([seed, li]).choice(len(pool), size=n, replace=False)
        bona = [pool[i] for i in sorted(pick)]
        for p in fake:
            rows.append((utt_id("mlaad", p.relative_to(mlaad)), p, 0, lang, p.parent.name))
        for p in bona:
            rows.append((utt_id("mailabs", p.relative_to(mailabs)), p, 1, lang, "bonafide"))
        summary["languages"][lang] = {
            "spoof": len(fake), "bonafide": n, "bonafide_pool": len(pool),
            "systems": len({p.parent.name for p in fake}),
            "imbalanced": n < len(fake)}
    ids = [r[0] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("utt_id collision after sanitising; refusing")
    out = [(u, p.relative_to(root).as_posix(), lab, lang, sysname)
           for u, p, lab, lang, sysname in rows]
    return out, summary


def main():
    import os
    ap = argparse.ArgumentParser()
    root0 = Path(os.environ.get("DATA_ROOT", "~/data/deepfake")).expanduser()
    ap.add_argument("--root", default=str(root0))
    ap.add_argument("--mlaad-dir", required=True)
    ap.add_argument("--mailabs-dir", required=True)
    ap.add_argument("--mlaad-revision", required=True, help="HF commit of the snapshot")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    root = Path(a.root).resolve()
    out_dir = root / "mlaad5"
    proto = out_dir / "protocol.csv"
    if proto.exists():
        raise SystemExit(f"REFUSING: {proto} exists (the protocol is frozen)")
    rows, summary = build(root, Path(a.mlaad_dir).resolve(), Path(a.mailabs_dir).resolve(), a.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(proto, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["utt_id", "path", "label", "lang", "system"])
        w.writerows(rows)
    summary.update({"mlaad_revision": a.mlaad_revision, "rows": len(rows),
                    "protocol_sha256": file_sha256(proto)})
    sums = Path(a.mailabs_dir) / "SHA256SUMS"
    if sums.is_file():
        summary["mailabs_archives_sha256"] = sums.read_text().split("\n")
    (out_dir / "protocol_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
