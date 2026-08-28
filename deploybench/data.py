"""Dataset registry + loaders. SKELETON: expected layouts below are from
public documentation; step 0.3 'Done when' adjusts them to the real
downloaded trees (run `python -m deploybench.data --check`)."""
from __future__ import annotations
import os, sys
from pathlib import Path

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "~/data/deepfake")).expanduser()

# name -> (subdir, protocol file hint, expected approx utterance count)
REGISTRY = {
    "asvspoof19_train": ("LA/ASVspoof2019_LA_train", "ASVspoof2019.LA.cm.train.trn.txt", 25380),
    "asvspoof19_dev":   ("LA/ASVspoof2019_LA_dev",   "ASVspoof2019.LA.cm.dev.trl.txt",   24844),
    "asvspoof19_eval":  ("LA/ASVspoof2019_LA_eval",  "ASVspoof2019.LA.cm.eval.trl.txt",  71237),
    "asvspoof21_df_eval": ("ASVspoof2021_DF_eval", "trial_metadata.txt", 611829),
    "asvspoof5_train":  ("ASVspoof5/train", None, None),
    "asvspoof5_dev":    ("ASVspoof5/dev", None, None),
    "asvspoof5_eval":   ("ASVspoof5/eval", None, None),
    "itw":              ("release_in_the_wild", "meta.csv", 31779),
    "mlaad":            ("mlaad", "meta.csv", None),
    "musan":            ("musan", None, None),
}


# Protocol-directory candidates for ASVspoof2019 LA, relative to DATA_ROOT.
# STILL LAYOUT GUESSES from the official package docs — step 0.3 replaces this
# tuple with the user's real tree (see `--check` output). Single source so
# check() and _iter_asvspoof19 can never disagree about where protocols live.
ASVSPOOF19_PROTOCOL_DIRS = (
    "LA/ASVspoof2019_LA_cm_protocols",
    "ASVspoof2019_LA_cm_protocols",
)

_AUDIO_EXT = {".flac", ".wav"}


def _count_audio(d: Path, cap: int = 1_000_000) -> int:
    """Recursive audio-file count; -1 if the directory does not exist."""
    if not d.exists():
        return -1
    n = 0
    for p in d.rglob("*"):
        if p.suffix.lower() in _AUDIO_EXT:
            n += 1
            if n >= cap:
                break
    return n


def _proto_candidates(name: str, root: Path):
    sub, proto, _ = REGISTRY[name]
    base = root / sub
    c = []
    if name.startswith("asvspoof19_"):
        c += [root / d / proto for d in ASVSPOOF19_PROTOCOL_DIRS]
        c += [base.parent / "ASVspoof2019_LA_cm_protocols" / proto]
    elif name == "asvspoof21_df_eval":
        c += [base / proto, base / "keys" / "DF" / "CM" / proto,
              root / "DF-keys-full" / "CM" / proto]
    return c + [base / proto, base.parent / proto]


def check(root: Path = DATA_ROOT, only=None) -> int:
    """Step 0.3 'done when': per dataset, print the resolved audio dir + file
    count (vs expected) and the resolved protocol path + line count. Reports
    rather than raises — this runs BEFORE the layout is known to be right.
    Returns the number of datasets whose audio directory is missing."""
    print(f"DATA_ROOT = {root}   (exists={root.exists()})\n")
    missing = 0
    for name, (sub, proto, expect) in REGISTRY.items():
        if only and name not in only:
            continue
        p = root / sub
        n_audio = _count_audio(p)
        missing += (n_audio < 0)
        exp = f"   expected~{expect}" if expect else ""
        print(f"[{'MISS' if n_audio < 0 else 'OK  '}] {name:22s} {p}")
        print(f"       audio_files={n_audio if n_audio >= 0 else 'n/a'}{exp}")
        if not proto:
            continue
        hit = next((c for c in _proto_candidates(name, root) if c.exists()), None)
        if hit is None:
            print("       protocol=MISS  tried: " + " | ".join(
                str(c) for c in _proto_candidates(name, root)))
        else:
            try:
                nl = sum(1 for _ in open(hit, encoding="utf-8", errors="replace"))
            except OSError as e:                                  # noqa: BLE001
                nl = f"unreadable ({e})"
            print(f"       protocol={hit}   lines={nl}")
    return missing


def _find(*candidates):
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("none of these exist: "
                            + " | ".join(str(c) for c in candidates))


def _iter_asvspoof19(part: str, root: Path):
    """ASVspoof2019.LA.cm.<part> protocol: `spk utt - attack label`,
    label bonafide/spoof -> 1/0. Audio: <partdir>/flac/<utt>.flac."""
    sub, proto, _ = REGISTRY[f"asvspoof19_{part}"]
    part_dir = root / sub
    proto_dir = _find(*[root / d for d in ASVSPOOF19_PROTOCOL_DIRS],
                      part_dir.parent / "ASVspoof2019_LA_cm_protocols")
    for line in open(proto_dir / proto):
        f = line.split()
        if len(f) < 5:
            continue
        utt, label = f[1], (1 if f[4] == "bonafide" else 0)
        yield utt, part_dir / "flac" / f"{utt}.flac", label


def _iter_asvspoof21_df(root: Path):
    """DF eval keys trial_metadata.txt (keys package): whitespace fields,
    utt = field 2; label = presence of 'bonafide'. Audio: flac/<utt>.flac."""
    sub, proto, _ = REGISTRY["asvspoof21_df_eval"]
    base = root / sub
    meta = _find(base / proto, base / "keys" / "DF" / "CM" / proto,
                 root / "DF-keys-full" / "CM" / proto)
    for line in open(meta):
        f = line.split()
        if len(f) < 2:
            continue
        utt = f[1]
        yield utt, base / "flac" / f"{utt}.flac", int("bonafide" in f)


def _iter_itw(root: Path):
    """In-the-Wild meta.csv: file,speaker,label with label bona-fide|spoof.
    Audio: <root>/release_in_the_wild/<file> (files are `N.wav`)."""
    import csv as _csv
    sub, proto, _ = REGISTRY["itw"]
    base = root / sub
    with open(base / proto, newline="") as f:
        for row in _csv.DictReader(f):
            fn = row.get("file") or row.get("filename")
            lab = (row.get("label") or "").strip().lower()
            yield Path(fn).stem, base / fn, int(lab in ("bona-fide", "bonafide"))


def _iter_mlaad(root: Path):
    """MLAAD meta.csv: tolerant reader — needs a path-like column; MLAAD is
    all-spoof (label 0) unless an explicit label column says otherwise."""
    import csv as _csv
    sub, proto, _ = REGISTRY["mlaad"]
    base = root / sub
    with open(base / proto, newline="") as f:
        sniff = _csv.Sniffer().sniff(f.read(4096), delimiters=",;|\t")
        f.seek(0)
        for row in _csv.DictReader(f, dialect=sniff):
            fn = row.get("path") or row.get("file") or row.get("filename")
            if not fn:
                continue
            lab = (row.get("label") or row.get("key") or "spoof").strip().lower()
            yield Path(fn).stem, base / fn, int(lab in ("bona-fide", "bonafide"))


def protocol_iter(dataset: str, root: Path = None):
    """Yield (utt_id, wav_path, label) with label 1 = bona fide, 0 = spoof
    (project-wide convention, matches wrapper score contract), -1 = unlabeled.
    Layout assumptions from official packages; `--check` validates paths."""
    root = Path(root) if root else DATA_ROOT
    if dataset.startswith("asvspoof19_"):
        yield from _iter_asvspoof19(dataset.split("_")[1], root)
    elif dataset == "asvspoof21_df_eval":
        yield from _iter_asvspoof21_df(root)
    elif dataset == "itw":
        yield from _iter_itw(root)
    elif dataset == "mlaad":
        yield from _iter_mlaad(root)
    elif dataset.startswith("asvspoof5_"):
        raise NotImplementedError(
            "ASVspoof5 is OUT OF SCOPE as of decision D30 (2026-07-29). The "
            "roster froze at 5 models, all Cohort-19, so every tau calibrates "
            "on ASVspoof19 LA dev and nothing trains on AS5. This is now a "
            "DECISION, not the deferred-implementation gap it used to be "
            "(D15). Re-adding AS5 as a transfer target is an open call in D30 "
            "and would be a logged deviation from PREREGISTRATION.md — if you "
            "do it, D24 applies unchanged (shifted target, report split by "
            "condition C00-C11, C00 only channel-matched, resolve the "
            "silence-trim mismatch first) and you add a TSV parser here "
            "mirroring _iter_asvspoof19.")
    else:
        raise KeyError(f"unknown dataset {dataset!r}; known: {list(REGISTRY)}")


if __name__ == "__main__":
    if "--check" in sys.argv:
        # optional: --datasets a,b   --root PATH
        only = root = None
        for i, a in enumerate(sys.argv):
            if a == "--datasets" and i + 1 < len(sys.argv):
                only = set(sys.argv[i + 1].split(","))
            if a == "--root" and i + 1 < len(sys.argv):
                root = Path(sys.argv[i + 1]).expanduser()
        sys.exit(1 if check(root or DATA_ROOT, only) else 0)
