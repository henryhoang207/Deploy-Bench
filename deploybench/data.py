"""Dataset registry + loaders. SKELETON: expected layouts below are from
public documentation; step 0.3 'Done when' adjusts them to the real
downloaded trees (run `python -m deploybench.data --check`)."""
from __future__ import annotations
import os, re, sys
import gzip
from pathlib import Path

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "~/data/deepfake")).expanduser()

# name -> (subdir, protocol file hint, expected approx utterance count)
REGISTRY = {
    "asvspoof19_train": ("LA/ASVspoof2019_LA_train", "ASVspoof2019.LA.cm.train.trn.txt", 25380),
    "asvspoof19_dev":   ("LA/ASVspoof2019_LA_dev",   "ASVspoof2019.LA.cm.dev.trl.txt",   24844),
    "asvspoof19_eval":  ("LA/ASVspoof2019_LA_eval",  "ASVspoof2019.LA.cm.eval.trl.txt",  71237),
    # Both DF21 surfaces read the same package; they differ only in which FLAC
    # integrity classes are excluded (D34, see DF21_SURFACES). Count = flac files.
    "asvspoof21_df_eval": ("ASVspoof2021_DF_eval", "trial_metadata.txt", 611829),
    "asvspoof21_df_eval_clean": ("ASVspoof2021_DF_eval", "trial_metadata.txt", 611829),
    # ASVspoof 5 (round 2, D54): Zenodo record 14498691, track-1 protocols. Audio: asvspoof5/flac_D, flac_E.
    # Train is registered but not loadable: nothing trains on AS5 here, and it was not downloaded.
    "asvspoof5_train":  ("asvspoof5/flac_T", "ASVspoof5.train.tsv", 182357),
    "asvspoof5_dev":    ("asvspoof5/flac_D", "ASVspoof5.dev.track_1.tsv", 140950),
    "asvspoof5_eval":   ("asvspoof5/flac_E", "ASVspoof5.eval.track_1.tsv", 680774),
    "itw":              ("release_in_the_wild", "meta.csv", 31779),
    # MLAAD-5 (D47/DEV-6): the frozen protocol from scripts/build_mlaad_protocol.py
    # (MLAAD spoof + count-matched M-AILABS bona fide), not MLAAD's own meta files.
    "mlaad":            ("mlaad5", "protocol.csv", None),
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
    elif name.startswith("asvspoof21_df_eval"):
        c += [base / proto, base / "keys" / "DF" / "CM" / proto,
              root / "keys" / "DF" / "CM" / proto,
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


# ASVspoof 2021 DF CM keys (`DF/CM/trial_metadata.txt`): 13 space-separated
# columns, names verbatim from asvspoof-challenge/2021 @9b33f5ec
# eval-package/config.py `p_names` + README. `subset` is eval|progress|hidden.
# Published DF21 numbers (B04 RawNet2 22.38%, Tak et al. 2.85%) pool the EVAL
# subset only, so that is all this loader yields (B4). `compr` (nocodec,
# low_mp3, ... = DF-C1..C9) is a property of the data, surfaced through
# native_conditions() into the score file's `condition` column.
DF21_COLS = ("speaker", "trial", "compr", "source", "attack", "label", "trim",
             "subset", "vocoder", "task", "team", "gender-pair", "language")
DF21_SUBSETS = ("eval", "progress", "hidden")
DF21_SUBSET = "eval"

# FLAC integrity classes of the public DF21 eval release (D34; evidence in
# phase_1_reproduction/df21_flac_integrity.md). 43% of eval trials are rejected
# by libFLAC >= 1.4. Class A: libFLAC 1.3.4 / ffmpeg decode verifies against the
# file's STREAMINFO MD5 (true audio). Class B: that decode FAILS the MD5 (audio is
# not the original), so it is excluded everywhere. Surface -> excluded classes:
DF21_SURFACES = {"asvspoof21_df_eval": ("B",),            # verified audio, 455,508
                 "asvspoof21_df_eval_clean": ("A", "B")}  # clean only, 305,525
DF21_CLASS_DIR = Path(__file__).resolve().parent.parent / "configs"


def _df21_class_ids(cls: str) -> set:
    """IDs of one integrity class from configs/df21_class<cls>.txt. The file's
    recorded sha256 is re-checked: a truncated or edited list would silently
    change the eval surface."""
    import hashlib
    p = DF21_CLASS_DIR / f"df21_class{cls}.txt"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: the DF21 surfaces are undefined "
                                "without it (D34). Refusing to guess.")
    lines = open(p, encoding="utf-8").read().splitlines()
    want = next((l.rsplit(":", 1)[1].strip() for l in lines
                 if l.startswith("# sha256")), None)
    ids = sorted(l.strip() for l in lines if l.strip() and not l.startswith("#"))
    got = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    if want != got:
        raise ValueError(f"{p}: recorded sha256 {want} != content {got}")
    return set(ids)


def _df21_rows(root: Path, dataset: str = "asvspoof21_df_eval"):
    """Validated eval-subset rows of the DF21 keys for one surface (D34).
    Malformed lines RAISE: a skipped or misparsed trial would silently change
    a pooled EER."""
    excluded = set().union(*(_df21_class_ids(c) for c in DF21_SURFACES[dataset]))
    sub, proto, _ = REGISTRY[dataset]
    base = root / sub
    meta = _find(base / proto, base / "keys" / "DF" / "CM" / proto,
                 root / "keys" / "DF" / "CM" / proto,
                 root / "DF-keys-full" / "CM" / proto)
    for n, line in enumerate(open(meta), 1):
        f = line.split()
        if not f:
            continue
        if len(f) != len(DF21_COLS):
            raise ValueError(f"{meta}:{n}: expected {len(DF21_COLS)} fields "
                             f"{DF21_COLS}, got {len(f)}: {line.strip()!r}")
        r = dict(zip(DF21_COLS, f))
        if r["label"] not in ("bonafide", "spoof"):
            raise ValueError(f"{meta}:{n}: label {r['label']!r} is neither "
                             "'bonafide' nor 'spoof'")
        if r["subset"] not in DF21_SUBSETS:
            raise ValueError(f"{meta}:{n}: subset {r['subset']!r} not in "
                             f"{DF21_SUBSETS}")
        if r["subset"] == DF21_SUBSET and r["trial"] not in excluded:
            yield base, r


def _iter_asvspoof21_df(root: Path, dataset: str):
    """EVAL subset of one D34 surface; label from the `label` column.
    Audio: flac/<trial>.flac."""
    for base, r in _df21_rows(root, dataset):
        yield (r["trial"], base / "flac" / f"{r['trial']}.flac",
               int(r["label"] == "bonafide"))


def native_conditions(dataset: str, root: Path = None):
    """{utt_id: condition} for datasets whose condition is a property of the
    DATA (DF21: `compr`), else None. score.py / score_onnx.py write it into the
    `condition` column so the per-codec breakdown survives (B4)."""
    root = Path(root) if root else DATA_ROOT
    if dataset in DF21_SURFACES:
        return {r["trial"]: r["compr"] for _, r in _df21_rows(root, dataset)}
    if dataset == "asvspoof5_eval":           # D24: AS5 eval is reported by codec condition
        return {r["utt"]: asvspoof5_condition(r) for r in asvspoof5_rows("eval", root)}
    return None


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
            # Refuse, never default: an unknown or empty label used to become
            # "spoof" silently (the B4 pattern). meta.csv has only these two.
            if lab not in ("bona-fide", "bonafide", "spoof") or not fn:
                raise ValueError(f"{base / proto}: unusable row {row!r}")
            yield Path(fn).stem, base / fn, int(lab != "spoof")


def _iter_mlaad(root: Path):
    """The frozen MLAAD-5 protocol (D47/DEV-6): utt_id,path,label,lang,system with
    path relative to DATA_ROOT and label 1 = bona fide. Strict: the old tolerant
    reader defaulted every unlabeled row to spoof and skipped path-less rows, the
    silent-default pattern this harness refuses."""
    import csv as _csv
    sub, proto, _ = REGISTRY["mlaad"]
    with open(root / sub / proto, newline="", encoding="utf-8") as f:
        rd = _csv.DictReader(f)
        if rd.fieldnames != ["utt_id", "path", "label", "lang", "system"]:
            raise ValueError(f"{root / sub / proto}: unexpected header {rd.fieldnames}")
        for row in rd:
            if row["label"] not in ("0", "1") or not row["path"] or not row["utt_id"]:
                raise ValueError(f"{root / sub / proto}: unusable row {row!r}")
            yield row["utt_id"], root / row["path"], int(row["label"])


# ASVspoof 5 track-1 protocol (Zenodo 14498691, ASVspoof5_protocols.tar.gz, md5 865d0e89...): 10 space-separated
# columns, SPEAKER_ID FLAC_FILE_NAME SPEAKER_GENDER CODEC CODEC_Q CODEC_SEED ATTACK_TAG ATTACK_LABEL KEY TMP
# (read on the release 2026-10-07: dev 140,950 rows, eval 680,774). KEY is bonafide|spoof. CODEC is "-" when no
# codec was applied (condition C00, the only channel-matched eval subset, D24) and C01..C11 otherwise. Dev and eval
# were post-processed by the organisers (peak normalisation, random trimming of leading/trailing non-speech,
# random 4-10 s chunk); train was not (ASVspoof 5 database paper, Sec. 4.2).
AS5_COLS = ("speaker", "utt", "gender", "codec", "codec_q", "codec_seed", "attack_tag", "attack", "key", "tmp")


def asvspoof5_rows(part: str, root: Path = None, *, protocol: Path = None):
    """Validated rows of one ASVspoof 5 track-1 protocol, as dicts keyed by AS5_COLS. Refuses a wrong column
    count, an unknown KEY or CODEC, and a row count other than the release's.
    An explicit protocol path supports analysis of bundled .tsv[.gz] metadata
    without a waveform directory; the default dataset layout is unchanged."""
    if part not in ("dev", "eval"):
        raise NotImplementedError(f"asvspoof5_{part}: only dev and eval are used (D54); "
                                  "nothing trains on ASVspoof 5 here and train was not downloaded")
    root = Path(root) if root else DATA_ROOT
    sub, proto, n_expected = REGISTRY[f"asvspoof5_{part}"]
    path = Path(protocol) if protocol is not None else (root / sub).parent / proto
    rows = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, mode="rt", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            p = line.split()
            if len(p) != len(AS5_COLS):
                raise ValueError(f"{path}:{i}: expected {len(AS5_COLS)} columns, got {len(p)}")
            r = dict(zip(AS5_COLS, p))
            if r["key"] not in ("bonafide", "spoof"):
                raise ValueError(f"{path}:{i}: unknown KEY {r['key']!r}")
            if r["codec"] != "-" and not re.fullmatch(r"C\d\d", r["codec"]):
                raise ValueError(f"{path}:{i}: unknown CODEC {r['codec']!r}")
            rows.append(r)
    if len(rows) != n_expected:
        raise ValueError(f"{path}: {len(rows)} rows, expected {n_expected}")
    return rows


def asvspoof5_condition(row) -> str:
    """C00 for no codec, else the protocol's codec ID (C01..C11)."""
    return "C00" if row["codec"] == "-" else row["codec"]


def _iter_asvspoof5(part: str, root: Path):
    sub, _, _ = REGISTRY[f"asvspoof5_{part}"]
    base = root / sub
    for r in asvspoof5_rows(part, root):
        yield r["utt"], base / f"{r['utt']}.flac", int(r["key"] == "bonafide")


def protocol_iter(dataset: str, root: Path = None):
    """Yield (utt_id, wav_path, label) with label 1 = bona fide, 0 = spoof
    (project-wide convention, matches wrapper score contract), -1 = unlabeled.
    Layout assumptions from official packages; `--check` validates paths."""
    root = Path(root) if root else DATA_ROOT
    if dataset.startswith("asvspoof19_"):
        yield from _iter_asvspoof19(dataset.split("_")[1], root)
    elif dataset in DF21_SURFACES:
        yield from _iter_asvspoof21_df(root, dataset)
    elif dataset == "itw":
        yield from _iter_itw(root)
    elif dataset == "mlaad":
        yield from _iter_mlaad(root)
    elif dataset in ("asvspoof5_dev", "asvspoof5_eval"):
        # Round 2 (D54, PREREGISTRATION_ROUND2.md): the ASVspoof 5-trained detector's source dev and eval. D24
        # applies to AS5 eval: shifted target, reported by condition C00-C11, C00 the only channel-matched subset.
        yield from _iter_asvspoof5(dataset.split("_")[1], root)
    elif dataset == "asvspoof5_train":
        raise NotImplementedError("asvspoof5_train is not used (D54: nothing trains on ASVspoof 5 here) "
                                  "and was not downloaded")
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
