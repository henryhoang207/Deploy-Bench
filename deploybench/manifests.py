"""Deterministic utt_id manifests (parity, PTQ calibration, D39 diagnostic subset).

One drawing rule for every manifest in the project, so a manifest can be
regenerated from its header and checked byte-for-byte. The rule is the one
scripts/make_parity_manifest.py has used since 2026-07-26: sort each class
first (the draw must not depend on protocol line order), draw without
replacement with numpy's default_rng(seed), then write bona fide ids sorted,
followed by spoof ids sorted.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def draw_balanced(bona, spoof, n_per_class: int, seed: int) -> list[str]:
    """n_per_class bona fide + n_per_class spoof ids, bona fide first."""
    bona, spoof = sorted(bona), sorted(spoof)
    if len(bona) < n_per_class or len(spoof) < n_per_class:
        raise ValueError(f"pool has {len(bona)} bona fide / {len(spoof)} spoof; "
                         f"cannot draw {n_per_class} of each")
    rng = np.random.default_rng(seed)
    pick_b = rng.choice(bona, n_per_class, replace=False)
    pick_s = rng.choice(spoof, n_per_class, replace=False)
    return sorted(pick_b.tolist()) + sorted(pick_s.tolist())


def ids_digest(ids) -> str:
    """sha256 of the ids joined by newlines: independent of header text."""
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def write_manifest(path: Path, ids, header_lines) -> str:
    """Write '# ...' header lines then one id per line. Returns ids_digest."""
    digest = ids_digest(ids)
    lines = [h if h.startswith("#") else f"# {h}" for h in header_lines]
    lines.append(f"# sha256    : {digest}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines + list(ids)) + "\n", encoding="utf-8")
    return digest


def read_ids(path: Path) -> list[str]:
    """Ids in file order; '#' comments ignored (same parse as score.read_manifest)."""
    ids = []
    for line in open(path, encoding="utf-8"):
        line = line.split("#")[0].strip()
        if line:
            ids.append(line.split()[0])
    return ids


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
