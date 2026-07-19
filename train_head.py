"""DEPRECATED (2026-07-19): superseded by train/train_heads.py.

This script trained a bare Linear/MLP head — NOT the MHFA backend the roster
specifies (reference_architectures.md #8, Peng et al. 2210.01273) — and saved a
checkpoint format models/wrap_wavlm.py cannot load. Kept as a tombstone because
the file is referenced in older docs; do not use.
"""
raise SystemExit("Deprecated: use train/train_heads.py (see docstring).")
