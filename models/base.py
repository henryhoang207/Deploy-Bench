"""Model wrapper interface + registry.

Each roster model gets a wrapper exposing ONE interface so score.py, the
conversion factory, and the parity suite never special-case models.

Vendored source expectation: clone the official repos into `vendor/`
(vendor/aasist, vendor/asvspoof2021_baselines, vendor/RawGAT-ST-antispoofing,
vendor/end-to-end-synthetic-speech-detection, vendor/SSL_Anti-spoofing).
Wrappers sys.path-inject the vendor dir. Entry points marked TODO-VERIFY were
written from public repo knowledge and need one check against the real clone.
"""
from __future__ import annotations
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

VENDOR = Path(__file__).resolve().parent.parent / "vendor"


@dataclass
class ModelSpec:
    name: str
    cohort: str            # "cohort19" for every in-roster model (D30 collapsed
                           # the two-cohort design). Kept as a field because the
                           # out-of-roster wrappers below still carry a label.
    input_samples: int     # fixed input length @16 kHz
    feature: str           # "raw" | "lfcc"
    params_approx: str
    wrapper: str           # module path under models/
    in_roster: bool = True  # False = kept for harness generality, NOT studied


# ---------------------------------------------------------------- ROSTER (D30)
# Frozen 2026-07-29 at FIVE models. Selection was by int8 failure MECHANISM:
#   graph attention (aasist, aasist_l) | recurrence (rawnet2)
#   plain conv (tssdnet)               | transformer (xlsr_aasist, UNRANKED)
# Every one is Cohort-19, so every tau is calibrated on ASVspoof19 LA dev.
# The three out-of-roster wrappers are DELIBERATELY RETAINED: they are audited
# against author source and guarded by tests/test_wrapper_contracts.py, and
# harness generality is part of the released artifact. Do not delete them, and
# do not re-add them to a grid without a new decision-log entry.
SPECS = {
    "aasist":      ModelSpec("aasist", "cohort19", 64600, "raw", "297K", "models.wrap_aasist"),
    "aasist_l":    ModelSpec("aasist_l", "cohort19", 64600, "raw", "85K", "models.wrap_aasist"),  # within-family replication for H6 — the only within-class control in the study
    "rawnet2":     ModelSpec("rawnet2", "cohort19", 64600, "raw", "17.6M", "models.wrap_rawnet2"),  # GRU = 16.1M of 17.6M params; int8 RNN fallback is the H6 headline here
    "tssdnet":     ModelSpec("tssdnet", "cohort19", 96000, "raw", "350K", "models.wrap_tssdnet"),  # 6 s: authors eval on 6-s clips; net hardcodes maxpool k=375 (=96000/4^4)
    "xlsr_aasist": ModelSpec("xlsr_aasist", "cohort19", 64600, "raw", "315M", "models.wrap_xlsr"),  # tau: asvspoof19_dev (D16) — ckpt fine-tuned on 19LA. UNRANKED server ref; enters no ranking claim; no published 19LA-eval anchor so reproduction is UNGATED
    "dummy":       ModelSpec("dummy", "cohort19", 64600, "raw", "0", "models.wrap_dummy"),
    # ---- OUT OF ROSTER (D30) — wrappers kept, models not studied ----
    "rawgat_st":   ModelSpec("rawgat_st", "cohort19", 64600, "raw", "437K", "models.wrap_rawgat", in_roster=False),
    "lcnn_lfcc":   ModelSpec("lcnn_lfcc", "cohort19", 64600, "lfcc", "~0.4M", "models.wrap_lcnn", in_roster=False),
    "wavlm_trunc": ModelSpec("wavlm_trunc", "cohort_ssl", 64600, "raw", "~51.9M", "models.wrap_wavlm", in_roster=False),
    "wavlm_mhfa":  ModelSpec("wavlm_mhfa", "cohort_ssl", 64600, "raw", "~97M", "models.wrap_wavlm", in_roster=False),
}

#: The frozen study roster, in reporting order. Anything iterating "the roster"
#: must use this, not SPECS.keys().
ROSTER = ["aasist", "aasist_l", "rawnet2", "tssdnet", "xlsr_aasist"]
#: Models that enter ranking claims (xlsr_aasist is an unranked reference, D16).
RANKED = ["aasist", "aasist_l", "rawnet2", "tssdnet"]


def load_vendor_module(alias: str, file_path, extra_sys_path=None):
    """Import a vendored source file under a unique alias.

    NEVER plain-import vendored entry points: several official repos name their
    module `model`/`models`, which collides with THIS package (repo/models) and
    with each other. importlib.spec_from_file_location sidesteps sys.modules.
    extra_sys_path is prepended for the vendored module's own sibling imports.
    """
    import importlib.util
    fp = Path(file_path)
    if not fp.exists():
        raise FileNotFoundError(f"vendored file missing: {fp} — clone the "
                                "official repo (phase_1 step_01).")
    if extra_sys_path is not None:
        sp = str(extra_sys_path)
        if sp not in sys.path:
            sys.path.insert(0, sp)
    spec = importlib.util.spec_from_file_location(alias, str(fp))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def load_model(name: str, checkpoint: str | None = None, device: str = "cpu"):
    """Returns object with .score_batch(wave_batch: np.ndarray [B, T]) -> np.ndarray [B]
    (higher = more bona fide). Wrapper modules define load(spec, checkpoint, device)."""
    spec = SPECS[name]
    _no_vendor = ("dummy", "wavlm_trunc", "wavlm_mhfa")   # HF-only, no vendor/
    if not VENDOR.exists() and name not in _no_vendor:
        raise FileNotFoundError(
            f"vendor/ not found at {VENDOR}. Clone the official repos there "
            "(see phase_1_reproduction/step_01_checkpoints.md).")
    if str(device).startswith("cuda"):
        # fp32 means fp32: cuDNN defaults to TF32 convolutions on Ampere+ (both
        # the GPU server RTX 4090s), which moved RawNet2 scores ~1.5e-3 relative off the
        # CPU result (2.4e-6 with TF32 off, 2026-09-16). Process-wide on purpose.
        import torch
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    mod = importlib.import_module(spec.wrapper)
    return mod.load(spec, checkpoint, device)
