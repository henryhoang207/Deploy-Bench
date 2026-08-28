"""TOMBSTONE (2026-07-29, decision D30): no roster model needs a trained head.

The roster was frozen at five models -- AASIST, AASIST-L, RawNet2, Res-TSSDNet,
XLSR+AASIST -- every one of which ships a published checkpoint. The two models
this script existed for (truncated-WavLM L1-6 + linear, WavLM-base+ + MHFA) were
dropped, and with them the entire ASVspoof 5 dependency chain: registration,
download, the deliberately-unimplemented AS5 loader (D15), six head checkpoints,
and a second tau surface.

Retaining the file as a tombstone rather than deleting it, because the project's
own rule is that history is appended to, not rewritten, and because two things
recorded here are worth not losing:

1. D14's contract -- frozen backbone kept in eval() during head training so
   dropout cannot corrupt the features the head learns from. If any future work
   revives head training, that is the non-obvious part.

2. A METHODOLOGICAL DEFECT this script contained, found in the 2026-07-29 review
   and never fixed because the models were dropped first: it early-stopped on
   the AS5 dev split (`if vl < best: torch.save(...)`), which D16 also used as
   the tau calibration surface. That is model selection on the calibration
   surface -- the same oracle leak the tier-2 reading pass criticised in
   AASIST's own best.pth, which was selected on eval t-DCF. Anyone reinstating
   head training must carve a separate selection split off train, or select at a
   fixed epoch count. Do not copy this file's loop as-is.

Superseded by: decision.md D30. Prior tombstone in this repo: repo/train_head.py
(deleted 2026-07-19, MLP-not-MHFA head + incompatible checkpoint format).
"""

raise SystemExit(
    "train/train_heads.py is a tombstone: D30 froze the roster at 5 models, "
    "none of which needs a trained head. See the module docstring."
)
