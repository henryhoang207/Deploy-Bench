"""XLSR-Mamba (Xiao & Das, IEEE SPL 2025) wrapper, round 2 (D54).
Verified against swagshaw/XLSR-Mamba @ 10ec738 (2026-10-07): model.py class Model(args, device) builds the fairseq
XLS-R 300M front end from ./xlsr2_300m.pt (relative to the CWD, as in SSL_Anti-spoofing) and a dual-column bidirectional
Mamba MixerModel (mamba_blocks.py; mamba-ssm 1.1.4, causal-conv1d 1.1.3.post1, fused add+RMSNorm through Triton) whose
classifier is nn.Linear(d_model, 2). Labels: bonafide = 1 (data_utils.genSpoof_list); eval score = batch_out[:, 1]
(main.py produce_evaluation_file). Eval input: data_utils.Dataset_eval cut = 66800 (the training length too) with
utils.pad (truncate from the start, repeat-tile short clips); the authors' In-the-Wild loader uses 64600, the one
place they differ -- round 2 uses 66800 everywhere (PREREGISTRATION_ROUND2.md). Checkpoint: the released DF model
(HF AustinXiao/XLSR-Mamba-DF @ 8aed3da, model.safetensors; tag Bmamba3_LA_WCE_1e-06_ES144_NE12, i.e. trained on LA19
train with RawBoost algo 3, emb_size 144, num_encoders 12, checkpoint chosen on LA19 dev loss).
Environment: envs/r2 (torch 2.2.1+cu118, the authors' pin). Vendor at: vendor/XLSR-Mamba (+ xlsr2_300m.pt symlink)."""
import os
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np

from .base import VENDOR, load_vendor_module

_DIR = VENDOR / "XLSR-Mamba"
_ARGS = SimpleNamespace(emb_size=144, num_encoders=12)


@contextmanager
def _chdir(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


class _Wrap:
    def __init__(self, model, torch):
        self.model, self.torch = model, torch
        self.device = next(model.parameters()).device  # A1: batch goes where the weights are

    def score_batch(self, wave_batch):
        t = self.torch
        with t.no_grad():
            x = t.tensor(np.asarray(wave_batch, dtype=np.float32), device=self.device)
            out = self.model(x)                     # [B, 2] logits
            # A1: the author SSLModel.extract_feat calls self.model.train() when the input's device/dtype differs
            # from the backbone's (same guard as SSL_Anti-spoofing); refuse if it fired.
            if any(m.training for m in self.model.modules()):
                raise RuntimeError("xlsr_mamba: submodule in train mode after forward (author device/dtype guard "
                                   "fired); scores would be nondeterministic")
            return out[:, 1].cpu().numpy()


def load(spec, checkpoint, device):
    import torch
    from safetensors.torch import load_file
    if checkpoint:                                  # resolve BEFORE the chdir below
        checkpoint = os.path.abspath(checkpoint)
    ckpt = checkpoint or str(_DIR / "hf_DF" / "model.safetensors")
    with _chdir(_DIR):                              # xlsr2_300m.pt (symlink) lives here
        mod = load_vendor_module("vendor_xlsrmamba_model", _DIR / "model.py", extra_sys_path=_DIR)
        model = mod.Model(_ARGS, device)
    sd = load_file(ckpt, device="cpu")
    # The HF upload used PyTorchModelHubMixin; strip a uniform "model." prefix only if EVERY key has it.
    if sd and all(k.startswith("model.") for k in sd):
        sd = {k[len("model."):]: v for k, v in sd.items()}
    model.load_state_dict(sd)                       # strict
    model.to(device).eval()
    return _Wrap(model, torch)
