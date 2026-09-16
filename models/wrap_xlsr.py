"""XLSR+AASIST (SSL SOTA reference) wrapper.
Verified against TakHemlata/SSL_Anti-spoofing@main (Jul 2026): entry model.py
class Model(args, device) — args unused in ctor, safe to pass None; forward
model(x[B,T]) -> [B,2] logits; eval score = out[:, 1] (bonafide=1); padding
repeat-tile / truncate-from-start @ 64600. GOTCHA: SSLModel hardcodes
cp_path='xlsr2_300m.pt' relative to CWD — we chdir to the vendor dir during
construction; put the fairseq XLS-R checkpoint there.
Vendor at: vendor/SSL_Anti-spoofing (+ fairseq installed)."""
import os
import numpy as np
from contextlib import contextmanager
from .base import VENDOR, load_vendor_module

_DIR = VENDOR / "SSL_Anti-spoofing"


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
            out = self.model(x)                     # [B, 2]
            # A1: author SSLModel.extract_feat calls self.model.train() when the
            # input's device/dtype differs from the backbone's, leaving layerdrop
            # live. Matching x to the weights avoids that; refuse if it fired anyway.
            if any(m.training for m in self.model.modules()):
                raise RuntimeError("xlsr_aasist: submodule in train mode after forward "
                                   "(author device/dtype guard fired); scores would "
                                   "be nondeterministic")
            return out[:, 1].cpu().numpy()


def load(spec, checkpoint, device):
    import torch
    with _chdir(_DIR):                              # xlsr2_300m.pt lives here
        mod = load_vendor_module("vendor_sslaasist_model", _DIR / "model.py",
                                 extra_sys_path=_DIR)
        model = mod.Model(None, device)
        ckpt = checkpoint or str(_DIR / "Best_LA_model_for_DF.pth")
        sd = torch.load(ckpt, map_location=device)
        # Best_LA_model_for_DF.pth was saved from nn.DataParallel (main_SSL_DF.py),
        # so every key carries "module."; LA_model.pth (main_SSL_LA.py) does not.
        # Strip only when ALL keys have it; load stays strict either way.
        if sd and all(k.startswith("module.") for k in sd):
            sd = {k[len("module."):]: v for k, v in sd.items()}
        model.load_state_dict(sd)
    model.to(device).eval()
    w = _Wrap(model, torch)

    def _export():
        class ScoreNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.m = model

            def forward(self, x):
                return self.m(x)[:, 1]
        return ScoreNet().eval()
    w.export_module = _export
    return w
