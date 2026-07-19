"""LFCC-LCNN (ASVspoof 2021 NII baseline) wrapper.
Verified against asvspoof-challenge/2021@main LA/Baseline-LFCC-LCNN (Jul 2026):
entry project/baseline_LA/model.py class Model(in_dim, out_dim, args, prj_conf,
mean_std); LFCC frontend INSIDE the model (frame 320/hop 160/fft 1024, 20 ceps
+ delta + energy, max_freq 0.5); inference score = raw pre-sigmoid scalar
(_compute_score(..., inference=True)), higher = bona fide (BCE target 1 =
bonafide). forward() needs NII fileinfo strings, so we call
_compute_embedding directly. Needs repo's sandbox/ + core_scripts/ on sys.path.
Vendor at: vendor/asvspoof2021_baselines/LA/Baseline-LFCC-LCNN."""
import numpy as np
from types import SimpleNamespace
from .base import VENDOR, load_vendor_module

_ROOT = VENDOR / "asvspoof2021_baselines" / "LA" / "Baseline-LFCC-LCNN"
_ENTRY = _ROOT / "project" / "baseline_LA" / "model.py"


class _Wrap:
    def __init__(self, model, torch):
        self.model, self.torch = model, torch

    def score_batch(self, wave_batch):
        t = self.torch
        with t.no_grad():
            x = t.tensor(np.asarray(wave_batch, dtype=np.float32)).unsqueeze(-1)
            emb = self.model._compute_embedding(x, [x.shape[1]] * x.shape[0])
            return self.model._compute_score(emb, inference=True).cpu().numpy()


def load(spec, checkpoint, device):
    import torch
    # NII framework: model.py imports sandbox.* and core_scripts.* from repo root
    mod = load_vendor_module("vendor_lcnn_model", _ENTRY, extra_sys_path=_ROOT)
    # prj_conf shim: protocol only needed for TRAINING targets; missing file is
    # caught by protocol_parse (OSError -> empty dict), fine for inference.
    prj_conf = SimpleNamespace(optional_argument=["__no_protocol__.txt"])
    model = mod.Model(in_dim=1, out_dim=1, args=None, prj_conf=prj_conf)
    ckpt = checkpoint or str(_ENTRY.parent / "__pretrained" /
                             "trained_network.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()
    w = _Wrap(model, torch)

    def _export():
        class ScoreNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.m = model

            def forward(self, x):                    # x: [B, T] waveform
                xx = x.unsqueeze(-1)
                emb = self.m._compute_embedding(xx, [xx.shape[1]] * xx.shape[0])
                return emb.squeeze(1)                # raw score, higher=bonafide
        return ScoreNet().eval()
    w.export_module = _export
    return w
