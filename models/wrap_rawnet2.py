"""RawNet2 (ASVspoof 2021 baseline) wrapper.
Verified against asvspoof-challenge/2021@main LA/Baseline-RawNet2 (Jul 2026):
entry `model.py` class RawNet, ctor RawNet(yaml['model'], device) with
model_config_RawNet.yaml; forward model(x[B,T]) -> [B,2]; eval score =
out[:, 1] (bonafide=1 — main.py produce_evaluation_file); eval padding
repeat-tile / truncate-from-start @ 64600; plain state_dict checkpoint.
Vendor at: vendor/asvspoof2021_baselines (clone of asvspoof-challenge/2021)."""
import numpy as np
from .base import VENDOR, load_vendor_module

_DIR = VENDOR / "asvspoof2021_baselines" / "LA" / "Baseline-RawNet2"


class _Wrap:
    def __init__(self, model, torch):
        self.model, self.torch = model, torch

    def score_batch(self, wave_batch):
        t = self.torch
        with t.no_grad():
            x = t.tensor(np.asarray(wave_batch, dtype=np.float32))
            out = self.model(x)                     # [B, 2]
            return out[:, 1].cpu().numpy()          # authors' eval score


def load(spec, checkpoint, device):
    import torch
    import yaml
    mod = load_vendor_module("vendor_rawnet2_model", _DIR / "model.py",
                             extra_sys_path=_DIR)
    with open(_DIR / "model_config_RawNet.yaml") as f:
        cfg = yaml.safe_load(f)
    model = mod.RawNet(cfg["model"], device)
    ckpt = checkpoint or str(_DIR / "pre_trained_DF_RawNet2.pth")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()
    w = _Wrap(model, torch)

    def _export():
        class ScoreNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.m = model

            def forward(self, x):
                return self.m(x)[:, 1]              # bonafide logit
        return ScoreNet().eval()
    w.export_module = _export
    return w
