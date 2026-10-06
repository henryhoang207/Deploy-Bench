"""RawGAT-ST wrapper.
Verified against eurecom-asp/RawGAT-ST-antispoofing@main (Jul 2026): entry
`model.py` class RawGAT_ST, ctor RawGAT_ST(yaml['model'], device) with
model_config_RawGAT_ST.yaml; forward model(x, Freq_aug=False) -> [B,2];
eval score = out[:, 1] (bonafide=1); padding repeat-tile / truncate-from-start
@ 64600; plain state_dict. Vendor at: vendor/RawGAT-ST-antispoofing."""
import numpy as np
from .base import VENDOR, load_vendor_module

_DIR = VENDOR / "RawGAT-ST-antispoofing"


class _Wrap:
    def __init__(self, model, torch):
        self.model, self.torch = model, torch
        self.device = next(model.parameters()).device  # A1: batch goes where the weights are

    def score_batch(self, wave_batch):
        t = self.torch
        with t.no_grad():
            x = t.tensor(np.asarray(wave_batch, dtype=np.float32), device=self.device)
            out = self.model(x, Freq_aug=False)     # [B, 2]
            return out[:, 1].cpu().numpy()


def load(spec, checkpoint, device):
    import torch
    import yaml
    mod = load_vendor_module("vendor_rawgat_model", _DIR / "model.py",
                             extra_sys_path=_DIR)
    with open(_DIR / "model_config_RawGAT_ST.yaml") as f:
        cfg = yaml.safe_load(f)
    model = mod.RawGAT_ST(cfg["model"], device)
    ckpt = checkpoint or str(_DIR / "Pre_trained_models" / "RawGAT_ST_mul" /
                             "Best_epoch.pth")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()
    w = _Wrap(model, torch)

    def _export():
        class ScoreNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.m = model

            def forward(self, x):
                return self.m(x, Freq_aug=False)[:, 1]
        return ScoreNet().eval()
    w.export_module = _export
    return w
