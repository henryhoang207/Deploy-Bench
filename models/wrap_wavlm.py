"""WavLM-based roster models (OURS — no upstream eval convention to match).
Backbone: HF `microsoft/wavlm-base-plus` (chen2022wavlm). Two variants:

  wavlm_trunc : CNN encoder + transformer layers 1..6 only (recipe per
                arXiv:2606.30780; layer evidence arXiv:2502.03559),
                mean-pool over frames -> Linear(768, 2).
  wavlm_mhfa  : full 12-layer backbone, MHFA backend per Peng et al.
                (arXiv:2210.01273): per-head layer-weighted keys/values,
                attentive pooling per head, concat -> Linear -> 2.

CONVENTION (ours, fixed): class index 1 = bona fide; score = out[:,1]-out[:,0].
train/train_heads.py trains the heads with THIS convention — keep in sync.
Head checkpoint format: {"head": state_dict, "variant": str, "seed": int}.
Backbone is frozen (never in the checkpoint) — loaded from HF by exact
revision pinned in configs/hf_revisions.json (create at first download)."""
import numpy as np
from .base import VENDOR  # noqa: F401  (kept for interface symmetry)

N_TRUNC_LAYERS = 6


def _build(variant, torch, device):
    import torch.nn as nn
    from transformers import WavLMModel

    backbone = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    if variant == "wavlm_trunc":
        backbone.encoder.layers = nn.ModuleList(
            list(backbone.encoder.layers)[:N_TRUNC_LAYERS])

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = backbone
                self.head = nn.Linear(768, 2)

            def forward(self, x):                    # x: [B, T]
                h = self.backbone(x).last_hidden_state    # [B, F, 768]
                return self.head(h.mean(dim=1))           # [B, 2]
        return Net()

    class MHFA(nn.Module):
        """Multi-Head Factorized Attentive pooling (Peng et al. 2022)."""

        def __init__(self, n_layers=13, hid=768, cmp_dim=128, heads=32,
                     out_dim=256):
            super().__init__()
            self.w_k = nn.Parameter(torch.zeros(n_layers))
            self.w_v = nn.Parameter(torch.zeros(n_layers))
            self.cmp_k = nn.Linear(hid, cmp_dim)
            self.cmp_v = nn.Linear(hid, cmp_dim)
            self.att = nn.Linear(cmp_dim, heads)
            self.pool_fc = nn.Linear(cmp_dim * heads, out_dim)

        def forward(self, layers):                   # [L, B, F, 768]
            k = (layers * torch.softmax(self.w_k, 0)[:, None, None, None]).sum(0)
            v = (layers * torch.softmax(self.w_v, 0)[:, None, None, None]).sum(0)
            k, v = self.cmp_k(k), self.cmp_v(v)      # [B, F, cmp]
            a = torch.softmax(self.att(k), dim=1)    # [B, F, heads]
            # per-head attentive sum of values: [B, heads, cmp]
            pooled = torch.einsum("bfh,bfc->bhc", a, v)
            return self.pool_fc(pooled.flatten(1))   # [B, out_dim]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.mhfa = MHFA()
            self.cls = nn.Linear(256, 2)

        def forward(self, x):
            out = self.backbone(x, output_hidden_states=True)
            layers = torch.stack(out.hidden_states, dim=0)  # [13, B, F, 768]
            return self.cls(self.mhfa(layers))
    return Net()


class _Wrap:
    def __init__(self, model, torch):
        self.model, self.torch = model, torch

    def score_batch(self, wave_batch):
        t = self.torch
        with t.no_grad():
            x = t.tensor(np.asarray(wave_batch, dtype=np.float32))
            out = self.model(x)
            return (out[:, 1] - out[:, 0]).cpu().numpy()


def load(spec, checkpoint, device):
    import torch
    net = _build(spec.name, torch, device)
    if checkpoint is None:
        raise FileNotFoundError(
            f"{spec.name}: head checkpoint required (train with "
            "train/train_heads.py — backbone is frozen, heads are ours).")
    ck = torch.load(checkpoint, map_location=device)
    missing, unexpected = net.load_state_dict(ck["head"], strict=False)
    assert not unexpected, f"unexpected keys in head ckpt: {unexpected}"
    net.to(device).eval()
    w = _Wrap(net, torch)

    def _export():
        class ScoreNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.m = net

            def forward(self, x):
                out = self.m(x)
                return out[:, 1] - out[:, 0]
        return ScoreNet().eval()
    w.export_module = _export
    return w
