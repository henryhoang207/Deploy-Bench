"""Pre-tag smoke tests for round 2 (allowed by D54: LA19 TRAIN clips only, no dev / eval / target scoring).
Run from the deploybench clone:  python -B /mnt/data8tb/henry/xfer/r2_smoke.py <stage>
Stages: kernels (envs/r2), xlsr_mamba (envs/r2), pascu (envs/r2), as5 (envs/db310)."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
STAGE = sys.argv[1]


def train_clips(n):
    from deploybench.data import protocol_iter
    from score import load_wav
    out = []
    for i, (u, p, lab) in enumerate(protocol_iter("asvspoof19_train")):
        if i % 2000 == 0:                       # spread over the protocol, both classes
            out.append((u, lab, load_wav(Path(p))))
        if len(out) >= n:
            break
    return out


if STAGE == "kernels":
    import torch
    print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0))
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
    from causal_conv1d import causal_conv1d_fn
    torch.manual_seed(0)
    b, d, L, n = 2, 64, 256, 16
    u = torch.randn(b, d, L, device="cuda")
    delta = torch.rand(b, d, L, device="cuda")
    A = -torch.rand(d, n, device="cuda")
    B_ = torch.randn(b, n, L, device="cuda")
    C_ = torch.randn(b, n, L, device="cuda")
    D = torch.randn(d, device="cuda")
    y1 = selective_scan_fn(u, delta, A, B_, C_, D)
    y2 = selective_scan_ref(u, delta, A, B_, C_, D)
    print("selective_scan max|cuda - ref| =", float((y1 - y2).abs().max()))
    x = torch.randn(2, 64, 300, device="cuda")
    w = torch.randn(64, 4, device="cuda")
    yc = causal_conv1d_fn(x, w, None)
    yr = torch.nn.functional.conv1d(torch.nn.functional.pad(x, (3, 0)), w.unsqueeze(1), groups=64)
    print("causal_conv1d max|cuda - ref| =", float((yc - yr).abs().max()))
    from mamba_ssm.ops.triton.layernorm import rms_norm_fn
    h = torch.randn(4, 100, 144, device="cuda")
    wt = torch.rand(144, device="cuda")
    yt = rms_norm_fn(h, wt, None, eps=1e-5)
    ref = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-5) * wt
    print("triton rms_norm max|triton - ref| =", float((yt - ref).abs().max()))

elif STAGE in ("xlsr_mamba", "as5"):
    from models.base import SPECS, load_model
    from score import fix_length
    name = "xlsr_mamba" if STAGE == "xlsr_mamba" else "as5_aasist"
    t0 = time.time()
    m = load_model(name, None, "cuda")
    print(f"{name} loaded in {time.time() - t0:.1f}s; params", sum(p.numel() for p in m.model.parameters()))
    clips = train_clips(6)
    n = SPECS[name].input_samples
    x = np.stack([fix_length(w, n) for _, _, w in clips])
    t0 = time.time()
    s = m.score_batch(x)
    print(f"scored {len(s)} TRAIN clips in {time.time() - t0:.2f}s; finite: {bool(np.all(np.isfinite(s)))}")
    print("labels", [lab for _, lab, _ in clips], "scores", np.round(s, 3).tolist())
    s2 = m.score_batch(x)
    print("repeat max|diff| =", float(np.abs(s - s2).max()))

elif STAGE == "pascu":
    from models.wrap_pascu import load_backbone
    t0 = time.time()
    bb = load_backbone("cuda")
    print(f"xls-r-2b loaded in {time.time() - t0:.1f}s; hidden", bb.model.config.hidden_size,
          "layers", bb.model.config.num_hidden_layers)
    clips = train_clips(8)
    t0 = time.time()
    F = np.stack([bb.features(w) for _, _, w in clips])
    el = time.time() - t0
    secs = sum(len(w) for _, _, w in clips) / 16000
    print(f"features {F.shape} dtype {F.dtype} finite {bool(np.all(np.isfinite(F)))}; {el:.2f}s for "
          f"{len(clips)} clips ({secs:.1f} s audio) -> {len(clips) / el:.1f} clips/s")
    F2 = np.stack([bb.features(w) for _, _, w in clips[:2]])
    print("repeat max|diff| =", float(np.abs(F[:2] - F2).max()))
else:
    raise SystemExit(STAGE)
