# ExecuTorch (XNNPACK) leg, desktop: D48 / DEVIATIONS.md DEV-7 (sol1, 2026-10-03)

Toolchain (every conversion log records it): Python 3.12.3, torch 2.14.0+cpu, executorch 1.5.1,
torchao 0.18.0, numpy 2.5.3, in `sol1:envs/et` (`pip freeze` in `sol1:envs/et_freeze.txt`).
The torch version is executorch 1.5.1's own pin (`torch_pin.py`, TORCH_VERSION = "2.14.0").
Lowering is `to_edge_transform_and_lower(torch.export.export(...), [XnnpackPartitioner()])`.
Launcher: `sol1:xfer/et_runs.sh` at HEAD `e2deaf7`. Exporter: `convert/to_executorch.py`.
Scorer: `score_pte.py` (refuses an artifact whose sidecar sha256/precision disagree).
The 2026-10-03 desktop parity reference is the same PyTorch fp32 CPU file as the ORT leg
(`conversion_sensitivity/<m>/parity_ref_fp32_pytorch.csv`).

**Disclosure: tooling shakedown.** Before the recorded run, the fp32 export of the four models was
run once into `exports/et_dryrun/` to catch tooling bugs. It produced the same four outcomes as
the recorded run (`dryrun_2026-10-03/`), and no code changed between the two.

## Results

| model | fp32 .pte | fp16 .pte | int8 .pte (5 seeds + rebuild) |
|---|---|---|---|
| tssdnet | **converts**, 1,428,872 B; **gate PASS**: flips 0/0/0, max abs score diff 3.0e-4 | **converts**, 1,408,772 B; flips 0/0/0; EER 1.40% vs 1.40%; max abs diff 0.50; r 0.999994 | **all 6 fail**: `RuntimeError: Weight and Input should both be quantized` (XNNPACK lowering, after 500-clip calibration) |
| aasist | fails: `amax.default only supports keep_dim == True` | fails: conv weight stays fp32 while the input is fp16 (`SpecViolationError`) | all 6 fail (amax, as fp32) |
| aasist_l | fails: same as aasist | fails: same as aasist | all 6 fail (amax) |
| rawnet2 | fails: `.numpy() is not supported for tensor subclasses` | fails: same | all 6 fail: same |

Canonical int8 (`tssdnet/canonical_int8.json`): **none**, since no seed produced an artifact
(fp32 diag EER 0.70%). No model has an ExecuTorch int8 artifact. Per DEV-7 nothing is retried
or substituted.

## What the failures are (from the logs; mechanisms, not repairs)

- **RawNet2:** the vendor `SincConv.forward` rebuilds its filter bank with numpy on every call.
  The ONNX TorchScript tracer freezes that as a constant. `torch.export` traces with fake
  tensors and cannot call `.numpy()`, so neither the float nor the PT2E path can capture the
  model.
- **AASIST / AASIST-L, fp32 and int8:** the max-pooling over time and frequency after the encoder
  (`AASIST.py` l. 542/549) and in the readout (l. 595/598) is `torch.max(x, dim)` without keepdim. XNNPACK's partitioner claims the resulting `amax` node, and its lowering then
  rejects keep_dim=False: a partitioner/lowering mismatch in executorch 1.5.1.
- **AASIST / AASIST-L, fp16:** the first conv (`conv_time`, a sinc layer) builds its filter as
  an fp32 tensor inside `forward`, so `.half()` does not reach it, and the edge dialect rejects
  the fp16-input x fp32-weight convolution. (ORT's graph-level fp16 pass converted these
  constants; it broke RawNet2's Cast instead, D46.)
- **Res-TSSDNet, int8:** calibration completes (500 clips per seed), then XNNPACK lowering
  rejects a quantized op whose weight and input are not both quantized. A graph-level check
  (`/tmp/et_diag.py` on sol1, diagnostic only) found no conv/linear with mismatched
  dequantized inputs at the aten level. The mismatch therefore arises inside XNNPACK's own
  fused patterns. Not localised further.

## Reading (exploratory; DEV-7)

ONNX Runtime converted all four ranked models at fp32 (zero flips) and three at fp16 out of the box.
ExecuTorch's `torch.export` path converts one model at fp32 and fp16 and none at int8 under its
documented default recipes. Where it does convert (Res-TSSDNet), it is as faithful as ORT:
zero flips at every operating point. The runtime choice changes **which models can be deployed at
all** before it changes any score.

## Pending

- Res-TSSDNet fp16 on ITW and the DF21 verified surface (`sol1:xfer/et_surfaces.sh`, running
  since 2026-10-03 11:19), then MLAAD-5 once its protocol exists (DEV-6).
- Device cells: `executorch` x {tssdnet fp32, tssdnet fp16} only. Every other executorch cell is
  refused by `deploybench.artifacts.et_artifact` (no gate pass or no canonical int8).
