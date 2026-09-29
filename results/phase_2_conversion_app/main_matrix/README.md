# Main conversion matrix: fp16 (and XLS-R fp32), sol1, 2026-09-29

Toolchain: torch 2.5.1 (TorchScript exporter, opset 17), onnx 1.22.0, onnxruntime 1.23.2,
onnxconverter-common 1.16.0 (`float16.convert_float_to_float16(keep_io_types=True)`,
installed `--no-deps` so numpy 1.23.5 stays). Launcher: `sol1:xfer/fp16_runs.sh`.
Checkpoints: the Phase-1 reproduction files (D31 for XLS-R).

| model | fp16 export | fp16 vs PyTorch fp32 on the 1,000-clip 19LA-eval parity manifest (diagnostic, never gates) |
|---|---|---|
| aasist | converts, 965,396 B | flips eer/frr1/frr5 = 0/0/0; EER 0.600% vs 0.600%; max abs score diff 0.147 |
| aasist_l | converts, 539,060 B | 0/0/1; EER 1.200% vs 1.200%; max abs score diff 0.132 |
| tssdnet | converts, 709,121 B | 0/0/1; EER 1.400% vs 1.400%; max abs score diff 0.197 |
| rawnet2 | **fails: ORT cannot load it** | – |
| xlsr_aasist | **fails** (fp32 and fp16) | – |

- **rawnet2 fp16.** The fp16 pass leaves `/m/Sinc_conv/Cast` declaring a float16 output
  while its `to` attribute is float. `onnx.checker` accepts that, but ORT refuses to load the
  model. The first export was logged `converts: true`, a plausible wrong result. `to_onnx.py`
  now also requires ORT to load the graph and run it (`26398c6`), and the re-export logs
  `converts: false`. The unloadable file is kept at `sol1:xfer/rawnet2_fp16_unloadable_2026-09-29.onnx`.
  No repair has been attempted; that would be repair effort, to be decided and logged.
- **xlsr_aasist.** The export fails in fairseq's `pad_to_multiple` (`utils.py`), which calls
  `.is_integer()` on a value that becomes a tensor under the tracer. With the pinned toolchain
  the standard exporter cannot export XLS-R at all, so this is a missing row (H6), not a
  harness crash. No workaround has been attempted.
- The aasist / aasist_l / tssdnet fp16 exports predate the ORT load check, but each loaded
  and scored all 1,000 clips.
- The tssdnet fp16 here is a fresh sol1 export. It is not the 2026-07-27 Windows pilot artifact.
