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
| rawnet2 | **fails: ORT cannot load it** (one-node repair, D46: converts, 35,641,282 B) | repaired: 0/0/1; EER 4.40% vs 4.40%; max abs score diff 0.049 |
| xlsr_aasist | **fails** (fp32 and fp16) | – |

- **rawnet2 fp16.** The fp16 pass leaves `/m/Sinc_conv/Cast` declaring a float16 output
  while its `to` attribute is float. `onnx.checker` accepts that, but ORT refuses to load the
  model. The first export was logged `converts: true`, a plausible wrong result. `to_onnx.py`
  now also requires ORT to load the graph and run it (`26398c6`), and the re-export logs
  `converts: false`. The unloadable file is kept at `sol1:xfer/rawnet2_fp16_unloadable_2026-09-29.onnx`.
  No repair has been attempted; that would be repair effort, to be decided and logged.
  **Update 2026-10-02: repaired once (D46 / DEV-5); see "RawNet2 fp16 repair" below.**
- **xlsr_aasist.** The export fails in fairseq's `pad_to_multiple` (`utils.py`), which calls
  `.is_integer()` on a value that becomes a tensor under the tracer. With the pinned toolchain
  the standard exporter cannot export XLS-R at all, so this is a missing row (H6), not a
  harness crash. No workaround has been attempted.
- The aasist / aasist_l / tssdnet fp16 exports predate the ORT load check, but each loaded
  and scored all 1,000 clips.
- The tssdnet fp16 here is a fresh sol1 export. It is not the 2026-07-27 Windows pilot artifact.

## RawNet2 fp16 repair (D46 / DEV-5, sol1, 2026-10-02)

The single permitted repair attempt (launcher `sol1:xfer/fp16_repair_d46.sh`, HEAD `6fe1c6d`):
`to_onnx.py --precision fp16 --fp16-node-block /m/Sinc_conv/Cast`, so the converter keeps that
one Cast in fp32 and wraps it in Casts. Everything else is the same toolchain as the table above.
Outputs are in `rawnet2/fp16_repaired_d46/`, under their own file names so that no script reads
them as plain fp16 (`transfer_matrix.py` labels them `fp16-repaired(D46)`).

- **Artifact:** `sol1:deploybench/exports/repair_d46/rawnet2_fp16.onnx`, 35,641,282 B, sha256
  `41734087411872b6df45648dc574a6e0970e503dcbc3cddfa9bb006ca5bd4257` (`artifact.json`). It loads
  in ORT 1.23.2 and passes the smoke inference. The 2026-09-29 unloadable graph
  (sha256 `efeb8b27...`) is kept at `sol1:xfer/rawnet2_fp16_unloadable_2026-09-29.onnx`.
- **Parity manifest (1,000 clips, diagnostic):** flips 0/0/1 at eer/frr1/frr5; EER 4.40% vs
  4.40%; max |score diff| 0.049; Pearson r 0.9999995 (`fp16_vs_fp32_diagnostic.json`).
- **Frozen 19LA-dev taus, against fp32 PyTorch on identical trials** (`transfer_matrix.txt`):

| surface | n | fp32 EER | fp16-repaired EER | dEER | flips at tau_eer |
|---|---|---|---|---|---|
| ITW | 31,779 | 49.00% | 49.00% | -0.00 pp | 18 |
| DF21 verified | 455,508 | 19.65% | 19.66% | +0.01 pp | 76 |
| DF21 clean only | 305,525 | 15.06% | 15.07% | +0.00 pp | 39 |

  DF21 decoded the same 149,983 class-A files through the MD5-verified fallback as every
  earlier run.
- **Reading:** lossless, like the other three fp16 rows. The registered result stays "fp16
  converts for 3 of 4 ranked models out of the box". This row is reported beside it as
  "fp16 (repaired: 1 node in fp32)", with its repair effort (1 attempt, 1 node). The failure is
  a converter defect: the fp16 pass rewrites a Cast's declared type without its `to`
  attribute. It is not a property of RawNet2.
