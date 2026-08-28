# TSSDNet precision comparison - FULL EVAL (H6)

ASVspoof19 LA eval, **71,237 clips** (7,355 bona fide). Frozen taus (D16). Bootstrap 1000 resamples, paired for deltas.

Supersedes the 1,000-clip manifest numbers, which resolved EER only to ~0.2 pp.

**Deltas for the converted precisions are against ONNX fp32, not PyTorch fp32** -- measuring them against PyTorch would fold conversion error into quantisation cost. The ONNX fp32 row's own delta IS the conversion error, reported rather than hidden.

| arm | EER % | 95% CI | vs | dEER pp | dEER 95% CI | flips e/f1/f5 | max\|d\| | Pearson |
|---|---|---|---|---|---|---|---|---|
| pt_fp32 | 1.6183 | 1.4937-1.7690 | -- | -- | -- | -- | -- | -- |
| onnx_fp32 | 1.6183 | 1.4937-1.7690 | pt_fp32 | +0.0000 | +0.0000 to +0.0000 | 0/0/0 | 3.285e-04 | 1.000000 |
| fp16 | 1.6183 | 1.4970-1.7675 | onnx_fp32 | +0.0000 | -0.0138 to +0.0138 | 34/19/6 | 2.364e-01 | 0.999998 |
| int8 | 2.5562 | 2.3929-2.7160 | onnx_fp32 | +0.9379 | +0.7817 to +1.0714 | 2254/1900/1351 | 2.323e+01 | 0.979417 |

Every EER computed twice (deploybench.metrics vs an independent numpy sweep); max disagreement 6.02e-03 pp.

The run GATES on the PyTorch fp32 arm reproducing the frozen reproduction anchor 1.6179% and aborts otherwise.
