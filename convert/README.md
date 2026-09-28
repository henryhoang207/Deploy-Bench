# convert/ — export, quantization, conversion-sensitivity study

| Script | Does |
|---|---|
| `to_onnx.py` | PyTorch wrapper -> `exports/<model>_<fp32\|fp16>.onnx` (+ conversion log) |
| `ptq.py` | ORT static int8 PTQ. Canonical mode writes next to the fp32 export; **run mode** (`--run-dir`) writes a new directory per attempt and appends to a ledger |
| `sensitivity_manifests.py` | D39 manifests in `configs/conversion_sensitivity/` (frozen; `--verify` re-checks them) |
| `rank_sensitivity.py` | Freezes `search_plan_<model>.json` from fp32-vs-int8 activation damage of attempt r01 |
| `select_recipe.py` | Applies the frozen selection rules: locked recipe (search), device artifact (canonical) |
| `run_sensitivity.py` | Runs the whole per-model sequence on sol1; resumable, never re-runs a logged attempt |
| `to_executorch.py` | ExecuTorch export (OD-4: not in the study) |

## Conversion-sensitivity study (D35, frozen by D39/D41 in PREREGISTRATION.md)

Per ranked model (`aasist`, `aasist_l`, `rawnet2`, `tssdnet`; XLS-R is out of scope):

```
python convert/run_sensitivity.py --model aasist all
```

1. **r01**: the default recipe (`ptq.py` defaults: QDQ, per-channel, S8S8, minmax,
   pre-process on, nothing excluded), calibrated on `calib_search_seed100.txt`.
2. **rank**: nodes ORT quantized are ranked by `damage_db` = min input SQNR minus
   output SQNR (fp32 vs int8, streamed over the 2,000-clip diagnostic subset). The
   plan (r02..r12 = top-k exclusions, k = 1,2,3,4,6,8,12,16,24,32,48, clipped) is
   frozen before any of those attempts runs.
3. **search**: every planned attempt is converted and scored on the diagnostic subset.
4. **select**: the smallest K with int8 EER - fp32 EER <= 1.0 pp (D36); ties go to the
   earlier attempt. The fp32 reference is the frozen PyTorch 19LA-dev score file.
5. **repeats**: the locked recipe on `calib_seed1..5` plus one identical rebuild of
   seed 1; each scored on the diagnostic subset and ASVspoof19 LA eval (D41).
6. **canonical**: the device artifact is the lowest seed that meets the budget, or
   none. `repeatability_<model>.json` compares seed 1 with its rebuild.

Rules the code enforces:
- A failed attempt is logged and counts. Nothing is retried, replaced or re-run.
- Selection refuses while any frozen attempt has no ledger entry or no scores.
- A mixed-precision artifact is labelled `int8-mixed (K=k)` (D38,
  `deploybench/artifacts.py`); an int8 row without its sidecar is refused.
- `--exclude-nodes` names missing from the quantized graph are refused (ORT would
  silently ignore them and yield a mislabelled, fully quantized artifact).
- A calibration manifest is used whole (the old reader truncated it to 500).

Outputs: `results/phase_2_conversion_app/conversion_sensitivity/` (ledgers, plans,
selections, per-run sidecars and scores; `.onnx` files are gitignored).
