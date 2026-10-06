"""Artifact labels (D38): how an int8 artifact must be named in every report.

The score schema's `precision` column stays "int8" for any QDQ graph, because
score_onnx.py checks that value against the graph itself. Whether the artifact
kept layers in fp32 is not visible in a score row, so reports must join it from
the conversion sidecar (`<artifact>_ptq_log.json`, written by convert/ptq.py).
A row whose sidecar is missing is refused: printing plain "int8" for a mixed
artifact is exactly the plausible wrong label D38 exists to prevent.
"""
from __future__ import annotations

import json
from pathlib import Path


class MissingSidecar(RuntimeError):
    pass


def precision_label(n_excluded: int) -> str:
    """"int8" only when nothing was kept in fp32 on purpose; else "int8-mixed (K=k)"."""
    if not isinstance(n_excluded, int) or n_excluded < 0:
        raise ValueError(f"exclusion count must be a non-negative int, got {n_excluded!r}")
    return "int8" if n_excluded == 0 else f"int8-mixed (K={n_excluded})"


class NotInt8(ValueError):
    pass


def label_from_sidecar(sidecar: dict) -> str:
    """Label from a ptq sidecar. Sidecars written before D38 have no
    n_fp32_excluded; their excluded_nodes list gives the same K.

    An artifact with no QuantizeLinear/DequantizeLinear at all is an fp32 graph
    whatever its exclusion count says (tssdnet r12 kept all 37 ranked nodes in
    fp32 and its sidecar was written as "int8-mixed (K=37)"). Refuse to label it int8."""
    if sidecar.get("n_quantize_linear") == 0 and sidecar.get("n_dequantize_linear") == 0:
        raise NotInt8("artifact has no Q/DQ ops: it is an fp32 graph, not int8")
    if "n_fp32_excluded" in sidecar:
        k = sidecar["n_fp32_excluded"]
        if "excluded_nodes" in sidecar and len(sidecar["excluded_nodes"]) != k:
            raise ValueError("sidecar disagrees with itself: n_fp32_excluded="
                             f"{k} but {len(sidecar['excluded_nodes'])} excluded_nodes")
        return precision_label(k)
    if "excluded_nodes" in sidecar:
        return precision_label(len(sidecar["excluded_nodes"]))
    raise MissingSidecar("sidecar has neither n_fp32_excluded nor excluded_nodes")


def report_label(precision: str, sidecar_path=None) -> str:
    """The precision string a table, figure or sentence may print.

    fp32 / fp16 pass through. int8 needs its sidecar; without one it is refused."""
    if precision != "int8":
        return precision
    if sidecar_path is None or not Path(sidecar_path).is_file():
        raise MissingSidecar(f"int8 row without its conversion sidecar ({sidecar_path}); "
                             "D38 forbids printing it as plain 'int8'")
    return label_from_sidecar(json.loads(Path(sidecar_path).read_text(encoding="utf-8")))


# ---- ExecuTorch device artifacts (D48 / DEVIATIONS.md DEV-7) ----------------
class NotHonourable(Exception):
    """An executorch cell this build must refuse rather than run under a wrong label."""


def et_artifact(model: str, precision: str, repo: Path) -> Path:
    """The .pte a device or scoring cell may use, or NotHonourable.

    DEV-7 rules: the model's ExecuTorch fp32 must have passed the desktop parity gate
    (zero flips vs PyTorch); fp32/fp16 use exports/et/<m>_<p>.pte; int8 uses ONLY the
    canonical seed from canonical_int8.json, never another seed. The artifact's sha256
    must match its to_executorch.py sidecar."""
    import hashlib
    res = repo / "results" / "phase_2_conversion_app" / "executorch" / model
    gate = res / "parity_report_fp32_conversion.json"
    if not gate.is_file() or json.loads(gate.read_text()).get("pass") is not True:
        raise NotHonourable(f"{model}: no passed ExecuTorch fp32 parity gate ({gate})")
    if precision in ("fp32", "fp16"):
        stem = f"{model}_{precision}"
    elif precision == "int8":
        can = res / "canonical_int8.json"
        seed = json.loads(can.read_text()).get("canonical_seed") if can.is_file() else None
        if seed is None:
            raise NotHonourable(f"{model}: no canonical ExecuTorch int8 artifact ({can})")
        stem = f"{model}_int8_seed{seed}"
    else:
        raise NotHonourable(f"unknown precision {precision!r}")
    pte = repo / "exports" / "et" / f"{stem}.pte"
    side = pte.with_name(f"conversion_log_{stem}_executorch.json")
    if not pte.is_file() or not side.is_file():
        raise NotHonourable(f"{pte} or its sidecar is missing")
    log = json.loads(side.read_text())
    sha = hashlib.sha256(pte.read_bytes()).hexdigest()
    if not log.get("converts") or log.get("sha256") != sha or log.get("precision") != precision:
        raise NotHonourable(f"{pte}: sidecar disagrees (converts/sha256/precision)")
    return pte


if __name__ == "__main__":
    # `python -m deploybench.artifacts et <model> <precision>`: print the .pte path or
    # exit 3 (the run_cell.sh refusal code) with the reason on stderr.
    import sys
    if len(sys.argv) != 4 or sys.argv[1] != "et":
        raise SystemExit("usage: python -m deploybench.artifacts et <model> <precision>")
    try:
        print(et_artifact(sys.argv[2], sys.argv[3], Path(__file__).resolve().parent.parent))
    except NotHonourable as e:
        print(e, file=sys.stderr)
        raise SystemExit(3)
