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
