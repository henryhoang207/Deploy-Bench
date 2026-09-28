"""Freeze a model's D39 recipe-search plan from fp32-vs-int8 activation differences.

Input: the fp32 export and the int8 artifact of search attempt r01 (the default
recipe, `ptq.py` defaults, calibrated on calib_search_seed100.txt). Output: a
FROZEN plan, `search_plan_<model>.json`, which lists attempts r02..rNN in the
order they must be run. The plan is written once and never overwritten
(PREREGISTRATION, D39: "that ordered sequence ... is frozen before step 2").

How nodes are ranked. Both graphs are run clip by clip on the diagnostic subset,
with every intermediate tensor exposed. Per tensor pair, SQNR = 10*log10(sum x^2 /
sum (x - y)^2) over all clips (streamed; nothing is held in memory per clip).
For each node that ORT quantized, the damage it adds is

    damage_db = min(SQNR of its activation inputs, as the int8 node consumed them)
                - SQNR of its output

i.e. how much fp32-vs-int8 agreement is lost across that node. Nodes are ranked
by damage, largest first; ties keep graph order. The exclusion attempts then keep
the top-k nodes in fp32 for k in --k-schedule, clipped to the number of
candidates, with at most 12 attempts in total including r01 (D39 cap).

The fp32 side is quant_pre_process'd exactly as ptq.py does it, so tensor and node
names match the graph ORT quantized (BatchNorm is folded there).

    python convert/rank_sensitivity.py --model aasist \\
        --fp32 exports/aasist_fp32.onnx \\
        --int8 runs/aasist/r01/aasist_int8.onnx \\
        --calib-dir $DATA_ROOT/LA/ASVspoof2019_LA_dev/flac \\
        --out results/phase_2_conversion_app/conversion_sensitivity/search_plan_aasist.json
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "convert"))

DIAG = REPO / "configs" / "conversion_sensitivity" / "diag_seed0.txt"
DEFAULT_K = "1,2,3,4,6,8,12,16,24,32,48"
MAX_ATTEMPTS = 12              # D39, including the default recipe r01
CAP_DB = 300.0                 # SQNR of an exact match (noise == 0)
QDQ = ("QuantizeLinear", "DequantizeLinear")


def sqnr_db(sig: float, noise: float) -> float:
    if noise <= 0.0:
        return CAP_DB
    if sig <= 0.0:
        return -CAP_DB
    return min(CAP_DB, 10.0 * math.log10(sig / noise))


def k_schedule(ks, n_candidates: int, max_attempts: int = MAX_ATTEMPTS) -> list[int]:
    """Clip the declared k values to the candidates; add 'all' once if the list
    overshoots; keep at most max_attempts - 1 (r01 is the default recipe)."""
    out = []
    for k in ks:
        k = min(int(k), n_candidates)
        if k >= 1 and k not in out:
            out.append(k)
        if k == n_candidates:
            break
    return out[: max_attempts - 1]


def qdq_pairs(qdq_graph, float_names: set):
    """Per quantized node: its activation inputs as (float_name, int8_name) pairs
    and its outputs. Weight inputs (DQ of an initializer) are skipped."""
    producer = {o: n for n in qdq_graph.node for o in n.output}
    inits = {i.name for i in qdq_graph.initializer}
    nodes = []
    for idx, n in enumerate(qdq_graph.node):
        if n.op_type in QDQ:
            continue
        ins, quantized = [], False
        for i in n.input:
            if not i:
                continue
            p = producer.get(i)
            if p is not None and p.op_type == "DequantizeLinear":
                quantized = True
                q = producer.get(p.input[0])
                src = q.input[0] if q is not None and q.op_type == "QuantizeLinear" else p.input[0]
                if src in inits:
                    continue                            # a weight, not an activation
                if src in float_names:
                    ins.append((src, i))
            elif i in float_names and i not in inits:
                ins.append((i, i))
        if not quantized:
            continue
        outs = [o for o in n.output if o in float_names]
        nodes.append({"idx": idx, "name": n.name, "op_type": n.op_type,
                      "inputs": ins, "outputs": outs})
    return nodes


def pre_process_like_ptq(fp32: Path, td: Path) -> Path:
    """Stage (ASCII, with any .onnx.data sidecar) and quant_pre_process, as ptq.py does."""
    from onnxruntime.quantization.shape_inference import quant_pre_process
    for f in sorted(fp32.parent.glob(fp32.stem + "*")):
        if f.is_file():
            shutil.copyfile(f, td / f.name)
    pre = td / (fp32.stem + "_pre.onnx")
    quant_pre_process(str(td / fp32.name), str(pre), skip_symbolic_shape=False)
    return pre


def expose_all(model: Path, out: Path) -> None:
    from onnxruntime.quantization.qdq_loss_debug import \
        modify_model_output_intermediate_tensors
    modify_model_output_intermediate_tensors(str(model), str(out))


def rank(fp32_pre: Path, int8: Path, reader, limit=None):
    import onnx
    import onnxruntime as ort
    g_f = onnx.load(str(fp32_pre)).graph
    g_q = onnx.load(str(int8)).graph
    float_names = {o for n in g_f.node for o in n.output} | {i.name for i in g_f.input}
    float_nodes = {n.name for n in g_f.node}
    cand = [c for c in qdq_pairs(g_q, float_names) if c["name"] in float_nodes]
    if not cand:
        raise SystemExit("no quantized nodes found in the int8 artifact")
    pairs = {(f, q) for c in cand for f, q in c["inputs"]} | \
            {(o, o) for c in cand for o in c["outputs"]}

    # Optimizations OFF on both sides so every intermediate tensor keeps its
    # graph name. The int8 side therefore runs as simulated quantization
    # (Q -> DQ -> fp32 op) rather than fused int8 kernels: right for locating
    # damage, and the reason this ranking is never reported as accuracy.
    td = Path(tempfile.mkdtemp(prefix="rank_ascii_"))
    aug_f, aug_q = td / "f.onnx", td / "q.onnx"
    expose_all(fp32_pre, aug_f)
    shutil.copyfile(int8, td / "q_in.onnx")
    expose_all(td / "q_in.onnx", aug_q)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s_f = ort.InferenceSession(str(aug_f), so, providers=["CPUExecutionProvider"])
    s_q = ort.InferenceSession(str(aug_q), so, providers=["CPUExecutionProvider"])
    # The augmented models expose tensor T as a flattened output "T" + postfix
    # (graph outputs keep their own name). Matching on bare names finds only
    # the graph output, so map every exposed output back to its tensor.
    from onnxruntime.quantization.qdq_loss_debug import _TENSOR_SAVE_POSTFIX as SFX

    def exposed(sess):
        m = {}
        for o in sess.get_outputs():
            m[o.name[: -len(SFX)] if o.name.endswith(SFX) else o.name] = o.name
        return m

    map_f, map_q = exposed(s_f), exposed(s_q)
    need_f = sorted({f for f, _ in pairs} & set(map_f))
    need_q = sorted({q for _, q in pairs} & set(map_q))
    sig = {p: 0.0 for p in pairs}
    noise = {p: 0.0 for p in pairs}
    n_clips, t0 = 0, time.time()
    while not (limit and n_clips >= limit):
        feed = reader.get_next()
        if feed is None:
            break
        of = dict(zip(need_f, s_f.run([map_f[t] for t in need_f], feed)))
        oq = dict(zip(need_q, s_q.run([map_q[t] for t in need_q], feed)))
        for f, q in pairs:
            if f in of and q in oq:
                x = np.asarray(of[f], np.float64).reshape(-1)
                y = np.asarray(oq[q], np.float64).reshape(-1)
                if x.shape != y.shape:
                    raise SystemExit(f"shape mismatch fp32 {f}{x.shape} vs int8 {q}{y.shape}")
                sig[(f, q)] += float(np.sum(x * x))
                noise[(f, q)] += float(np.sum((x - y) ** 2))
        n_clips += 1
    shutil.rmtree(td, ignore_errors=True)
    if n_clips == 0:
        raise SystemExit("no clips were read from the diagnostic manifest")
    measured = {p for p in pairs if sig[p] > 0 or noise[p] > 0}

    rows = []
    for c in cand:
        ins = [sqnr_db(sig[p], noise[p]) for p in c["inputs"] if p in measured]
        outs = [sqnr_db(sig[(o, o)], noise[(o, o)]) for o in c["outputs"] if (o, o) in measured]
        if not outs:
            continue                                  # nothing observable to rank on
        in_db = min(ins) if ins else CAP_DB
        out_db = min(outs)
        rows.append({"node": c["name"], "op_type": c["op_type"], "graph_idx": c["idx"],
                     "in_sqnr_db": round(in_db, 4), "out_sqnr_db": round(out_db, 4),
                     "damage_db": round(in_db - out_db, 4)})
    rows.sort(key=lambda r: (-r["damage_db"], r["graph_idx"]))
    unranked = sorted({c["name"] for c in cand} - {r["node"] for r in rows})
    return rows, unranked, n_clips, round(time.time() - t0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--int8", required=True, help="the r01 (default-recipe) artifact")
    ap.add_argument("--calib-dir", required=True, help="19LA dev audio (flac)")
    ap.add_argument("--diag-manifest", default=str(DIAG))
    ap.add_argument("--input-samples", type=int, default=None,
                    help="defaults to models.base.SPECS[model].input_samples")
    ap.add_argument("--k-schedule", default=DEFAULT_K)
    ap.add_argument("--calib-method", default="minmax",
                    help="calibration method of every exclusion attempt (r01's default)")
    ap.add_argument("--limit", type=int, default=None,
                    help="TESTS ONLY: stop after this many clips")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists: a search plan is frozen once written")
    fp32, int8 = Path(a.fp32), Path(a.int8)
    side_path = Path(str(int8).replace(".onnx", "_ptq_log.json"))
    if not side_path.is_file():
        raise SystemExit(f"{side_path} missing: r01 must be a ptq.py run-mode artifact")
    side = json.loads(side_path.read_text())
    if side.get("recipe_id") != "r01" or side.get("n_fp32_excluded") != 0 \
            or not side.get("pre_process", False):
        raise SystemExit("--int8 must be the default-recipe attempt r01 (no exclusions, "
                         f"pre_process on); its sidecar says recipe_id={side.get('recipe_id')}, "
                         f"K={side.get('n_fp32_excluded')}, pre_process={side.get('pre_process')}")
    if a.input_samples is None:
        from models.base import SPECS
        a.input_samples = SPECS[a.model].input_samples

    from ptq import WavCalibrationReader, artifact_hashes
    from deploybench.manifests import file_sha256, ids_digest, read_ids
    reader = WavCalibrationReader(Path(a.calib_dir), a.input_samples, "waveform",
                                  manifest=a.diag_manifest)
    with tempfile.TemporaryDirectory(prefix="rank_pre_") as td:
        pre = pre_process_like_ptq(fp32, Path(td))
        rows, unranked, n_clips, secs = rank(pre, int8, reader, a.limit)

    ks = k_schedule(a.k_schedule.split(","), len(rows))
    attempts = [{"recipe_id": "r01", "role": "default recipe (already run)",
                 "calib_method": side["calib_method"], "exclude_nodes": [],
                 "n_fp32_excluded": 0}]
    for j, k in enumerate(ks, start=2):
        attempts.append({"recipe_id": f"r{j:02d}", "role": f"top-{k} exclusion",
                         "calib_method": a.calib_method,
                         "exclude_nodes": [r["node"] for r in rows[:k]],
                         "n_fp32_excluded": k})
    plan = {
        "design": "PREREGISTRATION.md conversion sensitivity (D35/D39); D36 budget",
        "model": a.model, "frozen": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fp32_sha256": artifact_hashes(fp32), "r01_sha256": artifact_hashes(int8),
        "r01_sidecar_sha256": file_sha256(side_path),
        "search_calibration_manifest": side.get("calib_manifest"),
        "search_calibration_ids_sha256": side.get("calib_ids_sha256"),
        "diag_manifest": a.diag_manifest,
        "diag_ids_sha256": ids_digest(read_ids(Path(a.diag_manifest))),
        "diag_clips_used": n_clips, "rank_seconds": secs,
        "ranking_rule": "damage_db = min input SQNR - output SQNR (fp32 vs int8, "
                        "streamed over the diagnostic clips); largest first; ties by graph order",
        "k_schedule_declared": a.k_schedule, "max_attempts": MAX_ATTEMPTS,
        "budget": "int8 EER - fp32 EER <= 1.0 pp on the diagnostic subset (D36)",
        "selection_rule": "smallest K among qualifying attempts; ties -> earlier attempt (D39)",
        "attempts": attempts, "ranking": rows, "unranked_quantized_nodes": unranked,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(f"froze {len(attempts)} attempts for {a.model} "
          f"({len(rows)} ranked nodes, {n_clips} diag clips, {secs}s) -> {out}")
    for r in rows[:10]:
        print(f"  {r['node']:32s} {r['op_type']:12s} damage {r['damage_db']:8.2f} dB")


if __name__ == "__main__":
    main()
