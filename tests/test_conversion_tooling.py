"""D35/D38/D39 conversion-sensitivity tooling: manifests, labels, ptq run mode.

The ptq tests run the real ORT quantizer on a tiny synthetic Conv graph, so they
exercise the CLI end to end without any dataset or checkpoint."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "convert"))
from deploybench import artifacts  # noqa: E402
from deploybench.manifests import draw_balanced, ids_digest, read_ids  # noqa: E402
import sensitivity_manifests as sm  # noqa: E402

DEV_CSV = REPO / "results" / "phase_1_reproduction" / "scores_dev_aasist.csv"
D39 = REPO / "configs" / "conversion_sensitivity"


# ---- manifests -------------------------------------------------------------
def _dev_pool():
    rows, _ = sm.pool_from_csv(DEV_CSV)
    return rows


def test_shared_draw_reproduces_the_committed_tssdnet_calibration_manifest():
    rows = _dev_pool()
    bona = [u for u, lab in rows if lab == 1]
    spoof = [u for u, lab in rows if lab == 0]
    got = draw_balanced(bona, spoof, 250, 20260725)
    assert got == read_ids(REPO / "configs" / "calibration_500_tssdnet.txt")


def test_draw_does_not_depend_on_input_order():
    b, s = [f"b{i:03d}" for i in range(50)], [f"s{i:03d}" for i in range(80)]
    assert draw_balanced(b, s, 10, 7) == draw_balanced(b[::-1], s[::-1], 10, 7)


def test_d39_manifests_match_a_fresh_draw_and_the_index():
    built, info = sm.build(_dev_pool())
    index = json.loads((D39 / "index.json").read_text(encoding="utf-8"))
    assert index["pool"]["dev_rows_sha256"] == info["dev_rows_sha256"]
    for name, (ids, n, seed) in built.items():
        assert read_ids(D39 / name) == ids, name
        assert index["manifests"][name]["ids_sha256"] == ids_digest(ids)
        assert index["manifests"][name]["seed"] == seed
        assert len(ids) == 2 * n


def test_d39_design_sizes_and_disjointness():
    built, info = sm.build(_dev_pool())
    diag = set(built["diag_seed0.txt"][0])
    assert len(diag) == 2000
    assert (info["pool_bona"], info["pool_spoof"]) == (1548, 21296)
    for name in ["calib_search_seed100.txt"] + [f"calib_seed{s}.txt" for s in range(1, 6)]:
        ids = built[name][0]
        assert len(ids) == len(set(ids)) == 500
        assert not diag & set(ids), name


def test_d39_refuses_a_pool_that_is_not_19la_dev():
    rows = _dev_pool()[:-1]
    with pytest.raises(ValueError, match="ASVspoof19 LA dev is"):
        sm.build(rows)


# ---- D38 labels ------------------------------------------------------------
def test_precision_labels():
    assert artifacts.precision_label(0) == "int8"
    assert artifacts.precision_label(8) == "int8-mixed (K=8)"
    with pytest.raises(ValueError):
        artifacts.precision_label(-1)


def test_pre_d38_sidecar_gets_its_k_from_excluded_nodes(tmp_path):
    side = tmp_path / "tssdnet_int8_ptq_log.json"
    side.write_text(json.dumps({"excluded_nodes": [f"n{i}" for i in range(8)]}))
    assert artifacts.report_label("int8", side) == "int8-mixed (K=8)"


def test_int8_row_without_sidecar_is_refused(tmp_path):
    assert artifacts.report_label("fp16") == "fp16"
    with pytest.raises(artifacts.MissingSidecar):
        artifacts.report_label("int8", tmp_path / "absent.json")
    with pytest.raises(artifacts.MissingSidecar):
        artifacts.report_label("int8")


def test_self_contradicting_sidecar_is_refused():
    with pytest.raises(ValueError, match="disagrees"):
        artifacts.label_from_sidecar({"n_fp32_excluded": 2, "excluded_nodes": ["a"]})


# ---- ptq.py run mode (real ORT quantizer, synthetic graph) ------------------
onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
sf = pytest.importorskip("soundfile")
N = 64


def _tiny_model(path: Path):
    from onnx import TensorProto, helper, numpy_helper
    rng = np.random.default_rng(0)
    inits = [numpy_helper.from_array(np.array([1], np.int64), "ax"),
             numpy_helper.from_array(rng.normal(size=(4, 1, 3)).astype(np.float32), "w1"),
             numpy_helper.from_array(np.zeros(4, np.float32), "b1"),
             numpy_helper.from_array(rng.normal(size=(1, 4, 3)).astype(np.float32), "w2"),
             numpy_helper.from_array(np.zeros(1, np.float32), "b2")]
    nodes = [helper.make_node("Unsqueeze", ["waveform", "ax"], ["x"], name="unsq"),
             helper.make_node("Conv", ["x", "w1", "b1"], ["y"], name="conv_a"),
             helper.make_node("Relu", ["y"], ["r"], name="relu"),
             helper.make_node("Conv", ["r", "w2", "b2"], ["z"], name="conv_b"),
             helper.make_node("ReduceMean", ["z"], ["score"], name="mean",
                              axes=[1, 2], keepdims=0)]
    g = helper.make_graph(nodes, "tiny",
                          [helper.make_tensor_value_info("waveform", TensorProto.FLOAT, [1, N])],
                          [helper.make_tensor_value_info("score", TensorProto.FLOAT, [1])],
                          inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.save(m, str(path))


@pytest.fixture()
def calib(tmp_path):
    model = tmp_path / "tiny_fp32.onnx"
    _tiny_model(model)
    wavs = tmp_path / "wavs"
    wavs.mkdir()
    rng = np.random.default_rng(1)
    ids = [f"LA_D_{i:07d}" for i in range(6)]
    for u in ids:
        sf.write(str(wavs / f"{u}.wav"), rng.uniform(-0.5, 0.5, 200).astype(np.float32), 16000)
    man = tmp_path / "calib.txt"
    man.write_text("# test manifest\n" + "\n".join(ids) + "\n", encoding="utf-8")
    return tmp_path, model, wavs, man


def _ptq(tmp, model, wavs, man, run, *extra):
    cmd = [sys.executable, str(REPO / "convert" / "ptq.py"), "--onnx", str(model),
           "--calib-dir", str(wavs), "--calib-manifest", str(man),
           "--input-samples", str(N), "--run-dir", str(tmp / run),
           "--recipe-id", run, "--calib-seed", "1",
           "--ledger", str(tmp / "ledger.jsonl"), *extra]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(tmp))


def _ledger(tmp):
    return [json.loads(line) for line in (tmp / "ledger.jsonl").read_text().splitlines()]


def test_run_mode_writes_artifact_sidecar_and_ledger(calib):
    tmp, model, wavs, man = calib
    r = _ptq(tmp, model, wavs, man, "r01")
    assert r.returncode == 0, r.stderr[-2000:]
    run = tmp / "r01"
    assert (run / "tiny_int8.onnx").is_file()
    assert not list(run.glob("*_pre.onnx"))           # staging keeps it out
    side = json.loads((run / "tiny_int8_ptq_log.json").read_text())
    assert side["precision_label"] == "int8" and side["n_fp32_excluded"] == 0
    assert side["calib_clips"] == 6 and side["calib_seed"] == 1
    assert side["calib_ids_sha256"] == ids_digest(read_ids(man))
    assert set(side["dst_sha256"]) == {"tiny_int8.onnx"}
    (entry,) = _ledger(tmp)
    assert entry["outcome"] == "ok" and entry["recipe_id"] == "r01"


def test_run_mode_mixed_label_and_no_overwrite(calib):
    tmp, model, wavs, man = calib
    r = _ptq(tmp, model, wavs, man, "r02", "--exclude-nodes", "conv_a")
    assert r.returncode == 0, r.stderr[-2000:]
    side = json.loads((tmp / "r02" / "tiny_int8_ptq_log.json").read_text())
    assert side["precision_label"] == "int8-mixed (K=1)"
    again = _ptq(tmp, model, wavs, man, "r02")
    assert again.returncode != 0 and "already has files" in again.stderr


def test_unknown_exclusion_is_refused_and_logged_as_failed(calib):
    tmp, model, wavs, man = calib
    r = _ptq(tmp, model, wavs, man, "r03", "--exclude-nodes", "conv_typo")
    assert r.returncode != 0 and "not in the graph" in r.stderr
    (entry,) = _ledger(tmp)
    assert entry["outcome"] == "failed" and "conv_typo" in entry["error"]
    assert not (tmp / "r03" / "tiny_int8.onnx").exists()


def test_run_mode_requires_traceability(calib):
    tmp, model, wavs, man = calib
    cmd = [sys.executable, str(REPO / "convert" / "ptq.py"), "--onnx", str(model),
           "--calib-dir", str(wavs), "--input-samples", str(N),
           "--run-dir", str(tmp / "r04")]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(tmp))
    assert r.returncode != 0 and "--recipe-id" in r.stderr and "--calib-manifest" in r.stderr


def test_manifest_is_never_truncated(calib, tmp_path):
    tmp, model, wavs, man = calib
    ptq = __import__("ptq")
    reader = ptq.WavCalibrationReader(wavs, N, "waveform", limit=2, manifest=str(man))
    assert len(reader.files) == 6


# ---- rank_sensitivity.py (real ORT, synthetic graph) -------------------------
def _run(script, *args, cwd):
    return subprocess.run([sys.executable, str(REPO / "convert" / script), *args],
                          capture_output=True, text=True, cwd=str(cwd))


def test_k_schedule_clips_and_caps():
    import rank_sensitivity as rs
    assert rs.k_schedule([1, 2, 3, 4, 6, 8, 12], 5) == [1, 2, 3, 4, 5]
    assert rs.k_schedule([1, 2, 3], 1) == [1]
    assert len(rs.k_schedule(range(1, 100), 99)) == rs.MAX_ATTEMPTS - 1


def test_rank_freezes_a_plan_from_r01(calib):
    tmp, model, wavs, man = calib
    r = subprocess.run([sys.executable, str(REPO / "convert" / "ptq.py"), "--onnx", str(model),
                        "--calib-dir", str(wavs), "--calib-manifest", str(man),
                        "--input-samples", str(N), "--run-dir", str(tmp / "runs" / "r01"),
                        "--recipe-id", "r01", "--calib-seed", "100",
                        "--ledger", str(tmp / "ledger.jsonl")],
                       capture_output=True, text=True, cwd=str(tmp))
    assert r.returncode == 0, r.stderr[-2000:]
    int8 = tmp / "runs" / "r01" / "tiny_int8.onnx"
    plan_path = tmp / "plan.json"
    args = ["--model", "tiny", "--fp32", str(model), "--int8", str(int8),
            "--calib-dir", str(wavs), "--diag-manifest", str(man),
            "--input-samples", str(N), "--out", str(plan_path)]
    r = _run("rank_sensitivity.py", *args, cwd=tmp)
    assert r.returncode == 0, r.stderr[-2000:]
    plan = json.loads(plan_path.read_text())
    ranked = [x["node"] for x in plan["ranking"]]
    assert {"conv_a", "conv_b"} <= set(ranked)
    assert plan["attempts"][0]["recipe_id"] == "r01"
    assert plan["attempts"][0]["exclude_nodes"] == []
    for i, t in enumerate(plan["attempts"][1:], start=2):
        assert t["recipe_id"] == f"r{i:02d}"
        assert t["exclude_nodes"] == ranked[: t["n_fp32_excluded"]]
    assert plan["diag_clips_used"] == 6
    again = _run("rank_sensitivity.py", *args, cwd=tmp)
    assert again.returncode != 0 and "frozen" in again.stderr


def test_rank_refuses_an_artifact_that_is_not_r01(calib):
    tmp, model, wavs, man = calib
    r = _ptq(tmp, model, wavs, man, "r05", "--exclude-nodes", "conv_a")
    assert r.returncode == 0, r.stderr[-2000:]
    r = _run("rank_sensitivity.py", "--model", "tiny", "--fp32", str(model),
             "--int8", str(tmp / "r05" / "tiny_int8.onnx"), "--calib-dir", str(wavs),
             "--diag-manifest", str(man), "--input-samples", str(N),
             "--out", str(tmp / "plan.json"), cwd=tmp)
    assert r.returncode != 0 and "r01" in r.stderr


# ---- select_recipe.py (synthetic scores) -------------------------------------
def _write_scores(path, ids, labels, scores, model="m", precision="int8"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["utt_id", "label", "score", "model", "dataset", "precision",
                    "runtime", "condition", "seed"])
        for u, lab, s in zip(ids, labels, scores):
            w.writerow([u, lab, s, model, "asvspoof19_dev", precision, "ort", "clean", 0])


@pytest.fixture()
def sel(tmp_path):
    rng = np.random.default_rng(3)
    ids = [f"LA_D_{i:07d}" for i in range(400)]
    labels = [1] * 200 + [0] * 200
    diag = tmp_path / "diag.txt"
    diag.write_text("\n".join(ids) + "\n")
    base = np.where(np.array(labels) == 1, 1.0, -1.0) + rng.normal(0, 0.9, 400)
    _write_scores(tmp_path / "dev_fp32.csv", ids, labels, base, precision="fp32")
    # Noise added to the fp32 scores sets how far the int8 EER moves.
    variants = {n: base + rng.normal(0, n, 400) for n in (0.0, 0.2, 3.0)}
    return tmp_path, ids, labels, diag, variants


def _plan(tmp, diag, ks):
    atts = [{"recipe_id": f"r{i + 1:02d}", "n_fp32_excluded": k, "exclude_nodes": []}
            for i, k in enumerate(ks)]
    p = tmp / "plan.json"
    p.write_text(json.dumps({"model": "m", "diag_manifest": str(diag),
                             "diag_ids_sha256": ids_digest(read_ids(diag)),
                             "attempts": atts}))
    return p


def _ledger_line(f, rid, seed, k, outcome="ok"):
    f.write(json.dumps({"recipe_id": rid, "calib_seed": seed, "n_fp32_excluded": k,
                        "outcome": outcome,
                        "error": None if outcome == "ok" else "boom"}) + "\n")


def _select(tmp, *args):
    r = _run("select_recipe.py", *args, "--dev-scores", str(tmp / "dev_fp32.csv"),
             "--out", str(tmp / "sel.json"), cwd=tmp)
    return r, (json.loads((tmp / "sel.json").read_text()) if r.returncode == 0 else None)


def test_search_locks_smallest_qualifying_k_and_counts_failures(sel):
    tmp, ids, labels, diag, v = sel
    plan = _plan(tmp, diag, [0, 1, 2, 4])
    with open(tmp / "ledger.jsonl", "w") as f:
        _ledger_line(f, "r01", 100, 0)                 # heavy noise: misses budget
        _ledger_line(f, "r02", 100, 1, "failed")       # conversion failure
        _ledger_line(f, "r03", 100, 2)                 # exact: qualifies
        _ledger_line(f, "r04", 100, 4)                 # small noise
    _write_scores(tmp / "runs" / "r01" / "scores_diag.csv", ids, labels, v[3.0])
    _write_scores(tmp / "runs" / "r03" / "scores_diag.csv", ids, labels, v[0.0])
    _write_scores(tmp / "runs" / "r04" / "scores_diag.csv", ids, labels, v[0.2])
    r, out = _select(tmp, "search", "--plan", str(plan), "--ledger",
                     str(tmp / "ledger.jsonl"), "--runs-root", str(tmp / "runs"))
    assert r.returncode == 0, r.stderr
    assert out["locked_recipe"] == "r03" and out["locked_k"] == 2
    assert out["n_failed"] == 1
    by = {x["recipe_id"]: x for x in out["attempts"]}
    assert by["r01"]["qualifies"] is False and by["r02"]["outcome"] == "failed"
    assert abs(by["r03"]["delta_eer_pp"]) < 1e-9


def test_search_refuses_while_an_attempt_is_unaccounted(sel):
    tmp, ids, labels, diag, v = sel
    plan = _plan(tmp, diag, [0, 1])
    with open(tmp / "ledger.jsonl", "w") as f:
        _ledger_line(f, "r01", 100, 0)
    _write_scores(tmp / "runs" / "r01" / "scores_diag.csv", ids, labels, v[0.0])
    r, _ = _select(tmp, "search", "--plan", str(plan), "--ledger",
                   str(tmp / "ledger.jsonl"), "--runs-root", str(tmp / "runs"))
    assert r.returncode != 0 and "no ledger entry" in r.stderr


def test_search_refuses_a_retried_attempt(sel):
    tmp, ids, labels, diag, v = sel
    plan = _plan(tmp, diag, [0])
    with open(tmp / "ledger.jsonl", "w") as f:
        _ledger_line(f, "r01", 100, 0, "failed")
        _ledger_line(f, "r01", 100, 0)
    r, _ = _select(tmp, "search", "--plan", str(plan), "--ledger",
                   str(tmp / "ledger.jsonl"), "--runs-root", str(tmp / "runs"))
    assert r.returncode != 0 and "never retried" in r.stderr


def test_canonical_is_lowest_qualifying_seed_or_none(sel):
    tmp, ids, labels, diag, v = sel
    plan = _plan(tmp, diag, [0, 3])
    with open(tmp / "ledger.jsonl", "w") as f:
        for s in (1, 2, 3, 4, 5):
            _ledger_line(f, "r02", s, 3, "failed" if s == 1 else "ok")
    for s, noise in zip((2, 3, 4, 5), (3.0, 0.0, 0.0, 0.2)):
        _write_scores(tmp / "runs" / f"r02_seed{s}" / "scores_diag.csv", ids, labels, v[noise])
    args = ["canonical", "--plan", str(plan), "--ledger", str(tmp / "ledger.jsonl"),
            "--runs-root", str(tmp / "runs"), "--locked", "r02"]
    r, out = _select(tmp, *args)
    assert r.returncode == 0, r.stderr
    assert out["canonical_seed"] == 3 and out["n_failed"] == 1
    for s in (3, 4, 5):
        _write_scores(tmp / "runs" / f"r02_seed{s}" / "scores_diag.csv", ids, labels, v[3.0])
    r, out = _select(tmp, *args)
    assert r.returncode == 0, r.stderr
    assert out["canonical_seed"] is None and "no qualifying" in out["note"]


def test_plan_k_mismatch_with_ledger_is_refused(sel):
    tmp, ids, labels, diag, v = sel
    plan = _plan(tmp, diag, [0])
    with open(tmp / "ledger.jsonl", "w") as f:
        _ledger_line(f, "r01", 100, 3)
    r, _ = _select(tmp, "search", "--plan", str(plan), "--ledger",
                   str(tmp / "ledger.jsonl"), "--runs-root", str(tmp / "runs"))
    assert r.returncode != 0 and "frozen plan says" in r.stderr


# ---- run_sensitivity.py driver ------------------------------------------------
def test_driver_refuses_models_outside_d35_scope():
    import run_sensitivity as rsd
    with pytest.raises(SystemExit, match="outside the D35 scope"):
        rsd.Model("xlsr_aasist")


def test_driver_never_reruns_a_logged_attempt(tmp_path, monkeypatch):
    import run_sensitivity as rsd
    m = rsd.Model("tssdnet")
    m.ledger = tmp_path / "ledger.jsonl"
    m.ledger.write_text(json.dumps({"recipe_id": "r03", "calib_seed": 100,
                                    "outcome": "failed", "error": "boom"}) + "\n")
    calls = []
    monkeypatch.setattr(rsd, "py", lambda *a: calls.append(a) or 0)
    e = m.convert("r03", 100, tmp_path / "m.txt", "r03")
    assert e["outcome"] == "failed" and calls == []


def test_driver_stops_if_ptq_leaves_no_ledger_entry(tmp_path, monkeypatch):
    import run_sensitivity as rsd
    m = rsd.Model("tssdnet")
    m.ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(rsd, "py", lambda *a: 2)
    with pytest.raises(SystemExit, match="no ledger entry"):
        m.convert("r04", 100, tmp_path / "m.txt", "r04")


def test_driver_refuses_quantization_without_a_passing_parity_report(tmp_path):
    import run_sensitivity as rsd
    m = rsd.Model("tssdnet")
    m.parity = tmp_path / "parity.json"
    with pytest.raises(SystemExit, match="parity"):
        rsd.step_r01(m)
    m.parity.write_text(json.dumps({"pass": False}))
    with pytest.raises(SystemExit, match="parity"):
        rsd.step_r01(m)


def test_attempt_that_left_no_int8_ops_is_recorded_not_scored(sel):
    tmp, ids, labels, diag, v = sel
    plan = _plan(tmp, diag, [0, 5])
    with open(tmp / "ledger.jsonl", "w") as f:
        _ledger_line(f, "r01", 100, 0)
        _ledger_line(f, "r02", 100, 5)
    _write_scores(tmp / "runs" / "r01" / "scores_diag.csv", ids, labels, v[3.0])
    (tmp / "runs" / "r02").mkdir(parents=True)
    (tmp / "runs" / "r02" / "m_int8_ptq_log.json").write_text(
        json.dumps({"n_quantize_linear": 0, "n_dequantize_linear": 0}))
    r, out = _select(tmp, "search", "--plan", str(plan), "--ledger",
                     str(tmp / "ledger.jsonl"), "--runs-root", str(tmp / "runs"))
    assert r.returncode == 0, r.stderr
    by = {x["recipe_id"]: x for x in out["attempts"]}
    assert by["r02"]["outcome"] == "no_int8_ops" and by["r02"]["qualifies"] is False
    assert out["locked_recipe"] is None


# ---- DEV-1 percentile arm ------------------------------------------------------
CS_RESULTS = REPO / "results" / "phase_2_conversion_app" / "conversion_sensitivity"


@pytest.mark.parametrize("model,expect_n,dropped", [
    ("tssdnet", 11, ["r12"]),      # r12 = all 37 ranked nodes -> no int8 possible
    ("aasist", 12, []),
    ("aasist_l", 12, []),
    ("rawnet2", 12, []),
])
def test_percentile_plan_reuses_the_frozen_exclusions(model, expect_n, dropped):
    import run_sensitivity as rsd
    base = json.loads((CS_RESULTS / f"search_plan_{model}.json").read_text())
    plan = rsd.derive_percentile_plan(base)
    assert len(plan["attempts"]) == expect_n and plan["dropped_attempts"] == dropped
    kept = [t for t in base["attempts"] if t["recipe_id"] not in dropped]
    for p, b in zip(plan["attempts"], kept):
        assert p["calib_method"] == "percentile"
        assert p["exclude_nodes"] == b["exclude_nodes"]
        assert p["base_recipe_id"] == b["recipe_id"]
    assert plan["diag_ids_sha256"] == base["diag_ids_sha256"]
    assert plan["ranking"] == base["ranking"]


def test_percentile_arm_paths_never_touch_the_registered_arm():
    import run_sensitivity as rsd
    reg, dev = rsd.Model("rawnet2"), rsd.Model("rawnet2", arm="percentile")
    for attr in ("dir", "ledger", "plan", "selection", "canonical", "repeatability"):
        assert getattr(reg, attr) != getattr(dev, attr), attr
    assert reg.parity == dev.parity            # both arms quantize the same export


# ---- to_onnx: "converts" means ORT loads and runs it ---------------------------
def test_ort_smoke_accepts_a_good_graph_and_rejects_an_unloadable_one(tmp_path):
    from onnx import TensorProto, helper
    to_onnx = __import__("to_onnx")
    good = tmp_path / "good.onnx"
    _tiny_model(good)
    to_onnx.ort_smoke(good, N)
    # Cast declares FLOAT16 output but its `to` says FLOAT: onnx.checker accepts it,
    # ORT refuses to load it (the rawnet2 fp16 failure, 2026-09-29).
    g = helper.make_graph(
        [helper.make_node("Cast", ["waveform"], ["c"], to=TensorProto.FLOAT),
         helper.make_node("ReduceMean", ["c"], ["score"], axes=[1], keepdims=0)],
        "bad", [helper.make_tensor_value_info("waveform", TensorProto.FLOAT, [1, N])],
        [helper.make_tensor_value_info("score", TensorProto.FLOAT, [1])],
        value_info=[helper.make_tensor_value_info("c", TensorProto.FLOAT16, [1, N])])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    bad = tmp_path / "bad.onnx"
    onnx.save(m, str(bad))
    with pytest.raises(Exception):
        to_onnx.ort_smoke(bad, N)


def test_chunked_percentile_calibration_is_deterministic_and_recorded(calib):
    tmp, model, wavs, man = calib
    outs = []
    for run in ("c1", "c2"):
        r = _ptq(tmp, model, wavs, man, run, "--calib-method", "percentile", "--calib-chunk", "4")
        assert r.returncode == 0, r.stderr[-2000:]
        side = json.loads((tmp / run / "tiny_int8_ptq_log.json").read_text())
        assert side["calib_chunk"] == 4 and side["calib_method"] == "percentile"
        outs.append(side["dst_sha256"]["tiny_int8.onnx"])
    assert outs[0] == outs[1]


def test_calib_chunk_is_refused_for_minmax(calib):
    tmp, model, wavs, man = calib
    r = _ptq(tmp, model, wavs, man, "c3", "--calib-chunk", "4")
    assert r.returncode != 0 and "only to percentile" in r.stderr


def test_an_artifact_without_qdq_ops_is_never_labelled_int8():
    side = {"n_quantize_linear": 0, "n_dequantize_linear": 0, "n_fp32_excluded": 37,
            "excluded_nodes": [f"n{i}" for i in range(37)]}
    with pytest.raises(artifacts.NotInt8):
        artifacts.label_from_sidecar(side)


def test_ptq_labels_an_all_excluded_run_as_fp32(calib):
    tmp, model, wavs, man = calib
    r = _ptq(tmp, model, wavs, man, "all_ex", "--exclude-nodes", "unsq,conv_a,relu,conv_b,mean")
    assert r.returncode == 0, r.stderr[-2000:]
    side = json.loads((tmp / "all_ex" / "tiny_int8_ptq_log.json").read_text())
    assert side["n_quantize_linear"] == side["n_dequantize_linear"] == 0
    assert side["precision_label"] == "none: fp32 graph (no Q/DQ ops)"


# ---- fp16 node-block repair (D46) -------------------------------------------
def _cast_model():
    """The rawnet2 pattern: a weight Cast(to=FLOAT) feeding a Conv."""
    from onnx import TensorProto, helper, numpy_helper
    w = numpy_helper.from_array(np.ones((2, 1, 3), np.float32), "w")
    nodes = [helper.make_node("Cast", ["w"], ["wc"], name="/m/Sinc_conv/Cast", to=TensorProto.FLOAT),
             helper.make_node("Conv", ["waveform3d", "wc"], ["score"], name="conv")]
    g = helper.make_graph(nodes, "cast",
                          [helper.make_tensor_value_info("waveform3d", TensorProto.FLOAT, [1, 1, 8])],
                          [helper.make_tensor_value_info("score", TensorProto.FLOAT, [1, 2, 6])], [w])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    return m


def test_fp16_cast_bug_and_node_block_repair():
    pytest.importorskip("onnxconverter_common")
    import onnxruntime as ort
    from to_onnx import to_fp16

    def loads(m):
        try:
            s = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
        except Exception:                                       # noqa: BLE001
            return False
        out = s.run(None, {"waveform3d": np.ones((1, 1, 8), np.float32)})[0]
        return bool(np.allclose(out, 3.0))

    assert not loads(to_fp16(_cast_model()))                    # the 2026-09-29 failure
    assert loads(to_fp16(_cast_model(), ["/m/Sinc_conv/Cast"]))  # the repair
    with pytest.raises(ValueError, match="not in the graph"):
        to_fp16(_cast_model(), ["/m/Sinc_conv/Cast_typo"])
