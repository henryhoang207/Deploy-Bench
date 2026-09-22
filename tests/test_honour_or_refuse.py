"""Review B2 / B3 and the stress-label gap: every label a harness writes must be
something it actually did. Otherwise it refuses."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))
import score_cell  # noqa: E402

CELL = dict(model="tssdnet", precision="fp32", runtime="pytorch",
            dataset="asvspoof19_eval", condition="clean", seed=0, out="x.csv")


def test_score_cell_pytorch_fp32_goes_to_score_py():
    cmd = score_cell.build_cmd(dict(CELL))
    assert Path(cmd[1]).name == "score.py" and "--runtime" in cmd


def test_score_cell_ort_goes_to_score_onnx(tmp_path):
    onnx = tmp_path / "m.onnx"
    onnx.write_bytes(b"x")
    cmd = score_cell.build_cmd(dict(CELL, runtime="ort", precision="int8", onnx=str(onnx)))
    assert Path(cmd[1]).name == "score_onnx.py" and str(onnx) in cmd


@pytest.mark.parametrize("change", [
    dict(precision="int8"),                                  # pytorch is fp32-only
    dict(runtime="executorch"),                              # OD-4
    dict(runtime="ort", onnx="does/not/exist.onnx"),         # no artifact
    dict(condition="opus6k"),                                # stress not wired
    dict(runtime="tflite"),                                  # unknown runtime
])
def test_score_cell_refuses_what_it_cannot_honour(change):
    with pytest.raises(score_cell.Refused):
        score_cell.build_cmd(dict(CELL, **change))


def test_score_cell_requires_keys():
    cfg = dict(CELL)
    del cfg["seed"]
    with pytest.raises(score_cell.Refused):
        score_cell.build_cmd(cfg)


@pytest.mark.parametrize("extra", [["--precision", "int8"], ["--runtime", "ort"],
                                   ["--condition", "opus6k"]])
def test_score_py_refuses_labels_it_cannot_honour(tmp_path, extra):
    """Refused before any model or data is touched."""
    r = subprocess.run([sys.executable, str(REPO / "score.py"), "--model", "dummy",
                        "--dataset", "itw", "--out", str(tmp_path / "o.csv")] + extra,
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode != 0 and "refusing" in r.stderr
    assert not (tmp_path / "o.csv").exists()


def test_graph_precision_reads_the_artifact(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper
    import numpy as np
    from score_onnx import graph_precision

    def save(name, nodes, inits):
        g = helper.make_graph(nodes, "g",
                              [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
                              initializer=inits)
        p = tmp_path / name
        onnx.save(helper.make_model(g), str(p))
        return p

    ident = [helper.make_node("Identity", ["x"], ["y"])]
    w32 = numpy_helper.from_array(np.zeros(1, np.float32), "w")
    w16 = numpy_helper.from_array(np.zeros(1, np.float16), "w")
    q = [helper.make_node("QuantizeLinear", ["x", "s"], ["q"]),
         helper.make_node("DequantizeLinear", ["q", "s"], ["y"])]
    s = numpy_helper.from_array(np.ones(1, np.float32), "s")
    assert graph_precision(save("a.onnx", ident, [w32])) == "fp32"
    assert graph_precision(save("b.onnx", ident, [w16])) == "fp16"
    assert graph_precision(save("c.onnx", q, [s])) == "int8"


# ---------------------------------------------------------------- B3
def _db_and_taus(tmp_path):
    import run_analysis as RA
    db = RA.selftest_db()
    return RA, db, RA.selftest_freeze(db, tmp_path / "taus")


def test_analysis_uses_frozen_taus(tmp_path):
    RA, db, taus = _db_and_taus(tmp_path)
    ttg, qts = RA.analyze(db, tmp_path / "out", taus)
    frozen = json.loads((taus / "tau_goodmodel.json").read_text())["taus"]
    got = ttg[(ttg.model == "goodmodel") & (ttg.precision == "int8")
              & (ttg.policy == "eer")].tau.unique()
    assert list(got) == [frozen["eer"]]            # int8 cells use the FROZEN fp32 tau
    assert not qts.empty


def test_analysis_refuses_drifted_dev_rows(tmp_path):
    RA, db, taus = _db_and_taus(tmp_path)
    m = (db.model == "goodmodel") & (db.dataset == "asvspoof19_dev") & (db.precision == "fp32")
    db.loc[db[m].index[:50], "score"] += 0.5
    with pytest.raises(SystemExit):
        RA.analyze(db, tmp_path / "out", taus)


def test_analysis_refuses_second_dev_set(tmp_path):
    RA, db, taus = _db_and_taus(tmp_path)
    extra = db[db.dataset == "asvspoof19_dev"].head(10).copy()
    extra["dataset"] = "asvspoof5_dev"
    import pandas as pd
    with pytest.raises(SystemExit):
        RA.analyze(pd.concat([db, extra]), tmp_path / "out", taus)


def test_analysis_refuses_missing_frozen_tau(tmp_path):
    RA, db, taus = _db_and_taus(tmp_path)
    (taus / "tau_weakmodel.json").unlink()
    with pytest.raises(SystemExit):
        RA.analyze(db, tmp_path / "out", taus)
