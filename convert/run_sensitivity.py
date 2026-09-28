"""Driver for the D35/D39 conversion-sensitivity study, one model at a time (sol1).

Steps, in order. Each step is resumable and NEVER re-runs an attempt that already
has a ledger entry (D39: failures are results; nothing is retried or replaced).

  export     exports/<model>_fp32.onnx via convert/to_onnx.py, if absent
  parity     GATE: PyTorch fp32 (CPU) vs that ONNX fp32 export on the frozen
             1,000-clip 19LA-eval parity manifest, zero decision flips at the
             three frozen taus (tests/parity.py --mode conversion). Every later
             step refuses to run until this report exists and passed.
  r01        default recipe (ptq.py defaults) on calib_search_seed100.txt, then
             scored on the diagnostic subset
  rank       freeze search_plan_<model>.json from r01 (rank_sensitivity.py)
  search     every frozen attempt r02..rNN: convert, then score the diagnostic subset
  select     locked recipe (select_recipe.py search)
  repeats    the locked recipe on calib_seed1..5 + one identical-manifest rebuild
             of seed 1 (the repeatability control); each scored on the diagnostic
             subset AND on ASVspoof19 LA eval (D41)
  canonical  device artifact = lowest qualifying seed (select_recipe.py canonical);
             writes repeatability_<model>.json
  all        every step above, in order

Everything lands in results/phase_2_conversion_app/conversion_sensitivity/<model>/
(.onnx files are gitignored; sidecars, ledgers, plans and scores are tracked).

    source /mnt/data8tb/henry/env.sh && pick_gpu
    python convert/run_sensitivity.py --model aasist --checkpoint <ckpt> all

--checkpoint is required for export and parity: the same file the Phase-1
reproduction scored (xfer/repro_runs.sh), recorded with its sha256.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "convert"))
OUT = REPO / "results" / "phase_2_conversion_app" / "conversion_sensitivity"
MAN = REPO / "configs" / "conversion_sensitivity"
DIAG = MAN / "diag_seed0.txt"
SEARCH_MAN = MAN / "calib_search_seed100.txt"
PARITY_MANIFEST = REPO / "configs" / "parity_manifest_tssdnet.txt"   # model-independent ids
TAUS = REPO / "results" / "phase_1_reproduction" / "tau_{model}.json"
SEEDS = (1, 2, 3, 4, 5)
RANKED = ("aasist", "aasist_l", "rawnet2", "tssdnet")     # D35 scope; XLS-R excluded


def py(*args):
    """Run a repo script; a non-zero exit stops the driver (loud)."""
    cmd = [sys.executable, *map(str, args)]
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(REPO))
    return r.returncode


def must(*args):
    rc = py(*args)
    if rc != 0:
        raise SystemExit(f"step failed (exit {rc}); fix the cause, then re-run the step")


class Model:
    def __init__(self, name: str, checkpoint=None):
        if name not in RANKED:
            raise SystemExit(f"{name} is outside the D35 scope {RANKED}")
        from deploybench.data import DATA_ROOT, REGISTRY
        from models.base import SPECS
        self.name = name
        self.checkpoint = checkpoint
        self.n = SPECS[name].input_samples
        self.dev_flac = DATA_ROOT / REGISTRY["asvspoof19_dev"][0] / "flac"
        self.fp32 = REPO / "exports" / f"{name}_fp32.onnx"
        self.dir = OUT / name
        self.ledger = OUT / f"ledger_{name}.jsonl"
        self.plan = OUT / f"search_plan_{name}.json"
        self.selection = OUT / f"selection_{name}.json"
        self.canonical = OUT / f"canonical_{name}.json"
        self.parity = self.dir / "parity_report_fp32_conversion.json"

    def need_checkpoint(self):
        if not self.checkpoint or not Path(self.checkpoint).is_file():
            raise SystemExit(f"--checkpoint is required and must exist (got {self.checkpoint})")
        return self.checkpoint

    def logged(self, rid: str, seed: int):
        if not self.ledger.is_file():
            return None
        for line in open(self.ledger, encoding="utf-8"):
            e = json.loads(line) if line.strip() else {}
            if e.get("recipe_id") == rid and e.get("calib_seed") == seed:
                return e
        return None

    def convert(self, rid: str, seed: int, manifest: Path, run: str,
                calib_method: str = "minmax", exclude=()):
        """One PTQ attempt, unless the ledger already has it. A failed conversion
        is logged by ptq.py and returned as such; the driver moves on."""
        e = self.logged(rid, seed)
        if e is not None:
            print(f"= {rid} seed {seed}: already logged ({e['outcome']}), not re-run")
            return e
        py(REPO / "convert" / "ptq.py", "--onnx", self.fp32, "--calib-dir", self.dev_flac,
           "--calib-manifest", manifest, "--input-samples", self.n,
           "--calib-method", calib_method, "--exclude-nodes", ",".join(exclude),
           "--run-dir", self.dir / run, "--recipe-id", rid, "--calib-seed", seed,
           "--ledger", self.ledger)
        e = self.logged(rid, seed)
        if e is None:
            raise SystemExit(f"ptq.py left no ledger entry for {rid} seed {seed}")
        return e

    def score(self, run: str, dataset: str, out_name: str, manifest: Path = None):
        run_dir = self.dir / run
        out = run_dir / out_name
        if out.is_file():
            print(f"= {out.relative_to(REPO)} exists, not re-scored")
            return
        args = [REPO / "score_onnx.py", "--onnx", run_dir / f"{self.name}_int8.onnx",
                "--model", self.name, "--dataset", dataset, "--precision", "int8",
                "--runtime", "ort", "--out", out]
        if manifest:
            args += ["--manifest", manifest]
        must(*args)


def step_export(m: Model):
    if m.fp32.is_file():
        print(f"= {m.fp32.relative_to(REPO)} exists")
        return
    must(REPO / "convert" / "to_onnx.py", "--model", m.name, "--precision", "fp32",
         "--checkpoint", m.need_checkpoint())
    log = json.loads((REPO / "exports" / f"conversion_log_{m.name}_ort_fp32.json").read_text())
    if not log.get("converts"):
        raise SystemExit(f"fp32 export failed: {log.get('error')} (an H6 result; record it)")


def step_parity(m: Model):
    if m.parity.is_file():
        rep = json.loads(m.parity.read_text())
        print(f"= parity report exists: pass={rep.get('pass')}")
        return
    from deploybench.manifests import file_sha256
    m.dir.mkdir(parents=True, exist_ok=True)
    ref = m.dir / "parity_ref_fp32_pytorch.csv"
    test = m.dir / "parity_test_fp32_ort.csv"
    ck = m.need_checkpoint()
    if not ref.is_file():
        must(REPO / "score.py", "--model", m.name, "--checkpoint", ck,
             "--dataset", "asvspoof19_eval", "--manifest", PARITY_MANIFEST,
             "--device", "cpu", "--precision", "fp32", "--runtime", "pytorch", "--out", ref)
    if not test.is_file():
        must(REPO / "score_onnx.py", "--onnx", m.fp32, "--model", m.name,
             "--dataset", "asvspoof19_eval", "--manifest", PARITY_MANIFEST,
             "--precision", "fp32", "--runtime", "ort", "--out", test)
    taus = m.dir / "taus_flat.json"
    taus.write_text(json.dumps(json.loads(
        Path(str(TAUS).format(model=m.name)).read_text())["taus"]))
    r = subprocess.run([sys.executable, str(REPO / "tests" / "parity.py"), "--ref", str(ref),
                        "--test", str(test), "--taus", str(taus), "--mode", "conversion"],
                       cwd=str(REPO), capture_output=True, text=True)
    rep = json.loads(r.stdout)
    rep.update(checkpoint=str(ck), checkpoint_sha256=file_sha256(Path(ck)),
               fp32_export_sha256=__import__("ptq").artifact_hashes(m.fp32))
    m.parity.write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=2, default=str))
    if r.returncode != 0:
        raise SystemExit(f"{m.name}: fp32 export FAILS the conversion parity gate. "
                         "No quantization runs on a broken export.")


def gate(m: Model):
    if not m.parity.is_file() or json.loads(m.parity.read_text()).get("pass") is not True:
        raise SystemExit(f"{m.name}: no passing fp32 parity report at {m.parity}; run 'parity'")


def step_r01(m: Model):
    gate(m)
    e = m.convert("r01", 100, SEARCH_MAN, "r01")
    if e["outcome"] == "ok":
        m.score("r01", "asvspoof19_dev", "scores_diag.csv", DIAG)


def step_rank(m: Model):
    gate(m)
    if m.plan.is_file():
        print(f"= {m.plan.relative_to(REPO)} is frozen")
        return
    e = m.logged("r01", 100)
    if e is None or e["outcome"] != "ok":
        raise SystemExit("r01 did not convert, so there is nothing to rank against. "
                         "That is itself the result for this model; record it.")
    must(REPO / "convert" / "rank_sensitivity.py", "--model", m.name, "--fp32", m.fp32,
         "--int8", m.dir / "r01" / f"{m.name}_int8.onnx", "--calib-dir", m.dev_flac,
         "--out", m.plan)


def step_search(m: Model):
    gate(m)
    plan = json.loads(m.plan.read_text(encoding="utf-8"))
    for t in plan["attempts"][1:]:
        e = m.convert(t["recipe_id"], 100, SEARCH_MAN, t["recipe_id"],
                      t["calib_method"], t["exclude_nodes"])
        if e["outcome"] == "ok":
            m.score(t["recipe_id"], "asvspoof19_dev", "scores_diag.csv", DIAG)


def step_select(m: Model):
    must(REPO / "convert" / "select_recipe.py", "search", "--plan", m.plan,
         "--ledger", m.ledger, "--runs-root", m.dir, "--out", m.selection)


def locked(m: Model):
    sel = json.loads(m.selection.read_text(encoding="utf-8"))
    if sel["locked_recipe"] is None:
        raise SystemExit(f"{m.name}: no search attempt met the D36 budget. There is no "
                         "locked recipe, so no calibration repeats run. Report it.")
    plan = json.loads(m.plan.read_text(encoding="utf-8"))
    return next(t for t in plan["attempts"] if t["recipe_id"] == sel["locked_recipe"])


def step_repeats(m: Model):
    gate(m)
    t = locked(m)
    rid = t["recipe_id"]
    runs = [(rid, s, MAN / f"calib_seed{s}.txt", f"{rid}_seed{s}") for s in SEEDS]
    runs.append((f"{rid}_rebuild", 1, MAN / "calib_seed1.txt", f"{rid}_seed1_rebuild"))
    for r_id, seed, man, run in runs:
        e = m.convert(r_id, seed, man, run, t["calib_method"], t["exclude_nodes"])
        if e["outcome"] == "ok":
            m.score(run, "asvspoof19_dev", "scores_diag.csv", DIAG)
            m.score(run, "asvspoof19_eval", "scores_19la_eval.csv")


def _scores(path: Path) -> dict:
    return {r["utt_id"]: float(r["score"]) for r in csv.DictReader(open(path))}


def step_canonical(m: Model):
    rid = locked(m)["recipe_id"]
    must(REPO / "convert" / "select_recipe.py", "canonical", "--plan", m.plan,
         "--ledger", m.ledger, "--runs-root", m.dir, "--locked", rid, "--out", m.canonical)
    a, b = m.logged(rid, 1), m.logged(f"{rid}_rebuild", 1)
    rep = {"model": m.name, "recipe_id": rid, "seed": 1,
           "seed1_outcome": a and a["outcome"], "rebuild_outcome": b and b["outcome"]}
    if a and b and a["outcome"] == b["outcome"] == "ok":
        rep["artifact_sha256_identical"] = a["dst_sha256"] == b["dst_sha256"]
        for name in ("scores_diag.csv", "scores_19la_eval.csv"):
            sa = _scores(m.dir / f"{rid}_seed1" / name)
            sb = _scores(m.dir / f"{rid}_seed1_rebuild" / name)
            if set(sa) != set(sb):
                raise SystemExit(f"{name}: seed 1 and its rebuild scored different utterances")
            rep[f"{name}_max_abs_diff"] = max(abs(sa[u] - sb[u]) for u in sa)
    (OUT / f"repeatability_{m.name}.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=2))


STEPS = {"export": step_export, "parity": step_parity, "r01": step_r01, "rank": step_rank, "search": step_search,
         "select": step_select, "repeats": step_repeats, "canonical": step_canonical}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("step", choices=list(STEPS) + ["all"])
    a = ap.parse_args()
    m = Model(a.model, a.checkpoint)
    for name in (list(STEPS) if a.step == "all" else [a.step]):
        print(f"\n=== {m.name}: {name} ===", flush=True)
        STEPS[name](m)


if __name__ == "__main__":
    main()
