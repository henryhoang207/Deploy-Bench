"""Apply the frozen D39 selection rules. Nothing here is a judgment call.

  search     Locked recipe = among the plan's attempts that meet the D36 budget
             (int8 EER - fp32 EER <= 1.0 pp on the diagnostic subset), the one with
             the smallest K; ties go to the earlier attempt.
  canonical  Device artifact = the lowest-numbered calibration seed (1-5) whose
             artifact, built with the locked recipe, meets the same budget. If none
             does, the model has NO qualifying int8 artifact, and that is the result.

Both modes REFUSE to choose while any frozen attempt is unaccounted for: every
planned attempt needs a ledger entry, and every successful one needs its
diagnostic scores. A failed attempt is a result (it cannot qualify). It is never
retried, replaced or silently skipped. The fp32 reference is the frozen PyTorch
fp32 19LA-dev score file restricted to the diagnostic ids, the same scores the
taus were frozen from.

Layout (all under --runs-root, e.g. runs/<model>):
  <recipe_id>/scores_diag.csv            search attempts, seed 100
  <recipe_id>_seed<s>/scores_diag.csv    calibration repeats, seeds 1-5

    python convert/select_recipe.py search --plan .../search_plan_aasist.json \\
        --ledger .../ledger_aasist.jsonl --runs-root runs/aasist --out .../selection_aasist.json
    python convert/select_recipe.py canonical --plan ... --ledger ... --runs-root runs/aasist \\
        --locked r05 --out .../canonical_aasist.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from deploybench.manifests import ids_digest, read_ids  # noqa: E402
from deploybench.metrics import eer  # noqa: E402

BUDGET = 0.010              # D36: 1.0 pp, point estimate
SEARCH_SEED = 100
REPEAT_SEEDS = (1, 2, 3, 4, 5)
DEV_SCORES = REPO / "results" / "phase_1_reproduction" / "scores_dev_{model}.csv"


class Unaccounted(SystemExit):
    pass


def read_scores(path: Path, want_ids, model: str, precision: str) -> tuple[list, list]:
    """Scores in manifest order; refuses missing/extra ids or a wrong label on the file."""
    rows = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["model"] != model or r["precision"] != precision:
            raise SystemExit(f"{path}: row says model={r['model']} precision={r['precision']}, "
                             f"expected {model}/{precision}")
        if r["utt_id"] in rows:
            raise SystemExit(f"{path}: duplicate utt_id {r['utt_id']}")
        rows[r["utt_id"]] = (int(r["label"]), float(r["score"]))
    missing = [u for u in want_ids if u not in rows]
    if missing:
        raise SystemExit(f"{path}: {len(missing)} diagnostic ids missing (first {missing[:3]})")
    if precision != "fp32" and len(rows) != len(want_ids):
        raise SystemExit(f"{path}: {len(rows)} rows but the diagnostic subset has {len(want_ids)}")
    return [rows[u][1] for u in want_ids], [rows[u][0] for u in want_ids]


def ledger_index(path: Path) -> dict:
    idx = {}
    for n, line in enumerate(open(path, encoding="utf-8"), 1):
        if not line.strip():
            continue
        e = json.loads(line)
        key = (e.get("recipe_id"), e.get("calib_seed"))
        if key in idx:
            raise SystemExit(f"{path}:{n}: second ledger entry for recipe {key[0]} seed "
                             f"{key[1]}. Attempts are never retried or replaced; resolve "
                             "this by hand and document it before selecting anything")
        idx[key] = e
    return idx


def not_int8(run_dir: Path, model: str) -> bool:
    """True when ptq.py's sidecar says the artifact has no Q/DQ ops at all."""
    side = run_dir / f"{model}_int8_ptq_log.json"
    if not side.is_file():
        return False
    s = json.loads(side.read_text(encoding="utf-8"))
    return s.get("n_quantize_linear") == 0 and s.get("n_dequantize_linear") == 0


def evaluate(key, run_dir: Path, ledger, want_ids, model, fp32_eer, expect_k=None):
    e = ledger.get(key)
    if e is None:
        raise Unaccounted(f"attempt {key[0]} seed {key[1]} has no ledger entry: "
                          "run it (or log its failure) before selecting")
    row = {"recipe_id": key[0], "calib_seed": key[1], "outcome": e["outcome"],
           "n_fp32_excluded": e.get("n_fp32_excluded"),
           "precision_label": e.get("precision_label")}
    if expect_k is not None and e.get("n_fp32_excluded") != expect_k:
        raise SystemExit(f"{key[0]} seed {key[1]}: ledger K={e.get('n_fp32_excluded')} but the "
                         f"frozen plan says K={expect_k}")
    if e["outcome"] != "ok":
        row.update(error=e.get("error"), qualifies=False)
        return row
    if not_int8(run_dir, model):
        # Excluding every quantized node leaves an fp32 graph (tssdnet r12,
        # 2026-09-29). It is not an int8 artifact, so it cannot meet an int8
        # budget. score_onnx.py refuses to label it int8, so there are no scores.
        row.update(outcome="no_int8_ops", qualifies=False,
                   note="conversion left no QuantizeLinear/DequantizeLinear: an fp32 graph")
        return row
    scores_path = run_dir / "scores_diag.csv"
    if not scores_path.is_file():
        raise Unaccounted(f"{key[0]} seed {key[1]} converted but {scores_path} is missing: "
                          "score it on the diagnostic subset before selecting")
    s, y = read_scores(scores_path, want_ids, model, "int8")
    e_q = eer(s, y)[0]
    row.update(eer=e_q, delta_eer_pp=round((e_q - fp32_eer) * 100, 4),
               qualifies=(e_q - fp32_eer) <= BUDGET + 1e-12,
               artifact_sha256=e.get("dst_sha256"))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["search", "canonical"])
    ap.add_argument("--plan", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--locked", default=None, help="canonical mode: the locked recipe id")
    ap.add_argument("--dev-scores", default=None,
                    help="fp32 reference (default: the frozen scores_dev_<model>.csv)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))
    model = plan["model"]
    diag = Path(plan["diag_manifest"])
    if not diag.is_absolute():
        diag = REPO / diag
    want = read_ids(diag)
    if ids_digest(want) != plan["diag_ids_sha256"]:
        raise SystemExit(f"{diag} no longer matches the plan's diagnostic sha256")
    ref = Path(a.dev_scores) if a.dev_scores else Path(str(DEV_SCORES).format(model=model))
    fs, fy = read_scores(ref, want, model, "fp32")
    fp32_eer = eer(fs, fy)[0]
    ledger = ledger_index(Path(a.ledger))
    root = Path(a.runs_root)

    if a.mode == "search":
        rows = [evaluate((t["recipe_id"], SEARCH_SEED), root / t["recipe_id"], ledger,
                         want, model, fp32_eer, expect_k=t["n_fp32_excluded"])
                for t in plan["attempts"]]
        order = {t["recipe_id"]: i for i, t in enumerate(plan["attempts"])}
        ok = [r for r in rows if r["qualifies"]]
        ok.sort(key=lambda r: (r["n_fp32_excluded"], order[r["recipe_id"]]))
        result = {"mode": "search", "locked_recipe": ok[0]["recipe_id"] if ok else None,
                  "locked_k": ok[0]["n_fp32_excluded"] if ok else None}
    else:
        if not a.locked:
            raise SystemExit("canonical mode needs --locked (from the search selection)")
        k = {t["recipe_id"]: t["n_fp32_excluded"] for t in plan["attempts"]}.get(a.locked)
        if k is None:
            raise SystemExit(f"{a.locked} is not an attempt in the frozen plan")
        rows = [evaluate((a.locked, s), root / f"{a.locked}_seed{s}", ledger, want, model,
                         fp32_eer, expect_k=k) for s in REPEAT_SEEDS]
        ok = [r for r in rows if r["qualifies"]]
        result = {"mode": "canonical", "locked_recipe": a.locked,
                  "canonical_seed": min(r["calib_seed"] for r in ok) if ok else None}

    result.update(model=model, fp32_diag_eer=fp32_eer, budget_pp=BUDGET * 100,
                  fp32_reference=str(ref), n_diag=len(want),
                  n_attempts=len(rows), n_failed=sum(r["outcome"] != "ok" for r in rows),
                  attempts=rows)
    if result.get("locked_recipe", 1) is None or result.get("canonical_seed", 1) is None:
        result["note"] = ("no attempt meets the D36 budget: this model has no qualifying "
                          "int8 artifact under the frozen design (report it; do not substitute)")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in result if k != "attempts"}, indent=2))


if __name__ == "__main__":
    main()
