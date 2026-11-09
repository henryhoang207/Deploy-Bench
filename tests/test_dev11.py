"""Tests for the checks of DEV-11 (analysis/review_r1/run_lastepoch.py, the DF21 surface of run_smalllabel.py, Checks 6
and 7 of rv_checks.py, the step list of run_all.py). They run on synthetic scores only: R.load, R.taus, the committed
result files (run_lastepoch.REGISTERED and EXPLORE, rv_checks.EXPLORE) and rvlib.OUT are replaced, and R.read_scores
refuses, so no committed score file is read and nothing is written under results/. run_smalllabel.py executes at import,
so it is never imported: its source is parsed.

- A, direction and exactness: the contrast on synthetic runs, trimmed minus untrimmed (negative for the FRR line, whose
  registered sign is the opposite), p = 1/252 for complete separation, and a split with exactly 12 of 252 counted
  against a brute-force count; Holm at .05 over exactly the two family tests (p = 12/252 and 0.98 rejects nothing; 1/252
  rejects).
- A, inputs: r2_registered.perm_greater (untrimmed first, as h3 does) and _run_quantities are the ones used; seeds 1-5
  only (seed 1234 is never read and cannot change the result); importing the module computes nothing.
- A, refusals: the self-check against R2-H3 of r2_registered.json (written by r2_registered.h3 itself on the synthetic
  scores) and the cross-check against E2_E3 (written by r2_explore.e2_e3) each refuse when one committed value is
  altered, and then nothing is written; they pass within 1e-9.
- C: 1 - (1 - 1/G)^G (G = 2 gives 0.75, G = 67 is not the limit 1 - 1/e), the clamp rule at q = 0.02, 0.5 and 0.97 with
  edges equal to r2_explore.three_zone_edges, Check 7 over the eight detectors on their source-dev scores, and its
  refusal when an E7 edge differs.
- B: run_smalllabel.py has df21_verified last in the surface loop and default_rng(0) inside it; mutated copies of the
  source (df21_verified dropped, first, in the middle; the RNG moved out of the loop) are caught.
- run_all.py runs run_lastepoch.py directly before rv_checks.py; rv_checks.main appends Checks 6 and 7 after Check 5.
"""
import ast
import copy
import importlib.util
import itertools
import json
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis" / "round2"))
sys.path.insert(0, str(ROOT / "analysis" / "review_r1"))

import r2lib as R  # noqa: E402
import r2_explore as EX  # noqa: E402
import r2_registered as RR  # noqa: E402
import run_all  # noqa: E402
import run_lastepoch as LE  # noqa: E402
import rv_checks as RV  # noqa: E402
import rvlib  # noqa: E402

REVIEW = ROOT / "analysis" / "review_r1"
FRR, EER = LE.FAMILY
SMALLLABEL = REVIEW / "run_smalllabel.py"


# --------------------------------------------------------------------------- synthetic world
# model -> (class separation, tau_EER) of the synthetic ITW scores: bona fide ~ N(+sep, 1), spoof ~ N(-sep, 1). The trimmed
# arms are less separable (higher EER) and have a lower threshold (lower FRR), as in the paper; the arms of each pair
# differ so that each separates completely over seeds 1-5. Every other model: separation 1.5, tau 0.
PARAMS = {"aasist_l_r2": (2.5, 2.5), "aasist_l_trim": (1.2, -0.8),
          "aasist_l_r2_last": (2.0, 2.0), "aasist_l_trim_last": (1.0, -1.0)}
N = 400                                                      # clips per class
DEV_Q = {m: 0.90 for m in list(R.DETECTORS_R2) + list(R.ROUND1)}   # share of source-dev scores below tau_EER (= 0)
DEV_N = 1000


def _rng(*key):
    return np.random.default_rng(zlib.crc32("|".join(map(str, key)).encode()))


def dev_scores(model, surface, q):
    """DEV_N source-dev scores of which exactly round(q * DEV_N) lie below tau = 0."""
    k = round(q * DEV_N)
    rng = _rng("dev", model, surface)
    s = np.concatenate([-rng.uniform(0.1, 3.0, k), rng.uniform(0.1, 3.0, DEV_N - k)])
    y = rng.integers(0, 2, DEV_N)
    y[:2] = (0, 1)
    return np.array([f"{model}-{surface}-{i}" for i in range(DEV_N)]), y, s


class Fake:
    """Stands in for r2lib.load; records every request."""

    def __init__(self):
        self.calls = []
        self.pipeline_sep = -4.0                             # the seed-1234 run of the untrimmed last-epoch arm: inverted
        self.dev_q = dict(DEV_Q)

    def load(self, model, surface, seed=0, check_n=True):
        self.calls.append((model, surface, seed))
        if surface in ("la19_dev", "as5_dev") and model in self.dev_q:
            return dev_scores(model, surface, self.dev_q[model])
        sep = PARAMS.get(model, (1.5, 0.0))[0]
        if model == "aasist_l_r2_last" and seed == R.PIPELINE_SEED:
            sep = self.pipeline_sep
        rng = _rng(model, surface, seed)
        y = np.repeat([1, 0], N)
        s = np.concatenate([rng.normal(sep, 1.0, N), rng.normal(-sep, 1.0, N)])
        return np.array([f"{model}-{surface}-{seed}-{i}" for i in range(2 * N)]), y, s


def fake_taus(model, seed=0):
    tau = PARAMS.get(model, (1.5, 0.0))[1]
    return {"eer": tau, "frr1": tau - 1.0, "far1": tau + 1.0, "frr5": tau - 0.5, "far5": tau + 0.5}


def _no_real_scores(*args, **kwargs):
    raise AssertionError("a committed score file was requested")


def committed_from(fake):
    """r2_registered.json's R2-H3 and r2_explore.json's E2_E3, written by the registered code itself (r2_registered.h3
    and r2_explore.e2_e3) on the synthetic scores, as read back from disk."""
    registered, explore = {"R2-H3": RR.h3()}, {"E2_E3": EX.e2_e3()}
    return json.loads(json.dumps(registered)), json.loads(json.dumps(explore))


@pytest.fixture(scope="module")
def pristine():
    """The synthetic committed results, built once."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(R, "load", Fake().load)
        mp.setattr(R, "taus", fake_taus)
        mp.setattr(R, "read_scores", _no_real_scores)
        return committed_from(Fake())


@pytest.fixture
def world(tmp_path, monkeypatch, pristine):
    """Synthetic scores and thresholds in place of the committed ones; results go to tmp_path / 'review_r1', which does
    not exist until something is written. Returns (fake, out, registered, explore); the last two are private copies."""
    fake = Fake()
    monkeypatch.setattr(R, "load", fake.load)
    monkeypatch.setattr(R, "taus", fake_taus)
    monkeypatch.setattr(R, "read_scores", _no_real_scores)
    out = tmp_path / "review_r1"
    monkeypatch.setattr(rvlib, "OUT", out)
    monkeypatch.setattr(RV, "OUT", out)
    RV.LINES.clear()
    return fake, out, copy.deepcopy(pristine[0]), copy.deepcopy(pristine[1])


def run_main(tmp_path, monkeypatch, registered, explore):
    """LE.main() with the given committed files (written under tmp_path / 'committed')."""
    d = tmp_path / "committed"
    d.mkdir(exist_ok=True)
    (d / "r2_registered.json").write_text(json.dumps(registered), encoding="utf-8")
    (d / "r2_explore.json").write_text(json.dumps(explore), encoding="utf-8")
    monkeypatch.setattr(LE, "REGISTERED", d / "r2_registered.json")
    monkeypatch.setattr(LE, "EXPLORE", d / "r2_explore.json")
    LE.main()


def altered(obj, path, change):
    o = copy.deepcopy(obj)
    d = o
    for key in path[:-1]:
        d = d[key]
    d[path[-1]] = change(d[path[-1]])
    return o


def refused(excinfo):
    msg = str(excinfo.value.code)
    assert msg.startswith("REFUSED"), msg
    return msg


def nothing_written(tmp_path, out):
    assert not out.exists()
    assert not list(tmp_path.glob("**/lastepoch*"))


def arms_from():
    """The four-model synthetic arms as run_lastepoch.collect reads them: {'trimmed': {seed: ...}, 'untrimmed': ...}."""
    return {a: LE.collect(m) for a, m in LE.ARMS.items()}


# --------------------------------------------------------------------------- A. the contrast
def test_contrast_direction_sign_and_exact_p_under_complete_separation(world):
    arms = arms_from()
    t, u = arms["trimmed"], arms["untrimmed"]
    frr_t, frr_u = LE._values(t, "itw_frr"), LE._values(u, "itw_frr")
    eer_t, eer_u = LE._values(t, "itw_eer"), LE._values(u, "itw_eer")
    assert max(frr_t) < min(frr_u) and min(eer_t) > max(eer_u)      # complete separation, in opposite directions
    tests = LE.contrast(t, u)
    assert list(tests) == [FRR, EER]
    b, c = tests[FRR], tests[EER]
    # lower FRR when trimmed: the smallest possible one-sided p, 1 of the 252 splits
    assert (b["count"], b["n_splits"], b["p"]) == (1, 252, 1 / 252)
    assert b["diff_pp"] < 0                                        # trimmed minus untrimmed
    assert b["diff_pp"] == pytest.approx(100 * (np.mean(frr_t) - np.mean(frr_u)), abs=1e-12)
    assert b["diff_pp_registered_sign"] == -b["diff_pp"] > 0        # r2_registered.json's diff_pp has the opposite sign
    # lower EER when trimmed is contradicted: every split is at least as extreme
    assert (c["count"], c["n_splits"], c["p"]) == (252, 252, 1.0)
    assert c["diff_pp"] == pytest.approx(100 * (np.mean(eer_t) - np.mean(eer_u)), abs=1e-12) and c["diff_pp"] > 0


def twelve_of_252():
    """Synthetic arms (trimmed, untrimmed), each {seed: {"itw_frr", "itw_eer"}}, whose FRR contrast has exactly 12 of the
    252 splits at least as extreme as the observed one, with brute-force counts of both tests. The ten values are powers
    of two, so all subset sums differ and no split ties with another. Untrimmed FRR = the values of the 12th largest
    five-subset sum, trimmed FRR = the other five; the EERs the other way round."""
    vals = [2 ** i for i in range(10)]
    by_sum = sorted(itertools.combinations(range(10), 5), key=lambda c: -sum(vals[i] for i in c))
    top = by_sum[11]
    rest = [i for i in range(10) if i not in top]
    total = lambda idx: sum(vals[i] for i in idx)                   # noqa: E731
    count_b = sum(1 for c in by_sum if total(c) >= total(top))      # FRR: untrimmed = top
    count_c = sum(1 for c in by_sum if total(c) >= total(rest))     # EER: untrimmed = rest
    untrimmed = {k: {"itw_frr": vals[i] / 1024, "itw_eer": vals[j] / 1024} for k, (i, j) in enumerate(zip(top, rest), 1)}
    trimmed = {k: {"itw_frr": vals[j] / 1024, "itw_eer": vals[i] / 1024} for k, (i, j) in enumerate(zip(top, rest), 1)}
    return trimmed, untrimmed, count_b, count_c


def test_exact_count_of_12_of_252_matches_a_brute_force_count():
    trimmed, untrimmed, count_b, count_c = twelve_of_252()
    assert (count_b, count_c) == (12, 241)
    tests = LE.contrast(trimmed, untrimmed)
    assert (tests[FRR]["count"], tests[FRR]["p"]) == (12, 12 / 252)
    assert (tests[EER]["count"], tests[EER]["p"]) == (241, 241 / 252)
    assert LE.holm(tests) == {FRR: False, EER: False}               # 12/252 = 0.048 is above Holm's first step, 0.025


def test_holm_rejects_nothing_for_12_of_252_and_098_and_rejects_for_1_of_252():
    def tests(pb, pc):
        return {FRR: {"p": pb}, EER: {"p": pc}}
    assert LE.holm(tests(12 / 252, 0.98)) == {FRR: False, EER: False}
    assert LE.holm(tests(1 / 252, 0.98)) == {FRR: True, EER: False}
    assert LE.holm(tests(0.025, 0.98)) == {FRR: True, EER: False}            # alpha / 2 itself rejects
    assert LE.holm(tests(0.026, 0.98)) == {FRR: False, EER: False}           # alpha (= .05), not .026, is the second step
    assert LE.holm(tests(0.01, 0.04)) == {FRR: True, EER: True}              # both pass, the second at alpha
    assert LE.ALPHA == 0.05


def test_holm_is_applied_to_exactly_the_two_family_tests_and_nothing_else_is_tested(world, tmp_path, monkeypatch):
    fake, out, registered, explore = world
    seen = []
    real = R.holm
    monkeypatch.setattr(R, "holm", lambda pvals, alpha=0.05: seen.append((dict(pvals), alpha)) or real(pvals, alpha))
    run_main(tmp_path, monkeypatch, registered, explore)
    assert len(seen) == 2                                            # the self-check's, and the last-epoch contrast's
    for pvals, alpha in seen:
        assert sorted(pvals) == ["b", "c"] and alpha == 0.05
    j = json.loads((out / "lastepoch.json").read_text(encoding="utf-8"))
    assert list(j["cells"]["tests"]) == [FRR, EER]                   # no manipulation check (R2-H3 a)
    assert j["meta"]["family"] == [FRR, EER]
    assert all(set(q) == {"itw_frr", "itw_eer"} for arm in j["cells"]["runs"].values() for q in arm.values())


def test_perm_greater_and_run_quantities_are_the_registered_functions(world, monkeypatch):
    fake, *_ = world
    assert LE.RR is RR
    calls = []
    real = RR.perm_greater
    monkeypatch.setattr(RR, "perm_greater", lambda a, b: calls.append((list(a), list(b))) or real(a, b))
    arms = arms_from()
    LE.contrast(arms["trimmed"], arms["untrimmed"])
    # h3(): perm_greater(untrimmed, trimmed), the FRR test and then the EER test
    assert calls == [(LE._values(arms["untrimmed"], "itw_frr"), LE._values(arms["trimmed"], "itw_frr")),
                     (LE._values(arms["untrimmed"], "itw_eer"), LE._values(arms["trimmed"], "itw_eer"))]
    got = []
    real_q = RR._run_quantities
    monkeypatch.setattr(RR, "_run_quantities", lambda model, k: got.append((model, k)) or real_q(model, k))
    LE.collect("aasist_l_trim_last")
    assert got == [("aasist_l_trim_last", k) for k in (1, 2, 3, 4, 5)]


# --------------------------------------------------------------------------- A. seeds 1-5 only
def test_seed_1234_is_never_read_and_cannot_change_the_result(world, tmp_path, monkeypatch):
    fake, out, registered, explore = world
    run_main(tmp_path, monkeypatch, registered, explore)
    first = (out / "lastepoch.json").read_bytes()
    assert {k for _, _, k in fake.calls if k} <= {1, 2, 3, 4, 5}      # seed 0 is not requested either
    assert R.PIPELINE_SEED not in {k for _, _, k in fake.calls}
    j = json.loads(first)
    assert sorted(j["cells"]["runs"]["aasist_l_r2_last"]) == ["1", "2", "3", "4", "5"]
    # the same run again with another seed-1234 run, and a committed E2_E3 whose seed-1234 entry is nonsense
    fake.pipeline_sep, fake.calls = +4.0, []
    for arm in ("aasist_l_r2", "aasist_l_r2_last"):
        explore["E2_E3"][arm][str(R.PIPELINE_SEED)]["itw|eer"]["frr"] = 0.123
    (out / "lastepoch.json").unlink()
    run_main(tmp_path, monkeypatch, registered, explore)
    assert (out / "lastepoch.json").read_bytes() == first
    # the test bites: with the pipeline-check run in the untrimmed mean, the mean would be far off
    q1234 = RR._run_quantities("aasist_l_r2_last", R.PIPELINE_SEED)
    kept = [j["cells"]["runs"]["aasist_l_r2_last"][str(k)]["itw_frr"] for k in (1, 2, 3, 4, 5)]
    assert abs(np.mean(kept + [q1234["itw_frr"]]) - np.mean(kept)) > 0.01
    assert j["cells"]["means"]["aasist_l_r2_last"]["itw_frr"] == float(np.mean(kept))


def test_importing_run_lastepoch_computes_prints_and_writes_nothing(tmp_path, monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise AssertionError("R.load / R.taus called at import")
    monkeypatch.setattr(R, "load", boom)
    monkeypatch.setattr(R, "taus", boom)
    monkeypatch.setattr(rvlib, "OUT", tmp_path / "review_r1")
    capsys.readouterr()
    spec = importlib.util.spec_from_file_location("run_lastepoch_fresh", REVIEW / "run_lastepoch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert capsys.readouterr().out == ""
    assert callable(mod.main) and not (tmp_path / "review_r1").exists()
    tree = ast.parse((REVIEW / "run_lastepoch.py").read_text(encoding="utf-8"))
    guard = [n for n in tree.body if isinstance(n, ast.If)]
    assert len(guard) == 1 and ast.unparse(guard[0].test) == "__name__ == '__main__'"
    assert [ast.unparse(s) for s in guard[0].body] == ["main()"]
    assert not [n for n in tree.body if isinstance(n, ast.Expr) and not isinstance(n.value, ast.Constant)]


# --------------------------------------------------------------------------- A. what main writes
def test_main_writes_the_documented_result_and_records_both_checks(world, tmp_path, monkeypatch):
    fake, out, registered, explore = world
    run_main(tmp_path, monkeypatch, registered, explore)
    j = rvlib.read("lastepoch")
    meta, cells = j["meta"], j["cells"]
    assert meta["script"] == "analysis/review_r1/run_lastepoch.py" and meta["deviation"] == "DEV-11 A"
    assert meta["arms"] == {"trimmed": "aasist_l_trim_last", "untrimmed": "aasist_l_r2_last"}
    assert meta["seeds"] == [1, 2, 3, 4, 5] and meta["excluded_runs"] == {"aasist_l_r2_last": [1234]}
    assert meta["family"] == [FRR, EER] and meta["holm_alpha"] == 0.05
    assert "no RNG" in meta["test"] and "252" in meta["test"] and "perm_greater" in meta["test"]
    assert "exploratory" in meta["status"] and "before DEV-11 was logged" in meta["status"]
    assert "100 x (mean trimmed - mean untrimmed)" in meta["sign"] and "untrimmed minus trimmed" in meta["sign"]
    sc, cc = meta["self_check"], meta["cross_check"]
    assert sc["passed"] is True and sc["arms"] == {"trimmed": "aasist_l_trim", "untrimmed": "aasist_l_r2"}
    assert sorted(sc["tests"]) == [FRR, EER] and all(v["p_equal"] and v["n_splits_equal"] for v in sc["tests"].values())
    assert cc["passed"] is True and cc["values_checked"] == 20 and cc["max_abs_diff"] <= 1e-9
    arms = arms_from()
    for arm, model in LE.ARMS.items():
        for k in (1, 2, 3, 4, 5):
            assert cells["runs"][model][str(k)] == arms[arm][k]
        assert cells["means"][model]["itw_frr"] == float(np.mean(LE._values(arms[arm], "itw_frr")))
        assert cells["means"][model]["itw_eer"] == float(np.mean(LE._values(arms[arm], "itw_eer")))
    t = cells["tests"][FRR]
    assert (t["count"], t["n_splits"], t["p"], t["holm_reject"]) == (1, 252, 1 / 252, True)
    assert t["diff_pp_trimmed_minus_untrimmed"] < 0
    t = cells["tests"][EER]
    assert (t["count"], t["p"], t["holm_reject"]) == (252, 1.0, False) and t["diff_pp_trimmed_minus_untrimmed"] > 0


# --------------------------------------------------------------------------- A. refusals
SELF_ALTERATIONS = [("diff_pp", lambda v: v + 1e-6), ("diff_pp", lambda v: v - 1e-6), ("p", lambda v: v + 1e-12),
                    ("n_splits", lambda v: v + 1), ("holm_reject", lambda v: not v)]


@pytest.mark.parametrize("name", [FRR, EER])
@pytest.mark.parametrize("field,change", SELF_ALTERATIONS)
def test_self_check_refuses_when_a_committed_r2_h3_value_is_altered_and_writes_nothing(
        world, tmp_path, monkeypatch, name, field, change):
    fake, out, registered, explore = world
    bad = altered(registered, ("R2-H3", "tests", name, field), change)
    with pytest.raises(SystemExit) as e:
        run_main(tmp_path, monkeypatch, bad, explore)
    assert "self-check" in refused(e) and name in str(e.value.code)
    nothing_written(tmp_path, out)


def test_self_check_refuses_when_r2_h3_is_missing_and_passes_within_1e9(world, tmp_path, monkeypatch):
    fake, out, registered, explore = world
    with pytest.raises(SystemExit) as e:
        run_main(tmp_path, monkeypatch, {"R2-H3": {"evaluable": False, "missing": ["x"]}}, explore)
    assert "R2-H3 / tests" in refused(e)
    nothing_written(tmp_path, out)
    with pytest.raises(SystemExit) as e:                              # 1e-8 is beyond the tolerance
        run_main(tmp_path, monkeypatch, altered(registered, ("R2-H3", "tests", FRR, "diff_pp"), lambda v: v + 1e-8), explore)
    refused(e)
    nothing_written(tmp_path, out)
    run_main(tmp_path, monkeypatch, altered(registered, ("R2-H3", "tests", FRR, "diff_pp"), lambda v: v + 1e-10), explore)
    assert (out / "lastepoch.json").is_file()                         # 1e-10 is within it


def test_self_check_runs_on_the_prespecified_arms_not_on_the_last_epoch_ones(world, tmp_path, monkeypatch):
    """The registered file here is built from aasist_l_trim / aasist_l_r2; if the self-check read the last-epoch arms
    instead, it would refuse. Conversely it must refuse when the registered values come from the last-epoch arms."""
    fake, out, registered, explore = world
    run_main(tmp_path, monkeypatch, registered, explore)
    (out / "lastepoch.json").unlink()
    swapped = copy.deepcopy(registered)
    last = LE.contrast(LE.collect("aasist_l_trim_last"), LE.collect("aasist_l_r2_last"))
    for name in (FRR, EER):
        swapped["R2-H3"]["tests"][name]["diff_pp"] = last[name]["diff_pp_registered_sign"]
        swapped["R2-H3"]["tests"][name]["p"] = last[name]["p"]
    with pytest.raises(SystemExit) as e:
        run_main(tmp_path, monkeypatch, swapped, explore)
    assert "self-check" in refused(e)


EXPLORE_ALTERATIONS = [(model, seed, field) for model in ("aasist_l_trim_last", "aasist_l_r2_last") for seed in ("1", "5")
                       for field in ("frr", "eer")]


@pytest.mark.parametrize("model,seed,field", EXPLORE_ALTERATIONS)
def test_cross_check_refuses_when_an_e2_e3_value_is_altered_and_writes_nothing(
        world, tmp_path, monkeypatch, model, seed, field):
    fake, out, registered, explore = world
    bad = altered(explore, ("E2_E3", model, seed, "itw|eer", field), lambda v: v + 1e-6)
    with pytest.raises(SystemExit) as e:
        run_main(tmp_path, monkeypatch, registered, bad)
    msg = refused(e)
    assert "cross-check" in msg and model in msg and f"seed {seed}" in msg
    nothing_written(tmp_path, out)


def test_cross_check_refuses_on_a_missing_entry_and_passes_within_1e9(world, tmp_path, monkeypatch):
    fake, out, registered, explore = world
    gone = copy.deepcopy(explore)
    del gone["E2_E3"]["aasist_l_trim_last"]["3"]
    with pytest.raises(SystemExit) as e:
        run_main(tmp_path, monkeypatch, registered, gone)
    assert "E2_E3 / aasist_l_trim_last / 3" in refused(e)
    nothing_written(tmp_path, out)
    near = altered(explore, ("E2_E3", "aasist_l_r2_last", "2", "itw|eer", "frr"), lambda v: v + 1e-10)
    run_main(tmp_path, monkeypatch, registered, near)
    assert (out / "lastepoch.json").is_file()


def test_the_committed_files_are_required(world, tmp_path, monkeypatch):
    fake, out, *_ = world
    monkeypatch.setattr(LE, "REGISTERED", tmp_path / "absent.json")
    with pytest.raises(SystemExit) as e:
        LE.main()
    refused(e)
    nothing_written(tmp_path, out)


# --------------------------------------------------------------------------- A. Check 6 of rv_checks
def _write_lastepoch(trimmed, untrimmed):
    tests = LE.contrast(trimmed, untrimmed)
    sc = {"arms": dict(LE.PRESPEC), "tolerance": LE.TOL, "passed": True}
    cc = {"tolerance": LE.TOL, "values_checked": 20, "passed": True}
    meta, cells = LE.build({"trimmed": trimmed, "untrimmed": untrimmed}, tests, LE.holm(tests), sc, cc)
    rvlib.write("lastepoch", meta, cells)


def test_check6_prints_signed_differences_and_counts_over_the_splits(world, tmp_path, monkeypatch):
    fake, out, registered, explore = world
    run_main(tmp_path, monkeypatch, registered, explore)
    RV.check6()
    text = "\n".join(RV.LINES)
    lines = RV.LINES
    assert re.search(r"^ITW FRR at tau_EER lower when trimmed \| -\d+\.\d\d pp \| p = 1/252 = 0\.004 \| rejects$",
                     text, re.M)
    assert re.search(r"^ITW EER lower when trimmed \| \+\d+\.\d\d pp \| p = 252/252 = 1\.000 \| does not reject$", text, re.M)
    assert sum(" | seed " in ln for ln in lines) == 10 and sum(" | mean | " in ln for ln in lines) == 2
    assert "1/2 rejections (ITW FRR at tau_EER lower when trimmed)" in text
    assert text.count("passed") == 2 and "FAILED" not in text
    assert "trimmed minus untrimmed" in text and "excluded" in text and "seed 1234" in text


def test_check6_prints_12_of_252_as_0_048_and_no_rejection(tmp_path, monkeypatch):
    # needs no scores: the arms are built directly and the file goes to a folder under tmp_path
    monkeypatch.setattr(rvlib, "OUT", tmp_path / "review_r1")
    RV.LINES.clear()
    trimmed, untrimmed, *_ = twelve_of_252()
    _write_lastepoch(trimmed, untrimmed)
    RV.check6()
    text = "\n".join(RV.LINES)
    assert f"p = 12/252 = {12 / 252:.3f}" in text and "p = 12/252 = 0.048" in text
    assert "0/2 rejections (neither test rejects)" in text and "against the first step 0.025" in text
    assert re.search(r"lower when trimmed \| -\d+\.\d\d pp \| p = 12/252", text)


# --------------------------------------------------------------------------- C. distinct speakers
def test_distinct_speaker_share_formula():
    assert RV.distinct_share(2) == 0.75                               # 1 - (1/2)^2
    assert RV.distinct_share(1) == 1.0
    assert RV.distinct_share(3) == pytest.approx(1 - (2 / 3) ** 3)
    share = RV.distinct_share(67)
    assert share == pytest.approx(1 - (66 / 67) ** 67, abs=1e-15)
    assert 0.6348 < share < 0.6350
    assert abs(share - (1 - np.exp(-1.0))) > 0.002                    # not the limit 1 - 1/e
    assert 42.5 < 67 * share < 42.6                                   # about 42.5 distinct speakers of 67
    # a Monte Carlo of the cluster bootstrap itself agrees
    draws = np.random.default_rng(0).integers(0, 67, size=(20000, 67))
    mc = np.mean([len(set(row)) for row in draws]) / 67
    assert mc == pytest.approx(share, abs=0.002)


def _write_cv(out, g_values):
    out.mkdir(parents=True, exist_ok=True)
    cells = {f"m{i}|s0|la19_eval": {"model": "aasist", "seed": 0, "target": "la19_eval", "G": g}
             for i, g in enumerate(g_values)}
    cells["m|s0|itw"] = {"model": "aasist", "seed": 0, "target": "itw", "G": 5}      # other targets are ignored
    (out / "cv.json").write_text(json.dumps({"meta": {}, "cells": cells}), encoding="utf-8")


def _write_e7(tmp_path, monkeypatch, fake, shift=None):
    """A synthetic E7 (the committed key layout), computed with r2_explore.three_zone_edges on the synthetic dev."""
    e7 = {}
    for m in list(R.DETECTORS_R2) + list(R.ROUND1):
        _, _, sd = fake.load(m, rvlib.source_dev(m))
        e7[f"{m}|itw"] = {"0.1": {"edges_dev": list(EX.three_zone_edges(sd, 0.0, 0.10))}}
    fake.calls.clear()
    if shift:
        m, delta = shift
        e7[f"{m}|itw"]["0.1"]["edges_dev"][1] += delta
    path = tmp_path / "committed_e7.json"
    path.write_text(json.dumps({"E7": e7}), encoding="utf-8")
    monkeypatch.setattr(RV, "EXPLORE", path)
    return e7


def test_check7_takes_g_from_cv_and_prints_the_expected_share_and_count(world, tmp_path, monkeypatch):
    fake, out, *_ = world
    _write_cv(out, [2, 2, 2])
    _write_e7(tmp_path, monkeypatch, fake)
    RV.check7()
    text = "\n".join(RV.LINES)
    assert "G = 2 speakers" in text and "= 0.7500 (75.00%)" in text and "1.5 of 2 (1.50)" in text


def test_check7_with_g_67_prints_the_share_and_the_expected_number_to_one_decimal(world, tmp_path, monkeypatch):
    fake, out, *_ = world
    _write_cv(out, [67] * 14)
    _write_e7(tmp_path, monkeypatch, fake)
    RV.check7()
    share = 1 - (66 / 67) ** 67
    text = "\n".join(RV.LINES)
    assert f"= {share:.4f} ({100 * share:.2f}%)" in text
    assert f"{67 * share:.1f} of 67 ({67 * share:.2f})" in text


@pytest.mark.parametrize("g_values", [[67, 66], []])
def test_check7_refuses_without_one_common_la19_eval_speaker_count(world, g_values):
    fake, out, *_ = world
    _write_cv(out, g_values)
    with pytest.raises(SystemExit) as e:
        RV.check7()
    assert "la19_eval" in refused(e)


# --------------------------------------------------------------------------- C. the clamp rule
def share_below(q, n=1000):
    """n scores of which exactly q * n lie below 0."""
    k = round(q * n)
    return np.concatenate([-np.linspace(0.1, 3.0, k), np.linspace(0.1, 3.0, n - k)])


@pytest.mark.parametrize("q,low,high", [(0.02, True, False), (0.5, False, False), (0.97, False, True),
                                        (0.05, False, False), (0.95, False, False), (0.0, True, False), (1.0, False, True)])
def test_clamp_rule_and_edges_agree_with_three_zone_edges(q, low, high):
    sd = share_below(q)
    z = RV.zone_clamp(sd, 0.0)
    assert z["q"] == np.mean(sd < 0.0) == q
    assert z["lo_q"] == q - 0.05 and z["hi_q"] == q + 0.05
    assert (z["clamped_low"], z["clamped_high"]) == (low, high)       # q - 0.05 < 0 or q + 0.05 > 1, both strict
    assert z["edges"] == EX.three_zone_edges(sd, 0.0, 0.10)
    lo, hi = z["edges"]
    assert lo == float(np.quantile(sd, max(0.0, q - 0.05))) and hi == float(np.quantile(sd, min(1.0, q + 0.05)))
    if low:
        assert lo == sd.min()                                         # clamped to level 0: the lowest score
    if high:
        assert hi == sd.max()                                         # clamped to level 1: the highest score


def test_q_counts_scores_strictly_below_tau():
    z = RV.zone_clamp([-1.0, 0.0, 0.0, 1.0], 0.0)
    assert z["q"] == 0.25


def test_check7_covers_the_eight_detectors_on_their_source_dev_and_names_the_clamped_ones(world, tmp_path, monkeypatch):
    fake, out, *_ = world
    fake.dev_q.update({"aasist": 0.02, "pascu_lr": 0.97})
    _write_cv(out, [67])
    _write_e7(tmp_path, monkeypatch, fake)
    RV.check7()
    detectors = list(R.DETECTORS_R2) + list(R.ROUND1)
    assert len(detectors) == 8
    assert fake.calls.count(("as5_aasist", "as5_dev", 0)) == 1
    assert all((m, "la19_dev", 0) in fake.calls for m in detectors if m != "as5_aasist")
    assert not [c for c in fake.calls if c[1] not in ("la19_dev", "as5_dev")]
    # columns: detector | source dev | q | q - 0.05 | q + 0.05 | edge clamped | edges equal E7
    rows = [ln for ln in RV.LINES if ln.count(" | ") == 6 and re.search(r"\| (LA19|AS5) dev \|", ln)]
    assert [ln.split(" | ")[0] for ln in rows] == [RV.NAMES[m] for m in detectors]
    by_name = {ln.split(" | ")[0]: ln.split(" | ") for ln in rows}
    assert by_name["AS5 AASIST"][1:] == ["AS5 dev", "0.900", "0.850", "0.950", "no", "yes"]
    assert by_name["XLSR-Mamba"][1:] == ["LA19 dev", "0.900", "0.850", "0.950", "no", "yes"]
    assert by_name["AASIST"][1:] == ["LA19 dev", "0.020", "-0.030", "0.070", "q - 0.05 < 0", "yes"]
    assert by_name["Pascu LR"][1:] == ["LA19 dev", "0.970", "0.920", "1.020", "q + 0.05 > 1", "yes"]
    summary = [ln for ln in RV.LINES if ln.startswith("  detectors with a clamped edge")]
    assert len(summary) == 1
    assert "AASIST (q - 0.05 < 0)" in summary[0] and "Pascu LR (q + 0.05 > 1)" in summary[0] and "none" not in summary[0]


def test_check7_summary_says_none_when_no_edge_is_clamped(world, tmp_path, monkeypatch):
    fake, out, *_ = world
    _write_cv(out, [67])
    _write_e7(tmp_path, monkeypatch, fake)
    RV.check7()
    assert RV.LINES[-2:] == ["  detectors with a clamped edge (q - 0.05 < 0 or q + 0.05 > 1): none", ""]


def test_check7_refuses_when_a_committed_e7_edge_differs_and_passes_within_1e9(world, tmp_path, monkeypatch):
    fake, out, *_ = world
    _write_cv(out, [67])
    _write_e7(tmp_path, monkeypatch, fake, shift=("xlsr_mamba", 1e-6))
    with pytest.raises(SystemExit) as e:
        RV.check7()
    assert "E7 edges of xlsr_mamba" in refused(e)
    RV.LINES.clear()
    _write_e7(tmp_path, monkeypatch, fake, shift=("xlsr_mamba", 1e-10))
    RV.check7()
    assert RV.LINES[-1] == ""


def test_check7_refuses_when_e7_lacks_a_detector(world, tmp_path, monkeypatch):
    fake, out, *_ = world
    _write_cv(out, [67])
    e7 = _write_e7(tmp_path, monkeypatch, fake)
    del e7["rawnet2|itw"]
    (tmp_path / "committed_e7.json").write_text(json.dumps({"E7": e7}), encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        RV.check7()
    assert "E7 / rawnet2|itw" in refused(e)


# --------------------------------------------------------------------------- B. the DF21 surface of run_smalllabel.py
SURFACES = ["itw", "la19_eval", "df21_verified"]


def smalllabel_problems(src):
    """What is wrong with run_smalllabel.py's structure for DEV-11 B ([] if nothing): the surface loop must be exactly
    itw, la19_eval, df21_verified (df21_verified last), and default_rng(0) must be the first statement of that loop's
    body that touches the RNG, i.e. restarted for every detector and target, and the only default_rng call."""
    tree = ast.parse(src)
    problems = []
    outer = [n for n in tree.body if isinstance(n, ast.For) and ast.unparse(n.iter) == "ROUND1 + DETECTORS_R2"]
    if len(outer) != 1:
        return ["no single module-level loop over ROUND1 + DETECTORS_R2"]
    inner = [n for n in outer[0].body if isinstance(n, ast.For) and ast.unparse(n.target) == "surf"]
    if len(inner) != 1:
        return ["no single surface loop directly inside the detector loop"]
    inner = inner[0]
    if not isinstance(inner.iter, ast.Tuple):
        return ["the surface loop does not iterate over a literal tuple"]
    surfaces = [ast.literal_eval(e) for e in inner.iter.elts]
    if surfaces != SURFACES:
        problems.append(f"surfaces {surfaces}, expected {SURFACES}")
    rng_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func).endswith("default_rng")]
    in_loop = [n for n in ast.walk(inner) if n in rng_calls]
    if len(rng_calls) != 1 or len(in_loop) != 1 or ast.unparse(rng_calls[0]) != "np.random.default_rng(0)":
        problems.append("default_rng(0) is not called exactly once, inside the surface loop")
    else:
        direct = [i for i, st in enumerate(inner.body) if isinstance(st, ast.Assign) and rng_calls[0] in ast.walk(st)]
        uses = [i for i, st in enumerate(inner.body) if "rng." in ast.unparse(st)]
        if not direct or (uses and min(uses) < direct[0]):
            problems.append("default_rng(0) is not a direct statement of the surface loop before the first draw")
    return problems


def test_run_smalllabel_has_df21_last_and_restarts_the_rng_for_every_target():
    src = SMALLLABEL.read_text(encoding="utf-8")
    assert smalllabel_problems(src) == []
    doc = ast.get_docstring(ast.parse(src))
    assert "DEV-11 B" in doc and "df21_verified" in doc and "unchanged" in doc


def _mutate(src, old, new):
    assert src.count(old) == 1, old
    return src.replace(old, new)


MUTATIONS = {
    "df21_verified dropped": ('("itw", "la19_eval", "df21_verified")', '("itw", "la19_eval")'),
    "df21_verified first": ('("itw", "la19_eval", "df21_verified")', '("df21_verified", "itw", "la19_eval")'),
    "df21_verified in the middle": ('("itw", "la19_eval", "df21_verified")', '("itw", "df21_verified", "la19_eval")'),
    "la19_eval before itw": ('("itw", "la19_eval", "df21_verified")', '("la19_eval", "itw", "df21_verified")'),
    "df21_verified misspelt": ('"df21_verified")', '"df21")'),
}


@pytest.mark.parametrize("what", list(MUTATIONS))
def test_a_mutated_surface_loop_is_caught(what):
    old, new = MUTATIONS[what]
    bad = _mutate(SMALLLABEL.read_text(encoding="utf-8"), old, new)
    assert smalllabel_problems(bad), what


def test_moving_the_rng_out_of_the_surface_loop_is_caught():
    src = SMALLLABEL.read_text(encoding="utf-8")
    line = "        rng = np.random.default_rng(0)\n"
    assert src.count(line) == 1
    without = src.replace(line, "")
    before_surface_loop = _mutate(without, "    for surf in (", "    rng = np.random.default_rng(0)\n    for surf in (")
    assert smalllabel_problems(before_surface_loop)
    before_everything = _mutate(without, "for m in ROUND1 + DETECTORS_R2:", "rng = np.random.default_rng(0)\nfor m in ROUND1 + DETECTORS_R2:")
    assert smalllabel_problems(before_everything)
    after_first_draw = _mutate(without, "        res = {k: [] for k in KS}\n",
                               "        res = {k: [] for k in KS}\n        half0 = np.random.default_rng(1).permutation(G)\n"
                               "        rng = np.random.default_rng(0)\n")
    assert smalllabel_problems(after_first_draw)               # another seed, and a second call
    other_seed = _mutate(src, "np.random.default_rng(0)", "np.random.default_rng(1)")
    assert smalllabel_problems(other_seed)


# --------------------------------------------------------------------------- the two other patched scripts
def test_run_all_runs_run_lastepoch_directly_before_rv_checks():
    steps = list(run_all.STEPS)
    assert steps[-2:] == ["run_lastepoch.py", "rv_checks.py"]
    assert steps[:-2] == ["run_cv.py", "run_dev.py", "run_ladder.py", "run_smalllabel.py", "run_intervals.py",
                          "run_h2cv_fast.py", "run_h2cv_folds.py", "run_mofn.py", "run_atc.py"]
    here = Path(run_all.__file__).resolve().parent
    assert all((here / s).is_file() or (rvlib.REPO / "analysis" / "review_r1" / s).is_file() for s in steps)
    assert "DEV-11" in run_all.__doc__


def test_rv_checks_main_appends_checks_6_and_7_after_check_5():
    tree = ast.parse((REVIEW / "rv_checks.py").read_text(encoding="utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    loop = next(n for n in ast.walk(main) if isinstance(n, ast.For))
    parts = [ast.unparse(e) for e in loop.iter.elts]
    assert parts == ["check1", "check1b", "check1c", "check2", "check3", "smalllabel", "check4", "check4b", "check5",
                     "check6", "check7"]
    assert all(callable(getattr(RV, p)) for p in parts)
    assert "DEV-11" in RV.__doc__ and "r2_explore.json" in RV.__doc__
