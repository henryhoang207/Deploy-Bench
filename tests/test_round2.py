"""Round-2 tests (PREREGISTRATION_ROUND2.md; D54-D57). They run without vendor code, checkpoints or GPUs.

- The VAD trim is DEV-8's rule (synthetic signals).
- The ASVspoof 5 track-1 parser refuses what it cannot honour.
- The ported cluster / oracle / tie-interval machinery reproduces DEV-8's committed results exactly (round-1 scores
  are committed; the metadata-dependent checks need $R2_META, e.g. DFV_Conference/revision/meta, and skip otherwise).
- Static contracts for the round-2 wrappers and the training script (score lines, no eval set, swa.pth).
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis" / "round2"))

from deploybench import vad  # noqa: E402


# --------------------------------------------------------------------------- VAD (DEV-8 rule)
def _clip(lead_s, speech_s, trail_s, sr=16000):
    rng = np.random.default_rng(0)
    sil = lambda s: (1e-5 * rng.standard_normal(int(s * sr))).astype(np.float32)  # noqa: E731
    sp = (0.3 * np.sin(2 * np.pi * 220 * np.arange(int(speech_s * sr)) / sr)).astype(np.float32)
    return np.concatenate([sil(lead_s), sp, sil(trail_s)])


def test_vad_trims_both_ends_on_the_frame_grid():
    x = _clip(1.0, 2.0, 0.5)
    lead, trail = vad.bounds(x)
    assert lead % vad.HOP == 0
    assert abs(lead - 16000) <= vad.FRAME            # within one frame of the true onset
    assert abs(trail - 8000) <= vad.FRAME
    y, st = vad.trim(x)
    assert st == "trimmed" and len(y) == len(x) - lead - trail


def test_vad_guard_and_short_clip():
    x = _clip(1.0, 0.1, 1.0)                           # 0.1 s of speech: trimmed length < 0.25 s -> untouched
    y, st = vad.trim(x)
    assert st == "guard" and y is x
    z = np.zeros(300, dtype=np.float32)                 # shorter than one frame
    assert vad.bounds(z) == (0, 0) and vad.trim(z)[1] == "none"


def test_vad_equals_the_dev8_reference_formula():
    """The DEV-8 implementation (f_vad_stats.frame_energy_db / bounds), transcribed, on random clips."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        n = int(rng.integers(500, 40000))
        x = (rng.standard_normal(n) * np.exp(rng.standard_normal(n))).astype(np.float32)
        c = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
        nf = 1 + (n - 400) // 160
        i = np.arange(nf) * 160
        e = 10.0 * np.log10((c[i + 400] - c[i]) / 400 + 1e-12)
        act = np.nonzero(e >= e.max() - 35)[0]
        assert vad.bounds(x) == (int(act[0] * 160), int(n - (act[-1] * 160 + 400)))


# --------------------------------------------------------------------------- ASVspoof 5 parser
def _as5_tree(tmp_path, rows, part="dev"):
    d = tmp_path / "asvspoof5"
    (d / ("flac_D" if part == "dev" else "flac_E")).mkdir(parents=True)
    (d / f"ASVspoof5.{part}.track_1.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return tmp_path


def test_as5_parser_reads_track1_and_refuses_bad_rows(tmp_path, monkeypatch):
    from deploybench import data
    rows = ["D_0062 D_0000000001 F - - - AC1 A11 spoof -", "D_0755 D_0000000022 M - - - - bonafide bonafide -"]
    root = _as5_tree(tmp_path, rows)
    monkeypatch.setitem(data.REGISTRY, "asvspoof5_dev", ("asvspoof5/flac_D", "ASVspoof5.dev.track_1.tsv", 2))
    out = list(data.protocol_iter("asvspoof5_dev", root))
    assert [(u, l) for u, _, l in out] == [("D_0000000001", 0), ("D_0000000022", 1)]
    assert out[0][1] == root / "asvspoof5" / "flac_D" / "D_0000000001.flac"
    monkeypatch.setitem(data.REGISTRY, "asvspoof5_dev", ("asvspoof5/flac_D", "ASVspoof5.dev.track_1.tsv", 3))
    with pytest.raises(ValueError, match="expected 3"):
        list(data.protocol_iter("asvspoof5_dev", root))
    (root / "asvspoof5" / "ASVspoof5.dev.track_1.tsv").write_text("D_1 D_2 F - - - - A1 maybe -\n")
    monkeypatch.setitem(data.REGISTRY, "asvspoof5_dev", ("asvspoof5/flac_D", "ASVspoof5.dev.track_1.tsv", 1))
    with pytest.raises(ValueError, match="unknown KEY"):
        list(data.protocol_iter("asvspoof5_dev", root))
    with pytest.raises(NotImplementedError):
        list(data.protocol_iter("asvspoof5_train", root))


def test_as5_eval_conditions(tmp_path, monkeypatch):
    from deploybench import data
    rows = ["E_1607 E_0009538969 M C05 2 E_0009486171 AC1 A26 spoof -",
            "E_2192 E_0004993854 F - 0 - - bonafide bonafide -"]
    root = _as5_tree(tmp_path, rows, part="eval")
    monkeypatch.setitem(data.REGISTRY, "asvspoof5_eval", ("asvspoof5/flac_E", "ASVspoof5.eval.track_1.tsv", 2))
    assert data.native_conditions("asvspoof5_eval", root) == {"E_0009538969": "C05", "E_0004993854": "C00"}


# --------------------------------------------------------------------------- statistics
def test_permutation_test_is_exact():
    import r2_registered as T
    obs, p, n = T.perm_greater([5, 6, 7], [1, 2, 3])
    assert n == 20 and obs == 4 and p == pytest.approx(1 / 20)
    obs, p, n = T.perm_greater([1, 2, 3], [5, 6, 7])
    assert p == pytest.approx(1.0)


def test_tie_interval_and_point_reproduce_dev8_and_round1():
    """DEV-8 B3 intervals (b_thresholds.json) and the registered ITW TTGs at tau_eer (operating_points via h1_h4)."""
    import r2lib as R
    b3 = {"aasist": (-1.199649691581726, -0.3804270029067993), "aasist_l": (-1.0306272506713867, -0.3066258728504181),
          "rawnet2": (-7.159570217132568, -6.167920112609863), "tssdnet": (-8.500221252441406, -7.0918803215026855),
          "xlsr_aasist": (-3.382459878921509, 2.1434271335601807)}
    for m, (lo, hi) in b3.items():
        _, y, s = R.load(m, "la19_dev")
        t = R.taus(m)
        assert R.tie_interval(s, y, t["eer"]) == (lo, hi), m
    ttg = {"aasist": 2.94456, "aasist_l": 1.547921, "rawnet2": 5.436607, "tssdnet": 0.732725, "xlsr_aasist": 14.957636}
    for m, v in ttg.items():
        _, y, s = R.load(m, "itw")
        assert round(100 * R.point(s, y, R.taus(m)["eer"])["ttg"], 6) == v, m


@pytest.mark.skipif(not os.environ.get("R2_META"), reason="needs $R2_META (ITW meta.csv, LA19 eval protocol)")
def test_cluster_bootstrap_reproduces_dev8_a1_and_a2():
    import r2lib as R
    from deploybench.manifests import read_ids
    a1 = {"aasist": (1.856691480457534, [-9.842174937151176, 17.39022275262352], 0.36706329367063295),
          "tssdnet": (31.317214521204107, [14.322583814423211, 51.808940627818814], 9.999000099990002e-05)}
    for m, (d, ci, p) in a1.items():
        ids, y, s = R.load(m, "itw")
        g, names = R.clusters("itw", ids)
        C = R.cluster_counts(len(names), 10000, 0)
        fr, fa = R.cluster_rates(s, y, g, len(names), R.taus(m)["eer"], C)
        st = fr - fa
        assert R.pval_le0(st) == p and [100 * x for x in R.pct(st)] == ci, m
    a2 = {"aasist": ([8.428103604783841, 17.1245226570264], 9.999000099990002e-05),
          "rawnet2": ([0.9106654257225335, 4.8978460757469175], 0.0011998800119988001)}
    ids = read_ids(ROOT / "configs" / "parity_manifest_tssdnet.txt")
    import pandas as pd
    for m, (ci, p) in a2.items():
        h5 = ROOT / "results" / "phase_3_stress" / "h5"
        c = pd.read_csv(h5 / f"scores_{m}_clean.csv", dtype={"utt_id": str}, float_precision="round_trip").set_index("utt_id").reindex(ids)
        e = pd.read_csv(h5 / f"scores_{m}_encodec6k.csv", dtype={"utt_id": str}, float_precision="round_trip").set_index("utt_id").reindex(ids)
        y = c["label"].to_numpy(np.int64)
        g, names = R.clusters("stress_x", np.asarray(ids))
        C = R.cluster_counts(len(names), 10000, 0)
        _, _, dfrr, dfar = R.paired_cluster_test(y, c["score"].to_numpy(), e["score"].to_numpy(), g, len(names),
                                                 R.taus(m)["eer"], C)
        st = dfrr - dfar
        assert R.pval_le0(st) == p and [100 * x for x in R.pct(st)] == ci, m


# --------------------------------------------------------------------------- static contracts
def _src(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_round2_specs_and_roster():
    from models.base import ROSTER, ROUND2, SPECS
    assert ROSTER == ["aasist", "aasist_l", "rawnet2", "tssdnet", "xlsr_aasist"]
    assert ROUND2 == ["as5_aasist", "pascu_lr", "xlsr_mamba", "aasist_l_r2", "aasist_l_trim"]
    for m in ROUND2 + ["pascu_lr_gate", "aasist_l_r2_last", "aasist_l_trim_last"]:
        assert SPECS[m].round == 2, m
    assert SPECS["as5_aasist"].cohort == "cohort_as5"                     # tau on AS5 dev (D16)
    assert SPECS["aasist_l_trim"].preproc == "vad35" and SPECS["aasist_l_r2"].preproc == ""
    assert SPECS["aasist_l_trim_last"].preproc == "vad35" and SPECS["aasist_l_r2_last"].preproc == ""
    assert SPECS["pascu_lr"].input_samples == 0 and SPECS["xlsr_mamba"].input_samples == 66800
    assert all(SPECS[m].round == 1 for m in ROSTER)


def test_round2_score_lines():
    assert re.search(r"out\[:, 1\]\.cpu", _src("models/wrap_xlsr_mamba.py"))          # bonafide = 1 (author code)
    assert "load_vendor_module" in _src("models/wrap_xlsr_mamba.py")
    assert "m.training for m in self.model.modules()" in _src("models/wrap_xlsr_mamba.py")
    assert "strict=False" not in _src("models/wrap_xlsr_mamba.py")
    assert "out[i] = -(f @ self.coef + self.intercept)" in _src("models/wrap_pascu.py")   # their label 1 = fake
    s = _src("models/wrap_aasist.py")
    assert '"as5_aasist":    ("asvspoof5/Baseline-AASIST"' in s and "a checkpoint is required" in s


def test_training_script_never_touches_eval_and_keeps_the_recipe_rule():
    s = _src("train/aasist_r2_train.py")
    code = s.split('"""', 2)[2]                                            # skip the docstring
    for tok in ("eval_loader", "eval_trial", "cm.eval.trl", "LA_eval", "eval_score", "eval_all_best",
                "calculate_tDCF_EER", "is_eval=True"):
        assert tok not in code, f"the training script must not reference an evaluation set ({tok})"
    assert "best_dev_eer = 1." in code and "if best_dev_eer >= dev_eer:" in code
    assert '"swa.pth"' in code and '"best.pth"' not in code
    assert "optimizer_swa.bn_update(trn_loader, model, device=device)" in code
    assert "torch.backends.cudnn.allow_tf32 = False" in code


def test_conversion_scorers_refuse_round2():
    for f in ("score_onnx.py", "score_pte.py"):
        assert "SPECS[a.model].round != 1" in _src(f), f


def test_score_py_paths_for_round2():
    s = _src("score.py")
    assert "fix_length(wave, n) if n else wave" in s
    assert 'if spec.preproc == "vad35":' in s


# --------------------------------------------------------------------------- exploratory building blocks
def test_three_zone_budget_and_apply():
    import r2_explore as X
    rng = np.random.default_rng(0)
    s = rng.standard_normal(10000)
    lo, hi = X.three_zone_edges(s, 0.0, 0.10)
    q = np.mean(s < 0.0)
    assert np.mean((s >= lo) & (s < hi)) == pytest.approx(0.10, abs=0.002)
    assert np.mean(s < lo) == pytest.approx(q - 0.05, abs=0.002)
    y = (s > 0).astype(int)
    r = X.three_zone_apply(s, y, lo, hi)
    assert r["bona_auto_rejected"] == 0.0 and r["spoof_auto_accepted"] == 0.0     # perfectly separated at 0
    assert r["review_pooled"] == pytest.approx(0.10, abs=0.002)


def test_bayes_thresholds_recover_the_llr():
    """Scores that ARE log-likelihood ratios, at a 10% bona fide prior: the prior-corrected Platt map is ~identity, so the
    Bayes threshold for logit(pi) = L is ~ -L."""
    import r2_explore as X
    rng = np.random.default_rng(1)
    n_b, n_s = 2000, 18000
    sb, ss = rng.normal(1.0, np.sqrt(2.0), n_b), rng.normal(-1.0, np.sqrt(2.0), n_s)   # LLR = s for this pair
    s = np.concatenate([sb, ss]); y = np.concatenate([np.ones(n_b, int), np.zeros(n_s, int)])
    a, b, th = X.bayes_thresholds(s, y)
    assert a == pytest.approx(1.0, abs=0.1)
    for L in (-2.0, 0.0, 2.0):
        assert th[L] == pytest.approx(-L, abs=0.25)


def test_window_matches_the_closed_form():
    import r2_explore as X
    w = X.window(0.2, 0.1)
    for p in np.linspace(0.01, 0.99, 99):
        inside = w is not None and w[0] < p < w[1]
        assert inside == (X.norm_cost(p, 0.2, 0.1) < 1 - 1e-12) or abs(X.norm_cost(p, 0.2, 0.1) - 1) < 1e-9
    assert X.window(0.6, 0.6) is None                                                   # worse than chance: never


def test_platt_fit_is_not_used_for_e9_because_it_does_not_converge():
    """DEV-9 amendment 2: the harness platt_fit (200 fixed-step iterations) misses the ML fit; at a x20 score scale it
    settles on a wrong slope. E9 uses the exact fit, which is scale-equivariant."""
    import r2_explore as X
    from deploybench.metrics import platt_fit
    rng = np.random.default_rng(1)
    s = np.concatenate([rng.normal(1.0, np.sqrt(2.0), 2000), rng.normal(-1.0, np.sqrt(2.0), 18000)])
    y = np.concatenate([np.ones(2000, int), np.zeros(18000, int)])
    a1, _, _ = X.bayes_thresholds(s, y)
    a20, _, _ = X.bayes_thresholds(20 * s, y)
    assert a20 * 20 == pytest.approx(a1, rel=1e-3)
    assert abs(platt_fit(s, y)[0] - a1) > 0.05
    assert abs(platt_fit(20 * s, y, n_iter=20000)[0] * 20 - a1) > 0.2


def test_pascu_tiles_only_clips_shorter_than_the_xlsr_receptive_field():
    """DEV-9 amendment 5: a clip under 400 samples is repeated to exactly 400 (fix_length's tiling); others untouched."""
    from models.wrap_pascu import MIN_SAMPLES, _tile_to_min
    assert MIN_SAMPLES == 400
    x = np.arange(185, dtype=np.float32)
    y = _tile_to_min(x)
    assert len(y) == 400 and np.array_equal(y, np.tile(x, 3)[:400])
    z = np.arange(557, dtype=np.float32)
    assert _tile_to_min(z) is z
    assert _tile_to_min(np.arange(400, dtype=np.float32)).shape == (400,)
    with pytest.raises(ValueError):
        _tile_to_min(np.zeros(0, dtype=np.float32))


def test_round1_models_have_no_as5_scores_and_refuse_loudly():
    """r2_explore's have() skips a (model, surface) pair through SystemExit; a KeyError crashed E7 (2026-10-10)."""
    import r2lib as R
    with pytest.raises(SystemExit):
        R.r1_path("aasist", "as5_eval")
    assert R.r1_path("aasist", "itw").name == "scores_fp32_pytorch_itw.csv"
