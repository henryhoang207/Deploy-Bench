"""Static contract checks that run WITHOUT torch/vendor/checkpoints.
Execution-level checks happen on the user's machine once assets exist."""
import ast
import re
from pathlib import Path

M = Path(__file__).resolve().parent.parent / "models"
WRAPPERS = ["wrap_aasist", "wrap_rawnet2", "wrap_rawgat", "wrap_lcnn",
            "wrap_tssdnet", "wrap_wavlm", "wrap_xlsr"]


def _src(name):
    return (M / f"{name}.py").read_text(encoding="utf-8")


def test_all_wrappers_define_load():
    for w in WRAPPERS:
        tree = ast.parse(_src(w))
        fns = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert "load" in fns, f"{w}: missing load()"


def test_no_stubs_remain():
    for w in WRAPPERS:
        assert "NotImplementedError" not in _src(w), f"{w}: still a stub"


def test_no_plain_vendor_imports():
    """Vendored repos use module names `model`/`models` that collide with our
    package — wrappers must use load_vendor_module, never sys.path+import."""
    for w in WRAPPERS:
        s = _src(w)
        assert "sys.path.insert" not in s, f"{w}: raw sys.path injection"
        if w != "wrap_wavlm":  # wavlm is HF-based, no vendor module
            assert "load_vendor_module" in s, f"{w}: not collision-safe"


def test_score_conventions_documented():
    """The class-index order is per-repo (AASIST bonafide=1, TSSDNet
    bonafide=0). Guard the exact score lines against silent edits."""
    assert re.search(r"out\[:, 1\]\.cpu", _src("wrap_aasist"))       # raw logit
    assert re.search(r"out\[:, 0\] - out\[:, 1\]", _src("wrap_tssdnet"))
    assert re.search(r"out\[:, 1\]\.cpu", _src("wrap_rawnet2"))
    assert re.search(r"out\[:, 1\]\.cpu", _src("wrap_rawgat"))
    assert re.search(r"out\[:, 1\]\.cpu", _src("wrap_xlsr"))
    assert re.search(r"out\[:, 1\] - out\[:, 0\]", _src("wrap_wavlm"))


def test_fix_length_truncates_from_start():
    s = (M.parent / "score.py").read_text(encoding="utf-8")
    assert "wave[:n]" in s and "// 2" not in s.split("def fix_length")[1].split("def ")[0]


def test_vendor_check_exempts_hf_models():
    s = (M / "base.py").read_text(encoding="utf-8")
    assert "wavlm_trunc" in s.split("def load_model")[1], \
        "load_model must not require vendor/ for HF-only wavlm models"


def test_train_heads_is_tombstoned():
    """D30 froze the roster at 5 published-checkpoint models, so nothing trains
    a head any more. The old assertion here guarded D14's backbone-eval contract;
    it is replaced rather than deleted so that silently reviving the script
    without re-reading its docstring fails loudly. That docstring records a real
    defect (early-stopping on the same dev split D16 used as the tau surface)."""
    s = (M.parent / "train" / "train_heads.py").read_text(encoding="utf-8")
    assert "TOMBSTONE" in s and "raise SystemExit" in s, \
        "train/train_heads.py must stay tombstoned while D30 stands"


def test_roster_is_frozen_at_five():
    """The frozen roster is a decision (D30), not a convenience list. Anything
    iterating 'the roster' must read models.base.ROSTER, and the out-of-roster
    wrappers must stay present-but-flagged rather than deleted."""
    s = (M / "base.py").read_text(encoding="utf-8")
    tree = ast.parse(s)
    roster = next((n for n in ast.walk(tree)
                   if isinstance(n, ast.Assign)
                   and any(getattr(t, "id", "") == "ROSTER" for t in n.targets)), None)
    assert roster is not None, "models/base.py must define ROSTER"
    names = [e.value for e in roster.value.elts]
    assert names == ["aasist", "aasist_l", "rawnet2", "tssdnet", "xlsr_aasist"], \
        f"roster changed without a decision-log entry: {names}"
    for out in ("rawgat_st", "lcnn_lfcc", "wavlm_trunc", "wavlm_mhfa"):
        assert f'"{out}"' in s and "in_roster=False" in s, \
            f"{out} must remain defined but flagged out-of-roster, not deleted"
