"""Audit the Speech Communication draft from the paper worktree.

    python paper/speechcomm/audit_claims.py [--final]

Round-1 claims are rebuilt from the audited revision results; generated round-1
tables must equal the revision's bytes after checkout line-ending normalisation.
Main-file inputs and the supplement are
audited recursively. Quantitative prose and table/figure blocks require a
``% src:`` comment in the same paragraph or environment. Source comments accept
worktree paths, ``revision/...`` paths, and ``bib:<key>`` citations. Revision
numeric prose in the supplement is checked against its source numeric sequence.
Registered round-2 methods are checked against the registration and amendments.

Round 2 (integrated 2026-10-11): every gen/r2_*.tex and gen/r2s_*.tex must equal what
analysis/round2/r2_paper_tables.py writes from the committed results (regenerated into a
temporary folder and compared byte for byte), and the round-2 prose claims are rebuilt
from results/round2/{gates.json, analysis/r2_registered.json, analysis/r2_explore.json}.
Prose that cites a round-2 source may use only numbers that those files (rounded) or
the cited tables contain.

The read-only revision source is discovered under ~/OneDrive, or can be selected
with DFV_REVISION (revision directory, or its analysis directory). DEPLOYBENCH
is deliberately bound to THIS worktree before revlib imports its estimators.
--final refuses pending items, todos and missing/failed compiled PDFs.
Speech Communication has no page cap in this audit.
"""
from __future__ import annotations

import json
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
from argparse import ArgumentParser
from decimal import Decimal
from pathlib import Path

sys.dont_write_bytecode = True
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[1]


def revision_dir():
    selected = os.environ.get("DFV_REVISION")
    if selected:
        p = Path(selected).resolve()
        if p.name == "analysis":
            p = p.parent
        candidates = [p] if (p / "analysis" / "revlib.py").is_file() else []
    else:
        candidates = list((Path.home() / "OneDrive").glob(
            "*/CS material/Research/DFV_Conference/revision"))
        candidates = [p for p in candidates if (p / "analysis" / "revlib.py").is_file()]
    candidates = sorted({p.resolve() for p in candidates})
    if len(candidates) != 1:
        sys.exit("REFUSED: select the audited revision with DFV_REVISION; found "
                 f"{len(candidates)} usable revision directories")
    return candidates[0].resolve()


REVISION = revision_dir()
os.environ["DEPLOYBENCH"] = str(WORKTREE)
os.environ["DFV_META"] = str(REVISION / "meta")
sys.path.insert(0, str(WORKTREE))
spec = importlib.util.spec_from_file_location("speechcomm_revlib", REVISION / "analysis" / "revlib.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
if R.CLONE.resolve() != WORKTREE or R.RESULTS.resolve() != REVISION / "results":
    sys.exit("REFUSED: revlib did not resolve to this paper worktree and the selected revision")

RES = R.RESULTS
M5, RK = list(R.MODELS), list(R.RANKED)
B = json.loads((RES / "b_thresholds.json").read_text(encoding="utf-8"))
A = json.loads((RES / "a_cluster.json").read_text(encoding="utf-8"))
E = json.loads((RES / "e_cost.json").read_text(encoding="utf-8"))
C5 = json.loads((RES / "a5_cheapcheck.json").read_text(encoding="utf-8"))
G = json.loads((RES / "g_docs.json").read_text(encoding="utf-8"))
FS = json.loads((RES / "f_vad_stats_analysis.json").read_text(encoding="utf-8"))
FT = json.loads((RES / "f_vad_scores_analysis.json").read_text(encoding="utf-8"))
H2 = json.loads((R.COMMITTED / "h2_test.json").read_text(encoding="utf-8"))
H5 = json.loads((R.COMMITTED / "h5_test.json").read_text(encoding="utf-8"))
D = B["deterministic"]

# ---- round 2: the committed final results (collected from sol1 and SHA-256 verified, 1a51fdb)
R2 = WORKTREE / "results" / "round2"
GT = json.loads((R2 / "gates.json").read_text(encoding="utf-8"))
RG = json.loads((R2 / "analysis" / "r2_registered.json").read_text(encoding="utf-8"))
EX = json.loads((R2 / "analysis" / "r2_explore.json").read_text(encoding="utf-8"))
PH = json.loads((R2 / "pascu" / "head_train.json").read_text(encoding="utf-8"))
SEEDS = ("1", "2", "3", "4", "5")


def _numeric_leaves(o):
    if isinstance(o, bool):
        return
    if isinstance(o, (int, float)):
        yield float(o)
    elif isinstance(o, dict):
        for v in o.values():
            yield from _numeric_leaves(v)
    elif isinstance(o, list):
        for v in o:
            yield from _numeric_leaves(v)


def _forms(x):
    """Every printed form of a value that round-2 prose may use: x and 100x, 0-3 decimals, both signs."""
    out = set()
    for v in (x, 100 * x):
        for nd in (0, 1, 2, 3):
            s = str(Decimal(f"{abs(v):.{nd}f}").normalize())
            out.update((s, "-" + s if s != "0" else s))
    if 0 <= x < 1:                             # a p value printed without its leading zero (p=.881) reads as 881
        out.add(str(int(f"{x:.3f}"[2:])))
    return out


def arm_mean(arm, key, q):
    return sum(EX["E2_E3"][arm][s][key][q] for s in SEEDS) / len(SEEDS)


R2_DERIVED = [arm_mean(a, f"{t}|eer", q) for a in ("aasist_l_r2", "aasist_l_r2_last", "aasist_l_trim")
              for t in ("itw", "la19_eval") for q in ("frr", "far", "eer", "ttg")]
R2_DERIVED += [arm_mean("aasist_l_trim", "itw|eer", q) - arm_mean(a, "itw|eer", q)
               for a in ("aasist_l_r2", "aasist_l_r2_last") for q in ("frr", "far", "eer")]
R2_DERIVED += [EX["E2_E3"]["aasist_l_trim"][s]["itw|eer"]["frr"] - EX["E2_E3"]["aasist_l_r2_last"][s]["itw|eer"]["frr"]
               for s in SEEDS]
R2_DERIVED += [sum(r[c][k] for r in (EX["E4"][f"{a}|s{s}"] for s in SEEDS)) / 5 for a in ("aasist_l_r2", "aasist_l_trim")
               for c in EX["E4"]["aasist_l_r2|s1"] for k in ("dfrr", "dfar")]
R2_DERIVED += [1 / (1 + 2.718281828459045 ** -2)]          # logit 2 as a prior
R2_NUMBERS = set()
for _v in [*_numeric_leaves(GT), *_numeric_leaves(RG), *_numeric_leaves(EX), *_numeric_leaves(PH), *R2_DERIVED]:
    R2_NUMBERS |= _forms(_v)

# Report only literature values still present in the recursively read draft.
# These are attributed quotations, not estimates computed by this checker.
CITED = [("78.7%", "zhou26_eer_hides"), ("15.1%", "muller21_silence"),
         ("3.6%", "muller21_silence"), ("15.5%", "muller21_silence")]


def tgt(m, t):
    return D[m]["targets"][t]


def pol(m, t, p="eer"):
    return D[m]["targets"][t]["policies"][p]


def rng(vals, nd, scale=1.0):
    return f"{min(vals) * scale:.{nd}f}-{max(vals) * scale:.{nd}f}"


def f(x, nd=1):
    return f"{x:.{nd}f}"


def sgn(x, nd=1):
    return f"{x:+.{nd}f}"


def norm_tex(s):
    s = re.sub(r"(?m)(?<!\\)%.*$", "", s)
    s = s.replace("\\%", "%").replace("\\(", "").replace("\\)", "").replace("\\,", "").replace("~", "")
    s = s.replace("{,}", ",").replace("$-$", "-").replace("$+$", "+").replace("--", "-")
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = re.sub(r"[{}$\\]", "", s)
    return re.sub(r"\s+", "", s)


def norm_claim(s):
    return re.sub(r"\s+", "", s.replace("--", "-"))


def command_args(tex, command):
    """Read balanced arguments, including nested TeX in pending placeholders."""
    for match in re.finditer(r"\\" + re.escape(command) + r"\s*\{", tex):
        start, depth, i = match.end(), 1, match.end()
        while i < len(tex) and depth:
            if tex[i] == "{" and (i == 0 or tex[i - 1] != "\\"):
                depth += 1
            elif tex[i] == "}" and (i == 0 or tex[i - 1] != "\\"):
                depth -= 1
            i += 1
        if depth:
            raise ValueError(f"unclosed \\{command} argument")
        yield match.start(), i, tex[start:i - 1]


def remove_commands(tex, names):
    for name in names:
        spans = list(command_args(tex, name))
        for start, end, _ in reversed(spans):
            tex = tex[:start] + " " + tex[end:]
    return tex


def visible_numbers(tex):
    """Numeric claims, omitting identifiers, references and layout dimensions."""
    tex = re.sub(r"(?m)(?<!\\)%.*$", "", tex)
    tex = re.sub(r"(?m)^\s*\\(?:begin\{tabular\}|setlength|includegraphics|resizebox).*$", " ", tex)
    tex = remove_commands(tex, ("todo", "cite", "citep", "citet", "ref", "label", "url",
                               "input", "gentab", "wide", "includegraphics", "setlength",
                               "vspace", "hspace", "bibliography", "bibliographystyle"))
    tex = re.sub(r"\\begin\{tabular\}.*", " ", tex)
    tex = re.sub(r"\\(?:begin|end)\{[^}]+\}(?:\[[^\]]*\])?", " ", tex)
    tex = re.sub(r"(?m)^.*\\(?:setlength|resizebox)\b.*$", " ", tex)
    tex = re.sub(r"\\includegraphics\[[^\]]*\]", " ", tex)
    tex = re.sub(r"\\(?:section|subsection|subsubsection)\*?\{[^}]*\}", " ", tex)
    tex = tex.replace("{,}", ",").replace("--", "-")
    tex = re.sub(r"(\d+(?:\.\d+)?)\s*\\times\s*10\^\{(-?\d+)\}", r"\1e\2", tex)
    tex = re.sub(r"10\^\{(-?\d+)\}", r"1e\1", tex)
    tex = re.sub(r"\\[A-Za-z]+", " ", tex)
    tex = re.sub(r"\bamr(\d+\.\d+)\b", r" \1", tex)
    tex = re.sub(r"\bround(?:\s|~|-)+[12]\b", " ", tex, flags=re.I)
    tex = re.sub(r"\b(?:LA|DF|H|R2-H|E|C|DEV-)\d+\b", " ", tex)
    from decimal import Decimal, InvalidOperation
    nums = []
    for value in re.findall(r"(?<![A-Za-z\d])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:e[-+]?\d+)?", tex):
        try:
            nums.append(str(Decimal(value.replace(",", "")).normalize()))
        except InvalidOperation:
            pass
    return nums


def prose_blocks(tex):
    """Source scope is one paragraph, or a complete table/figure environment."""
    block, depth, start = [], 0, 1
    for nline, line in enumerate(tex.splitlines(), 1):
        if not block:
            start = nline
        depth += len(re.findall(r"\\begin\{(?:table\*?|figure\*?)\}", line))
        depth -= len(re.findall(r"\\end\{(?:table\*?|figure\*?)\}", line))
        if not line.strip() and depth == 0:
            if block:
                yield start, "\n".join(block)
                block = []
        else:
            block.append(line)
    if block:
        yield start, "\n".join(block)


def source_paths(comment, bibkeys):
    """Resolve every named file; do not silently accept a nonexistent source."""
    paths, errors = [], []
    keys = re.findall(r"\bbib:([\w-]+)", comment)
    for key in keys:
        if key not in bibkeys:
            errors.append(f"unknown bibliography key {key}")
    named = re.findall(r"[A-Za-z0-9_./\\-]+\.(?:md|tex|jsonl|json|txt|csv|py)\b", comment)
    for name in named:
        name = name.replace("\\", "/")
        if "DFV_Conference/revision/" in name:
            path = REVISION / name.split("DFV_Conference/revision/", 1)[1]
        elif name.startswith("revision/"):
            path = REVISION / name[9:]
        elif name.startswith("paper/speechcomm/"):
            path = WORKTREE / name
        elif name.startswith("gen/"):
            path = HERE / name
        elif (WORKTREE / name).is_file():
            path = WORKTREE / name
        elif "revision" in comment.lower() and name.startswith("results/"):
            path = REVISION / name
        elif "revision" in comment.lower() and "/" not in name:
            options = [RES / name, REVISION / "analysis" / name, REVISION / "paper" / name]
            path = next((p for p in options if p.is_file()), options[0])
        else:
            path = WORKTREE / name
        if not path.is_file():
            errors.append(f"source file not found: {path}")
        else:
            paths.append(path.resolve())
    # The legacy comment names the revision as a whole; bind it to actual files.
    if "revision" in comment.lower() and not any(p.is_relative_to(REVISION) for p in paths):
        paths.extend((REVISION / "paper" / "main.tex", REVISION / "paper" / "supplement.tex"))
    if not paths and not keys:
        errors.append("unrecognised source; use a worktree path, revision/path or bib:key")
    for amendment in re.findall(r"DEV-9 amendment\s+(\d+)", comment, flags=re.I):
        deviations = (WORKTREE / "DEVIATIONS.md").read_text(encoding="utf-8")
        if f"### DEV-9 amendment {amendment} " not in deviations:
            errors.append(f"DEV-9 amendment {amendment} is not in the worktree deviation log")
    return paths, errors


LINE_OFFSETS = {}


def collect_documents(pending, fails):
    docs, active = {}, set()

    def visit(path):
        path = path.resolve()
        if path in active:
            fails.append(("input.cycle", str(path)))
            return
        if path in docs:
            return
        if not path.is_relative_to(HERE):
            fails.append(("input.outside_paper", str(path)))
            return
        if not path.is_file():
            fails.append(("input.missing", str(path)))
            return
        tex = path.read_text(encoding="utf-8")
        body = tex.split(r"\begin{document}", 1)[-1]
        LINE_OFFSETS[path] = tex[:len(tex) - len(body)].count("\n")
        docs[path] = body
        active.add(path)
        if path.parent.name != "gen":
            for command in ("input", "gentab", "wide"):
                for _, _, target in command_args(body, command):
                    if "#" in target or target.startswith("\\"):
                        continue
                    if command in ("gentab", "wide"):
                        target = "gen/" + target
                    child = HERE / (target if target.endswith(".tex") else target + ".tex")
                    if not child.exists() and command == "gentab":
                        pending.append(f"generated table {target} is missing")
                    else:
                        visit(child)
            for _, _, text in command_args(body, "todo"):
                pending.append(f"{path.name}: " + " ".join(text.split()))
        active.remove(path)

    visit(HERE / "main.tex")
    visit(HERE / "supplement.tex")
    return docs


def audit_provenance(docs, fails):
    bib = (HERE / "refs.bib").read_text(encoding="utf-8")
    bibkeys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib))
    n = 0
    revision_supp = visible_numbers((REVISION / "paper" / "supplement.tex").read_text(encoding="utf-8"))
    inherited_numbers = set(revision_supp + visible_numbers(
        (WORKTREE / "paper" / "tbiom" / "main.tex").read_text(encoding="utf-8")))
    mlaad_dir = WORKTREE / "results" / "phase_2_conversion_app" / "mlaad5"
    mlaad = json.loads((mlaad_dir / "mlaad_report.json").read_text(encoding="utf-8"))["models"]
    census = json.loads((mlaad_dir / "sample_rate_census.json").read_text(encoding="utf-8"))["by_lang_class_sr_ch"]
    bona = [r for r in census if r["class"] == "bonafide"]
    native16 = 100 * sum(r["n"] for r in bona if r["sr"] == 16000) / sum(r["n"] for r in bona)
    computed_mlaad = [r["sr"] / 1000 for r in census] + [native16]
    for model in M5:
        row = mlaad[model]["fp32"]
        computed_mlaad.extend(100 * x for x in (row["eer"], row["ttg_eer"], *row["at_eer"].values()))
    mlaad_numbers = set(visible_numbers(" ".join(f"{x:.{nd}f}" for x in computed_mlaad for nd in (1, 2))))
    for path, tex in docs.items():
        if path.parent.name == "gen":
            continue
        for line, block in prose_blocks(tex):
            nums = visible_numbers(block)
            has_table = bool(re.search(r"\\(?:input|gentab|wide)\{(?:gen/)?(?:s_)?tab_", block))
            has_figure = bool(re.search(r"\\includegraphics", block))
            if not nums and not has_table and not has_figure:
                continue
            n += 1
            label = f"{path.name}:{line + LINE_OFFSETS.get(path, 0)}"
            comments = re.findall(r"(?m)%\s*src:\s*(.+)$", block)
            if not comments:
                fails.append(("src.missing", label))
                continue
            sources = []
            for comment in comments:
                paths, errors = source_paths(comment, bibkeys)
                sources.extend(paths)
                fails.extend(("src.invalid", f"{label}: {error}") for error in errors)
            cites_revision = any(p.is_relative_to(REVISION) for p in sources)
            if nums and (path.name not in ("supplement.tex", "round2_design.tex")
                         or (path.name == "supplement.tex" and not cites_revision)):
                allowed = set(inherited_numbers) if path.name != "supplement.tex" else set()
                for source in sources:
                    if source.suffix in (".md", ".tex", ".py", ".json", ".jsonl", ".txt"):
                        allowed.update(visible_numbers(source.read_text(encoding="utf-8")))
                # Newly rendered descriptive MLAAD values have separate computed checks below.
                if any(p.name in ("mlaad_report.json", "sample_rate_census.json") for p in sources):
                    allowed.update(visible_numbers(docs.get(HERE / "gen" / "r1_mlaad.tex", "")))
                    allowed.update(mlaad_numbers)
                # TTG at the ends of a dev-equivalent interval: HTER there minus the oracle HTER (explicit claim below).
                if any(p.name == "b_thresholds.json" for p in sources):
                    for mm in M5:
                        for tt, cell in D[mm]["targets"].items():
                            if "tau_lo" in cell:
                                for k in ("tau_lo", "tau_hi"):
                                    v = (cell[k]["frr_pct"] + cell[k]["far_pct"]) / 2 - cell["oracle_hter_pct"]
                                    allowed.update(_forms(v / 100))
                # Round-2 prose: the committed result files, rounded as printed (the explicit claims below are exact).
                if any(p.is_relative_to(R2) or p.name.startswith(("r2_", "r2s_")) for p in sources):
                    allowed.update(R2_NUMBERS)
                unsupported = sorted(set(nums) - allowed)
                if unsupported:
                    fails.append(("src.unsupported_number", f"{label}: numbers absent from the cited source or audited draft: {unsupported}"))
            if nums and path.name == "supplement.tex" and cites_revision:
                if not any(revision_supp[i:i + len(nums)] == nums for i in range(len(revision_supp))):
                    fails.append(("supplement.numeric_sequence", f"{label}: numbers differ from the audited revision"))
            if nums and path.name == "round2_design.tex":
                # Methods numbers must exist in the documents that the paragraph cites.
                allowed = set()
                for source in sources:
                    if source.name in ("PREREGISTRATION_ROUND2.md", "DEVIATIONS.md"):
                        allowed.update(visible_numbers(source.read_text(encoding="utf-8")))
                unsupported = sorted(set(nums) - allowed)
                if unsupported:
                    fails.append(("r2.method_number", f"{label}: unsupported numbers {unsupported}"))
    return n


def regenerated_r2_tables():
    """Run the round-2 table generator of THIS worktree into a temporary folder; return {name: bytes}."""
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, R2_GEN=tmp, PYTHONDONTWRITEBYTECODE="1")
        r = subprocess.run([sys.executable, str(WORKTREE / "analysis" / "round2" / "r2_paper_tables.py")], env=env,
                           capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"REFUSED: the round-2 table generator failed:\n{r.stderr}")
        return {p.name: p.read_bytes() for p in Path(tmp).glob("*.tex")}


def audit_round2(chk, truth, schk):
    """Round-2 prose claims, rebuilt from the committed results."""
    pct = lambda x, nd=1: f"{100 * x:.{nd}f}"  # noqa: E731
    pv = lambda p: f"{p:.3f}".lstrip("0")  # noqa: E731
    span = lambda vals, nd=1, s=100: f"{min(vals) * s:.{nd}f}-{max(vals) * s:.{nd}f}"  # noqa: E731

    # gates
    a, x, p = GT["as5_aasist"], GT["xlsr_mamba"], GT["pascu_lr"]
    truth("r2.gates_pass", a["passed"] and x["passed"] and p["passed"])
    chk("r2.gate_as5", f"EER ({a['eer_pct']:.2f}% against {a['published_pct']}%")
    chk("r2.gate_mamba", f"is {x['eer_ours_pct']:.2f}% against {x['eer_theirs_pct']:.2f}% for the authors' released scores "
                         f"on the same {x['n_overlap']:,} trials")
    chk("r2.gate_pascu", f"reaches {p['eer_pct']:.2f}% on LA19 eval against the published {p['published_pct']}%")

    # E1 operating behaviour
    e1 = EX["E1_E8_E10"]
    c = lambda m, t, q="eer": e1[f"{m}|{t}"]["policies"][q]  # noqa: E731
    pa, xm, a5 = "pascu_lr", "xlsr_mamba", "as5_aasist"
    chk("r2.itw_eer", f"(EER {pct(e1[pa + '|itw']['eer'])}% for Pascu's detector and {pct(e1[xm + '|itw']['eer'])}% for "
                      f"XLSR-Mamba, against 34-49%)")
    m = c(xm, "itw")
    chk("r2.mamba_itw", f"XLSR-Mamba rejects {pct(m['frr'])}% of genuine ITW clips and accepts {pct(m['far'])}% of spoofs, so "
                        f"{100 * m['transfer_share']:.0f}% of its deployed HTER is transfer (TTG {pct(m['ttg'], 2)}pp "
                        f"[{pct(m['ttg_ci95'][0], 2)}, {pct(m['ttg_ci95'][1], 2)}])")
    q = c(pa, "itw")
    chk("r2.pascu_itw", f"Pascu's detector rejects {pct(q['frr'])}% and accepts {pct(q['far'])}% (TTG {pct(q['ttg'], 2)}pp "
                        f"[{pct(q['ttg_ci95'][0], 2)}, {pct(q['ttg_ci95'][1], 2)}], {100 * q['transfer_share']:.0f}%)")
    chk("r2.df21_dir", f"({pct(c(pa, 'df21_verified')['frr'])}% against {pct(c(pa, 'df21_verified')['far'])}%, and "
                       f"{pct(c(xm, 'df21_verified')['frr'])}% against {pct(c(xm, 'df21_verified')['far'])}%)")
    chk("r2.la19_share", f"({100 * c(pa, 'la19_eval')['transfer_share']:.0f}% and "
                         f"{100 * c(xm, 'la19_eval')['transfer_share']:.0f}%)")
    tie = lambda m: sorted(e1[f"{m}|itw"]["tie"]["frr"])  # noqa: E731
    chk("r2.tie", f"its ITW FRR runs from {pct(tie(xm)[0])}% to {pct(tie(xm)[1])}%, and from {pct(tie(pa)[0])}% to "
                  f"{pct(tie(pa)[1])}%")
    surf5 = ("la19_eval", "itw", "df21_verified", "mlaad5", "as5_eval")
    chk("r2.as5_eer", f"Its EER is {span([e1[f'{a5}|{t}']['eer'] for t in surf5], 0)}% on every target, including its own "
                      f"evaluation set ({pct(e1[a5 + '|as5_eval']['eer'])}%)")
    chk("r2.as5_ttg", f"(TTG {pct(c(a5, 'as5_eval')['ttg'], 2)}pp on ASVspoof 5 eval) and to LA19 eval and DF21 "
                      f"({pct(c(a5, 'la19_eval')['ttg'], 2)} and {pct(c(a5, 'df21_verified')['ttg'], 2)}pp)")
    chk("r2.as5_dir", f"accepts {pct(c(a5, 'la19_eval')['far'])}% of spoofs and rejects {pct(c(a5, 'la19_eval')['frr'])}% of "
                      f"genuine clips, whereas on ITW it rejects {pct(c(a5, 'itw')['frr'])}% of genuine speech and accepts "
                      f"{pct(c(a5, 'itw')['far'])}% of spoofs (TTG {pct(c(a5, 'itw')['ttg'], 2)}pp), and on MLAAD-5 it "
                      f"rejects {pct(c(a5, 'mlaad5')['frr'])}%")
    # E8 / E10
    chk("r2.e8_far", f"accept {pct(c(pa, 'itw', 'far5')['far'])}% and {pct(c(xm, 'itw', 'far5')['far'])}% of ITW spoofs, so "
                     f"their BPCER of {pct(c(pa, 'itw', 'far5')['frr'])}% and {pct(c(xm, 'itw', 'far5')['frr'])}%")
    bo = lambda m: e1[f"{m}|itw"]["bpcer_at_apcer"]["0.05"]["target_oracle"]  # noqa: E731
    chk("r2.e8_oracle", f"their BPCER is {pct(bo(pa))}% and {pct(bo(xm))}%")
    wins = [c(m, "itw")["window"] for m in (a5, pa, xm)]
    truth("r2.e10_windows", all(w and w[0] < 1 / 11 < w[1] and not w[0] < 0.95 < w[1] for w in wins))

    # R2-H1
    h1 = RG["R2-H1"]["per_model"]
    truth("r2.h1", RG["R2-H1"]["supported"] and RG["R2-H1"]["n_supporting"] == 2 and h1[pa]["supports"]
          and h1[xm]["supports"] and not h1[a5]["supports"])
    truth("r2.h1_all4", all(h1[m][k]["holm_reject"] for m in (pa, xm) for k in ("frr1", "far5")))
    chk("r2.h1_frr1", f"FRR {h1[pa]['frr1']['frr_pct']:.1f}% and {h1[xm]['frr1']['frr_pct']:.1f}% against FAR "
                      f"{h1[pa]['frr1']['far_pct']:.1f}% and {h1[xm]['frr1']['far_pct']:.1f}%")
    chk("r2.h1_far5", f"FAR {h1[pa]['far5']['far_pct']:.1f}% and {h1[xm]['far5']['far_pct']:.1f}% against FRR "
                      f"{h1[pa]['far5']['frr_pct']:.1f}% and {h1[xm]['far5']['frr_pct']:.1f}%")
    f1 = h1[a5]["frr1"]
    chk("r2.h1_as5", f"({f1['frr_minus_far_pp']:+.1f}pp [{f1['ci95_pp'][0]:+.1f}, {f1['ci95_pp'][1]:+.1f}])")
    chk("r2.h1_as5_far5", f"it rejects {h1[a5]['far5']['frr_pct']:.1f}% of genuine ITW speech and accepts "
                          f"{h1[a5]['far5']['far_pct']:.1f}% of spoofs")
    # R2-H2
    h2 = RG["R2-H2"]["per_run"]
    truth("r2.h2", RG["R2-H2"]["supported"] and RG["R2-H2"]["n_rejected"] == 4 and not h2["1"]["holm_reject"])
    ttg = [h2[s]["ttg_pp"] for s in SEEDS]
    chk("r2.h2_range", f"of {min(ttg):.2f} to {max(ttg):.2f}pp")
    chk("r2.h2_s1", f"({h2['1']['ttg_pp']:.2f}pp [{h2['1']['ci95_pp'][0]:.2f}, {h2['1']['ci95_pp'][1]:.2f}], "
                    f"p={pv(h2['1']['p'])})")
    chk("r2.h2_share", f"Transfer accounts for {span([h2[s]['transfer_share'] for s in SEEDS], 0)}%")
    chk("r2.h2_released", f"LA19-eval gap ({pct(EX['E2_E3']['released_aasist_l']['la19_eval']['ttg'], 2)}pp)")
    truth("r2.h2_threefold", max(ttg) / min(ttg) > 3)
    # R2-H3
    h3 = RG["R2-H3"]["tests"]
    ta, tb, tc = (h3[k] for k in ("a_dev_eer_higher_when_trimmed", "b_itw_frr_lower_when_trimmed",
                                  "c_itw_eer_lower_when_trimmed"))
    truth("r2.h3", RG["R2-H3"]["evaluable"] and tb["holm_reject"] and not tc["holm_reject"]
          and abs(ta["p"] - 1 / 252) < 1e-12 and ta["n_splits"] == 252)
    chk("r2.h3_manip", f"({ta['diff_pp']:+.2f}pp, p={pv(ta['p'])}")
    chk("r2.h3_frr", f"(run means {arm_mean('aasist_l_trim', 'itw|eer', 'frr') * 100:.1f}% against "
                     f"{arm_mean('aasist_l_r2', 'itw|eer', 'frr') * 100:.1f}%, -{tb['diff_pp']:.2f}pp, p={pv(tb['p'])})")
    chk("r2.h3_eer", f"{-tc['diff_pp']:.2f}pp higher, p={pv(tc['p'])}")
    # R2-H4
    h4 = RG["R2-H4"]
    cond = h4["conditions"]
    d4 = lambda cc, k: [cond[cc]["per_model"][mm][k] for mm in R.RANKED]  # noqa: E731
    chk("r2.h4_opus6k", f"Opus at 6kbps raises FRR by {rng(d4('opus6k', 'dfrr_pp'), 0)}pp")
    chk("r2.h4_amr", f"AMR-NB at 4.75kbps by {rng(d4('amr4.75', 'dfrr_pp'), 0)}pp across the four primary models")
    truth("r2.h4_far_falls", all(v < 0 for cc in ("opus6k", "amr4.75") for v in d4(cc, "dfar_pp")))
    truth("r2.h4_4of4", all(cond[cc]["n_rejected"] == 4 for cc in ("opus6k", "amr4.75", "opus24k", "mic_ir")))
    chk("r2.h4_ir", f"raises FRR by {rng(d4('mic_ir', 'dfrr_pp'), 0)}pp (four of four)")
    truth("r2.h4_opus24k_small", max(abs(v) for v in d4("opus24k", "dfrr_pp")) <= 1.0)
    o24 = [-v for v in d4("opus24k", "dfar_pp")]
    chk("r2.h4_opus24k", f"FAR falls by {min(o24):.1f}-{max(o24):.1f}pp")
    truth("r2.h4_mp3", cond["mp3_32"]["n_rejected"] == 1)
    aa = [cond[cc]["per_model"][mm]["dfrr_pp"] for cc in ("snr10", "snr5") for mm in ("aasist", "aasist_l")]
    chk("r2.h4_noise_frr", f"AASIST and AASIST-L reject {rng(aa, 0)}pp more genuine speech at 10 and 5dB SNR")
    rn = [cond[cc]["per_model"]["rawnet2"]["dfar_pp"] for cc in ("snr20", "snr10", "snr5")]
    ts = [cond[cc]["per_model"]["tssdnet"]["dfar_pp"] for cc in ("snr20", "snr10", "snr5")]   # all three noise levels
    chk("r2.h4_noise_far", f"RawNet2 (by {rng(rn, 0)}pp) and of Res-TSSDNet (by {rng(ts, 0)}pp, at all three noise levels)")
    truth("r2.h4_noise_2of4", all(cond[cc]["n_rejected"] == 2 for cc in ("snr20", "snr10", "snr5")))
    truth("r2.h4_total", h4["n_conditions_supported"] == 4)
    # reading rule
    rl = RG["R2-RL"]
    truth("r2.rl", rl[a5]["category"] == "discrimination-limited" and rl[pa]["category"] == "transfer-limited"
          and rl[xm]["category"] == "transfer-limited" and rl[pa]["transfer_share"] < 0.5 <= rl[xm]["transfer_share"])
    chk("r2.rl_as5", f"discrimination-limited on ITW (EER {rl[a5]['itw_eer_pct']:.1f}%)")
    chk("r2.rl_mamba", f"(transfer share {rl[xm]['transfer_share']:.2f}; its dev-equivalent interval spans "
                       f"{rl[xm]['tie_itw_frr_span_pp']:.1f}pp of ITW FRR)")
    chk("r2.rl_pascu", f"({rl[pa]['tie_itw_frr_span_pp']:.1f}pp, share {rl[pa]['transfer_share']:.2f})")

    # E2 / E3 / trimmed arm
    e = EX["E2_E3"]
    rel = e["released_aasist_l"]
    r2v = lambda arm, key, qq: [e[arm][s][key][qq] for s in SEEDS]  # noqa: E731
    chk("r2.e2_la19", f"({span(r2v('aasist_l_r2', 'la19_eval|eer', 'eer'), 2)}% against {pct(rel['la19_eval']['eer'], 2)}%)")
    chk("r2.e2_frr", f"({pct(rel['itw']['frr'])}% against {span(r2v('aasist_l_r2', 'itw|eer', 'frr'))}%)")
    truth("r2.e2_outside", rel["itw"]["frr"] > max(r2v("aasist_l_r2", "itw|eer", "frr"))
          and rel["itw"]["ttg"] < min(r2v("aasist_l_r2", "itw|eer", "ttg")))
    chk("r2.e2_ttg", f"({pct(rel['itw']['ttg'], 2)} against {span(r2v('aasist_l_r2', 'itw|eer', 'ttg'), 2)}pp)")
    chk("r2.e2_eer", f"({span(r2v('aasist_l_r2', 'itw|eer', 'eer'))}% against {pct(rel['itw']['eer'])}%)")
    p1234 = e["aasist_l_r2"]["1234"]
    chk("r2.e2_1234", f"(LA19-eval TTG {pct(p1234['la19_eval|eer']['ttg'], 2)}pp, ITW FRR {pct(p1234['itw|eer']['frr'])}%)")
    truth("r2.e2_1234_outside", p1234["itw|eer"]["frr"] < min(r2v("aasist_l_r2", "itw|eer", "frr"))
          and p1234["la19_eval|eer"]["ttg"] > max(r2v("aasist_l_r2", "la19_eval|eer", "ttg")))
    runs = GT["runs"]
    nswa = [runs[f"aasist_l_r2_s{s}"]["n_swa_update"] for s in SEEDS]
    chk("r2.e3_swa", f"absorbed {min(nswa)} to {max(nswa)} development-selected snapshots")
    truth("r2.trim_no_swa", all(runs[f"aasist_l_trim_s{s}"]["n_swa_update"] == 0 for s in SEEDS))
    chk("r2.e3_la19eer", f"similar LA19-eval EER ({span(r2v('aasist_l_r2_last', 'la19_eval|eer', 'eer'), 2)}%)")
    chk("r2.e3_ttg", f"on LA19 eval {span(r2v('aasist_l_r2_last', 'la19_eval|eer', 'ttg'), 2)}pp (mean "
                     f"{100 * arm_mean('aasist_l_r2_last', 'la19_eval|eer', 'ttg'):.2f}, against "
                     f"{100 * arm_mean('aasist_l_r2', 'la19_eval|eer', 'ttg'):.2f} for the averages), and on ITW a mean of "
                     f"{100 * arm_mean('aasist_l_r2_last', 'itw|eer', 'ttg'):.2f}pp against "
                     f"{100 * arm_mean('aasist_l_r2', 'itw|eer', 'ttg'):.2f}")
    best = []
    for s in SEEDS:
        lines = (R2 / "runs" / f"aasist_l_trim_s{s}" / "epochs.jsonl").read_text(encoding="utf-8").splitlines()
        best.append(min(json.loads(x)["dev_eer"] for x in lines if x.strip()))
    truth("r2.trim_never_1pct", min(best) > 1.0)
    chk("r2.trim_dev", f"(best epoch {min(best):.2f}-{max(best):.2f}%, scored model "
                       f"{span([e['aasist_l_trim'][s]['dev_eer'] for s in SEEDS], 2)}%)")
    chk("r2.trim_la19", f"raises LA19-eval EER from {span(r2v('aasist_l_r2', 'la19_eval|eer', 'eer'), 2)}% to "
                        f"{span(r2v('aasist_l_trim', 'la19_eval|eer', 'eer'))}%")
    m_t, m_l, m_s = (100 * arm_mean(arm, "itw|eer", "frr") for arm in ("aasist_l_trim", "aasist_l_r2_last", "aasist_l_r2"))
    chk("r2.e3_frr", f"ITW FRR at is {m_t:.1f}% for the trimmed and {m_l:.1f}% for the untrimmed runs (-{m_l - m_t:.1f}pp, "
                     f"against -{m_s - m_t:.1f}pp as registered)")
    per = [100 * (e["aasist_l_r2_last"][s]["itw|eer"]["frr"] - e["aasist_l_trim"][s]["itw|eer"]["frr"]) for s in SEEDS]
    truth("r2.e3_each_seed", min(per) > 0)
    chk("r2.e3_seeds", f"lower for each of the five seeds by {min(per):.1f} to {max(per):.1f}pp")
    share = (m_l - m_t) / (m_s - m_t)
    truth("r2.two_thirds", 0.6 < 1 - share < 0.7, f"selection share {1 - share:.3f}")
    fa = [100 * arm_mean(arm, "itw|eer", "far") for arm in ("aasist_l_trim", "aasist_l_r2", "aasist_l_r2_last")]
    chk("r2.e3_far", f"(ITW FAR {fa[0]:.1f}% when trimmed, against {fa[1]:.1f}% for the averaged and {fa[2]:.1f}% for the "
                     f"last-epoch untrimmed models)")
    de = [100 * (arm_mean("aasist_l_trim", "itw|eer", "eer") - arm_mean(arm, "itw|eer", "eer"))
          for arm in ("aasist_l_r2", "aasist_l_r2_last")]
    chk("r2.e3_eer", f"(+{de[0]:.1f} and +{de[1]:.1f}pp)")

    # E4 channels
    e4 = EX["E4"]
    w = lambda key, cc, k="dfrr": round(100 * e4[key][cc][k])  # noqa: E731
    am = lambda arm, cc, k="dfrr": round(100 * sum(e4[f"{arm}|s{s}"][cc][k] for s in SEEDS) / 5)  # noqa: E731
    ssl = [w(f"{mm}|s0", cc) for mm in ("pascu_lr", "xlsr_mamba", "xlsr_aasist") for cc in ("opus6k", "amr4.75")]
    chk("r2.e4_ssl", f"raise FRR by {min(ssl)}-{max(ssl)}pp for Pascu's detector, XLSR-Mamba and the round-1 reference")
    chk("r2.e4_encodec", f"raises it by {w('pascu_lr|s0', 'encodec6k')} and {w('xlsr_mamba|s0', 'encodec6k')}pp")
    n5 = [w(f"{mm}|s0", "snr5") for mm in ("pascu_lr", "xlsr_mamba", "xlsr_aasist")]
    chk("r2.e4_noise", f"5-dB noise raises it by {min(n5)}-{max(n5)}pp")
    a5c = sorted([w("as5_aasist|s0", "opus6k"), w("as5_aasist|s0", "amr4.75")])
    chk("r2.e4_as5_codec", f"(+{a5c[0]} and +{a5c[1]}pp)")
    a5n = [w("as5_aasist|s0", cc) for cc in ("snr20", "snr10", "snr5")]
    chk("r2.e4_as5_noise", f"(+{min(a5n)} to +{max(a5n)}pp)")
    truth("r2.e4_as5_ir", w("as5_aasist|s0", "mic_ir") < 0 < w("as5_aasist|s0", "mic_ir", "dfar"))
    chk("r2.e4_as5_ir_far", f"raises its FAR (+{w('as5_aasist|s0', 'mic_ir', 'dfar')}pp)")
    chk("r2.e4_runs", f"(Opus 6kbps +{am('aasist_l_r2', 'opus6k')} against +{w('aasist_l|s0', 'opus6k')}pp, AMR-NB "
                      f"+{am('aasist_l_r2', 'amr4.75')} against +{w('aasist_l|s0', 'amr4.75')}pp)")
    chk("r2.e4_trim_noise", f"(+{am('aasist_l_trim', 'snr10')} and +{am('aasist_l_trim', 'snr5')}pp at 10 and 5dB, against "
                            f"+{am('aasist_l_r2', 'snr10')} and +{am('aasist_l_r2', 'snr5')}pp)")
    chk("r2.e4_trim_encodec", f"(+{am('aasist_l_trim', 'encodec6k')} against +{am('aasist_l_r2', 'encodec6k')}pp)")
    chk("r2.e4_trim_far", f"lower their FAR by {-am('aasist_l_trim', 'opus6k', 'dfar')}, "
                          f"{-am('aasist_l_trim', 'amr4.75', 'dfar')} and {-am('aasist_l_trim', 'encodec6k', 'dfar')}pp")

    # E5 MLAAD-5
    e5 = EX["E5"]
    chk("r2.e5_eer", f"pooled EERs of {span([e1[f'{mm}|mlaad5']['eer'] for mm in (a5, pa, xm)], 0)}%")
    fr = lambda mm, k: [round(100 * v[k]) for v in e5[mm].values()]  # noqa: E731
    chk("r2.e5_frr", f"(FRR {min(fr(a5, 'frr'))}-{max(fr(a5, 'frr'))}% and {min(fr(pa, 'frr'))}-{max(fr(pa, 'frr'))}%)")
    chk("r2.e5_mamba", f"accepts {min(fr(xm, 'far'))}-{max(fr(xm, 'far'))}% of spoofs at FRR "
                       f"{min(fr(xm, 'frr'))}-{max(fr(xm, 'frr'))}%")
    chk("r2.e5_rerun", f"repeated two 186-sample spoofs to 400 samples and scored all {e1[pa + '|mlaad5']['n']:,} trials")

    # E6 ASVspoof 5
    e6 = EX["E6"]
    chk("r2.e6_c00", f"(EER {pct(e6['C00']['eer'])}%, TTG {pct(e6['C00']['ttg'], 2)}pp)")
    wide = [e6[f"C0{i}"]["ttg"] for i in range(1, 8)]
    chk("r2.e6_wide", f"(TTG at most {100 * max(wide):.2f}pp)")
    nb = [e6[k] for k in ("C08", "C09", "C10", "C11")]
    chk("r2.e6_nb", f"raise FRR to {span([v['frr'] for v in nb])}% and the transfer gap to {span([v['ttg'] for v in nb], 2)}pp")
    truth("r2.e6_nb_worst", min(v["frr"] for v in nb) > max(e6[f"C0{i}"]["frr"] for i in range(0, 8)))
    sil = e6["silence"]
    bona = {sil[f"asvspoof5_{k}|bona"]["lead_median_s"] for k in ("dev", "eval")}
    spoof = [sil[f"asvspoof5_{k}|spoof"]["lead_median_s"] for k in ("dev", "eval")]
    truth("r2.e6_sil_bona", len(bona) == 1)
    chk("r2.e6_sil", f"the median is {bona.pop():.2f}s for bona fide clips and {min(spoof):.2f}-{max(spoof):.2f}s for spoofs")

    # E7 three-zone
    e7 = EX["E7"]
    la19_trained = ("pascu_lr", "xlsr_mamba", "aasist", "aasist_l", "rawnet2", "tssdnet", "xlsr_aasist")
    dv = lambda mm, k, t="itw", b="0.1", kind="dev": e7[f"{mm}|{t}"][b][kind][k]  # noqa: E731
    chk("r2.e7_review", f"it sends {span([dv(mm, 'review_pooled') for mm in la19_trained])}% of ITW clips, and "
                        f"{span([dv(mm, 'review_bona') for mm in la19_trained])}% of genuine ones, to review")
    chk("r2.e7_autorej", f"reject {span([dv(mm, 'bona_auto_rejected') for mm in la19_trained])}% of genuine clips")
    truth("r2.e7_noacc", max(dv(mm, "spoof_auto_accepted") for mm in la19_trained) < 0.001)
    chk("r2.e7_as5", f"({pct(dv(a5, 'review_pooled'))}%) only because most genuine ITW scores fall below the band, so it "
                     f"auto-rejects {pct(dv(a5, 'bona_auto_rejected'))}%")
    lf = [dv(mm, "bona_auto_rejected", kind="label_free") for mm in (a5, *la19_trained)]
    chk("r2.e7_labelfree", f"{span(lf)}% of genuine ITW clips are then rejected without review")
    chk("r2.e7_oracle", f"auto-reject {pct(dv(xm, 'bona_auto_rejected', kind='oracle'))}% and "
                        f"{pct(dv(pa, 'bona_auto_rejected', kind='oracle'))}%")
    truth("r2.e7_collapse", all(v["error_budget"]["collapse"] == (not k.startswith(a5)) for k, v in e7.items()))
    chk("r2.e7_eb", f"which then holds {pct(e7[a5 + '|itw']['error_budget']['review_pooled'])}% of ITW")

    # E9 calibration
    e9 = EX["E9"]
    at = lambda mm, t, lg=0.0: next(r for r in e9[f"{mm}|{t}"]["rows"] if abs(r[0] - lg) < 1e-9)  # noqa: E731
    truth("r2.e9_act_ge_min", all(r[1] >= r[2] - 1e-12 for v in e9.values() for r in v["rows"] if r[1] is not None))
    chk("r2.e9_ssl", f"({at(xm, 'itw')[1]:.2f} against {at(xm, 'itw')[2]:.2f} for XLSR-Mamba, {at('xlsr_aasist', 'itw')[1]:.2f} "
                     f"against {at('xlsr_aasist', 'itw')[2]:.2f} for the round-1 reference, {at(pa, 'itw')[1]:.2f} against "
                     f"{at(pa, 'itw')[2]:.2f} for Pascu's detector)")
    chk("r2.e9_primary_min", f"near 1 ({span([at(mm, 'itw')[2] for mm in R.RANKED], 2, 1)})")
    chk("r2.e9_trivial", f"RawNet2 on ITW ({at('rawnet2', 'itw')[1]:.2f}) and Pascu's detector on MLAAD-5 "
                         f"({at(pa, 'mlaad5')[1]:.2f})")
    truth("r2.e9_trivial_gt1", at("rawnet2", "itw")[1] > 1 and at(pa, "mlaad5")[1] > 1)
    p2 = [at(mm, "itw", 2.0)[1] for mm in (a5, *la19_trained)]
    truth("r2.e9_logit2_all", min(p2) > 1)
    chk("r2.e9_logit2", f"({min(p2):.2f}-{max(p2):.2f})")
    chk("r2.e9_logit2_min", f"({span([at(mm, 'itw', 2.0)[2] for mm in (pa, xm, 'xlsr_aasist')], 2, 1)})")

    # discussion and abstract
    words = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight"]
    chk("disc.mamba_span", f"XLSR-Mamba's ITW FRR spans {rl[xm]['tie_itw_frr_span_pp']:.1f}pp")
    chk("disc.two_thirds", "about two thirds of that reduction comes from checkpoint selection")
    chk("abs.h1", f"sets the dominant error for {words[RG['R2-H1']['n_supporting']]} of three current detectors")
    chk("abs.h2", f"recurs across retraining ({words[RG['R2-H2']['n_rejected']]} of five seeds)")
    chk("abs.h4", f"({words[h4['n_conditions_supported']]} of eight channel conditions)")
    chk("abs.h2_r1", f"for three of four primary models when clips are resampled, as registered, but for "
                     f"{words[sum(1 for mm in RK if A['a1_h2']['per_model'][mm]['holm_reject_h0'])]} of four when speakers "
                     f"are resampled")
    share1 = sorted(100 * pol(mm, "19la_eval")["transfer_share"] for mm in ("aasist", "aasist_l"))
    chk("abs.r1_share", f"transfer causes {share1[0]:.0f}-{share1[1]:.0f}% of the deployed error of two detectors whose "
                        f"equal error rates are below 1%")
    truth("abs.r1_eer_below_1", all(tgt(mm, "19la_eval")["eer_pct"] < 1 for mm in ("aasist", "aasist_l")))

    # supplement (round-2 section)
    schk("s.r2_far1", f"gives an APCER of {pct(c(pa, 'itw', 'far1')['far'])}% for Pascu's detector and "
                      f"{pct(c(xm, 'itw', 'far1')['far'])}% for XLSR-Mamba")
    truth("s.r2_nowindow", not c(pa, "mlaad5").get("window"))
    schk("s.r2_last", f"(LA19 eval mean {100 * arm_mean('aasist_l_r2', 'la19_eval|eer', 'ttg'):.2f} against "
                      f"{100 * arm_mean('aasist_l_r2_last', 'la19_eval|eer', 'ttg'):.2f}pp, ITW "
                      f"{100 * arm_mean('aasist_l_r2', 'itw|eer', 'ttg'):.2f} against "
                      f"{100 * arm_mean('aasist_l_r2_last', 'itw|eer', 'ttg'):.2f}pp)")
    schk("s.r2_last_frr", f"from {m_l:.1f}% to {m_t:.1f}%")
    schk("s.r2_last_far", f"from {fa[2]:.1f}% to {fa[0]:.1f}%")
    schk("s.r2_last_eer", f"from {100 * arm_mean('aasist_l_r2_last', 'itw|eer', 'eer'):.1f}% to "
                          f"{100 * arm_mean('aasist_l_trim', 'itw|eer', 'eer'):.1f}%")
    schk("s.r2_la19_review", f"({span([dv(mm, 'review_pooled', 'la19_eval') for mm in la19_trained])}%)")
    schk("s.r2_mlaad_review", f"it reaches {span([dv(mm, 'review_pooled', 'mlaad5') for mm in la19_trained])}%")
    schk("s.r2_budget20", f"still sends {pct(dv(xm, 'review_pooled', b='0.2'))}% of ITW clips to review")
    schk("s.r2_oracle_primary", f"leaves {span([dv(mm, 'bona_auto_rejected', kind='oracle') for mm in ('aasist', 'aasist_l', 'rawnet2')])}% "
                                f"of genuine clips auto-rejected")
    truth("s.r2_noise_each_seed", all(
        max(e4[f"aasist_l_trim|s{s}"][cc]["dfrr"] for s in SEEDS) < min(e4[f"aasist_l_r2|s{s}"][cc]["dfrr"] for s in SEEDS)
        for cc in ("snr10", "snr5")) and max(e4[f"aasist_l_r2|s{s}"]["encodec6k"]["dfrr"] for s in SEEDS)
        < min(e4[f"aasist_l_trim|s{s}"]["encodec6k"]["dfrr"] for s in SEEDS))
    mla = [at(mm, "mlaad5", -2.0)[2] for mm in (a5, *la19_trained)]
    truth("s.r2_mlaad_min1", all(abs(v - 1) < 0.005 for v in mla))
    schk("s.r2_pascu_fit", f"fitted on the {PH['n']:,} LA19 training clips (L-BFGS, {PH['n_iter'][0]} iterations, no "
                           f"convergence warning, training accuracy {PH['train_accuracy']:.1f})")
    truth("s.r2_pascu_fit_clean", PH["convergence_warnings"] == [] and PH["splits"] == ["train"])


def main(final: bool):
    tex = (HERE / "main.tex").read_text(encoding="utf-8")
    fails, pending, n = [], [], 0
    docs = collect_documents(pending, fails)
    body = norm_tex("\n".join(t for p, t in docs.items() if p.parent.name != "gen" and p.name != "supplement.tex"))
    sbody = norm_tex(docs.get(HERE / "supplement.tex", ""))
    n += audit_provenance(docs, fails)

    def chk(name, text):
        nonlocal n
        n += 1
        if norm_claim(text) not in body:
            fails.append((name, text))

    def truth(name, ok, detail=""):
        nonlocal n
        n += 1
        if not ok:
            fails.append((name, detail or "false"))

    def schk(name, text):
        nonlocal n
        n += 1
        if norm_claim(text) not in sbody:
            fails.append((name, text))

    # ---- generated round-2 tables: byte-identical to a fresh run of the generator on the committed results
    fresh = regenerated_r2_tables()
    present = {g.name for g in (HERE / "gen").glob("r2*.tex")}
    truth("gen.r2_complete", present == set(fresh), f"gen/ has {sorted(present ^ set(fresh))} beyond or missing from "
                                                    f"the generator's output")
    for name in sorted(fresh):
        g = HERE / "gen" / name
        truth(f"gen.{g.stem}", g.is_file() and g.read_bytes().replace(b"\r\n", b"\n") == fresh[name].replace(b"\r\n", b"\n"),
              f"{name} differs from analysis/round2/r2_paper_tables.py's output")
    for name in sorted(fresh):
        if (HERE / "gen" / name) not in docs:
            fails.append(("gen.r2_unused", f"{name} is generated but neither the paper nor the supplement includes it"))

    # ---- generated round-1 tables: byte-identical to the audited revision's
    for g in sorted((HERE / "gen").glob("*.tex")):
        if g.name.startswith(("r2_", "r2s_")) or g.name == "r1_mlaad.tex":
            continue
        src = R.REV / "paper" / "gen" / g.name
        truth(f"gen.{g.stem}", src.exists() and src.read_bytes().replace(b"\r\n", b"\n") ==
              g.read_bytes().replace(b"\r\n", b"\n"), f"{g.name} differs from {src} (ignoring checkout line endings only)")

    # ---- study design
    dv = D["aasist"]["dev"]
    chk("design.dev_n", f"({dv['n_bona'] + dv['n_spoof']:,} trials, {dv['n_bona']:,} bona fide)")
    _, y, _ = R.load("aasist", "19la_eval")
    chk("design.la19", f"({len(y):,} trials; 11 of 13 attacks unseen)")
    _, y, _ = R.load("aasist", "itw")
    chk("design.itw", f"({len(y):,} clips of public figures with {A['a1_h2']['clusters']} distinct speaker labels")
    gt = G["total"]
    n_eval, n_a, n_b, n_ver = gt["eval"][0], gt["A"][0], gt["B"][0], gt["verified"][0]
    chk("design.df21", f"({n_ver:,} trials; libFLAC 1.4 rejects {100 * (n_a + n_b) / n_eval:.0f}% of the public evaluation "
                       f"files and for {n_b:,} of them")
    hm = G["h5_manifest"]
    chk("design.h5", f"1,000 LA19-eval clips (500 per class, all {hm['speakers']} speakers)")

    # ---- round 1: unseen attacks
    la, it, df = "19la_eval", "itw", "df21_verified"
    ttg = lambda m, t, p="eer": pol(m, t, p)["ttg_pp"]  # noqa: E731
    cl = lambda m, t, p="eer": A["a3_e2"]["ttg"][t]["cells"][f"{m}|{p}"]["cluster_ci95_pp"]  # noqa: E731
    chk("r1.la_far_frr", f"accept {rng([pol(m, la)['far_pct'] for m in RK], 1)}% of spoofs and reject "
                         f"{rng([pol(m, la)['frr_pct'] for m in RK], 1)}% of genuine clips")
    chk("r1.la_ttg", f"lose {f(ttg('aasist', la), 2)} and {f(ttg('aasist_l', la), 2)} HTER points to transfer (speaker-level "
                     f"95% CIs [{f(cl('aasist', la)[0], 2)}, {f(cl('aasist', la)[1], 2)}] and [{f(cl('aasist_l', la)[0], 2)}, "
                     f"{f(cl('aasist_l', la)[1], 2)}]), {100 * pol('aasist', la)['transfer_share']:.0f}% and "
                     f"{100 * pol('aasist_l', la)['transfer_share']:.0f}% of their deployed HTER")
    chk("r1.la_eer_hter", f"the EER hides ({f(tgt('aasist', la)['eer_pct'], 2)}% against "
                          f"{f(pol('aasist', la)['hter_pct'], 2)}% for AASIST)")
    chk("r1.la_rank", f"Res-TSSDNet ({f(pol('tssdnet', la)['hter_pct'], 2)}%) overtakes both")
    truth("r1.la_rank_order", all(pol("tssdnet", la)["hter_pct"] < pol(m, la)["hter_pct"] and
                                  tgt("tssdnet", la)["eer_pct"] > tgt(m, la)["eer_pct"] for m in ("aasist", "aasist_l")))

    # ---- round 1: shifted targets
    for t in (it, df):
        exc = {m for m in M5 if pol(m, t)["frr_pct"] <= pol(m, t)["far_pct"]}
        truth(f"r1.four_of_five.{t}", len(exc) == 1, f"FRR <= FAR at tau_EER for {exc}")
    chk("r1.itw_eer", f"(EER {rng([tgt(m, it)['eer_pct'] for m in RK], 0)}%; transfer only "
                      f"{rng([pol(m, it)['transfer_share'] for m in RK], 0, 100)}% of the deployed HTER)")
    x = "xlsr_aasist"
    chk("r1.xlsr_share", f"({100 * pol(x, it)['transfer_share']:.0f}% and {100 * pol(x, df)['transfer_share']:.0f}% of its "
                         f"HTER on ITW and DF21)")
    truth("r1.xlsr_largest_gap", all(ttg(x, t) > ttg(m, t) for t in (it, df) for m in RK),
          "the reference's TTG is not the largest on ITW and DF21")
    chk("r1.fewest", f"(HTER {f(pol(x, it)['hter_pct'])}% on ITW against {rng([pol(m, it)['hter_pct'] for m in RK], 1)}%)")
    truth("r1.fewest_order", pol(x, it)["hter_pct"] < min(pol(m, it)["hter_pct"] for m in RK))

    # ---- round 1: registered tests
    h2 = A["a1_h2"]["per_model"]
    truth("r1.h2_clip", H2["n_models_frr_gt_far"] == 3 and H2["h2_supported"], f"clip-level H2: {H2['n_models_frr_gt_far']}")
    chk("r1.h2_clusters", f"resampling ITW's {A['a1_h2']['clusters']} speaker labels")
    chk("r1.h2_res", f"only Res-TSSDNet's FRR - FAR excludes zero ({sgn(h2['tssdnet']['delta_pp'])}pp "
                     f"[{sgn(h2['tssdnet']['ci95_pp'][0])}, {sgn(h2['tssdnet']['ci95_pp'][1])}])")
    truth("r1.h2_count", sum(1 for m in RK if h2[m]["holm_reject_h0"]) == 1, "cluster H2 is not 1 of 4")
    chk("r1.h2_loo", f"moves AASIST's difference between {sgn(h2['aasist']['loo_min_pp'])} and "
                     f"{sgn(h2['aasist']['loo_max_pp'])}pp")
    h5 = A["a2_h5"]["per_model"]
    truth("r1.h5_dirs", all(h5[m]["dfrr_pp"] > 0 and h5[m]["dfar_pp"] < 0 for m in RK), "EnCodec FRR up / FAR down")
    truth("r1.h5_holm", all(h5[m]["holm_dfrr"] and h5[m]["holm_dfar"] and h5[m]["holm_stat"] for m in RK))
    chk("r1.h5_68", f"Res-TSSDNet rejects {100 * H5['per_model']['tssdnet']['frr_encodec']:.1f}% of resynthesised bona fide")

    # ---- round 1: policy, determinacy
    chk("r1.frr1", f"rejects {rng([pol(m, it, 'frr1')['frr_pct'] for m in M5], 1)}% of genuine ITW speech")
    chk("r1.far5", f"a dev FAR of 5% becomes {rng([pol(m, it, 'far5')['far_pct'] for m in M5], 1)}% FAR with FAR above FRR "
                   f"for all five models")
    truth("r1.far5_dir", all(pol(m, it, "far5")["far_pct"] > pol(m, it, "far5")["frr_pct"] for m in M5))
    ivx, iv = D[x]["eer_equivalent_interval"], D[x]["targets"]["itw"]
    chk("r1.interval", f"every threshold in [{sgn(ivx['tau_lo'], 2)}, {sgn(ivx['tau_hi'], 2)}] ties")
    chk("r1.frr_range", f"its ITW FRR runs from {f(iv['tau_lo']['frr_pct'])}% to {f(iv['tau_hi']['frr_pct'])}% over that "
                        f"interval")
    ends = [(iv[k]["frr_pct"] + iv[k]["far_pct"]) / 2 - iv["oracle_hter_pct"] for k in ("tau_lo", "tau_hi")]
    chk("r1.interval_ttg", f"its ITW TTG could be anywhere from {min(ends):.1f} to {max(ends):.1f}pp")
    chk("r1.dev_attacks", "whose spoofs come from the six training attacks (A01-A06)")
    fsd = FS["distributions"]
    chk("r1.lead_eval", f"(evaluation: {f(fsd['asvspoof19_eval|bona']['lead_median_s'], 2)}s against "
                        f"{f(fsd['asvspoof19_eval|spoof']['lead_median_s'], 2)}s)")

    # ---- round 1: silence
    dist = FS["distributions"]
    leads = {k: (dist[f"asvspoof19_{k}|bona"]["lead_median_s"], dist[f"asvspoof19_{k}|spoof"]["lead_median_s"])
             for k in ("dev", "eval")}
    hits = [k for k, (b, s) in leads.items()
            if norm_claim(f"begin with a median {f(b, 2)}s of silence against {f(s, 2)}s for spoofs") in body]
    truth("r1.lead", bool(hits), f"LA19 lead medians dev {leads['dev']} / eval {leads['eval']} not in the text")
    chk("r1.itw_lead", f"ITW genuine clips begin with {f(dist['itw|bona']['lead_median_s'], 2)}s")
    fold = [FT[m]["19la_eval"]["eer_trim"] / FT[m]["19la_eval"]["eer_untrim"] for m in M5]
    chk("r1.trim_fold", f"of all five models {min(fold):.0f}-{max(fold):.0f} fold")
    dfrr = [100 * (FT[m]["itw"]["reg_untrim"]["frr_fixed"] - FT[m]["itw"]["tautrim_trim"]["frr_fixed"]) for m in M5]
    dfar = [100 * (FT[m]["itw"]["tautrim_trim"]["far_fixed"] - FT[m]["itw"]["reg_untrim"]["far_fixed"]) for m in M5]
    deer = [FT[m]["itw"]["eer_trim"] - FT[m]["itw"]["eer_untrim"] for m in M5]
    chk("r1.trim", f"lowers ITW FRR ({min(dfrr):.0f}-{max(dfrr):.0f}pp) but raises FAR ({min(dfar):.0f}-{max(dfar):.0f}pp)")
    chk("r1.trim_eer", f"moves ITW EER by only {sgn(min(deer))} to {sgn(max(deer))}pp")

    # ---- round 1: checks for deployers
    res = C5["results"]
    fp = [res[m][k]["held_far_new"][0] for m in RK for k in ("50", "100", "500")]
    chk("r1.reset", f"accepts {min(fp):.0f}-{max(fp):.1f}% of held-out spoofs for the primary models")
    w = E["windows"]
    truth("r1.windows", not w["rawnet2|itw"]["eer"] and all(w[f"{m}|itw"]["eer"] for m in M5 if m != "rawnet2"),
          "break-even windows: RawNet2 must have none, the others one")

    # ---- existing fp32 MLAAD surface (descriptive; source files are round 1)
    mlaad_dir = WORKTREE / "results" / "phase_2_conversion_app" / "mlaad5"
    mlaad = json.loads((mlaad_dir / "mlaad_report.json").read_text(encoding="utf-8"))["models"]
    census = json.loads((mlaad_dir / "sample_rate_census.json").read_text(encoding="utf-8"))["by_lang_class_sr_ch"]
    rows = {m: mlaad[m]["fp32"] for m in M5}
    chk("r1.mlaad_eer", f"EER spans {rng([rows[m]['eer'] for m in M5], 1, 100)}%")
    truth("r1.mlaad_direction", sum(rows[m]["at_eer"]["far"] > rows[m]["at_eer"]["frr"] for m in M5) == 4)
    chk("r1.mlaad_exception", f"Res-TSSDNet is the exception (FRR {100 * rows['tssdnet']['at_eer']['frr']:.2f}%, FAR "
        f"{100 * rows['tssdnet']['at_eer']['far']:.2f}%)")
    bona = [r for r in census if r["class"] == "bonafide"]
    spoof = [r for r in census if r["class"] == "spoof"]
    native16 = 100 * sum(r["n"] for r in bona if r["sr"] == 16000) / sum(r["n"] for r in bona)
    truth("r1.mlaad_census_counts", sum(r["n"] for r in bona) == rows["aasist"]["n_bona"] and
          sum(r["n"] for r in spoof) + sum(r["n"] for r in bona) == rows["aasist"]["n"])
    truth("r1.mlaad_spoof_rate", {r["sr"] for r in spoof} == {22050})
    chk("r1.mlaad_confound", f"Every spoof file is native 22.05kHz, whereas {native16:.1f}% of bona fide files are native 16kHz")
    mlaad_table = HERE / "gen" / "r1_mlaad.tex"
    truth("gen.r1_mlaad_exists", mlaad_table in docs, "the fp32 MLAAD table is not included")
    if mlaad_table in docs:
        table = docs[mlaad_table]
        names = ("AASIST", "AASIST-L", "RawNet2", "Res-TSSDNet", "XLS-R+AASIST (ref.)")
        wanted = []
        for m, name in zip(M5, names):
            row = rows[m]
            point = row["at_eer"]
            vals = [row["eer"], point["frr"], point["far"], point["hter"], row["ttg_eer"]]
            wanted.append(name + " & " + " & ".join(f"{100 * v:.2f}" for v in vals))
        observed = [line.split(r"\\")[0].strip() for line in table.splitlines()
                    if re.match(r"(?:AASIST|RawNet2|Res-TSSDNet|XLS-R)", line)]
        truth("gen.r1_mlaad_rows", observed == wanted, "MLAAD table does not equal the existing fp32 source values")
        truth("gen.r1_mlaad_provenance", "% src: results/phase_2_conversion_app/mlaad5/mlaad_report.json" in table)

    # ---- registered round-2 methods (never reads the loop session's results)
    design_path = HERE / "round2_design.tex"
    truth("r2.design_input", design_path in docs, "round2_design.tex is not an input of main.tex")
    design = norm_tex(docs.get(design_path, ""))
    registration = (WORKTREE / "PREREGISTRATION_ROUND2.md").read_text(encoding="utf-8")
    deviations = (WORKTREE / "DEVIATIONS.md").read_text(encoding="utf-8")

    def method(name, source_pattern, manuscript_phrase):
        truth("r2.source." + name, re.search(source_pattern, registration, re.S) is not None,
              "the registered source differs; inspect the registration before changing this audit")
        truth("r2.method." + name, norm_claim(manuscript_phrase) in design, manuscript_phrase)

    method("bootstrap", r"B = 10,000, seed 0.*?alpha = 0\.05", "B=10,000, seed 0")
    method("alpha", r"Holm at alpha = 0\.05", "=0.05")
    method("seeds", r"Seeds 1, 2, 3, 4, 5 for each arm; seed 1234", "seeds 1-5 in each of two arms")
    method("pipeline_seed", r"seed 1234.*?pipeline check", "default seed, 1234")
    method("epochs", r"100 epochs, batch 24", "100 epochs, batches of 24")
    method("input_aasist", r"64,600, pad", "use 64,600 samples")
    method("input_mamba", r"66,800 everywhere", "uses 66,800 samples on every surface")
    method("vad_frames", r"25 ms frames, 10 ms hop", "25-ms frames and a 10-ms hop")
    method("vad_threshold", r"within 35 dB", "35 dB of the clip's loudest frame")
    method("vad_minimum", r"shorter than 0\.25 s", "shorter than 0.25 s")
    method("h1_family", r"Holm over the six tests", "Holm correction covers all six tests")
    method("h1_support", r"at least 2 of 3 detectors", "at least two of the three detectors")
    method("h2_margin", r"eval > 1 pp", "exceeds 1 pp")
    method("h2_family", r"Holm over five; supported if at least 4 of 5", "covers the five runs, and support requires at least four rejections")
    method("h3_permutations", r"over the 252 relabellings", "over the 252 reassignments")
    method("h3_minimum", r"fewer than three finished runs in either arm", "fewer than three completed runs in either arm")
    method("h4_family", r"Holm over the four models within each\s+condition", "covers the four models within each condition")
    method("h4_support", r"at least 3 of 4 reject", "at least three models reject")
    method("reading_discrimination", r"ITW EER >= 20%", "EER of at least 20%")
    method("reading_transfer", r"TTG / HTER >= 0\.5.*?spans >= 10 pp", "TTG/HTER is at least 0.5")
    method("reading_span", r"spans >= 10 pp", "spans at least 10 pp")
    method("reading_well", r"share is < 0\.2 and the span < 5 pp", "share is below 0.2 and the span below 5 pp")
    method("e1_interval", r"speaker-cluster 95% intervals \(B = 1,000\)", "95% intervals with B=1,000")
    method("review_primary", r"review budget of 10%", "primary review budget is 10%")
    method("review_secondary", r"budgets of 5% and 20%", "Secondary budgets are 5% and 20%")
    method("prior_grid", r"\[-4, 4\] in steps of 0\.25", "from -4 to 4 in steps of 0.25")
    method("mlaad_confound", r"every spoof 22\.05 kHz,.*?98\.1% of bona fide 16 kHz", "sample-rate confound with every numerical result")
    truth("r2.amendment2_source", "exact unpenalised logistic fit" in deviations)
    truth("r2.amendment2_method", "unpenalisedlogisticcalibrationfittedonsourcedevalone" in design)
    truth("r2.amendments", all(f"DEV-9 amendment {i}" in docs.get(design_path, "") for i in range(1, 6)),
          "all five DEV-9 amendments must be disclosed")
    truth("r2.amendment5_source", "### DEV-9 amendment 5 " in deviations and "repeated to exactly\n400 samples" in deviations)
    truth("r2.amendment5_method", norm_claim("repeats a clip shorter than 400 samples to exactly 400 samples") in design)

    # ---- round-2 results (registered tests run once; exploratory analyses), rebuilt from the committed files
    audit_round2(chk, truth, schk)

    # ---- Speech Communication's guide for authors: abstract <= 250 words; 3-5 highlights of <= 85 characters
    abstract = tex.split("\\begin{abstract}", 1)[1].split("\\end{abstract}", 1)[0]
    words = re.sub(r"(?m)^\s*%.*$", "", abstract).replace("~", " ").split()
    truth("venue.abstract_words", len(words) <= 250, f"{len(words)} words")
    hl = [x for x in (HERE / "highlights.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
    truth("venue.highlights", 3 <= len(hl) <= 5 and all(len(x) <= 85 for x in hl),
          f"{len(hl)} highlights, longest {max(map(len, hl))} characters")

    if final:
        truth("final.todo", not any(r"\todo{" in t for t in docs.values()), "unresolved manuscript placeholders")
        for stem in ("main", "supplement"):
            log, pdf = HERE / f"{stem}.log", HERE / f"{stem}.pdf"
            content = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
            truth(f"final.{stem}_compiled", pdf.is_file() and "Output written on" in content,
                  "a successful local PDF build is required")
            truth(f"final.{stem}_clean_log", not re.search(r"(?m)^!|Undefined control sequence|Fatal error|undefined references|Citation .* undefined", content),
                  "LaTeX errors or unresolved references/citations")
            inputs = [*docs, HERE / "refs.bib"]
            truth(f"final.{stem}_current_pdf", pdf.is_file() and all(pdf.stat().st_mtime >= p.stat().st_mtime for p in inputs),
                  "PDF predates an audited source")
        for item in pending:
            fails.append(("final.pending", item))
    all_body = norm_tex("\n".join(t for p, t in docs.items() if p.parent.name != "gen"))
    quoted = sum(norm_claim(value) in all_body for value, _ in CITED)
    print(f"{n} claims checked, {len(fails)} failed; {len(pending)} items pending; {quoted} quoted values (not computed)")
    for name, text in fails:
        print(f"  FAIL {name}: {text}")
    for p in pending:
        print(f"  pending: {p}")
    return len(fails)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--final", action="store_true", help="refuse all pending work and require current compiled PDFs")
    args = parser.parse_args()
    R.gate(verbose=False)
    sys.exit(1 if main(args.final) else 0)
