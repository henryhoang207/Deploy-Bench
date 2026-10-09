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

The read-only revision source is discovered under ~/OneDrive, or can be selected
with DFV_REVISION (revision directory, or its analysis directory). DEPLOYBENCH
is deliberately bound to THIS worktree before revlib imports its estimators.
No results/round2 file is read. Filling round-2 results needs an explicit audited
result integration; the draft passes with pending items, while --final refuses
pending items, todos, missing result audits, and missing/failed compiled PDFs.
Speech Communication has no page cap in this audit.
"""
from __future__ import annotations

import json
import importlib.util
import os
import re
import sys
from argparse import ArgumentParser
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
    named = re.findall(r"[A-Za-z0-9_./\\-]+\.(?:md|tex|json|txt|csv|py)\b", comment)
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
        if path.resolve().is_relative_to(WORKTREE / "results" / "round2"):
            errors.append("round-2 results are reserved for audited integration by the loop session")
        elif not path.is_file():
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
                    if child.name.startswith("r2_") or child.name.startswith("r2s_"):
                        pending.append(f"round-2 result table {target}: numerical audit not integrated")
                        # Keep the loop session's result material entirely unread.
                    elif not child.exists() and command == "gentab":
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
            if nums and path.name not in ("supplement.tex", "round2_design.tex"):
                allowed = set(inherited_numbers)
                for source in sources:
                    if source.suffix in (".md", ".tex", ".py", ".json", ".txt"):
                        allowed.update(visible_numbers(source.read_text(encoding="utf-8")))
                # Newly rendered descriptive MLAAD values have separate computed checks below.
                if any(p.name in ("mlaad_report.json", "sample_rate_census.json") for p in sources):
                    allowed.update(visible_numbers(docs.get(HERE / "gen" / "r1_mlaad.tex", "")))
                    allowed.update(mlaad_numbers)
                unsupported = sorted(set(nums) - allowed)
                if unsupported:
                    fails.append(("src.unsupported_number", f"{label}: numbers absent from the cited source or audited draft: {unsupported}"))
            if nums and path.name == "supplement.tex" and any(p.is_relative_to(REVISION) for p in sources):
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


def main(final: bool):
    tex = (HERE / "main.tex").read_text(encoding="utf-8")
    fails, pending, n = [], [], 0
    docs = collect_documents(pending, fails)
    body = norm_tex("\n".join(t for p, t in docs.items() if p.parent.name != "gen" and p.name != "supplement.tex"))
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
    truth("r2.amendments", all(f"DEV-9 amendment {i}" in docs.get(design_path, "") for i in range(1, 5)),
          "all four DEV-9 amendments must be disclosed")
    pending.append("round-2 numerical results and result audit have not been integrated")

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
