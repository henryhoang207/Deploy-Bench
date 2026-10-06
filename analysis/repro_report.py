"""Reproduction report + tau freeze for ONE model (phase_1 steps 1.3 / 1.5).

Does four things, and REPORTS disagreements instead of reconciling them:
  A. EER on the eval scores via deploybench.metrics
  B. EER again by a plain numpy descending-sort ROC sweep importing NOTHING
     from deploybench — if A and B disagree the report says so and fails
  C. tau calibrated on the DEV scores at eer / frr1 / frr5 (deploybench.tau is
     the only place thresholds are computed), then FROZEN to a json file this
     script refuses to overwrite without --refreeze
  D. verification: achieved dev FRR at each frozen tau, score-sign check
     (bona fide must score higher than spoof), in-domain TTG

    python analysis/repro_report.py --model tssdnet \
        --eval-scores results/phase_1_reproduction/scores_repro_tssdnet.csv \
        --dev-scores  results/phase_1_reproduction/scores_dev_tssdnet.csv \
        --published 1.6590 --published-note "Linux pretrained, ghua-ac README" \
        --out-report  results/phase_1_reproduction/repro_report_tssdnet.md \
        --out-tau     results/phase_1_reproduction/tau_tssdnet.json

    python analysis/repro_report.py --selftest      # no data needed
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deploybench import metrics as M          # noqa: E402
from deploybench import tau as T              # noqa: E402


# ------------------------------------------------------------------ loading

def load_scores(path: Path):
    """Read the project score schema. Returns (scores, labels, meta)."""
    scores, labels = [], []
    cols = {k: set() for k in ("model", "dataset", "precision", "runtime",
                               "condition")}
    seen, dups = set(), 0
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        need = {"utt_id", "label", "score"}
        if not need <= set(rd.fieldnames or []):
            raise SystemExit(f"{path}: header {rd.fieldnames} lacks {need}")
        for row in rd:
            uid = row["utt_id"]
            dups += uid in seen
            seen.add(uid)
            labels.append(int(row["label"]))
            scores.append(float(row["score"]))
            for k in cols:
                cols[k].add(row.get(k, ""))
    meta = {"path": str(path), "n": len(scores), "duplicate_utt_ids": dups,
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    meta.update({k: sorted(v) for k, v in cols.items()})
    return np.asarray(scores, float), np.asarray(labels, int), meta


# ------------------------------------------- INDEPENDENT EER (no deploybench)

def eer_numpy_sweep(scores, labels):
    """Plain descending-sort ROC sweep. Deliberately shares NO code with
    deploybench.metrics (which sweeps with searchsorted over two sorted
    per-class arrays). Decision rule is the project's: accept as bona fide
    iff score >= tau, so FRR(tau) = P(bona < tau), FAR(tau) = P(spoof >= tau)."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    nb, ns = int((y == 1).sum()), int((y == 0).sum())
    if nb == 0 or ns == 0:
        raise ValueError("need both classes present")
    order = np.argsort(-s, kind="mergesort")
    s_d, y_d = s[order], y[order]
    acc_b = np.cumsum(y_d == 1)                  # bona with score >= this score
    acc_s = np.cumsum(y_d == 0)                  # spoof with score >= this score
    last = np.append(np.diff(s_d) != 0, True)    # collapse tied scores
    thr = np.concatenate([[s.max() + 1.0], s_d[last]])
    frr = np.concatenate([[1.0], 1.0 - acc_b[last] / nb])
    far = np.concatenate([[0.0], acc_s[last] / ns])
    d = frr - far                                # thr descending => d decreasing
    below = np.nonzero(d <= 0)[0]
    if len(below) == 0:
        i = len(d) - 1
        return float((frr[i] + far[i]) / 2), float(thr[i])
    i = int(below[0])
    if i == 0:
        return float((frr[0] + far[0]) / 2), float(thr[0])
    d0, d1 = d[i - 1], d[i]
    w = 0.0 if d1 == d0 else d0 / (d0 - d1)
    e = float((1 - w) * (frr[i - 1] + far[i - 1]) / 2
              + w * (frr[i] + far[i]) / 2)
    t = float((1 - w) * thr[i - 1] + w * thr[i])
    return e, t


def brute_frr_far(tau, scores, labels):
    """The definition, computed directly — audits tau.py's outputs."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    return float(np.mean(s[y == 1] < tau)), float(np.mean(s[y == 0] >= tau))


# ------------------------------------------------------------------ freezing

def freeze_tau(taus, path: Path, dev_meta, model, refreeze):
    """tau is frozen ONCE (methodology C3/P5). No silent re-freezing."""
    payload = {"model": model,
               "dev_dataset": (dev_meta["dataset"] or ["?"])[0],
               "n_dev": dev_meta["n"],
               "dev_scores_sha256": dev_meta["sha256"],
               "frozen_utc": datetime.now(timezone.utc)
                             .strftime("%Y-%m-%dT%H:%M:%SZ"),
               "taus": {k: float(v) for k, v in taus.items()}}
    if path.exists() and not refreeze:
        old = json.loads(path.read_text())
        drift = {k: (old["taus"].get(k), payload["taus"][k])
                 for k in payload["taus"]
                 if abs(old["taus"].get(k, float("nan"))
                        - payload["taus"][k]) > 1e-12}
        return old, ("ALREADY FROZEN — existing file kept; "
                     + ("recomputation agrees exactly." if not drift else
                        f"**RECOMPUTATION DRIFTS**: {drift}. Investigate; do "
                        "not re-freeze without a decision-log entry."))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return payload, ("RE-FROZEN (--refreeze given)" if refreeze else "frozen now")


# ------------------------------------------------------------------- report

def build(model, ev, dv, published, published_note, gate, tau_path, refreeze,
          n_boot):
    ev_s, ev_y, ev_m = ev
    lines, verdicts = [], []

    # --- score sign -------------------------------------------------------
    mb, ms = float(ev_s[ev_y == 1].mean()), float(ev_s[ev_y == 0].mean())
    medb = float(np.median(ev_s[ev_y == 1]))
    meds = float(np.median(ev_s[ev_y == 0]))
    sign_ok = bool(mb > ms and medb > meds)
    verdicts.append(("score sign: bona fide scores above spoof", sign_ok))

    # --- EER, two ways ----------------------------------------------------
    e_lib, t_lib = M.eer(ev_s, ev_y)
    e_ind, t_ind = eer_numpy_sweep(ev_s, ev_y)
    agree = bool(abs(e_lib - e_ind) < 1e-6)
    verdicts.append(("EER: metrics.py agrees with independent sweep", agree))

    rel = ((e_lib * 100 - published) / published) if published else float("nan")
    gate_pass = bool(abs(rel) <= gate) if published else None
    if published:
        verdicts.append((f"reproduction gate |rel| <= {gate:.0%}", gate_pass))

    ci = (M.bootstrap_ci(ev_s, ev_y, lambda s, l: M.eer(s, l)[0], n_boot=n_boot)
          if n_boot else None)

    lines += [f"# Reproduction report — {model}", "",
              "Generated "
              f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%MZ')} by "
              "`analysis/repro_report.py`. Score contract: label 1 = bona fide, "
              "higher score = bona fide, SPOOF if score < tau.", "",
              "## Inputs", "",
              "| role | file | n | sha256[:16] | model | dataset | precision | "
              "runtime | condition | dup ids |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for tag, m in (("eval", ev_m), ("dev", dv[2] if dv else None)):
        if m:
            lines.append(
                f"| {tag} | `{Path(m['path']).name}` | {m['n']} | "
                f"`{m['sha256'][:16]}` | {','.join(m['model'])} | "
                f"{','.join(m['dataset'])} | {','.join(m['precision'])} | "
                f"{','.join(m['runtime'])} | {','.join(m['condition'])} | "
                f"{m['duplicate_utt_ids']} |")

    lines += ["", "## A/B — EER computed two independent ways", "",
              "| method | EER % | tau at EER |", "|---|---|---|",
              f"| `deploybench.metrics.eer` | {e_lib*100:.4f} | {t_lib:+.6f} |",
              f"| independent numpy sort-and-sweep | {e_ind*100:.4f} | "
              f"{t_ind:+.6f} |",
              f"| **absolute difference** | **{abs(e_lib-e_ind)*100:.6f}** | "
              f"{abs(t_lib-t_ind):.6f} |", "",
              ("Agreement below 1e-6 — the metrics module is not the source of "
               "this number." if agree else
               "**DISAGREEMENT.** The two implementations differ by more than "
               "1e-6. Report neither number until it is resolved; the likeliest "
               "causes are tie handling at the crossing and the threshold grid "
               "endpoints.")]

    lines += ["", "## Reproduction gate", ""]
    if published:
        lines += [f"- Published anchor: **{published:.4f}%**"
                  + (f" ({published_note})" if published_note else ""),
                  f"- Ours: **{e_lib*100:.4f}%**"
                  + (f"  (95% bootstrap CI {ci['lo']*100:.4f}–"
                     f"{ci['hi']*100:.4f}%, {n_boot} resamples)" if ci else ""),
                  f"- Relative difference: **{rel*100:+.2f}%**  "
                  f"(gate |rel| <= {gate:.0%})", "",
                  f"### VERDICT: {'PASS' if gate_pass else 'FAIL'}", ""]
        if not gate_pass:
            lines += ["Diagnose in this order before touching any wrapper:", "",
                      "1. score sign / class-index order (next section)",
                      "2. input length — TSSDNet needs exactly 96,000 samples",
                      "3. padding — repeat-tile short, truncate long FROM THE START",
                      "4. checkpoint identity (byte size, filename EER string)",
                      "5. which protocol file was actually iterated (dev vs eval)",
                      ""]
    else:
        lines += ["No `--published` anchor supplied; gate not evaluated.", ""]

    lines += ["## Score-sign check", "",
              "| class | n | mean | median |", "|---|---|---|---|",
              f"| bona fide (label 1) | {int((ev_y==1).sum())} | {mb:+.4f} | "
              f"{medb:+.4f} |",
              f"| spoof (label 0) | {int((ev_y==0).sum())} | {ms:+.4f} | "
              f"{meds:+.4f} |", "",
              ("Bona fide scores higher on both mean and median — convention "
               "holds." if sign_ok else
               "**SIGN INVERTED.** Spoof scores above bona fide. The wrapper is "
               "NOT the first thing to change: verify the protocol label column "
               "and the checkpoint first (wrapper_audit.md finding 2). An "
               "inverted sign yields EER ~ 100 - EER."), ""]

    # --- tau --------------------------------------------------------------
    frozen = None
    if dv:
        dv_s, dv_y, dv_m = dv
        taus = T.calibrate_tau(dv_s, dv_y)
        frozen, freeze_note = freeze_tau(taus, tau_path, dv_m, model, refreeze)
        use = frozen["taus"]
        lines += ["## Frozen operating points (tau)", "",
                  f"Calibrated on `{(dv_m['dataset'] or ['?'])[0]}` "
                  f"(n={dv_m['n']}) — D16: tau follows the model's FINE-TUNING "
                  "data; TSSDNet is Cohort-19, so ASVspoof19 LA dev.",
                  f"Artifact `{tau_path}` — {freeze_note}", "",
                  "| point | tau | dev target | dev FRR | dev FAR | eval FRR | "
                  "eval FAR | eval HTER |",
                  "|---|---|---|---|---|---|---|---|"]
        targets = {"eer": "FRR = FAR", "frr1": "FRR <= 1%", "frr5": "FRR <= 5%"}
        for name in ("eer", "frr1", "frr5"):
            t = use[name]
            dfrr, dfar = brute_frr_far(t, dv_s, dv_y)
            efrr, efar = brute_frr_far(t, ev_s, ev_y)
            lines.append(f"| {name} | {t:+.6f} | {targets[name]} | "
                         f"{dfrr*100:.3f}% | {dfar*100:.3f}% | {efrr*100:.3f}% | "
                         f"{efar*100:.3f}% | {(efrr+efar)/2*100:.3f}% |")
        f1 = brute_frr_far(use["frr1"], dv_s, dv_y)[0]
        f5 = brute_frr_far(use["frr5"], dv_s, dv_y)[0]
        # FRR rises with tau (a higher bar rejects more bona fide), so the
        # stricter 1% budget must sit at the LOWER threshold.
        mono = bool(use["frr1"] <= use["frr5"])
        verdicts.append(("frr1 tau achieves dev FRR <= 1%",
                         bool(f1 <= 0.01 + 1e-9)))
        verdicts.append(("frr5 tau achieves dev FRR <= 5%",
                         bool(f5 <= 0.05 + 1e-9)))
        verdicts.append(("tau(frr1) <= tau(frr5)", mono))
        lines += ["",
                  "All dev FRR/FAR above are recomputed by brute force "
                  "(`np.mean(bona < tau)`), not by the module that produced "
                  "tau.", "",
                  f"- FRR-1% point achieves {f1*100:.4f}% dev FRR "
                  f"({'meets' if f1 <= 0.01 + 1e-9 else 'MISSES'} the 1% target)",
                  f"- FRR-5% point achieves {f5*100:.4f}% dev FRR "
                  f"({'meets' if f5 <= 0.05 + 1e-9 else 'MISSES'} the 5% target)",
                  f"- Ordering tau(frr1) {'<=' if mono else '>'} tau(frr5) — "
                  + ("correct: FRR rises with tau, so the stricter 1% "
                     "rejection budget sits at the lower threshold"
                     if mono else "WRONG"),
                  "",
                  "These three taus are now the ONLY thresholds this model may "
                  "use downstream (conversion parity, device runs, stress "
                  "conditions). Nothing re-tunes them.", ""]

        lines += ["## Threshold transfer dev -> eval (seen corpus, unseen attacks)", "",
                  "| point | HTER at frozen tau | oracle HTER on eval | TTG |",
                  "|---|---|---|---|"]
        for name in ("eer", "frr1", "frr5"):
            r = M.ttg(ev_s, ev_y, use[name])
            lines.append(f"| {name} | {r['hter_fixed']*100:.3f}% | "
                         f"{r['hter_oracle']*100:.3f}% | {r['ttg']*100:+.3f} pp |")
        # D37: 19LA dev -> eval is NOT in-domain. Eval attacks A07-A19 are unseen in
        # train/dev, and the measured transfer is large for the GAT models.
        lines += ["", "Dev and eval share a corpus, but eval attacks A07-A19 are "
                  "unseen in train/dev (A16/A19 = A04/A06), so this TTG measures an "
                  "attack shift, not a sanity floor (D37). Report it and keep it "
                  "separate from the cross-corpus TTGs.", ""]
    else:
        lines += ["## Frozen operating points (tau)", "",
                  "**Not computed — no `--dev-scores` supplied.** tau must be "
                  "calibrated on ASVspoof19 LA dev (D16) before any conversion "
                  "or device run.", ""]

    lines += ["## Verification summary", "", "| check | result |", "|---|---|"]
    for name, good in verdicts:
        lines.append(f"| {name} | {'PASS' if good else 'FAIL'} |")
    all_ok = all(g for _, g in verdicts)
    lines += ["", f"**Overall: {'PASS' if all_ok else 'FAIL'}**", ""]
    return "\n".join(lines), all_ok, {"eer_lib": e_lib, "eer_ind": e_ind,
                                      "rel": rel, "frozen": frozen}


# ------------------------------------------------------------------ selftest

def selftest(tmp: Path):
    """Synthetic two-Gaussian dev/eval CSVs -> full report, no real data.
    mu = +-2.1, sigma = 1 gives an analytic EER of Phi(-2.1) = 1.786%."""
    rng = np.random.default_rng(7)

    def mk(path, n, dataset):
        y = np.r_[np.ones(n, int), np.zeros(n, int)]
        s = np.r_[rng.normal(2.1, 1.0, n), rng.normal(-2.1, 1.0, n)]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["utt_id", "label", "score", "model", "dataset",
                        "precision", "runtime", "condition", "seed"])
            for i, (yy, ss) in enumerate(zip(y, s)):
                w.writerow([f"U_{dataset}_{i}", yy, ss, "tssdnet", dataset,
                            "fp32", "pytorch", "clean", 0])

    tmp.mkdir(parents=True, exist_ok=True)
    de, dd = tmp / "eval.csv", tmp / "dev.csv"
    mk(de, 3000, "asvspoof19_eval")
    mk(dd, 3000, "asvspoof19_dev")
    md, ok_, extra = build("tssdnet", load_scores(de), load_scores(dd),
                           1.786, "analytic Phi(-2.1)", 0.15,
                           tmp / "tau.json", True, 200)
    (tmp / "report.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\nselftest dir: {tmp}")
    print(f"lib={extra['eer_lib']*100:.4f}%  independent="
          f"{extra['eer_ind']*100:.4f}%  overall={'PASS' if ok_ else 'FAIL'}")
    return 0 if ok_ else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tssdnet")
    ap.add_argument("--eval-scores")
    ap.add_argument("--dev-scores")
    ap.add_argument("--published", type=float, default=None,
                    help="published EER in PERCENT, e.g. 1.6590")
    ap.add_argument("--published-note", default="")
    ap.add_argument("--gate", type=float, default=0.15,
                    help="relative tolerance; 0.15 = 15%%")
    ap.add_argument("--out-report", default=None)
    ap.add_argument("--out-tau", default=None)
    ap.add_argument("--refreeze", action="store_true",
                    help="overwrite an existing frozen tau file — requires a "
                         "decision-log entry, tau is frozen once")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        import tempfile
        return selftest(Path(tempfile.mkdtemp(prefix="repro_selftest_")))
    if not a.eval_scores:
        raise SystemExit("--eval-scores is required (or use --selftest)")

    ev = load_scores(Path(a.eval_scores))
    dv = load_scores(Path(a.dev_scores)) if a.dev_scores else None
    tau_path = Path(a.out_tau) if a.out_tau else Path(f"tau_{a.model}.json")
    md, all_ok, extra = build(a.model, ev, dv, a.published, a.published_note,
                              a.gate, tau_path, a.refreeze, a.n_boot)
    if a.out_report:
        Path(a.out_report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_report).write_text(md, encoding="utf-8")
        print(f"report -> {a.out_report}")
    else:
        print(md)
    print(f"EER {extra['eer_lib']*100:.4f}%  (independent "
          f"{extra['eer_ind']*100:.4f}%)   overall: "
          f"{'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
