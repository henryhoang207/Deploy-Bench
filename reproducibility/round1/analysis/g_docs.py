"""DEV-8 G (S6): documentation tables (descriptive).

    python g_docs.py      # writes results/g_docs.json and results/g_docs.txt

DF21: per-condition and per-class counts of the eval subset, integrity class A (libFLAC >= 1.4 rejects, libFLAC-1.3-equivalent
decode matches the stored MD5), class B (decode fails the MD5: excluded), the verified surface and the clean surface.
H5: composition of the 1,000-clip manifest. ITW: speaker-label summary.
"""
from __future__ import annotations

import collections

import numpy as np
import pandas as pd

import revlib as R

CFG = R.CLONE / "configs"


def class_ids(name):
    ids = [l.strip() for l in open(CFG / f"df21_class{name}.txt", encoding="utf-8") if l.strip() and not l.startswith("#")]
    return set(ids)


def main():
    L = []
    meta = R.df21_meta()
    ev = meta[meta.subset == "eval"].copy()
    A, B = class_ids("A"), class_ids("B")
    ev["cls"] = np.where(ev.index.isin(B), "B", np.where(ev.index.isin(A), "A", "ok"))
    L.append(f"== DF21 eval subset: {len(ev)} trials (bona fide {int((ev.label == 'bonafide').sum())}, spoof {int((ev.label == 'spoof').sum())}); "
             f"class A {int((ev.cls == 'A').sum())}, class B {int((ev.cls == 'B').sum())}, rejected by libFLAC >= 1.4 = A + B = {int((ev.cls != 'ok').sum())}")
    L.append("   verified surface = eval - B = %d; clean surface = eval - A - B = %d" % (int((ev.cls != 'B').sum()), int((ev.cls == 'ok').sum())))
    assert len(ev) == 533928 and (ev.cls == "A").sum() == 149983 and (ev.cls == "B").sum() == 78420, "counts differ from the registered surfaces"
    L.append(f"\n{'condition':10s} {'eval':>8s} {'bona%':>6s} | {'A':>7s} {'A bona%':>7s} | {'B':>7s} {'B bona%':>7s} | {'verified':>8s} {'bona%':>6s} | {'clean':>7s} {'bona%':>6s}")
    res = {"by_condition": {}}
    for c, d in sorted(ev.groupby("compr")):
        row = {}
        for name, mask in (("eval", d.index == d.index), ("A", d.cls == "A"), ("B", d.cls == "B"), ("verified", d.cls != "B"), ("clean", d.cls == "ok")):
            sub = d[mask]
            row[name] = (int(len(sub)), float(np.mean(sub.label == "bonafide")) if len(sub) else float("nan"))
        res["by_condition"][c] = row
        L.append(f"{c:10s} {row['eval'][0]:8d} {100 * row['eval'][1]:6.2f} | {row['A'][0]:7d} {100 * row['A'][1]:7.2f} | {row['B'][0]:7d} {100 * row['B'][1]:7.2f} | "
                 f"{row['verified'][0]:8d} {100 * row['verified'][1]:6.2f} | {row['clean'][0]:7d} {100 * row['clean'][1]:6.2f}")
    tot = {n: (int(m.sum()), float(np.mean(ev[m].label == "bonafide"))) for n, m in (("eval", ev.index == ev.index), ("A", ev.cls == "A"), ("B", ev.cls == "B"), ("verified", ev.cls != "B"), ("clean", ev.cls == "ok"))}
    res["total"] = tot
    L.append(f"{'all':10s} {tot['eval'][0]:8d} {100 * tot['eval'][1]:6.2f} | {tot['A'][0]:7d} {100 * tot['A'][1]:7.2f} | {tot['B'][0]:7d} {100 * tot['B'][1]:7.2f} | "
             f"{tot['verified'][0]:8d} {100 * tot['verified'][1]:6.2f} | {tot['clean'][0]:7d} {100 * tot['clean'][1]:6.2f}")
    L.append("\n   by source (eval / class B / verified): " + "; ".join(
        f"{s}: {int((ev.source == s).sum())} / {int(((ev.source == s) & (ev.cls == 'B')).sum())} ({100 * ((ev.source == s) & (ev.cls == 'B')).sum() / (ev.source == s).sum():.1f}%) / {int(((ev.source == s) & (ev.cls != 'B')).sum())}"
        for s in sorted(ev.source.unique())))
    # chi-square style imbalance check: share of B by condition
    bshare = {c: float(np.mean(d.cls == "B")) for c, d in ev.groupby("compr")}
    L.append("   class-B share by condition: " + ", ".join(f"{c} {100 * v:.1f}%" for c, v in sorted(bshare.items())))
    res["class_b_share_by_condition"] = bshare
    L.append(f"   DF21 other subsets in the key file (not in any registered surface): hidden {int((meta.subset == 'hidden').sum())} (all 'trim'), "
             f"progress {int((meta.subset == 'progress').sum())}")

    proto = R.la19_protocol("eval")
    ids = R.read_ids(CFG / "parity_manifest_tssdnet.txt")
    att = collections.Counter(proto[u][1] for u in ids if proto[u][2] == 0)
    spk = collections.Counter(proto[u][0] for u in ids)
    L.append(f"\n== H5 manifest: {len(ids)} clips = {sum(proto[u][2] == 1 for u in ids)} bona fide + {sum(proto[u][2] == 0 for u in ids)} spoof, uniform seeded draw within class from LA19 eval "
             f"(seed 20260725, frozen 2026-07-26 for the conversion-parity gate, before H5 was specified); {len(spk)} of 67 eval speakers (3-{max(spk.values())} clips each, median {int(np.median(list(spk.values())))})")
    L.append("   spoofs per attack: " + ", ".join(f"{a} {n}" for a, n in sorted(att.items())))
    res["h5_manifest"] = {"attacks": dict(att), "speakers": len(spk), "clips_per_speaker": [min(spk.values()), float(np.median(list(spk.values()))), max(spk.values())]}

    im = R.itw_meta()
    c = im.groupby("speaker").label.value_counts().unstack(fill_value=0)
    L.append(f"\n== ITW meta.csv: {len(im)} clips, {len(c)} distinct speaker labels (the dataset paper reports 58 public figures); bona fide {int(c['bona-fide'].sum())}, spoof {int(c['spoof'].sum())}; "
             f"every label has both classes: {bool(((c > 0).all(axis=1)).all())}; clips per label min {int((c.sum(axis=1)).min())}, median {int(c.sum(axis=1).median())}, max {int(c.sum(axis=1).max())}")
    R.write_json("g_docs.json", res)
    (R.RESULTS / "g_docs.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
