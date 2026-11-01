"""Label-free accuracy estimation by average thresholded confidence (ATC; Garg et al. 2022) at tau_EER
(DEVIATIONS.md, DEV-10 F3; specified and deposited before it ran):

    python analysis/review_r1/run_atc.py       # needs results/review_r1/ladder.json -> atc.json

For the 25 detector-target cells of C: the confidence of a decision at tau_EER is the clip's margin |s - tau_EER|. On
the detector's source dev set, acc_dev is the share of dev clips decided correctly at tau_EER, and the ATC threshold t
is the (1 - acc_dev) quantile of the dev margins (numpy.quantile, linear interpolation). The predicted target accuracy
is the share of target clips whose margin exceeds t; the true target accuracy is the share decided correctly at tau_EER
(one minus the total error rate at the target's class mix, checked against ladder.json). Success criterion, fixed in
DEV-10: |predicted - true| <= 5 pp in a cell. Reading rule: ATC counts as a usable label-free monitor of this shift only
if it meets the criterion on ITW for all four strong detectors.
"""
import numpy as np

from rvlib import NAMES, R, ROUND1, DETECTORS_R2, read, source_dev, targets_for, write

CRITERION = 0.05
STRONG = ("xlsr_aasist", "xlsr_mamba", "as5_aasist", "pascu_lr")
META = {"script": "analysis/review_r1/run_atc.py", "method": "ATC (Garg et al. 2022) with the margin |s - tau_EER| "
        "as confidence", "threshold": "(1 - acc_dev) quantile of the source-dev margins, numpy.quantile (linear)",
        "criterion": f"|predicted - true| <= {CRITERION}", "strong_detectors": list(STRONG),
        "reading": "usable only if every strong detector meets the criterion on ITW"}
LADDER = read("ladder")["cells"]


def correct(s, y, tau):
    """Decisions at tau that are right: bona fide accepted (s >= tau), spoof rejected (s < tau)."""
    return np.where(y == 1, s >= tau, s < tau)


CELLS = {}
for m in ROUND1 + DETECTORS_R2:
    _, yd, sd = R.load(m, source_dev(m))
    tau = R.taus(m)["eer"]
    acc_dev = float(np.mean(correct(sd, yd, tau)))
    t = float(np.quantile(np.abs(sd - tau), 1 - acc_dev))
    for surf in targets_for(m):
        _, y, s = R.load(m, surf)
        pred = float(np.mean(np.abs(s - tau) > t))
        true = float(np.mean(correct(s, y, tau)))
        if abs((1 - LADDER[f"{m}|{surf}"]["ter"]["static"]) - true) > 1e-12:
            R.die(f"{m}/{surf}: true accuracy {true} disagrees with ladder.json's static total error rate")
        CELLS[f"{m}|{surf}"] = {"model": m, "target": surf, "acc_dev": acc_dev, "t": t, "predicted": pred,
                                "true": true, "error": pred - true, "meets": bool(abs(pred - true) <= CRITERION)}
        c = CELLS[f"{m}|{surf}"]
        print(f"{NAMES[m]:11s} {surf[:10]:10s} acc_dev {100 * acc_dev:6.2f}  predicted {100 * pred:6.2f}  true "
              f"{100 * true:6.2f}  error {100 * c['error']:+7.2f} pp  {'meets' if c['meets'] else 'fails'}", flush=True)
itw = [c for c in CELLS.values() if c["target"] == "itw"]
summary = {"cells": len(CELLS), "meeting": sum(c["meets"] for c in CELLS.values()),
           "itw_cells": len(itw), "itw_meeting": sum(c["meets"] for c in itw),
           "usable": all(CELLS[f"{m}|itw"]["meets"] for m in STRONG)}
print(f"{summary['meeting']}/{summary['cells']} cells meet the criterion ({summary['itw_meeting']}/{summary['itw_cells']}"
      f" on ITW); usable as a label-free monitor: {'yes' if summary['usable'] else 'no'}")
write("atc", META, CELLS, summary)
