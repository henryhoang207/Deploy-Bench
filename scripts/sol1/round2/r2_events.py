#!/usr/bin/env python3
"""Event check for the round-2 run -> verify -> next loop (W1-W3 on sol1). Read-only apart from its own state
(xfer/r2_events.state.json) and history (xfer/r2_events.log). System python3, no third-party packages:

    python3 /mnt/data8tb/henry/xfer/r2_events.py

Prints every event since the previous call and exits 10 if one needs attention, 0 if none (first call: records the
current state and exits 0).
  FAIL      a new line with FAIL, REJECTED, REFUSED or Traceback in a watched log
  DL        the ASVspoof 5 download and extraction script ended (R2_DL_DONE)
  E6        the ASVspoof 5 silence statistics ended (AS5_SILENCE_DONE)
  QUEUE     queue B or the W1-finish chain reported a result
  RUN_END   a training run wrote run_final.json
  SCHED_END a scheduler printed its DONE line
  GATE_IN   a gate's input exists (as5_aasist: as5_dev and its tau; xlsr_mamba: df21_verified)
  SCORED    every registered score file (and tau) of a detector or of a training run exists
  DEAD      a run, scheduler or queue process is gone without its end mark
  STALL     a live run has written no epoch for 60 minutes
  DISK      less than 300 GB free on /mnt/data8tb
"""
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

H = Path("/mnt/data8tb/henry")
C = H / "deploybench"
SC, TAUS = C / "results/round2/scores", C / "results/round2/taus"
RUNS = H / "r2_runs"
STATE, HIST = H / "xfer/r2_events.state.json", H / "xfer/r2_events.log"
LOGS = {"downloads": "r2_downloads.log", "queueB": "r2_queueB.log", "w1_finish": "r2_w1_finish.log",
        "train_sched": "r2_train_sched.log", "score_sched": "r2_score_sched.log",
        "as5_silence": "r2_as5_silence.log", "pascu_rerun": "r2_pascu_mlaad5_rerun.log",
        "pascu_a5": "r2_pascu_mlaad5_a5.log", "w4": "r2_w4.log"}
WAKE = {"downloads": re.compile(r"R2_DL_DONE"), "queueB": re.compile(r"\] done |QUEUE_B_DONE"),
        "w1_finish": re.compile(r"."), "train_sched": re.compile(r"TRAIN_RUNS_DONE"),
        "score_sched": re.compile(r"SCORING_DONE"), "as5_silence": re.compile(r"AS5_SILENCE_DONE"),
        "pascu_rerun": re.compile(r"PASCU_MLAAD5_DONE|^start "), "pascu_a5": re.compile(r"PASCU_MLAAD5_DONE|^start "),
        "w4": re.compile(r"W4_RUN_DONE")}
FAILRE = re.compile(r"FAIL|REJECTED|REFUSED|Traceback")
# process (args regex) that must be alive until its log shows the end mark
PROCS = {"train_sched": (r"r2_train_sched\.py", "TRAIN_RUNS_DONE"),
         "score_sched": (r"r2_score_sched\.py", "SCORING_DONE"),
         "queueB": (r"r2_score_detectors\.sh", "QUEUE_B_DONE"),
         "w1_finish": (r"r2_w1_finish\.sh", "W1_FINISH_DONE"),
         "as5_silence": (r"r2_as5_silence\.py", "AS5_SILENCE_DONE"),
         "pascu_rerun": (r"r2_pascu_mlaad5_rerun\.py", "PASCU_MLAAD5_"),
         "pascu_a5": (r"r2_pascu_mlaad5_rerun\.py", "PASCU_MLAAD5_"),
         "w4": (r"r2_w4_run\.sh", "W4_RUN_DONE")}
CONDS = ["clean", "opus6k", "opus24k", "amr4.75", "mp3_32", "snr20", "snr10", "snr5", "mic_ir", "encodec6k"]
STRESS = [f"stress_{c}" for c in CONDS]
DET = {"as5_aasist": ["as5_dev", "as5_eval", "la19_eval", "itw", "df21_verified", "mlaad5"] + STRESS,
       "xlsr_mamba": ["la19_dev", "la19_eval", "itw", "df21_verified", "mlaad5"] + STRESS,
       "pascu_lr": ["la19_dev", "la19_eval", "itw", "df21_verified", "mlaad5"] + STRESS}
RUN_IDS = [("r2", k) for k in (1, 2, 3, 4, 5, 1234)] + [("trim", k) for k in (1, 2, 3, 4, 5)]


def units():
    """unit -> list of paths that must all exist."""
    u = {}
    for m, surfs in DET.items():
        u[f"det:{m}"] = [SC / m / "s0" / f"{s}.csv" for s in surfs] + [TAUS / f"{m}__s0.json"]
    for arm, k in RUN_IDS:
        run = RUNS / f"aasist_l_{arm}_s{k}"
        swa, last = f"aasist_l_{arm}", f"aasist_l_{arm}_last"
        u[f"run:aasist_l_{arm}_s{k}"] = (
            [run / "run_final.json"]
            + [SC / swa / f"s{k}" / f"{s}.csv" for s in ["la19_dev", "la19_eval", "itw"] + STRESS]
            + [SC / last / f"s{k}" / f"{s}.csv" for s in ["la19_dev", "la19_eval", "itw"]]
            + [TAUS / f"{swa}__s{k}.json", TAUS / f"{last}__s{k}.json"])
    return u


def ps():
    out = subprocess.run(["ps", "-eww", "-o", "args="], capture_output=True, text=True).stdout
    return out.splitlines()


def main():
    first = not STATE.exists()
    st = json.loads(STATE.read_text()) if not first else {"offsets": {}, "reported": []}
    reported = set(st["reported"])
    wake, info = [], []

    # 1. new log lines
    for key, name in LOGS.items():
        p = H / "logs" / name
        if not p.exists():
            continue
        size = p.stat().st_size
        off = st["offsets"].get(key, size if first else 0)
        if size < off:
            off = 0
        with open(p, "rb") as f:
            f.seek(off)
            chunk = f.read()
        done = chunk.rfind(b"\n") + 1
        st["offsets"][key] = off + done
        for line in chunk[:done].decode("utf-8", "replace").splitlines():
            if not line.strip():
                continue
            if FAILRE.search(line):
                wake.append(f"FAIL [{key}] {line}")
            elif WAKE[key].search(line):
                kind = {"downloads": "DL", "queueB": "QUEUE", "w1_finish": "QUEUE", "as5_silence": "E6", "pascu_rerun": "QUEUE", "pascu_a5": "QUEUE", "w4": "W4"}.get(key, "SCHED_END")
                wake.append(f"{kind} [{key}] {line}")
            else:
                info.append(f"[{key}] {line}")

    # 2. completed units, gate inputs, finished runs
    for unit, paths in units().items():
        if unit not in reported and all(x.exists() for x in paths):
            wake.append(f"SCORED {unit}")
            reported.add(unit)
    gate_in = {"gate:as5_aasist": [SC / "as5_aasist/s0/as5_dev.csv", TAUS / "as5_aasist__s0.json"],
               "gate:xlsr_mamba": [SC / "xlsr_mamba/s0/df21_verified.csv"]}
    for g, paths in gate_in.items():
        if g not in reported and all(x.exists() for x in paths):
            wake.append(f"GATE_IN {g[5:]}")
            reported.add(g)
    args = ps()
    now = datetime.now().timestamp()
    for d in sorted(RUNS.glob("aasist_l_*_s*")):
        if not d.is_dir():
            continue
        if (d / "run_final.json").exists():
            if f"end:{d.name}" not in reported:
                wake.append(f"RUN_END {d.name}")
                reported.add(f"end:{d.name}")
            continue
        alive = any(str(d) in a.split() for a in args)    # whole token: .../aasist_l_r2_s1 is a prefix of ..._s1234
        if not alive and f"dead:{d.name}" not in reported:
            wake.append(f"DEAD run {d.name} (no process, no run_final.json)")
            reported.add(f"dead:{d.name}")
        ep = d / "epochs.jsonl"
        t_last = max(d.stat().st_mtime, ep.stat().st_mtime if ep.exists() else 0)
        n_ep = sum(1 for _ in open(ep)) if ep.exists() else 0
        if alive and now - t_last > 3600 and f"stall:{d.name}:{n_ep}" not in reported:
            wake.append(f"STALL run {d.name}: no epoch for {(now - t_last) / 60:.0f} min (epochs {n_ep})")
            reported.add(f"stall:{d.name}:{n_ep}")

    # 3. processes that ended without their end mark (ps first, then the log: the mark is written before exit)
    for key, (pat, mark) in PROCS.items():
        if any(re.search(pat, a) for a in args):
            continue
        p = H / "logs" / LOGS[key]
        if not p.exists():          # never launched yet: nothing can have died
            continue
        if mark not in p.read_text(errors="replace") and f"dead:{key}" not in reported:
            wake.append(f"DEAD {key}: no process and no {mark} in {p.name}")
            reported.add(f"dead:{key}")

    s = os.statvfs("/mnt/data8tb")
    free_gb = s.f_bavail * s.f_frsize / 1e9
    if free_gb < 300 and "disk" not in reported:
        wake.append(f"DISK {free_gb:.0f} GB free on /mnt/data8tb")
        reported.add("disk")

    st["reported"] = sorted(reported)
    if first:
        # baseline: what is already complete or dead now is recorded without waking
        STATE.write_text(json.dumps(st, indent=1))
        print(f"initialized {datetime.now().isoformat(timespec='seconds')}: {len(reported)} keys "
              f"({', '.join(sorted(reported)) or 'none'}); not reported as events: {wake}")
        return 0
    STATE.write_text(json.dumps(st, indent=1))
    ts = datetime.now().astimezone().isoformat(timespec="seconds")
    if wake or info:
        with open(HIST, "a") as f:
            for x in wake:
                f.write(f"{ts} EVENT {x}\n")
            for x in info:
                f.write(f"{ts} info  {x}\n")
    if wake:
        print(f"{ts}  {len(wake)} event(s)")
        print("\n".join(wake))
        if info:
            print("-- context --")
            print("\n".join(info[-15:]))
        return 10
    return 0


if __name__ == "__main__":
    sys.exit(main())
