"""Round-2 scoring scheduler (PREREGISTRATION_ROUND2.md Sec. 5-6): every registered score file not produced by
r2_score_detectors.sh, started when its inputs exist and a GPU has room, plus the frozen thresholds (r2_taus.py) as
soon as each source-dev file exists. Run from the clone after the round2-freeze tag:

    setsid nohup /mnt/data8tb/henry/envs/r2/bin/python -B scripts/sol1/round2/r2_score_sched.py > ../logs/r2_score_sched.log 2>&1 &

A task's output is written as <out>.partial and renamed on success; an existing output is never redone; a failed task
is reported and not retried. GPU room: the task's estimate (MEM) + MARGIN_GB must fit beside every measured process and
the estimates of tasks launched in the last 3 minutes. At most MAX_TASKS GPU tasks at once. Ends with SCORING_DONE.
"""
import datetime
import os
import subprocess
import sys
import time

H = "/mnt/data8tb/henry"
DATA = f"{H}/data/deepfake"
RUNS = f"{H}/r2_runs"
V = f"{H}/deploybench/vendor"
PY = {"db310": f"{H}/envs/db310/bin/python", "r2": f"{H}/envs/r2/bin/python"}
DS = {"la19_dev": "asvspoof19_dev", "la19_eval": "asvspoof19_eval", "itw": "itw", "df21_verified": "asvspoof21_df_eval",
      "mlaad5": "mlaad", "as5_dev": "asvspoof5_dev", "as5_eval": "asvspoof5_eval"}
R2SET = f"{DATA}/stress/r2_parity/manifest.csv"
H5SET = f"{DATA}/stress/h5_parity/manifest.csv"
CONDS = ["clean", "opus6k", "opus24k", "amr4.75", "mp3_32", "snr20", "snr10", "snr5", "mic_ir"]
MEM = {"aasist": 3.0, "xlsr_mamba": 4.5, "pascu": 11.0}
MARGIN_GB, MAX_TASKS, GAP_S = 2.0, 3, 45
PASCU_HEAD = f"{RUNS}/pascu/head_train.npz"


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def out_path(model, seed, surface):
    return f"results/round2/scores/{model}/s{seed}/{surface}.csv"


def score_task(model, seed, surface, env, ckpt, mem, batch, prereq=()):
    out = out_path(model, seed, surface)
    if surface.startswith("stress_"):
        cond = surface[len("stress_"):]
        man = H5SET if cond == "encodec6k" else R2SET
        data = ["--dataset", "asvspoof19_eval", "--stress-manifest", man, "--condition", cond]
        prereq = tuple(prereq) + (man,)
    else:
        data = ["--dataset", DS[surface]]
    cmd = [PY[env], "-B", "score.py", "--model", model, *(["--checkpoint", ckpt] if ckpt else []), *data,
           "--device", "cuda", "--batch", str(batch), "--seed", str(seed), "--progress", "50000", "--out", out + ".partial"]
    return {"name": f"{model}/s{seed}/{surface}", "out": out, "cmd": cmd, "mem": mem, "prereq": tuple(prereq) +
            ((ckpt,) if ckpt else ()), "gpu": True}


def tau_task(model, seed, surface):
    dev = out_path(model, seed, surface)
    return {"name": f"tau {model}/s{seed}", "out": f"results/round2/taus/{model}__s{seed}.json",
            "cmd": [PY["r2"], "-B", "analysis/round2/r2_taus.py", "--model", model, "--seed", str(seed)],
            "mem": 0, "prereq": (dev,), "gpu": False}


def tasks():
    T = []
    # as5_aasist (db310): its source dev, its own eval, and every registered target
    for s in ("as5_dev", "as5_eval", "la19_eval", "itw", "df21_verified", "mlaad5"):
        T.append(score_task("as5_aasist", 0, s, "db310", None, MEM["aasist"], 8,
                            prereq=(f"{DATA}/asvspoof5/.extracted_flac_E_aj.tar",)))
    T.append(tau_task("as5_aasist", 0, "as5_dev"))
    # the stress set for the three detectors (the W1 queue does their other surfaces)
    for c in CONDS + ["encodec6k"]:
        T.append(score_task("as5_aasist", 0, f"stress_{c}", "db310", None, MEM["aasist"], 8))
        T.append(score_task("xlsr_mamba", 0, f"stress_{c}", "r2", None, MEM["xlsr_mamba"], 16))
        T.append(score_task("pascu_lr", 0, f"stress_{c}", "r2", PASCU_HEAD, MEM["pascu"], 1))
    T.append(tau_task("xlsr_mamba", 0, "la19_dev"))
    T.append(tau_task("pascu_lr", 0, "la19_dev"))
    # the AASIST-L runs (db310 for scoring), each once its training has finished
    for arm in ("r2", "trim"):
        seeds = (1, 2, 3, 4, 5, 1234) if arm == "r2" else (1, 2, 3, 4, 5)
        for k in seeds:
            run = f"{RUNS}/aasist_l_{arm}_s{k}"
            fin = (f"{run}/run_final.json",)
            for wname, model in (("swa", f"aasist_l_{arm}"), ("last", f"aasist_l_{arm}_last")):
                ck = f"{run}/weights/{wname}.pth"
                surfs = ["la19_dev", "la19_eval", "itw"] + ([f"stress_{c}" for c in CONDS + ["encodec6k"]]
                                                            if wname == "swa" else [])
                for s in surfs:
                    T.append(score_task(model, k, s, "db310", ck, MEM["aasist"], 8, prereq=fin))
                T.append(tau_task(model, k, "la19_dev"))
    return T


def gpu_free():
    q = lambda a: [[x.strip() for x in l.split(",")] for l in subprocess.run(  # noqa: E731
        ["nvidia-smi", *a, "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout.splitlines() if l.strip()]
    gpus = {u: (int(i), float(t) / 1024) for i, u, t in q(["--query-gpu=index,uuid,memory.total"])}
    used = {i: 0.0 for i, _ in gpus.values()}
    for pid, uuid, mem in q(["--query-compute-apps=pid,gpu_uuid,used_memory"]):
        used[gpus[uuid][0]] += float(mem) / 1024
    return {i: t - used[i] for i, t in gpus.values()}


def main():
    if subprocess.run(["git", "merge-base", "--is-ancestor", "round2-freeze", "HEAD"]).returncode != 0:
        sys.exit("REFUSED: the clone does not contain tag round2-freeze")
    env = dict(os.environ, DATA_ROOT=DATA, CC=f"{H}/envs/cc/bin/gcc", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               PATH=f"{H}/tools/bin:" + os.environ.get("PATH", ""))
    os.makedirs(f"{H}/logs/r2_score", exist_ok=True)
    pending = [t for t in tasks() if not os.path.exists(t["out"])]
    print(f"{len(pending)} tasks pending {now()}", flush=True)
    running, recent, fails, last = {}, [], 0, 0.0
    while pending or running:
        for name in list(running):
            p, t, fh = running[name]
            rc = p.poll()
            if rc is None:
                continue
            fh.close()
            del running[name]
            if rc == 0 and t["gpu"]:
                os.replace(t["out"] + ".partial", t["out"])
            if rc == 0:
                print(f"done   {name} {now()}", flush=True)
            else:
                fails += 1
                print(f"FAILED {name} rc={rc} {now()} (log {H}/logs/r2_score/{name.replace('/', '_')}.log)", flush=True)
        recent = [(ts, g, m) for ts, g, m in recent if time.time() - ts < 180]
        if time.time() - last >= GAP_S:
            ready = [t for t in pending if all(os.path.exists(x) for x in t["prereq"])
                     and not os.path.exists(t["out"] + ".partial")]
            for t in ready:
                if not t["gpu"]:
                    pass
                elif sum(1 for p, tt, _ in running.values() if tt["gpu"]) >= MAX_TASKS:
                    continue
                gpu = None
                if t["gpu"]:
                    free = gpu_free()
                    for g in sorted(free, key=lambda g: -free[g]):
                        if free[g] - sum(m for ts, gg, m in recent if gg == g) >= t["mem"] + MARGIN_GB:
                            gpu = g
                            break
                    if gpu is None:
                        continue
                os.makedirs(os.path.dirname(t["out"]), exist_ok=True)
                fh = open(f"{H}/logs/r2_score/{t['name'].replace('/', '_')}.log", "w")
                e = dict(env, CUDA_VISIBLE_DEVICES=str(gpu)) if t["gpu"] else env
                p = subprocess.Popen(t["cmd"], env=e, stdout=fh, stderr=subprocess.STDOUT)
                running[t["name"]] = (p, t, fh)
                pending.remove(t)
                if t["gpu"]:
                    recent.append((time.time(), gpu, t["mem"]))
                    last = time.time()
                print(f"start  {t['name']}" + (f" on GPU {gpu}" if t["gpu"] else "") + f" {now()}", flush=True)
                if t["gpu"]:
                    break
        time.sleep(20)
    print(f"SCORING_DONE {now()} fails={fails}", flush=True)


if __name__ == "__main__":
    main()
