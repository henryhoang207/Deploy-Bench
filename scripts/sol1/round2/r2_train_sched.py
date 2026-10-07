"""W2 launcher, version 2 (DEV-9 amendment 1): start the eleven AASIST-L runs as GPU memory allows.

Each of our runs is counted at RUN_GB from launch to exit (the trimmed runs allocate little during their ~13-minute
trim-status pre-count; a run alone on a GPU reserves ~15.5 GB while training); runs already started (their directory
exists) are skipped; every other process on the GPU at its measured use; MARGIN_GB kept
free; at most CAP runs per GPU. Runs start in the interleaved order below, one start per minute at most. Each run refuses
an existing directory (train/aasist_r2_train.py); none is restarted or relaunched by this script. Run from the clone:

    setsid nohup /mnt/data8tb/henry/envs/r2/bin/python -B scripts/sol1/round2/r2_train_sched.py > ../logs/r2_train_sched.log 2>&1 &

Ends with TRAIN_RUNS_DONE fails=<n>.
"""
import datetime
import os
import subprocess
import sys
import time

H = "/mnt/data8tb/henry"
ORDER = [("r2", 1), ("trim", 1), ("r2", 2), ("trim", 2), ("r2", 3), ("trim", 3), ("r2", 4), ("trim", 4),
         ("r2", 5), ("trim", 5), ("r2", 1234)]
RUN_GB, MARGIN_GB, CAP, START_GAP_S = 16.0, 2.0, 4, 60   # 16 GB: a run alone on a GPU reserved 15.5 GB
OUT = f"{H}/r2_runs"


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def q(args):
    out = subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"], capture_output=True, text=True,
                         check=True).stdout
    return [[x.strip() for x in line.split(",")] for line in out.strip().splitlines() if line.strip()]


def gpu_state():
    gpus = {u: (int(i), float(t) / 1024) for i, u, t in q(["--query-gpu=index,uuid,memory.total"])}
    used = {}
    for row in q(["--query-compute-apps=pid,gpu_uuid,used_memory"]):
        pid, uuid, mem = int(row[0]), row[1], float(row[2]) / 1024
        used.setdefault(gpus[uuid][0], []).append((pid, mem))
    total = {i: t for i, t in gpus.values()}
    return total, used


def main():
    if subprocess.run(["git", "merge-base", "--is-ancestor", "round2-freeze", "HEAD"]).returncode != 0:
        sys.exit("REFUSED: the clone does not contain tag round2-freeze")
    env = dict(os.environ, DATA_ROOT=f"{H}/data/deepfake", CC=f"{H}/envs/cc/bin/gcc", OMP_NUM_THREADS="2",
               MKL_NUM_THREADS="2", PATH=f"{H}/tools/bin:" + os.environ.get("PATH", ""))
    py = f"{H}/envs/r2/bin/python"
    os.makedirs(f"{OUT}/logs", exist_ok=True)
    queue = list(ORDER)
    running = {}                                   # name -> (Popen, gpu, log handle)
    fails, last_start = 0, 0.0
    while queue or running:
        for name in list(running):
            p, g, fh = running[name]
            rc = p.poll()
            if rc is not None:
                fh.close()
                del running[name]
                if rc == 0:
                    print(f"finished {name} on GPU {g} {now()}", flush=True)
                else:
                    fails += 1
                    print(f"FAILED {name} on GPU {g} rc={rc} {now()}", flush=True)
        if queue and time.time() - last_start >= START_GAP_S:
            total, used = gpu_state()
            ours = {n: (p.pid, g) for n, (p, g, _) in running.items()}
            our_pids = {pid for pid, _ in ours.values()}
            for g in sorted(total):
                n_ours = sum(1 for _, gg in ours.values() if gg == g)
                other = sum(m for pid, m in used.get(g, []) if pid not in our_pids)
                free = total[g] - other - RUN_GB * n_ours - MARGIN_GB
                while queue and os.path.exists(f"{OUT}/aasist_l_{queue[0][0]}_s{queue[0][1]}"):
                    print(f"skip aasist_l_{queue[0][0]}_s{queue[0][1]} (directory exists: started earlier)", flush=True)
                    queue.pop(0)
                if queue and n_ours < CAP and free >= RUN_GB:
                    arm, seed = queue.pop(0)
                    name = f"aasist_l_{arm}_s{seed}"
                    cmd = [py, "-B", "train/aasist_r2_train.py", "--seed", str(seed),
                           "--trim", "vad35" if arm == "trim" else "none",
                           "--database-path", f"{H}/data/deepfake/LA", "--out-dir", f"{OUT}/{name}"]
                    fh = open(f"{OUT}/logs/{name}.log", "w")
                    p = subprocess.Popen(cmd, env=dict(env, CUDA_VISIBLE_DEVICES=str(g)), stdout=fh,
                                         stderr=subprocess.STDOUT)
                    running[name] = (p, g, fh)
                    last_start = time.time()
                    print(f"started {name} on GPU {g} (pid {p.pid}; other {other:.1f} GB, ours {n_ours}) {now()}",
                          flush=True)
                    break
        time.sleep(30)
    print(f"TRAIN_RUNS_DONE {now()} fails={fails}", flush=True)


if __name__ == "__main__":
    main()
