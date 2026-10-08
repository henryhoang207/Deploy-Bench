"""DEV-9 amendment 4: rerun pascu_lr on MLAAD-5 once, unchanged, on a GPU without a training run.

Waits until every one of the eleven AASIST-L runs has started (the training queue is empty) and a GPU carries no
train/aasist_r2_train.py process, then scores there with the registered settings (whole utterance, batch 1, fp32) plus
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. Output: results/round2/scores/pascu_lr/s0/mlaad5.csv (via .partial).
Ends with PASCU_MLAAD5_DONE or PASCU_MLAAD5_FAILED. Run from the clone (envs/r2 python):

    setsid nohup /mnt/data8tb/henry/envs/r2/bin/python -B ../xfer/r2_pascu_mlaad5_rerun.py > ../logs/r2_pascu_mlaad5_rerun.log 2>&1 &
"""
import datetime
import os
import subprocess
import sys
import time

H = "/mnt/data8tb/henry"
OUT = "results/round2/scores/pascu_lr/s0/mlaad5.csv"
RUNS = [f"{H}/r2_runs/aasist_l_r2_s{k}" for k in (1, 2, 3, 4, 5, 1234)] + \
       [f"{H}/r2_runs/aasist_l_trim_s{k}" for k in (1, 2, 3, 4, 5)]


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def smi(args):
    out = subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"], capture_output=True, text=True,
                         check=True).stdout
    return [[x.strip() for x in line.split(",")] for line in out.splitlines() if line.strip()]


def free_gpu():
    """Index of a GPU with no training process and >= 16 GB free, or None."""
    train = {int(p) for p in subprocess.run(["pgrep", "-f", "train/aasist_r2_train.py"], capture_output=True,
                                            text=True).stdout.split()}
    gpus = {u: (int(i), float(t) - float(used)) for i, u, t, used in
            smi(["--query-gpu=index,uuid,memory.total,memory.used"])}
    busy = {gpus[u][0] for pid, u in smi(["--query-compute-apps=pid,gpu_uuid"]) if int(pid) in train}
    for u, (i, free_mib) in sorted(gpus.items(), key=lambda kv: kv[1][0]):
        if i not in busy and free_mib >= 16 * 1024:
            return i
    return None


def main():
    if subprocess.run(["git", "merge-base", "--is-ancestor", "round2-freeze", "HEAD"]).returncode != 0:
        sys.exit("REFUSED: the clone does not contain tag round2-freeze")
    if "_flac_read_verified" not in open("score.py").read():
        sys.exit("REFUSED: score.py lacks the verified FLAC decoder")
    if os.path.exists(OUT):
        print(f"{OUT} exists; nothing to do", flush=True)
        return
    print(f"waiting for all 11 runs to start and a GPU without training {now()}", flush=True)
    while True:
        if all(os.path.isdir(r) for r in RUNS):
            g = free_gpu()
            if g is not None:
                break
        time.sleep(120)
    env = dict(os.environ, DATA_ROOT=f"{H}/data/deepfake", CC=f"{H}/envs/cc/bin/gcc", OMP_NUM_THREADS="4",
               MKL_NUM_THREADS="4", PATH=f"{H}/tools/bin:" + os.environ.get("PATH", ""), CUDA_VISIBLE_DEVICES=str(g),
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    print(f"start pascu_lr mlaad5 on GPU {g} {now()}", flush=True)
    cmd = [f"{H}/envs/r2/bin/python", "-B", "score.py", "--model", "pascu_lr", "--checkpoint",
           f"{H}/r2_runs/pascu/head_train.npz", "--dataset", "mlaad", "--device", "cuda", "--batch", "1",
           "--progress", "50000", "--out", OUT + ".partial"]
    with open(f"{H}/logs/r2_pascu_lr_mlaad5_rerun_score.log", "w") as fh:
        rc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc == 0:
        os.replace(OUT + ".partial", OUT)
        print(f"PASCU_MLAAD5_DONE {now()}", flush=True)
    else:
        print(f"PASCU_MLAAD5_FAILED rc={rc} {now()} (log {H}/logs/r2_pascu_lr_mlaad5_rerun_score.log)", flush=True)


if __name__ == "__main__":
    main()
