"""Generate one YAML per evaluation cell (phase_3 step_02).

    python gen_configs.py --out configs/grid/ [--models aasist rawnet2 ...]
"""
import argparse
import itertools
from pathlib import Path
import yaml

MODELS = ["aasist", "aasist_l", "rawnet2", "rawgat_st", "lcnn_lfcc",
          "tssdnet", "wavlm_trunc", "wavlm_mhfa", "xlsr_aasist"]
PRECISIONS = ["fp32", "fp16", "int8"]
RUNTIMES = ["ort", "executorch"]
# dev sets first (tau source), then transfer/stress
EVAL_DATASETS = ["asvspoof19_eval", "asvspoof5_eval", "itw",
                 "asvspoof21_df_eval", "mlaad"]


def tau_source(model):
    return "asvspoof5_dev" if model.startswith(("wavlm", "xlsr")) else "asvspoof19_dev"
CONDITIONS = ["clean", "opus6k", "opus24k", "amr4.75", "mp3_32",
              "encodec6k", "snr20", "snr10", "snr5", "mic_ir"]
STRESS_DATASETS = {"itw", "asvspoof21_df_eval"}   # stress applies here only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for m, p, r, s in itertools.product(args.models, PRECISIONS, RUNTIMES,
                                        args.seeds):
        datasets = [tau_source(m)] + EVAL_DATASETS   # dev cell FIRST (tau source)
        for d in datasets:
            conds = CONDITIONS if d in STRESS_DATASETS else ["clean"]
            for c in conds:
                name = f"{m}_{p}_{r}_{d}_{c}_s{s}"
                cfg = dict(model=m, precision=p, runtime=r, dataset=d,
                           condition=c, seed=s, tau_source=tau_source(m),
                           out=f"results/scores/{name}.csv")
                (out / f"{name}.yaml").write_text(yaml.dump(cfg, sort_keys=False))
                n += 1
    print(f"{n} cell configs -> {out}")


if __name__ == "__main__":
    main()
