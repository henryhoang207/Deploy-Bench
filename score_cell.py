"""Score one grid cell from its YAML config (used by run_grid.sh).
Thin adapter: resolves dataset+condition to a wav manifest, then calls the
same machinery as score.py. Dataset resolution completed in step 1.2."""
import argparse
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(open(ap.parse_args().config))
    import subprocess, sys
    cmd = [sys.executable, "score.py", "--model", cfg["model"],
           "--dataset", cfg["dataset"], "--out", cfg["out"],
           "--precision", cfg["precision"], "--runtime", cfg["runtime"],
           "--condition", cfg["condition"], "--seed", str(cfg["seed"])]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
