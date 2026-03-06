
# =============================================================================
# run_all.py  —  Run all 5 stages in sequence
# =============================================================================
# Usage:
#   python run_all.py                  # all stages, no tuning
#   python run_all.py --tune           # all stages + Optuna UKF tuning
#   python run_all.py --tune --trials 300
#   python run_all.py --from ecm       # restart from stage 3 (skips 1 & 2)
#
# Stages:
#   1  load_data    — discover CSVs, load, denoise, save parquets
#   2  soc_ocv      — Coulomb SoC + OCV–SoC spline fit
#   3  ecm          — fit R0, R1, C1 per cycle from voltage data
#   4  ukf          — UKF SoC estimation (+ optional Optuna tuning)
#   5  summary      — aggregate table + cross-battery comparison plots

import sys, argparse, subprocess, time,os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)
STAGES = {
    "load":    "src/load_data.py",
    "soc":     "src/soc_ocv.py",
    "ecm":     "src/ecm_modelling.py",
    "ukf":     "src/ukf.py",
    "summary": "src/summary.py",
}
STAGE_ORDER = ["load", "soc", "ecm", "ukf", "summary"]
STAGE_LABEL = {
    "load":    "1 — Load & Denoise",
    "soc":     "2 — Coulomb SoC & OCV–SoC Fit",
    "ecm":     "3 — ECM Parameter Fitting",
    "ukf":     "4 — UKF SoC Estimation",
    "summary": "5 — Summary & Results",
}
VALIDATE_SCRIPT = "src/validate.py" 


def run_stage(script, extra_args=()):
    cmd = [sys.executable, script] + list(extra_args)
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Battery SoC Estimation — full pipeline")
    parser.add_argument("--tune",   action="store_true", help="Run Optuna tuning in UKF stage")
    parser.add_argument("--trials", type=int,   default=150, help="Optuna trials (default 150)")
    parser.add_argument("--from",   dest="start_from", default="load",
                        choices=STAGE_ORDER,
                        help="Start from this stage (skip earlier stages)")
    args = parser.parse_args()

    start_idx = STAGE_ORDER.index(args.start_from)
    stages    = STAGE_ORDER[start_idx:]

    print("=" * 60)
    print("  Battery SoC Estimation — Full Pipeline")
    print(f"  Stages to run: {', '.join(stages)}")
    if args.tune:
        print(f"  Optuna tuning: ON ({args.trials} trials)")
    print("=" * 60)

    # Always validate first (skip only if restarting mid-pipeline)
    if args.start_from == "load":
        print(f"\n{'─' * 60}")
        print("  Validating dataset …")
        print(f"{'─' * 60}")
        if not run_stage(VALIDATE_SCRIPT):
            print("\n  Dataset validation failed. Fix the errors above, then re-run.")
            sys.exit(1)

    t_total = time.time()

    for stage in stages:
        print(f"\n{'─' * 60}")
        print(f"  Running Stage {STAGE_LABEL[stage]}")
        print(f"{'─' * 60}")

        extra = []
        if stage == "ukf" and args.tune:
            extra = ["--tune", "--trials", str(args.trials)]

        t0      = time.time()
        success = run_stage(STAGES[stage], extra)
        elapsed = time.time() - t0

        if success:
            print(f"\n  ✓  Stage done in {elapsed:.1f}s")
        else:
            print(f"\n  ✗  Stage FAILED — stopping pipeline")
            sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"  All stages complete in {time.time()-t_total:.1f}s")
    print(f"  Results in:  results/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
