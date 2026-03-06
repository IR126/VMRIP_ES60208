# =============================================================================
# src/summary.py  —  Stage 5: aggregate results, all required metrics, plots
# =============================================================================
# Run:  python src/summary.py
# In:   results/<sub>/<bat>/ukf_metrics.json
#       results/<sub>/<bat>/ecm_params.csv
#       results/<sub>/<bat>/soc.parquet
# Out:  results/summary.csv           (RMSE/MAE for SOC & voltage, cap error)
#       results/comparison_rmse_mae.png
#       results/comparison_capacity.png

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from config import RESULTS_DIR, RATED_CAP_AH

COLORS = ["#3498db","#e74c3c","#2ecc71","#f39c12",
          "#9b59b6","#1abc9c","#e67e22","#34495e"]


def load_all_results():
    manifest = pd.read_csv(RESULTS_DIR / "manifest.csv")
    manifest = manifest[manifest["status"] == "ok"]
    rows = []

    for _, bat in manifest.iterrows():
        bid, sub = bat.battery_id, bat.subfolder
        out_dir  = RESULTS_DIR / sub / bid

        if not (out_dir / "ukf_metrics.json").exists():
            print(f"  [SKIP] no ukf_metrics for {bid} — run ukf.py first")
            continue

        with open(out_dir / "ukf_metrics.json") as f:
            m = json.load(f)

        soc_df = pd.read_parquet(out_dir / "soc.parquet")
        ecm_df = pd.read_csv(out_dir / "ecm_params.csv") if (out_dir / "ecm_params.csv").exists() else None

        # ── Capacity metrics ──────────────────────────────────────────────────
        caps         = soc_df.groupby("cycle_num")["total_capacity_ah"].first()
        cap_first    = float(caps.iloc[0])
        cap_last     = float(caps.iloc[-1])
        cap_drop     = cap_first - cap_last
        # Capacity prediction error: |measured first-cycle cap - rated cap| / rated cap
        cap_pred_err_pct = abs(cap_first - RATED_CAP_AH) / RATED_CAP_AH * 100

        # ── Voltage RMSE from ECM fit ─────────────────────────────────────────
        # ecm_params.csv stores rmse_fit (V) per cycle from the optimizer
        if ecm_df is not None and "rmse_fit" in ecm_df.columns:
            v_rmse_mv = float(ecm_df["rmse_fit"].mean() * 1000)   # mV
            v_rmse_mv_best  = float(ecm_df["rmse_fit"].min() * 1000)
            v_rmse_mv_worst = float(ecm_df["rmse_fit"].max() * 1000)
        else:
            v_rmse_mv = v_rmse_mv_best = v_rmse_mv_worst = float("nan")

        row = dict(
            battery          = bid,
            subfolder        = sub,
            n_cycles         = int(soc_df["cycle_num"].nunique()),

            # ── Capacity ──────────────────────────────────────────────────────
            cap_first_Ah     = round(cap_first, 4),
            cap_last_Ah      = round(cap_last,  4),
            cap_drop_Ah      = round(cap_drop,  4),
            cap_fade_pct     = round(cap_drop / cap_first * 100, 2),
            cap_pred_err_pct = round(cap_pred_err_pct, 3),   # vs rated capacity

            # ── Voltage (ECM fit quality) ─────────────────────────────────────
            V_RMSE_mV_mean   = round(v_rmse_mv,       3),
            V_RMSE_mV_best   = round(v_rmse_mv_best,  3),
            V_RMSE_mV_worst  = round(v_rmse_mv_worst, 3),

            # ── SoC (UKF estimation quality) ─────────────────────────────────
            SOC_RMSE_pct     = round(m["overall_RMSE"] * 100, 4),
            SOC_MAE_pct      = round(m["overall_MAE"]  * 100, 4),
        )

        # per-cycle SoC RMSE / MAE
        for cyc, pm in m["per_cycle"].items():
            row[f"c{cyc}_soc_rmse_pct"] = round(pm["RMSE"] * 100, 4)
            row[f"c{cyc}_soc_mae_pct"]  = round(pm["MAE"]  * 100, 4)

        rows.append(row)

    return pd.DataFrame(rows)


def plot_soc_metrics(summary):
    bats = summary["battery"].tolist()
    subs = summary["subfolder"].tolist()
    x    = np.arange(len(bats)); w = 0.35
    sub_names = sorted(set(subs))
    sub_color = {s: COLORS[i % len(COLORS)] for i, s in enumerate(sub_names)}
    bar_c     = [sub_color[s] for s in subs]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("UKF SoC Estimation — Cross-Battery Metrics", fontsize=13, fontweight="bold")

    axes[0].bar(x, summary["SOC_RMSE_pct"], w, color=bar_c, alpha=0.85)
    axes[0].set_xticks(x); axes[0].set_xticklabels(bats, rotation=35, ha="right", fontsize=8)
    axes[0].set_ylabel("SoC RMSE [%]"); axes[0].set_title("SoC RMSE per Battery")
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(x, summary["V_RMSE_mV_mean"], w, color=bar_c, alpha=0.85)
    axes[1].set_xticks(x); axes[1].set_xticklabels(bats, rotation=35, ha="right", fontsize=8)
    axes[1].set_ylabel("Voltage RMSE [mV]"); axes[1].set_title("ECM Voltage RMSE per Battery (mean across cycles)")
    axes[1].grid(True, alpha=0.3, axis="y")

    handles = [Line2D([0],[0], color=sub_color[s], lw=8, label=s) for s in sub_names]
    fig.legend(handles=handles, loc="upper right", fontsize=8)
    plt.tight_layout()
    path = RESULTS_DIR / "comparison_rmse_mae.png"
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved → {path}")


def plot_capacity(summary):
    bats = summary["battery"].tolist()
    subs = summary["subfolder"].tolist()
    x    = np.arange(len(bats))
    sub_names = sorted(set(subs))
    sub_color = {s: COLORS[i % len(COLORS)] for i, s in enumerate(sub_names)}
    bar_c     = [sub_color[s] for s in subs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Capacity Analysis", fontsize=13, fontweight="bold")

    ax1.bar(x, summary["cap_first_Ah"], 0.5, color=bar_c, alpha=0.9)
    ax1.bar(x, summary["cap_last_Ah"],  0.5, color=bar_c, alpha=0.4, hatch="//")
    ax1.axhline(RATED_CAP_AH * 0.8, color="red", ls="--", lw=1.5,
                label=f"EOL 80% ({RATED_CAP_AH*0.8:.2f} Ah)")
    ax1.set_xticks(x); ax1.set_xticklabels(bats, rotation=35, ha="right", fontsize=8)
    ax1.set_ylabel("Capacity [Ah]"); ax1.set_title("Capacity Fade\n(solid=first, hatched=last cycle)")
    sub_h = [Line2D([0],[0], color=sub_color[s], lw=8, label=s) for s in sub_names]
    ax1.legend(handles=sub_h + [Line2D([0],[0], color="red", ls="--", lw=1.5, label="EOL 80%")], fontsize=7)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(x, summary["cap_pred_err_pct"], 0.5, color=bar_c, alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(bats, rotation=35, ha="right", fontsize=8)
    ax2.set_ylabel("Error [%]"); ax2.set_title("Capacity Prediction Error\n(|measured − rated| / rated × 100)")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = RESULTS_DIR / "comparison_capacity.png"
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved → {path}")


def print_table(summary):
    print("\n" + "=" * 80)
    print("SUMMARY — KEY METRICS")
    print("=" * 80)
    cols = ["battery","subfolder","n_cycles",
            "cap_first_Ah","cap_last_Ah","cap_fade_pct","cap_pred_err_pct",
            "V_RMSE_mV_mean","SOC_RMSE_pct","SOC_MAE_pct"]
    print(summary[cols].to_string(index=False))
    print("=" * 80)
    print(f"\n  Mean SoC RMSE : {summary['SOC_RMSE_pct'].mean():.4f} %")
    print(f"  Mean V  RMSE  : {summary['V_RMSE_mV_mean'].mean():.3f} mV")
    print(f"  Mean cap error: {summary['cap_pred_err_pct'].mean():.3f} %")


def main():
    print("=" * 60)
    print("STAGE 5 — Summary & Results")
    print("=" * 60)

    summary = load_all_results()
    if summary.empty:
        print("\n[ERROR] No results found. Run all previous stages first.")
        sys.exit(1)

    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)
    print(f"\n  Saved → {RESULTS_DIR / 'summary.csv'}")

    plot_soc_metrics(summary)
    plot_capacity(summary)
    print_table(summary)
    print(f"\nAll outputs in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
