# =============================================================================
# src/soc_ocv.py  —  Stage 2: Coulomb counting → SoC, OCV–SoC spline fit, plots
# =============================================================================
# Run:  python src/soc_ocv.py
# In:   results/<sub>/<bat>/clean.parquet
# Out:  results/<sub>/<bat>/soc.parquet
#       results/<sub>/<bat>/ocv_lookup.csv
#       results/<sub>/<bat>/ocv_soc_curve.png

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from config import (
    RESULTS_DIR, PACK_V_MIN, PACK_V_MAX,
    MIN_DISCHARGE_CURRENT, MIN_VALID_CAP_AH,
    SOC_BINS, SPLINE_SMOOTHING,
    NOMINAL_DISCHARGE_CURRENT_A,
)


# ── Stage 4: Coulomb counting ─────────────────────────────────────────────────
def compute_soc(df):
    """
    Per reference discharge segment:
      dt  = time diff (clipped to 5 s to handle gaps)
      SoC = 1 - cumAh / totalAh   (1.0 full → 0.0 empty)

    Returns soc_df with columns:
      segment_id, cycle_num, total_capacity_ah, time, SOC, voltage, temperature
    """
    ref = df[df["is_ref_discharge"]].copy()
    records, cycle_num = [], 0

    for seg_id, seg in ref.groupby("segment_id"):
        seg = seg.sort_values("time").reset_index(drop=True)
        valid = (
            (seg["voltage_load"].fillna(0)  > PACK_V_MIN) &
            (seg["current_load"].fillna(0)  > MIN_DISCHARGE_CURRENT)
        )
        seg = seg[valid].reset_index(drop=True)
        if len(seg) < 10:
            continue

        dt       = seg["time"].diff().fillna(1.0).clip(upper=5.0)
        cum_ah   = (seg["current_load"] * dt / 3600.0).cumsum()
        total_ah = float(cum_ah.iloc[-1])
        if total_ah < MIN_VALID_CAP_AH:
            continue

        records.append(pd.DataFrame({
            "segment_id":        seg_id,
            "cycle_num":         cycle_num,
            "total_capacity_ah": total_ah,
            "time":              seg["time"].values,
            "SOC":               (1.0 - cum_ah / total_ah).clip(0.0, 1.0).values,
            "voltage":           seg["voltage_load"].values,
            "temperature":       seg["temperature_battery"].values,
        }))
        cycle_num += 1

    if not records:
        raise RuntimeError("No valid reference discharge segments found.")
    return pd.concat(records, ignore_index=True)


# ── Stage 5: OCV–SoC spline ───────────────────────────────────────────────────
def fit_ocv_soc(soc_df):
    """
    1. Bin SOC into SOC_BINS equal-width bins, take median voltage per bin
    2. Savitzky-Golay smooth
    3. Fit UnivariateSpline
    Returns (ocv_lookup DataFrame, fitted spline callable)
    """
    mask  = soc_df["SOC"].between(0.0, 1.0) & soc_df["voltage"].between(PACK_V_MIN, PACK_V_MAX)
    clean = soc_df[mask].copy()

    clean["soc_bin"] = pd.cut(clean["SOC"], bins=SOC_BINS, labels=False)
    binned = (
        clean.groupby("soc_bin")["voltage"]
        .agg(median="median", count="count")
        .reset_index()
    )
    binned["soc_center"] = (binned["soc_bin"] + 0.5) / SOC_BINS
    binned = binned.dropna()[binned["count"] >= 2]

    win = min(31, (len(binned) // 4) * 2 + 1)
    binned["v_smooth"] = savgol_filter(binned["median"], window_length=win, polyorder=3)

    spline = UnivariateSpline(
        binned["soc_center"], binned["v_smooth"], s=SPLINE_SMOOTHING, k=4
    )

    soc_axis   = np.linspace(0.0, 1.0, 100)
    ocv_lookup = pd.DataFrame({"SOC": soc_axis, "OCV_V": spline(soc_axis)})
    return ocv_lookup, spline, binned


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_ocv_soc(soc_df, ocv_lookup, binned, out_path, battery_id):
    CYCLE_COLORS = [
        "#e74c3c","#3498db","#2ecc71","#f39c12",
        "#9b59b6","#1abc9c","#e67e22","#34495e",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{battery_id} — SoC & OCV–SoC", fontsize=13, fontweight="bold")

    # Left: SoC vs time per cycle
    ax = axes[0]
    for cyc, grp in soc_df.groupby("cycle_num"):
        t = (grp["time"].values - grp["time"].values[0]) / 60.0
        c = CYCLE_COLORS[cyc % len(CYCLE_COLORS)]
        ax.plot(t, grp["SOC"].values, color=c, lw=1.5, label=f"Cycle {cyc}")
    ax.set_xlabel("Time in cycle [min]")
    ax.set_ylabel("SoC (Coulomb counting)")
    ax.set_title("SoC vs Time per Cycle")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Right: OCV–SoC curve
    ax = axes[1]
    ax.scatter(binned["soc_center"], binned["median"],
               s=6, alpha=0.4, color="#95a5a6", label="Binned median")
    ax.plot(ocv_lookup["SOC"], ocv_lookup["OCV_V"],
            color="#e74c3c", lw=2.5, label="Spline fit")
    ax.invert_xaxis()
    ax.set_xlabel("SoC")
    ax.set_ylabel("OCV [V]")
    ax.set_title("OCV–SoC Curve")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("STAGE 2 — Coulomb SoC + OCV–SoC Fit")
    print("=" * 60)

    manifest = pd.read_csv(RESULTS_DIR / "manifest.csv")
    manifest = manifest[manifest["status"] == "ok"]

    for _, bat in manifest.iterrows():
        bid, sub = bat.battery_id, bat.subfolder
        out_dir  = RESULTS_DIR / sub / bid
        parquet  = out_dir / "clean.parquet"

        print(f"\n  {sub}/{bid} …", end=" ", flush=True)
        try:
            df     = pd.read_parquet(parquet)
            soc_df = compute_soc(df)

            n_cyc  = soc_df["cycle_num"].nunique()
            caps   = soc_df.groupby("cycle_num")["total_capacity_ah"].first()
            print(f"{n_cyc} cycles | cap {caps.iloc[0]:.3f} → {caps.iloc[-1]:.3f} Ah")

            ocv_lookup, spline, binned = fit_ocv_soc(soc_df)

            soc_df.to_parquet(out_dir / "soc.parquet", index=False)
            ocv_lookup.to_csv(out_dir / "ocv_lookup.csv", index=False)
            plot_ocv_soc(soc_df, ocv_lookup, binned,
                         out_dir / "ocv_soc_curve.png", bid)
            print(f"    Saved soc.parquet, ocv_lookup.csv, ocv_soc_curve.png")

        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"    FAILED: {e}")

    print(f"\nDone — outputs in {RESULTS_DIR}/<subfolder>/<battery_id>/")


if __name__ == "__main__":
    main()
