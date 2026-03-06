# =============================================================================
# src/load_data.py  —  Stage 1: discover CSVs, load, label, denoise
# =============================================================================
# Run:  python src/load_data.py
# Out:  results/<subfolder>/<battery_id>/clean.parquet
#       results/manifest.csv

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from config import (
    DATASET_ROOT, BATTERY_SUBFOLDERS, RESULTS_DIR,
    MIN_SEG_ROWS, MIN_SEG_DURATION_S, ADC_VOLTAGE_FLOOR,
    MIN_DISCHARGE_CURRENT, SPIKE_WINDOW,
    VOLTAGE_SPIKE_THRESH, TEMP_SPIKE_THRESH,
)


# ── Discovery ─────────────────────────────────────────────────────────────────
def discover_batteries():
    manifest = []
    for sub in BATTERY_SUBFOLDERS:
        folder = DATASET_ROOT / sub
        if not folder.exists():
            print(f"  [SKIP] not found: {folder}")
            continue
        for csv_path in sorted(folder.glob("*.csv")):
            if csv_path.stem.lower().startswith("readme"):
                continue
            manifest.append(dict(
                battery_id = csv_path.stem,
                subfolder  = sub,
                csv_path   = str(csv_path),
            ))
    return pd.DataFrame(manifest)


# ── Stage 1: load ─────────────────────────────────────────────────────────────
def load_raw(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["start_time"], low_memory=False)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    num_cols = ["time","mode","voltage_charger","temperature_battery",
                "voltage_load","current_load","temperature_mosfet",
                "temperature_resistor","mission_type"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("time").reset_index(drop=True)


# ── Stage 2: label reference discharge segments ───────────────────────────────
def label_segments(df):
    df = df.copy()
    is_ref = (df["mode"] == -1) & (df["mission_type"] == 0)
    df["is_ref_discharge"] = is_ref
    block_start = is_ref & ~is_ref.shift(fill_value=False)
    df["segment_id"] = block_start.cumsum()
    df.loc[~is_ref, "segment_id"] = -1
    return df


# ── Stage 3: noise removal ────────────────────────────────────────────────────
def remove_noise(df):
    df = df.copy()

    # 1 — phantom segments
    ref = df[df["is_ref_discharge"]]
    phantom = {
        sid for sid, seg in ref.groupby("segment_id")
        if (len(seg) < MIN_SEG_ROWS or
            seg["time"].iloc[-1] - seg["time"].iloc[0] < MIN_SEG_DURATION_S)
    }
    df = df[~df["segment_id"].isin(phantom)].reset_index(drop=True)

    # 2 — ADC floor
    mask = (df["mode"] == -1) & (df["voltage_load"].fillna(-1) <= ADC_VOLTAGE_FLOOR)
    df.loc[mask, "voltage_load"] = np.nan

    # 3 — sub-threshold current
    mask = (df["mode"] == -1) & (df["current_load"].fillna(0) < MIN_DISCHARGE_CURRENT)
    df.loc[mask, "current_load"] = np.nan

    # 4 — voltage spikes
    v_med = df["voltage_load"].rolling(2*SPIKE_WINDOW+1, center=True, min_periods=1).median()
    df.loc[df["voltage_load"].notna() & ((df["voltage_load"]-v_med).abs() > VOLTAGE_SPIKE_THRESH),
           "voltage_load"] = np.nan

    # 5 — temperature spikes
    t_med = df["temperature_battery"].rolling(2*SPIKE_WINDOW+1, center=True, min_periods=1).median()
    df.loc[df["temperature_battery"].notna() & ((df["temperature_battery"]-t_med).abs() > TEMP_SPIKE_THRESH),
           "temperature_battery"] = np.nan

    # 6 — charge-mode bleed
    df.loc[(df["mode"] == 1) & (df["voltage_load"].fillna(0) > 0),
           ["voltage_load", "current_load"]] = np.nan

    for col in ["voltage_load", "current_load", "temperature_battery", "voltage_charger"]:
        if col in df.columns:
            df[col] = df[col].interpolate(method="linear", limit=10, limit_direction="both")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("STAGE 1 — Load, Label, Denoise")
    print("=" * 60)

    manifest = discover_batteries()
    if manifest.empty:
        print(f"\n[ERROR] No CSVs found under {DATASET_ROOT}")
        print("  Check DATASET_ROOT in config.py")
        sys.exit(1)

    print(f"\nFound {len(manifest)} battery file(s):")
    for _, row in manifest.iterrows():
        print(f"  [{row.subfolder}]  {row.battery_id}")

    rows = []
    for _, bat in manifest.iterrows():
        bid, sub = bat.battery_id, bat.subfolder
        out_dir = RESULTS_DIR / sub / bid
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  Processing {sub}/{bid} …", end=" ", flush=True)
        try:
            df = load_raw(bat.csv_path)
            df = label_segments(df)
            df = remove_noise(df)
            df.to_parquet(out_dir / "clean.parquet", index=False)

            n_ref = df["is_ref_discharge"].sum()
            print(f"OK  ({len(df):,} rows, {n_ref:,} ref-discharge rows)")
            rows.append(dict(battery_id=bid, subfolder=sub,
                             total_rows=len(df), ref_rows=n_ref, status="ok"))
        except Exception as e:
            print(f"FAILED: {e}")
            rows.append(dict(battery_id=bid, subfolder=sub, status="error", error=str(e)))

    # Save manifest with status
    manifest_out = pd.DataFrame(rows)
    manifest_out.to_csv(RESULTS_DIR / "manifest.csv", index=False)
    print(f"\nManifest saved → {RESULTS_DIR / 'manifest.csv'}")
    print(f"Clean parquets  → {RESULTS_DIR}/<subfolder>/<battery_id>/clean.parquet")


if __name__ == "__main__":
    main()
