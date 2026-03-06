# =============================================================================
# src/validate.py  —  Check your dataset before running the pipeline
# =============================================================================
# Run:  python src/validate.py
#
# Checks:
#   1. DATASET_ROOT exists and subfolders are found
#   2. CSVs can be read and have the required columns
#   3. Reference discharge segments exist (mode=-1, mission_type=0)
#   4. Enough valid rows per segment for Coulomb counting
#   5. Prints a per-file summary so you know what the pipeline will see

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from config import (
    DATASET_ROOT, BATTERY_SUBFOLDERS,
    PACK_V_MIN, PACK_V_MAX,
    MIN_DISCHARGE_CURRENT, MIN_VALID_CAP_AH,
    MIN_SEG_ROWS, MIN_SEG_DURATION_S,
    NOMINAL_DISCHARGE_CURRENT_A,
)

REQUIRED_COLUMNS = {
    "time", "mode", "mission_type",
    "voltage_load", "current_load", "temperature_battery",
}

PASS = "  ✓"
WARN = "  ⚠"
FAIL = "  ✗"


def check_dataset():
    errors   = []
    warnings = []

    print("=" * 60)
    print("  Dataset Validation")
    print("=" * 60)

    # ── 1. Root folder ────────────────────────────────────────────────────────
    print(f"\n[1] Dataset root: {DATASET_ROOT}")
    if not DATASET_ROOT.exists():
        print(f"{FAIL} NOT FOUND: {DATASET_ROOT.resolve()}")
        print(f"\n  → Edit DATASET_ROOT in config.py to point at your data folder.")
        errors.append("DATASET_ROOT not found")
        return errors, warnings
    print(f"{PASS} Exists")

    # ── 2. Subfolders ─────────────────────────────────────────────────────────
    print(f"\n[2] Subfolders in BATTERY_SUBFOLDERS:")
    found_any = False
    csv_files = []
    for sub in BATTERY_SUBFOLDERS:
        folder = DATASET_ROOT / sub
        if not folder.exists():
            print(f"{WARN} Not found (will be skipped): {folder}")
            warnings.append(f"Subfolder missing: {sub}")
        else:
            csvs = [f for f in folder.glob("*.csv")
                    if not f.stem.lower().startswith("readme")]
            print(f"{PASS} {sub}  ({len(csvs)} CSV files)")
            csv_files.extend(csvs)
            found_any = True

    if not found_any:
        print(f"{FAIL} No subfolders found at all.")
        print(f"\n  → Edit BATTERY_SUBFOLDERS in config.py to match your folder names.")
        errors.append("No subfolders found")
        return errors, warnings

    if not csv_files:
        print(f"{FAIL} Subfolders exist but contain no CSV files.")
        errors.append("No CSV files found")
        return errors, warnings

    print(f"\n  Total: {len(csv_files)} CSV file(s) to process")

    # ── 3. Per-file checks ────────────────────────────────────────────────────
    print(f"\n[3] Per-file schema & segment check:")
    print(f"  {'File':<35} {'Rows':>8} {'Ref segs':>10} {'Est cycles':>12} {'Status'}")
    print("  " + "─" * 75)

    file_summaries = []
    for csv_path in sorted(csv_files):
        bid  = csv_path.stem
        sub  = csv_path.parent.name
        diag = []

        try:
            df = pd.read_csv(csv_path, low_memory=False, nrows=None)
            df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

            # Column check
            missing = REQUIRED_COLUMNS - set(df.columns)
            if missing:
                diag.append(f"MISSING COLUMNS: {missing}")

            # Numeric cast
            for col in REQUIRED_COLUMNS:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Reference discharge segments
            has_ref = ("mode" in df.columns and "mission_type" in df.columns)
            if has_ref:
                ref = df[(df["mode"] == -1) & (df["mission_type"] == 0)]
                if ref.empty:
                    diag.append("NO reference discharge rows (mode=-1, mission_type=0)")
            else:
                ref = pd.DataFrame()

            # Count segments that are long enough
            n_valid_segs = 0
            if not ref.empty:
                df["_is_ref"] = (df["mode"] == -1) & (df["mission_type"] == 0)
                df["_seg"]    = (df["_is_ref"] & ~df["_is_ref"].shift(fill_value=False)).cumsum()
                df.loc[~df["_is_ref"], "_seg"] = -1
                for _, seg in df[df["_is_ref"]].groupby("_seg"):
                    dur = seg["time"].iloc[-1] - seg["time"].iloc[0] if "time" in seg else 0
                    valid = (
                        (seg["voltage_load"].fillna(0) > PACK_V_MIN) &
                        (seg["current_load"].fillna(0) > MIN_DISCHARGE_CURRENT)
                    )
                    seg_valid = seg[valid]
                    if len(seg) >= MIN_SEG_ROWS and dur >= MIN_SEG_DURATION_S:
                        dt = seg_valid["time"].diff().fillna(1.0).clip(upper=5.0)
                        cap = float((seg_valid["current_load"].fillna(0) * dt / 3600.0).sum())
                        if cap >= MIN_VALID_CAP_AH:
                            n_valid_segs += 1

            if n_valid_segs == 0 and has_ref:
                diag.append("0 valid discharge cycles after filtering")

            # Voltage range sanity
            if "voltage_load" in df.columns:
                v = df["voltage_load"].dropna()
                if not v.empty:
                    vmin, vmax = v.min(), v.max()
                    if vmax < PACK_V_MIN:
                        diag.append(f"voltage_load max={vmax:.2f}V < PACK_V_MIN={PACK_V_MIN}V — check PACK_V_MIN in config.py")
                    if vmin > PACK_V_MAX:
                        diag.append(f"voltage_load min={vmin:.2f}V > PACK_V_MAX={PACK_V_MAX}V — check PACK_V_MAX in config.py")

            status = "FAIL" if any("MISSING" in d or "NO ref" in d or "0 valid" in d
                                    or "check PACK" in d for d in diag) else \
                     "WARN" if diag else "OK"

            symbol = {"OK": PASS, "WARN": WARN, "FAIL": FAIL}[status]
            label  = f"[{sub}] {bid}"
            print(f"  {label:<35} {len(df):>8,} {len(ref):>10,} {n_valid_segs:>12}   {symbol.strip()}")

            for d in diag:
                print(f"       → {d}")
                if status == "FAIL":
                    errors.append(f"{bid}: {d}")
                else:
                    warnings.append(f"{bid}: {d}")

            file_summaries.append(dict(
                battery_id=bid, subfolder=sub, status=status,
                n_rows=len(df), n_valid_cycles=n_valid_segs,
            ))

        except Exception as e:
            print(f"  {bid:<35} {'':>8} {'':>10} {'':>12}   {FAIL.strip()}  (read error: {e})")
            errors.append(f"{bid}: could not read file — {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    ok   = sum(1 for f in file_summaries if f["status"] == "OK")
    warn = sum(1 for f in file_summaries if f["status"] == "WARN")
    fail = sum(1 for f in file_summaries if f["status"] == "FAIL")

    print(f"\n{'=' * 60}")
    print(f"  Result:  {ok} OK   {warn} warnings   {fail} errors")

    if errors:
        print(f"\n  Pipeline will NOT run until errors are fixed:")
        for e in errors:
            print(f"    • {e}")
    elif warnings:
        print(f"\n  Pipeline will run. Warnings to be aware of:")
        for w in warnings:
            print(f"    • {w}")
    else:
        print(f"\n  All checks passed. Run:  python run_all.py")

    print("=" * 60)
    return errors, warnings


if __name__ == "__main__":
    errors, _ = check_dataset()
    sys.exit(1 if errors else 0)
