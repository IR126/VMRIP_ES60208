# =============================================================================
# config.py  —  THE ONLY FILE YOU NEED TO EDIT FOR A NEW DATASET
# =============================================================================
#
# TO USE ON A NEW DATASET:
#   1. Change DATASET_ROOT to point at your data folder
#   2. Change BATTERY_SUBFOLDERS to match your subfolder names
#   3. Run: python src/validate.py   ← checks everything before you run
#   4. Run: python run_all.py
#
# Everything else (column names, mode values, pipeline logic) stays the same
# because all datasets share the same schema.
# =============================================================================

from pathlib import Path

# ── CHANGE THESE FOR A NEW DATASET ───────────────────────────────────────────

# Path to the folder that CONTAINS your subfolders
# Example: if your layout is  my_data/group_a/*.csv  and  my_data/group_b/*.csv
#          then set DATASET_ROOT = Path("my_data")
DATASET_ROOT = Path("battery_alt_dataset")

# List every subfolder name that contains battery CSVs
# Remove entries that don't exist — missing folders are skipped with a warning
BATTERY_SUBFOLDERS = [
    "recommissioned_batteries",
    "regular_alt_batteries",
    "second_life_batteries",
]

# ── CHANGE THIS IF YOUR PACK IS DIFFERENT ────────────────────────────────────

PACK_V_MIN   = 5.0    # V   minimum valid terminal voltage (pack-level cutoff)
PACK_V_MAX   = 8.7    # V   maximum valid terminal voltage (pack-level full)
RATED_CAP_AH = 2.5    # Ah  nameplate capacity of one cell (not pack)
NOMINAL_DISCHARGE_CURRENT_A = 2.5   # A   constant CC rate of reference discharges

# ── LEAVE THESE ALONE UNLESS YOU KNOW WHAT YOU'RE DOING ──────────────────────

RESULTS_DIR = Path("results")

# Noise removal
MIN_SEG_ROWS          = 30
MIN_SEG_DURATION_S    = 60
ADC_VOLTAGE_FLOOR     = 0.0
MIN_DISCHARGE_CURRENT = 0.5
SPIKE_WINDOW          = 5
VOLTAGE_SPIKE_THRESH  = 0.5
TEMP_SPIKE_THRESH     = 15.0

# Coulomb counting
MIN_VALID_CAP_AH = 0.5

# OCV–SoC spline fit
SPLINE_SMOOTHING = 0.02 
SOC_BINS = 100           

# ECM optimiser — initial guess and bounds for [R0 (Ω), R1 (Ω), C1 (F)]
ECM_INIT   = [0.01, 0.01, 500.0]
ECM_BOUNDS = [(1e-4, 0.2), (1e-4, 0.2), (10.0, 50000.0)]

# UKF
UKF_DEFAULTS = dict(
    Q_soc=1e-5, Q_vrc=1e-5, R_meas=1e-3,
    P0_soc=0.01, P0_vrc=1e-4,
    alpha_ukf=1e-3, beta_ukf=2.0, kappa_ukf=0.0,
)
OPTUNA_TRIALS = 50
