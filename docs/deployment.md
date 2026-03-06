# Deployment Instructions

## Quick deployment (offline batch)

```bash
git clone https://github.com/<you>/battery-soc-estimation
cd battery-soc-estimation
pip install -r requirements.txt

# 1. Point at your data
#    Edit config.py: set DATASET_ROOT and BATTERY_SUBFOLDERS

# 2. Validate
python src/validate.py

# 3. Run
python run_all.py --tune --trials 200
```

All outputs land in `results/`. The key files are:

| File | Use |
|------|-----|
| `results/summary.csv` | All metrics for every battery |
| `results/<sub>/<bat>/ukf_params.json` | Best hyperparameters (if --tune used) |
| `results/<sub>/<bat>/ocv_lookup.csv` | OCV–SoC table |
| `results/<sub>/<bat>/ecm_params.csv` | R₀, R₁, C₁, τ per cycle |

---

## Deploying the UKF in real time (embedded / streaming)

The filter itself has no file-system dependencies at inference time. Minimum code to run it on a new session:

```python
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from src.ukf import build_ocv_grid, run_ukf_segment
import json

# Load pre-computed artefacts for this battery
ocv_lookup = pd.read_csv("results/<sub>/<bat>/ocv_lookup.csv")
ecm_df     = pd.read_csv("results/<sub>/<bat>/ecm_params.csv")
with open("results/<sub>/<bat>/ukf_params.json") as f:
    params = json.load(f)     # omit if using UKF_DEFAULTS from config.py

# Build OCV grid (run once at startup)
build_ocv_grid(ocv_lookup)

# Pick ECM params for the relevant cycle (or use cycle 0 as default)
row = ecm_df.iloc[0]
R0, R1, tau = row["R0"], row["R1"], row["tau"]

# Run on a discharge segment
result = run_ukf_segment(
    time_arr    = time_array,      # shape (N,), seconds
    current_arr = current_array,   # shape (N,), amps
    voltage_arr = voltage_array,   # shape (N,), volts
    cap_ah      = 2.5,             # measured or rated capacity
    R0=R0, R1=R1, tau=tau,
    **params,
)

soc_estimate = result["soc"]   # shape (N,), values 0–1
```

**Per-timestep latency:** < 0.5 ms on a standard CPU. No GPU needed.

---

## Adapting to new hardware

Only `config.py` needs changes:

```python
# New pack
PACK_V_MIN   = 3.0    # new lower cutoff
PACK_V_MAX   = 4.2    # new upper limit  (single cell example)
RATED_CAP_AH = 3.0    # new nameplate capacity
NOMINAL_DISCHARGE_CURRENT_A = 1.5   # new CC reference rate
```

Then re-run `src/validate.py` → `run_all.py`. The ECM fitting and UKF will automatically re-train on the new data — no code changes.

---

## Minimum dataset requirements for a new battery

| Requirement | Value |
|-------------|-------|
| Columns present | `time, mode, mission_type, voltage_load, current_load, temperature_battery` |
| Reference discharge segments | ≥ 1 per battery (`mode=-1, mission_type=0`) |
| Minimum segment length | 30 rows, ≥ 60 seconds |
| Minimum discharged capacity | ≥ 0.5 Ah per segment |

Run `python src/validate.py` — it checks all of these and reports exactly what's missing.
