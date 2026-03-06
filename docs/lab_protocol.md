# OCV–SoC Lab Protocol
## Samsung INR18650-25R  |  2S Series Pack

This describes the measurement procedure used to generate the OCV–SoC lookup table from field discharge data. No special lab equipment is required — the protocol extracts OCV from standard reference discharge cycles already present in the dataset.

---

## What counts as a "reference discharge"

A reference discharge segment is identified automatically from the CSV by two flags:

| Column | Value | Meaning |
|--------|-------|---------|
| `mode` | `-1` | Discharge mode active |
| `mission_type` | `0` | Reference (not mission) cycle |

These segments are constant-current discharges at **2.5 A** from full charge (~8.4 V) to cutoff (~5.0 V). Terminal voltage at this rate serves as the OCV proxy (ohmic drop = R₀ × 2.5 A ≈ 125 mV, acceptable for a lookup table).

---

## Step-by-step extraction (automated in `src/soc_ocv.py`)

### 1. Select valid rows
- `voltage_load > 5.0 V` (above cutoff)
- `current_load > 0.5 A` (real discharge load present)

### 2. Coulomb counting → SoC
```
dt     = time[k+1] - time[k]   (clipped to 5 s to handle gaps)
cum_Ah = Σ (I × dt / 3600)
SoC[k] = 1 - cum_Ah[k] / total_Ah    (1.0 = full, 0.0 = empty)
```

### 3. Bin and smooth
- Divide SoC ∈ [0, 1] into **200 equal bins**
- Compute **median** voltage per bin (robust to noise)
- Apply **Savitzky-Golay filter** (window = 31, polyorder = 3)

### 4. Fit spline
- `scipy.interpolate.UnivariateSpline` with `k=4`, `s=0.02`
- Export 100-point table to `ocv_lookup.csv`

---

## Output file: `ocv_lookup.csv`

| Column | Unit | Description |
|--------|------|-------------|
| `SOC`  | —    | 0.00 to 1.00 in steps of 0.01 |
| `OCV_V`| V    | Open-circuit voltage at that SoC |

---

## If you are doing a fresh lab characterisation

For a new cell chemistry or pack configuration where no reference discharge data exists yet:

1. **Fully charge** the pack to the upper voltage limit using the standard CC-CV protocol.
2. **Discharge at C/20** (0.125 A for a 2.5 Ah cell) to the lower cutoff. This rate is slow enough that terminal voltage ≈ true OCV throughout.
3. Log `time`, `voltage`, `current` at ≥ 1 Hz.
4. Feed the resulting CSV through `src/soc_ocv.py` — the column schema is identical to the field data.
5. Update `PACK_V_MIN`, `PACK_V_MAX`, `RATED_CAP_AH`, and `NOMINAL_DISCHARGE_CURRENT_A` in `config.py` to match the new pack.

---

## Sanity checks after fitting

| Check | Expected |
|-------|----------|
| OCV at SoC = 1.0 | ≈ PACK_V_MAX (8.4 V for 2S) |
| OCV at SoC = 0.0 | ≈ PACK_V_MIN (5.0 V for 2S) |
| Monotonicity | OCV must be strictly increasing with SoC |
| dOCV/dSoC | Should be smooth — no sharp kinks |

`src/validate.py` checks the voltage range automatically before the pipeline runs.
