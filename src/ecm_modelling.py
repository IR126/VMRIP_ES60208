# =============================================================================
# src/ecm_modelling.py  —  Stage 3: fit 1RC ECM parameters from data
# =============================================================================
# Run:  python src/ecm_modelling.py
# In:   results/<sub>/<bat>/soc.parquet + ocv_lookup.csv
# Out:  results/<sub>/<bat>/ecm_params.csv
#       results/<sub>/<bat>/ecm_fit.png
#
# ECM equations (from teammate's Marimo model):
#   v[k]      = OCV(z[k]) - R1*i_R1[k] - R0*I[k]
#   z[k+1]    = z[k] - (dt/Q)*I[k]
#   i_R1[k+1] = exp(-dt/τ)*i_R1[k] + (1-exp(-dt/τ))*I[k]    τ = R1*C1
#
# VECTORIZED: reference discharges are constant CC, so z and i_R1 are
# solved analytically with cumsum / cumprod — zero Python loops in the
# forward simulator. This makes the optimizer ~30-50x faster.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from config import (
    RESULTS_DIR, NOMINAL_DISCHARGE_CURRENT_A,
    ECM_INIT, ECM_BOUNDS,
)


# ── Vectorized ECM forward simulator ─────────────────────────────────────────
def simulate_ecm_vectorized(time_arr, I, soc0, cap_ah, R0, R1, C1, ocv_fn):
    """
    Constant-current forward simulation of the 1RC ECM.
    Because I is constant, z and i_R1 recurrences have closed-form solutions
    that are computed with cumsum/cumprod — no Python loop.

    z[k]    = soc0 - I * cumsum(dt) / Q_As
    i_R1[k] = I * (1 - prod(α[0:k]))     where α[k] = exp(-dt[k]/τ)

    Terminal voltage:
    v[k] = OCV(z[k]) - R1*i_R1[k] - R0*I
    """
    dt   = np.diff(time_arr, prepend=time_arr[0])
    dt   = np.clip(dt, 0.0, 5.0)
    tau  = R1 * C1
    QAs  = cap_ah * 3600.0

    # SoC: exact cumulative integral
    z = np.clip(soc0 - I * np.cumsum(dt) / QAs, 0.0, 1.0)

    # i_R1: product of alpha chain  →  1 - cumprod
    alpha      = np.exp(-dt / tau)
    alpha[0]   = 1.0          # first step: no decay yet
    cumprod_a  = np.cumprod(alpha)
    i_R1       = I * (1.0 - cumprod_a)

    v = ocv_fn(z) - R1 * i_R1 - R0 * I
    return v


# ── Per-cycle fitting ─────────────────────────────────────────────────────────
def fit_ecm_cycle(grp, ocv_fn):
    """
    Minimise RMSE(v_measured, v_simulated) over [R0, R1, C1].
    Uses the vectorized simulator — no inner Python loop.
    """
    t    = grp["time"].values.astype(np.float64)
    v    = grp["voltage"].values.astype(np.float64)
    soc0 = float(grp["SOC"].iloc[0])
    cap  = float(grp["total_capacity_ah"].iloc[0])
    I    = NOMINAL_DISCHARGE_CURRENT_A   # scalar — constant CC

    def objective(p):
        R0, R1, C1 = p
        v_sim = simulate_ecm_vectorized(t, I, soc0, cap, R0, R1, C1, ocv_fn)
        return float(np.sqrt(np.mean((v - v_sim) ** 2)))

    res = minimize(objective, x0=ECM_INIT, bounds=ECM_BOUNDS,
                   method="L-BFGS-B", options={"maxiter": 400, "ftol": 1e-10})
    R0, R1, C1 = res.x
    return dict(R0=R0, R1=R1, C1=C1, tau=R1*C1, rmse_fit=res.fun, success=res.success)


# ── Plot ECM fit quality ──────────────────────────────────────────────────────
def plot_ecm_fit(soc_df, ecm_params, ocv_fn, out_path, battery_id):
    COLORS = ["#e74c3c","#3498db","#2ecc71","#f39c12",
               "#9b59b6","#1abc9c","#e67e22","#34495e"]
    n_cyc = soc_df["cycle_num"].nunique()
    ncols = min(n_cyc, 4)
    nrows = (n_cyc + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    fig.suptitle(f"{battery_id} — ECM Fit per Cycle", fontsize=12, fontweight="bold")

    for cyc, grp in soc_df.groupby("cycle_num"):
        grp  = grp.sort_values("time").reset_index(drop=True)
        p    = ecm_params[cyc]
        t    = grp["time"].values
        v_m  = grp["voltage"].values
        v_s  = simulate_ecm_vectorized(
            t, NOMINAL_DISCHARGE_CURRENT_A,
            float(grp["SOC"].iloc[0]), float(grp["total_capacity_ah"].iloc[0]),
            p["R0"], p["R1"], p["C1"], ocv_fn,
        )
        ax   = axes[cyc // ncols][cyc % ncols]
        t_m  = (t - t[0]) / 60.0
        c    = COLORS[cyc % len(COLORS)]
        ax.plot(t_m, v_m, color=c,       lw=1.8, label="Measured")
        ax.plot(t_m, v_s, color="black", lw=1.2, ls="--", label="ECM fit")
        ax.set_title(f"Cycle {cyc}  RMSE={p['rmse_fit']*1000:.2f} mV", fontsize=9)
        ax.set_xlabel("t [min]"); ax.set_ylabel("V [V]")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # hide unused subplots
    for idx in range(n_cyc, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("STAGE 3 — ECM Parameter Fitting")
    print("=" * 60)

    manifest = pd.read_csv(RESULTS_DIR / "manifest.csv")
    manifest = manifest[manifest["status"] == "ok"]

    for _, bat in manifest.iterrows():
        bid, sub = bat.battery_id, bat.subfolder
        out_dir  = RESULTS_DIR / sub / bid

        print(f"\n  {sub}/{bid} …")
        try:
            soc_df     = pd.read_parquet(out_dir / "soc.parquet")
            ocv_lookup = pd.read_csv(out_dir / "ocv_lookup.csv")
            ocv_fn     = interp1d(ocv_lookup["SOC"], ocv_lookup["OCV_V"],
                                   fill_value="extrapolate")

            ecm_params = {}
            rows       = []
            for cyc, grp in soc_df.groupby("cycle_num", sort=True):
                grp = grp.sort_values("time").reset_index(drop=True)
                p   = fit_ecm_cycle(grp, ocv_fn)
                ecm_params[int(cyc)] = p
                print(f"    Cycle {cyc:2d}  R0={p['R0']*1000:.2f}mΩ  "
                      f"R1={p['R1']*1000:.2f}mΩ  C1={p['C1']:.0f}F  "
                      f"τ={p['tau']:.1f}s  RMSE={p['rmse_fit']*1000:.2f}mV")
                rows.append(dict(cycle=cyc, **{k: round(v, 8) for k, v in p.items()}))

            pd.DataFrame(rows).to_csv(out_dir / "ecm_params.csv", index=False)
            plot_ecm_fit(soc_df, ecm_params, ocv_fn,
                         out_dir / "ecm_fit.png", bid)
            print(f"    Saved ecm_params.csv, ecm_fit.png")

        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"    FAILED: {e}")

    print(f"\nDone — outputs in {RESULTS_DIR}/<subfolder>/<battery_id>/")


if __name__ == "__main__":
    main()
