# =============================================================================
# src/ukf.py  —  Stage 4: UKF SoC estimation + Optuna hyperparameter tuning
# =============================================================================
# Run:  python src/ukf.py                 # default hyperparameters
#       python src/ukf.py --tune          # + Optuna search
#       python src/ukf.py --tune --trials 300
#
# In:   results/<sub>/<bat>/soc.parquet
#       results/<sub>/<bat>/ocv_lookup.csv
#       results/<sub>/<bat>/ecm_params.csv
# Out:  results/<sub>/<bat>/ukf_results.parquet
#       results/<sub>/<bat>/ukf_params.json
#       results/<sub>/<bat>/ukf_soc.png
#       results/<sub>/<bat>/ukf_diagnostics.png

import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
from config import (
    RESULTS_DIR, NOMINAL_DISCHARGE_CURRENT_A,
    UKF_DEFAULTS, OPTUNA_TRIALS,
)


# ── OCV grid (rebuilt per battery for fast np.interp in filter loop) ──────────
_SOC_GRID = None
_OCV_GRID = None
_DOCV_GRID = None

def build_ocv_grid(ocv_lookup):
    global _SOC_GRID, _OCV_GRID, _DOCV_GRID
    _SOC_GRID  = np.linspace(0.0, 1.0, 2000)
    fn         = interp1d(ocv_lookup["SOC"], ocv_lookup["OCV_V"], fill_value="extrapolate")
    _OCV_GRID  = fn(_SOC_GRID).astype(np.float64)
    _DOCV_GRID = np.gradient(_OCV_GRID, _SOC_GRID).astype(np.float64)

def _ocv(soc):       return np.interp(soc, _SOC_GRID, _OCV_GRID)
def _docv_dsoc(soc): return np.interp(soc, _SOC_GRID, _DOCV_GRID)


# ── PD repair ─────────────────────────────────────────────────────────────────
def _nearest_pd(P, min_eig=1e-8):
    P = 0.5 * (P + P.T)
    w, v = np.linalg.eigh(P)
    return v @ np.diag(np.maximum(w, min_eig)) @ v.T


# ── UKF segment (VECTORIZED sigma point ops) ──────────────────────────────────
def run_ukf_segment(time_arr, current_arr, voltage_arr, cap_ah,
                    R0, R1, tau,
                    Q_soc=1e-5, Q_vrc=1e-5, R_meas=1e-3,
                    P0_soc=0.01, P0_vrc=1e-4,
                    alpha_ukf=1e-3, beta_ukf=2.0, kappa_ukf=0.0,
                    soc0=1.0):
    """
    Unscented Kalman Filter — state x = [SOC, V_RC]

    Sigma point ops (generation + process propagation + observation) are
    written as matrix ops — the inner 'for j in range(5)' loops are gone.

    Key vectorizations vs the old version:
      • Sigma column generation: x ± sq * S_chol done with broadcasting
      • Process update: applied element-wise across all 5 columns at once
      • Observation: np.interp called once on the 5 sigma SOCs (not in a loop)
    """
    N   = len(time_arr)
    n   = 2
    QAs = cap_ah * 3600.0

    # Pre-compute dt array and RC decay array for the whole segment
    dt_arr    = np.empty(N);  dt_arr[0] = 1.0;  dt_arr[1:] = np.diff(time_arr)
    dt_arr    = np.clip(dt_arr, 0.0, 5.0)
    arc_arr   = np.where(tau > 1e-9, np.exp(-dt_arr / tau), 0.0)  # shape (N,)

    # Weights
    lam = alpha_ukf**2 * (n + kappa_ukf) - n
    sc  = n + lam
    Wm  = np.full(2*n+1, 1.0 / (2.0 * sc));  Wm[0] = lam / sc
    Wc  = Wm.copy();  Wc[0] = lam / sc + (1.0 - alpha_ukf**2 + beta_ukf)
    sq  = np.sqrt(abs(sc))

    Q = np.diag([Q_soc, Q_vrc])

    # Output arrays
    soc_e = np.empty(N); vrc_e = np.empty(N)
    p_s   = np.empty(N); p_v   = np.empty(N)
    inn   = np.empty(N); nees  = np.empty(N)

    x = np.array([soc0, 0.0])
    P = _nearest_pd(np.diag([P0_soc, P0_vrc]))
    soc_e[0]=x[0]; vrc_e[0]=x[1]; p_s[0]=P[0,0]; p_v[0]=P[1,1]; inn[0]=0; nees[0]=0

    for k in range(1, N):
        dt  = dt_arr[k]
        arc = arc_arr[k]
        Ip  = current_arr[k-1]
        Ik  = current_arr[k]

        # ── Generate sigma points (vectorized) ────────────────────────────────
        P       = _nearest_pd(P)
        SC      = np.linalg.cholesky(P)           # (2,2)
        cols    = sq * SC                          # (2,2) — two offset columns
        # X: (2, 5)   center | +cols | -cols
        X       = np.hstack([x[:,None],
                              x[:,None] + cols,
                              x[:,None] - cols])   # (2, 5)
        X[0,:]  = np.clip(X[0,:], 0.0, 1.0)

        # ── Process propagation (vectorized over 5 sigma columns) ─────────────
        Xp      = np.empty_like(X)
        Xp[0,:] = np.clip(X[0,:] - Ip * dt / QAs, 0.0, 1.0)
        Xp[1,:] = arc * X[1,:] + R1 * (1.0 - arc) * Ip

        xp      = Xp @ Wm
        xp[0]   = np.clip(xp[0], 0.0, 1.0)
        dXp     = Xp - xp[:,None]                 # (2, 5)
        Pp      = _nearest_pd((Wc * dXp) @ dXp.T + Q)

        # ── Re-sigma from predicted covariance ───────────────────────────────
        SC2     = np.linalg.cholesky(Pp)
        cols2   = sq * SC2
        X2      = np.hstack([xp[:,None],
                              xp[:,None] + cols2,
                              xp[:,None] - cols2])  # (2, 5)
        X2[0,:] = np.clip(X2[0,:], 0.0, 1.0)

        # ── Observation sigma points (single np.interp call on 5 SOCs) ───────
        Z       = _ocv(X2[0,:]) - Ik * R0 - X2[1,:]   # (5,)
        zp      = float(Wm @ Z)
        dZ      = Z - zp                                # (5,)
        Szz     = float(Wc @ (dZ * dZ)) + R_meas
        Pxz     = (Wc * (X2 - xp[:,None])) @ dZ        # (2,)
        Kg      = Pxz / Szz

        y       = float(voltage_arr[k]) - zp
        x       = xp + Kg * y;  x[0] = np.clip(x[0], 0.0, 1.0)
        P       = _nearest_pd(Pp - Szz * np.outer(Kg, Kg))

        soc_e[k]=x[0]; vrc_e[k]=x[1]; p_s[k]=P[0,0]; p_v[k]=P[1,1]
        inn[k]=y;       nees[k]=(y**2) / max(Szz, 1e-12)

    return dict(soc=soc_e, vrc=vrc_e, p_soc=p_s, p_vrc=p_v, innov=inn, nees=nees)


# ── Run UKF over all cycles ───────────────────────────────────────────────────
def run_ukf_all_cycles(soc_df, ecm_df, **params):
    ecm = ecm_df.set_index("cycle").to_dict("index")
    max_cyc = max(ecm.keys())

    results = []
    for cyc, grp in soc_df.groupby("cycle_num", sort=True):
        grp    = grp.sort_values("time").reset_index(drop=True)
        p      = ecm[min(int(cyc), max_cyc)]
        R0, R1, tau = p["R0"], p["R1"], p["tau"]
        I_arr  = np.full(len(grp), NOMINAL_DISCHARGE_CURRENT_A)

        r = run_ukf_segment(
            grp["time"].values, I_arr, grp["voltage"].values,
            float(grp["total_capacity_ah"].iloc[0]),
            R0, R1, tau, **params,
        )
        out             = grp.copy()
        out["soc_ukf"]  = r["soc"];    out["p_soc"]  = r["p_soc"]
        out["innov"]    = r["innov"];  out["nees"]   = r["nees"]
        out["soc_err"]  = out["soc_ukf"] - out["SOC"]
        results.append(out)

    return pd.concat(results, ignore_index=True)


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(df):
    per = {}
    for cyc, g in df.groupby("cycle_num"):
        e = g["soc_err"].values
        per[int(cyc)] = dict(
            RMSE  = float(np.sqrt(np.mean(e**2))),
            MAE   = float(np.mean(np.abs(e))),
            MaxAE = float(np.max(np.abs(e))),
        )
    e_all = df["soc_err"].values
    return dict(
        overall_RMSE = float(np.sqrt(np.mean(e_all**2))),
        overall_MAE  = float(np.mean(np.abs(e_all))),
        per_cycle    = per,
    )


# ── Optuna tuning ─────────────────────────────────────────────────────────────
def tune_hyperparameters(soc_df, ecm_df, n_trials):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = dict(
            Q_soc     = trial.suggest_float("Q_soc",     1e-8, 1e-2, log=True),
            Q_vrc     = trial.suggest_float("Q_vrc",     1e-8, 1e-2, log=True),
            R_meas    = trial.suggest_float("R_meas",    1e-5, 1e-1, log=True),
            P0_soc    = trial.suggest_float("P0_soc",    1e-4, 0.25, log=True),
            P0_vrc    = trial.suggest_float("P0_vrc",    1e-6, 1e-2, log=True),
            alpha_ukf = trial.suggest_float("alpha_ukf", 1e-4, 1.0,  log=True),
            beta_ukf  = trial.suggest_float("beta_ukf",  0.0,  4.0),
            kappa_ukf = trial.suggest_float("kappa_ukf", 0.0,  3.0),
        )
        df_ = run_ukf_all_cycles(soc_df, ecm_df, **params)
        return float(df_.groupby("cycle_num")["soc_err"]
                       .apply(lambda e: np.sqrt(np.mean(e**2))).mean())

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params, study.best_value


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_soc(df, out_path, battery_id):
    COLORS = ["#e74c3c","#3498db","#2ecc71","#f39c12",
              "#9b59b6","#1abc9c","#e67e22","#34495e"]
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(f"{battery_id} — UKF SoC Estimation", fontsize=12, fontweight="bold")

    for cyc, grp in df.groupby("cycle_num"):
        grp = grp.sort_values("time")
        t   = (grp["time"].values - grp["time"].values[0]) / 60.0
        c   = COLORS[cyc % len(COLORS)]
        ax.plot(t, grp["SOC"].values,     color=c, lw=1.5,  alpha=0.9)
        ax.plot(t, grp["soc_ukf"].values, color=c, lw=1.5,  ls="--", alpha=0.85)

    ax.set_xlabel("Time in cycle [min]"); ax.set_ylabel("SoC")
    ax.grid(True, alpha=0.3)
    n_cyc = df["cycle_num"].nunique()
    ch = [Line2D([0],[0], color=COLORS[i%len(COLORS)], lw=2, label=f"Cycle {i}")
          for i in range(n_cyc)]
    th = [Line2D([0],[0], color="grey", lw=1.5,        label="Coulomb ref"),
          Line2D([0],[0], color="grey", lw=1.5, ls="--", label="UKF")]
    ax.legend(handles=ch+th, fontsize=7, ncol=2, loc="upper right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)


def plot_diagnostics(df, out_path, battery_id):
    n_cyc = df["cycle_num"].nunique()
    cmap  = cm.get_cmap("plasma", n_cyc)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"{battery_id} — UKF Diagnostics", fontsize=13, fontweight="bold")
    ap, ae, ai = axes[0]
    an, au, ab = axes[1]
    rmse_list, cyc_list = [], []

    for cyc, grp in df.groupby("cycle_num"):
        grp = grp.sort_values("time"); c = cmap(cyc / max(n_cyc-1, 1))
        s   = grp["SOC"].values
        ap.plot(s, grp["SOC"].values,     color=c, lw=0.7, alpha=0.4)
        ap.plot(s, grp["soc_ukf"].values, color=c, lw=1.1, ls="--")
        ae.plot(s, grp["soc_err"].values*100, color=c, lw=0.9, alpha=0.8)
        ai.plot(s, grp["innov"].values,   color=c, lw=0.7, alpha=0.7)
        an.plot(s, grp["nees"].values,    color=c, lw=0.7, alpha=0.7)
        au.plot(s, np.sqrt(np.abs(grp["p_soc"].values))*100, color=c, lw=0.9)
        e = grp["soc_err"].values
        rmse_list.append(float(np.sqrt(np.mean(e**2))))
        cyc_list.append(cyc)

    xi = np.arange(len(cyc_list))
    ab.bar(xi, np.array(rmse_list)*100, color="#e74c3c", alpha=0.85)
    ab.set_xticks(xi); ab.set_xticklabels([f"C{c}" for c in cyc_list])
    ab.set_ylabel("RMSE [%]"); ab.set_title("RMSE per Cycle"); ab.grid(True, alpha=0.3)

    ap.plot([0,1],[0,1],"k--",lw=0.7,alpha=0.3); ap.invert_xaxis()
    ap.set_title("UKF vs Coulomb SoC"); ap.set_xlabel("SoC ref"); ap.set_ylabel("SoC"); ap.grid(True,alpha=0.3)
    ap.legend(handles=[Line2D([0],[0],color="grey",lw=1.5,label="Ref"),
                        Line2D([0],[0],color="grey",lw=1.5,ls="--",label="UKF")], fontsize=8)
    ae.axhline(0,color="k",lw=0.8,ls="--"); ae.invert_xaxis()
    ae.set_title("SoC Error [%]"); ae.set_xlabel("SoC"); ae.set_ylabel("Error [%]"); ae.grid(True,alpha=0.3)
    ai.axhline(0,color="k",lw=0.8,ls="--"); ai.invert_xaxis()
    ai.set_title("Innovation [V]"); ai.set_xlabel("SoC"); ai.set_ylabel("[V]"); ai.grid(True,alpha=0.3)
    an.axhline(1,color="k",lw=1.0,ls="--",label="NEES=1"); an.invert_xaxis()
    an.set_title("NEES (1=ideal)"); an.set_xlabel("SoC"); an.set_ylabel("NEES"); an.legend(fontsize=8); an.grid(True,alpha=0.3)
    au.invert_xaxis(); au.set_title("Posterior Uncertainty √P_soc [%]")
    au.set_xlabel("SoC"); au.set_ylabel("1σ [%]"); au.grid(True,alpha=0.3)

    sm = cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(0, max(n_cyc-1,1)))
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), label="Cycle", shrink=0.45, pad=0.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="UKF SoC estimation")
    parser.add_argument("--tune",   action="store_true", help="Run Optuna tuning")
    parser.add_argument("--trials", type=int, default=OPTUNA_TRIALS,
                        help=f"Optuna trials (default {OPTUNA_TRIALS})")
    args = parser.parse_args()

    print("=" * 60)
    print("STAGE 4 — UKF SoC Estimation" +
          (" + Optuna Tuning" if args.tune else ""))
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
            ecm_df     = pd.read_csv(out_dir / "ecm_params.csv")
            build_ocv_grid(ocv_lookup)

            # ── Default run ───────────────────────────────────────────────────
            params = UKF_DEFAULTS.copy()
            params_path = out_dir / "ukf_params.json"
            if params_path.exists():
                with open(params_path) as f:
                    params = json.load(f)
                print(f"    Loaded saved params from ukf_params.json")

            df = run_ukf_all_cycles(soc_df, ecm_df, **params)
            m  = compute_metrics(df)
            print(f"    Default  RMSE={m['overall_RMSE']*100:.4f}%  "
                  f"MAE={m['overall_MAE']*100:.4f}%")

            # ── Optuna tuning ─────────────────────────────────────────────────
            if args.tune:
                print(f"    Optuna ({args.trials} trials) …")
                best_params, best_val = tune_hyperparameters(soc_df, ecm_df, args.trials)
                print(f"    Tuned   RMSE={best_val*100:.4f}%")
                print(f"    Best params: {best_params}")

                with open(params_path, "w") as f:
                    json.dump(best_params, f, indent=2)

                df     = run_ukf_all_cycles(soc_df, ecm_df, **best_params)
                m      = compute_metrics(df)
                params = best_params

            # ── Save ──────────────────────────────────────────────────────────
            df.to_parquet(out_dir / "ukf_results.parquet", index=False)
            with open(out_dir / "ukf_metrics.json", "w") as f:
                json.dump(m, f, indent=2)

            plot_soc(df,         out_dir / "ukf_soc.png",         bid)
            plot_diagnostics(df, out_dir / "ukf_diagnostics.png", bid)
            print(f"    Saved ukf_results.parquet, ukf_metrics.json, plots")

        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"    FAILED: {e}")

    print(f"\nDone — outputs in {RESULTS_DIR}/<subfolder>/<battery_id>/")


if __name__ == "__main__":
    main()
