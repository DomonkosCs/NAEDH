"""Monte-Carlo sweep across measurement-noise levels → `results.pkl`.

Run as:  python NAEDH_final/run_experiments.py

The script declares one `FILTERS` list of (name, module, hyperparameters)
tuples. A generic Monte-Carlo runner consumes each tuple through the
uniform `init / predict / update` interface — it has no knowledge of what
kind of filter is running. Adding a new filter is one file + one tuple.

RNG discipline: every per-MC RNG is constructed from a `SeedSequence` whose
inputs are (base_seed, mc_index, sigma_index, salt), where `salt` is either
`0` for the measurement-noise RNG or `crc32(filter_name)` for a filter's
private RNG. Reordering / adding / removing filters does not perturb any
other filter's numbers.
"""

import pickle
import subprocess
import time
import warnings
import zlib
from pathlib import Path

import numpy as np

# BLAS-level overflow/underflow fires for divergent runs (the divergence
# threshold catches them downstream); silencing keeps the progress log readable.
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="(divide by zero|overflow|invalid value) encountered in matmul",
)

import problem as problem_mod
import naedh
import edh
import ekf
import bootstrap_pf
import gromov
from results import (
    FilterStats,
    SigmaSweepEntry,
    SubstepTrace,
    Results,
)

# ============================================================================
# Parameters — match the paper's Example-2 setup.
# ============================================================================

BASE_SEED = 42
N_MC = 100  # Monte-Carlo trials per σ_θ
SIGMA_THETA_LIST_DEG = [0.001, 0.05, 1.0]  # bearing-noise std-dev sweep [deg]
SIGMA_A = 0.1  # continuous-time process accel noise std-dev
T_TOTAL_SEC = 15.0  # simulation horizon [s]
SINE_PERIOD = 6.0  # truth sine-y perturbation period [s]
SINE_Y_AMP = 2.0  # truth sine-y perturbation amplitude [m]

SENSOR_POSITIONS = np.array(
    [[10.0, 0.0], [30.0, 0.0]]
)  # bearing sensors (n_sensors, 2) [m]
X0_TRUTH = np.array(
    [5.0, 20.0, 0.0, 0.0, 2.0, 0.0]
)  # true initial state [x, y, vx, vy, ax, ay]
PRIOR_BIAS = np.array(
    [-10.0, 15.0, 0.0, 0.0, 0.0, 0.0]
)  # prior-mean offset from truth (filter's prior = truth + bias)

NP_FLOW = 500  # particle count for flow filters (NAEDH/EDH/Gromov)
NP_PF_SMALL = 10_000  # bootstrap PF small-config particle count
NP_PF_BIG = 100_000  # bootstrap PF large-config particle count

N_STEPS_FIXED = 10  # homotopy substeps for fixed flows; ΔL-calibration target
ALPHA = 0.1  # Gromov optimised-diffusion coefficient see \cite{Zhang2025Importance} (paper symbol α)
ESS_FRAC = 0.5  # PF resamples when ESS/Np falls below this fraction

DIVERGENCE_THRESHOLD = 100.0  # per-run RMSE [m] above which a run is discarded

FILTERS = [
    (
        "NAEDH-lin",
        naedh,
        dict(n_steps=N_STEPS_FIXED, schedule=naedh.LINEAR, Np=NP_FLOW),
    ),
    (
        "NAEDH-ccr",
        naedh,
        dict(n_steps=N_STEPS_FIXED, schedule=naedh.CONSTANT_RATE, Np=NP_FLOW),
    ),
    ("EDH-lin", edh, dict(n_steps=N_STEPS_FIXED, Np=NP_FLOW)),
    ("EDH-adaptive", edh, dict(n_steps=N_STEPS_FIXED, delta_L="calibrate", Np=NP_FLOW)),
    ("Gromov", gromov, dict(delta_L="calibrate", alpha=ALPHA, Np=NP_FLOW)),
    ("EKF", ekf, dict()),
    ("PF (Np=10,000)", bootstrap_pf, dict(Np=NP_PF_SMALL, ess_frac=ESS_FRAC)),
    ("PF (Np=100,000)", bootstrap_pf, dict(Np=NP_PF_BIG, ess_frac=ESS_FRAC)),
]


# ============================================================================
# RNG plumbing — name-derived seeds isolate filters from each other.
# ============================================================================


def _name_to_key(name: str) -> int:
    """Stable, deterministic 32-bit hash (Python's `hash` is salt-randomized)."""
    return int(zlib.crc32(name.encode("utf-8")))


def measurement_rng(base_seed, mc, sigma_idx) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence(
            [base_seed, mc, sigma_idx, 0],
        )
    )


def filter_rng(base_seed, mc, sigma_idx, filter_name) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence(
            [base_seed, mc, sigma_idx, _name_to_key(filter_name)],
        )
    )


# ============================================================================
# Generic MC runner — knows nothing about specific filter kinds.
# ============================================================================


def run_mc(filter_module, problem, hp, base_seed, mc, sigma_idx, filter_name):
    """One MC trial: predict + update at each measurement timestep."""
    Z = problem.generate_Z(measurement_rng(base_seed, mc, sigma_idx))
    rng = filter_rng(base_seed, mc, sigma_idx, filter_name)
    state = filter_module.init(rng, problem, problem.x0_hat, problem.P0, **hp)

    T = problem.T_steps
    means = np.empty((T, problem.nx))
    n_evals = np.empty(T)
    ms = np.empty(T)
    for t in range(T):
        t0 = time.perf_counter()
        state = filter_module.predict(state, problem, rng)
        state, x_post, n = filter_module.update(state, problem, Z[t], rng, **hp)
        ms[t] = (time.perf_counter() - t0) * 1000.0
        means[t] = x_post
        n_evals[t] = n
    return means, n_evals, ms


# ============================================================================
# ΔL calibration — log-bisect until avg substep count matches the target.
# ============================================================================


def calibrate_delta_L(
    filter_module,
    hp_template,
    problem,
    target_ne,
    base_seed,
    sigma_idx,
    filter_name,
    dL_lo=1e-3,
    dL_hi=200.0,
    n_iter=12,
    n_seeds=3,
):
    """Bisect ΔL so the 3-seed average n_eval is close to `target_ne`."""
    tol = max(0.5, 0.02 * target_ne)
    delta_L = float(np.sqrt(dL_lo * dL_hi))
    for _ in range(n_iter):
        hp = {**hp_template, "delta_L": delta_L}
        avg_ne = 0.0
        for k in range(n_seeds):
            _, ne, _ = run_mc(
                filter_module,
                problem,
                hp,
                base_seed=base_seed + 9000,
                mc=k,
                sigma_idx=sigma_idx,
                filter_name=filter_name,
            )
            avg_ne += float(ne.mean())
        avg_ne /= n_seeds
        if abs(avg_ne - target_ne) <= tol:
            return delta_L
        if avg_ne > target_ne:
            dL_lo = delta_L
        else:
            dL_hi = delta_L
        delta_L = float(np.sqrt(dL_lo * dL_hi))
    return delta_L


# ============================================================================
# Stats aggregation — one cell of the table.
# ============================================================================


def aggregate(name, all_means, all_ne, all_ms, truth):
    """Compute the FilterStats record for one (filter, σ_θ) cell.

    Convergence flag = per-run time-RMS position error > threshold. The same
    scalar drives both the divergence flag and the row's mean/std, so the
    table is internally consistent (median can never disagree with the cutoff).
    """
    errs = all_means - truth[None, :, :]
    pos_err = np.sqrt(np.sum(errs[..., 0:2] ** 2, axis=-1))
    run_rmse = np.sqrt(np.mean(pos_err**2, axis=1))
    is_div = run_rmse > DIVERGENCE_THRESHOLD
    keep = ~is_div
    n_div = int(is_div.sum())

    T = pos_err.shape[1]
    if keep.any():
        rmse_k = run_rmse[keep]
        ne_k = all_ne[keep]
        ms_k = all_ms[keep]
        mean_rmse = float(rmse_k.mean())
        std_rmse = float(rmse_k.std(ddof=1)) if rmse_k.size > 1 else 0.0
        n_eval = float(ne_k.mean())
        ms_call = float(ms_k.mean())
        pt_mean = pos_err[keep].mean(axis=0)
    else:
        mean_rmse = std_rmse = n_eval = ms_call = float("nan")
        pt_mean = np.full(T, np.nan)

    return FilterStats(
        name=name,
        mean_rmse=mean_rmse,
        std_rmse=std_rmse,
        n_eval=n_eval,
        ms_per_update=ms_call,
        frac_diverged=n_div / max(1, all_means.shape[0]),
        n_diverged=n_div,
        per_time_pos_err_mean=pt_mean,
    )


# ============================================================================
# First-update substep traces — used by figure panel a (one realization).
# ============================================================================


def first_step_traces(problem, primary_dL_edh):
    meas_rng = measurement_rng(BASE_SEED, mc=0, sigma_idx=0)
    Z = problem.generate_Z(meas_rng)

    naedh_rng = filter_rng(BASE_SEED, mc=0, sigma_idx=0, filter_name="NAEDH-ccr")
    X0 = problem.x0_hat[:, None] + problem.L_P0 @ naedh_rng.standard_normal(
        (problem.nx, NP_FLOW),
    )
    X_pred = problem.F @ X0 + problem.Q_sqrt @ naedh_rng.standard_normal(
        (problem.nx, NP_FLOW),
    )
    P_pred = problem.F @ problem.P0 @ problem.F.T + problem.Q
    P_pred = 0.5 * (P_pred + P_pred.T)
    x_pred = X_pred.mean(axis=1)

    _, lams_ccr, means_ccr = naedh.flow(
        x_pred,
        P_pred,
        X_pred,
        Z[0],
        problem.R,
        problem.h,
        problem.dh,
        n_steps=N_STEPS_FIXED,
        schedule=naedh.CONSTANT_RATE,
        record=True,
    )

    _, lams_avg, means_avg, _ = edh.flow_adaptive(
        x_pred,
        P_pred,
        X_pred,
        Z[0],
        problem.R,
        problem.h,
        problem.dh,
        delta_L=primary_dL_edh,
        record=True,
    )

    # Bisect ΔL so EDH-adaptive uses exactly n_steps_fixed substeps on this
    # one realization (figure panel a's middle curve).
    dL_lo, dL_hi = 1e-4, 1e4
    dL_n10 = float(np.sqrt(dL_lo * dL_hi))
    lams_n10 = means_n10 = None
    for _ in range(50):
        _, lams_n10, means_n10, n_taken = edh.flow_adaptive(
            x_pred,
            P_pred,
            X_pred,
            Z[0],
            problem.R,
            problem.h,
            problem.dh,
            delta_L=dL_n10,
            record=True,
        )
        if n_taken == N_STEPS_FIXED:
            break
        if n_taken > N_STEPS_FIXED:
            dL_lo = dL_n10
        else:
            dL_hi = dL_n10
        dL_n10 = float(np.sqrt(dL_lo * dL_hi))

    return [
        SubstepTrace(label="naedh_ccr", lams=lams_ccr, means=means_ccr),
        SubstepTrace(label="edh_adapt_n10", lams=lams_n10, means=means_n10),
        SubstepTrace(label="edh_adapt_avg", lams=lams_avg, means=means_avg),
    ]


# ============================================================================
# Main entry point.
# ============================================================================


def main():
    out_path = Path(__file__).parent / "results.pkl"
    print(f"NAEDH-final experiment driver — writing {out_path}")
    print(
        f"  base_seed={BASE_SEED}, n_mc={N_MC}, "
        f"Np_flow={NP_FLOW}, n_steps={N_STEPS_FIXED}"
    )
    print(f"  σ_θ sweep = {SIGMA_THETA_LIST_DEG} deg")
    print(f"  {len(FILTERS)} filters: {[name for name, *_ in FILTERS]}")

    sweep: list[SigmaSweepEntry] = []
    primary_problem: problem_mod.Problem | None = None

    for sigma_idx, sd in enumerate(SIGMA_THETA_LIST_DEG):
        print(f"\nσ_θ = {sd}°")
        prob = problem_mod.make_example(
            sigma_theta_deg=sd,
            sigma_a=SIGMA_A,
            T_total=T_TOTAL_SEC,
            sensor_positions=SENSOR_POSITIONS,
            x0_truth=X0_TRUTH,
            prior_bias=PRIOR_BIAS,
            sine_period=SINE_PERIOD,
            sine_y_amp=SINE_Y_AMP,
        )
        if sigma_idx == 0:
            primary_problem = prob

        # -- ΔL calibration for the adaptive filters at this σ_θ ----------------
        dL_edh = float("nan")
        dL_gromov = float("nan")
        for name, mod, hp in FILTERS:
            if hp.get("delta_L") != "calibrate":
                continue
            hp_no_cal = {k: v for k, v in hp.items() if k != "delta_L"}
            dL = calibrate_delta_L(
                mod,
                hp_no_cal,
                prob,
                target_ne=float(N_STEPS_FIXED),
                base_seed=BASE_SEED,
                sigma_idx=sigma_idx,
                filter_name=name,
            )
            if mod is edh:
                dL_edh = dL
            elif mod is gromov:
                dL_gromov = dL
            print(f"  calibrated {name}: ΔL = {dL:.4g}")

        # -- MC sweep over filters ----------------------------------------------
        filter_stats: list[FilterStats] = []
        for _, (name, mod, hp) in enumerate(FILTERS):
            hp_run = dict(hp)
            if hp_run.get("delta_L") == "calibrate":
                hp_run["delta_L"] = dL_edh if mod is edh else dL_gromov

            all_means = np.empty((N_MC, prob.T_steps, prob.nx))
            all_ne = np.empty((N_MC, prob.T_steps))
            all_ms = np.empty((N_MC, prob.T_steps))
            t0 = time.perf_counter()
            for mc in range(N_MC):
                means, ne, ms = run_mc(
                    mod,
                    prob,
                    hp_run,
                    base_seed=BASE_SEED,
                    mc=mc,
                    sigma_idx=sigma_idx,
                    filter_name=name,
                )
                all_means[mc] = means
                all_ne[mc] = ne
                all_ms[mc] = ms
            stats = aggregate(name, all_means, all_ne, all_ms, prob.truth[1:])
            elapsed = time.perf_counter() - t0
            print(
                f"  {name:<20s}  mean RMSE={stats.mean_rmse:8.3f}"
                f"  n_eval={stats.n_eval:5.2f}"
                f"  ms/upd={stats.ms_per_update:6.2f}"
                f"  %div={100*stats.frac_diverged:5.1f}"
                f"  ({elapsed:.1f}s)"
            )
            filter_stats.append(stats)

        sweep.append(
            SigmaSweepEntry(
                sigma_theta_deg=sd,
                delta_L_edh=dL_edh,
                delta_L_gromov=dL_gromov,
                filters=filter_stats,
            )
        )

    # -- Substep traces at the primary σ_θ ----------------------------------
    primary_sd = SIGMA_THETA_LIST_DEG[0]
    print(f"\nBuilding first-step substep traces at σ_θ = {primary_sd}°")
    traces = first_step_traces(primary_problem, sweep[0].delta_L_edh)

    config = dict(
        base_seed=BASE_SEED,
        n_mc=N_MC,
        sigma_theta_list_deg=SIGMA_THETA_LIST_DEG,
        sigma_a=SIGMA_A,
        dt=primary_problem.dt,
        T_total_sec=T_TOTAL_SEC,
        T_steps=primary_problem.T_steps,
        sine_period=SINE_PERIOD,
        sine_y_amp=SINE_Y_AMP,
        Np_flow=NP_FLOW,
        Np_pf=[NP_PF_SMALL, NP_PF_BIG],
        n_steps_fixed=N_STEPS_FIXED,
        alpha=ALPHA,
        ess_resample_frac=ESS_FRAC,
        divergence_threshold=DIVERGENCE_THRESHOLD,
        x0_truth=primary_problem.x0_truth.tolist(),
        prior_bias=primary_problem.prior_bias.tolist(),
        filter_order=[name for name, *_ in FILTERS],
    )

    results = Results(
        config=config,
        s_truth=primary_problem.truth.copy(),
        s0_hat=primary_problem.x0_hat.copy(),
        sensor_positions=primary_problem.sensor_positions.copy(),
        P0=primary_problem.P0.copy(),
        sweep=sweep,
        first_step_substeps=traces,
        primary_sigma_theta_deg=primary_sd,
    )
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
