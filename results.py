"""Typed records pickled into `results.pkl`.

Kept in its own module so the table/figure consumers can unpickle without
importing `run_experiments.py` (and therefore without instantiating every
filter module). The dataclasses are frozen — the pickle is read-only output.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FilterStats:
    """One cell of the table: aggregate stats for one (filter, σ_θ) combo."""
    name: str
    mean_rmse: float
    std_rmse: float
    n_eval: float
    ms_per_update: float
    frac_diverged: float
    n_diverged: int
    per_time_pos_err_mean: np.ndarray   # (T_steps,) — mean over converged runs


@dataclass(frozen=True)
class SigmaSweepEntry:
    """One σ_θ column: calibrated ΔLs + per-filter stats in declaration order."""
    sigma_theta_deg: float
    delta_L_edh: float
    delta_L_gromov: float
    filters: list                       # list[FilterStats]


@dataclass(frozen=True)
class SubstepTrace:
    """One curve in figure panel a — substep means from a single first-update run."""
    label: str                          # 'naedh_ccr', 'edh_adapt_n10', 'edh_adapt_avg'
    lams: np.ndarray                    # (n+1,)
    means: np.ndarray                   # (n+1, nx)


@dataclass(frozen=True)
class Results:
    """The complete output of run_experiments.py."""
    config: dict[str, Any]
    s_truth: np.ndarray                 # (T_steps + 1, nx)
    s0_hat: np.ndarray                  # (nx,)
    sensor_positions: np.ndarray
    P0: np.ndarray
    sweep: list                         # list[SigmaSweepEntry]
    first_step_substeps: list           # list[SubstepTrace]
    primary_sigma_theta_deg: float      # σ_θ used by the figure panels
