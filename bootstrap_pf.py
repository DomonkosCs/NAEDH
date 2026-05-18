"""Bootstrap particle filter with ESS-adaptive systematic resampling."""

import numpy as np


def reweight_and_resample(X, log_w, z, R_inv, h, rng, ess_thresh):
    """Likelihood reweight + ESS-conditional systematic resample.

    Returns (X, w, log_w).
    """
    d = z[:, None] - h(X)
    d = np.arctan2(np.sin(d), np.cos(d))  # bearing-wrap
    log_w = log_w - 0.5 * np.sum(d * (R_inv @ d), axis=0)
    log_w = log_w - log_w.max()  # numerical stability
    w = np.exp(log_w)
    w = w / w.sum()

    Np = X.shape[1]
    ess = 1.0 / np.sum(w * w)
    if ess < ess_thresh:
        cdf = np.cumsum(w)
        cdf[-1] = 1.0  # guard searchsorted
        u = rng.uniform(0.0, 1.0 / Np) + np.arange(Np) / Np
        idx = np.searchsorted(cdf, u)
        X = X[:, idx]
        w = np.full(Np, 1.0 / Np)
        log_w = np.zeros(Np)
        return X, w, log_w
    return X, w, log_w


def init(rng, problem, x0_hat, P0, Np, **_):
    X = x0_hat[:, None] + problem.L_P0 @ rng.standard_normal((problem.nx, Np))
    return {
        "X": X,
        "log_w": np.zeros(Np),
        "R_inv": np.linalg.inv(problem.R),
    }


def predict(state, problem, rng):
    X = state["X"]
    state["X"] = problem.F @ X + problem.Q_sqrt @ rng.standard_normal(X.shape)
    return state


def update(state, problem, z, rng, Np, ess_frac=0.5, **_):
    ess_thresh = ess_frac * Np
    X, w, log_w = reweight_and_resample(
        state["X"],
        state["log_w"],
        z,
        state["R_inv"],
        problem.h,
        rng,
        ess_thresh,
    )
    state["X"] = X
    state["log_w"] = log_w
    x_post = X @ w
    return state, x_post, 1
