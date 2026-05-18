"""N-step Analytic EDH: the paper's contribution.

The N-step homotopy [0, 1] is partitioned into N segments. At each boundary
the Jacobian is relinearized at the current particle mean.
Within each segment the linear-Gaussian flow ODE is
solved *exactly* in closed form via an eigendecomposition of
D = L⁻¹ H P Hᵀ L⁻ᵀ — replacing per-substep numerical ODE stepping with one
matrix-vector evaluation.

The math primitive is `flow(...)`. The bottom of the file wraps it into the
uniform filter-module interface (`init`, `predict`, `update`).
"""

import numpy as np
import scipy.linalg

_THRESH = 1e-8


def LINEAR(n: int, eigvals: np.ndarray) -> np.ndarray:
    """Uniform [0, 1] grid of n+1 points (spectrum ignored)."""
    del eigvals
    return np.linspace(0.0, 1.0, n + 1)


def CONSTANT_RATE(n: int, eigvals: np.ndarray) -> np.ndarray:
    """Geometric grid in (1 + λ d_max) — equal per-segment contraction along
    the dominant eigendirection. Falls back to linear when the dominant
    eigenvalue is near zero (L'Hôpital limit).
    """
    d_max = float(np.max(eigvals))
    if d_max < _THRESH:
        return np.linspace(0.0, 1.0, n + 1)
    k = np.arange(n + 1)
    return ((1.0 + d_max) ** (k / n) - 1.0) / d_max


def _substep(x_bar, P, X, z_eff, H, R, lam_a, lam_b):
    """Closed-form flow on [lam_a, lam_b] with frozen linearization (H, z_eff)."""
    nz = z_eff.shape[0]
    L = np.linalg.cholesky(R)
    L_inv = scipy.linalg.solve_triangular(L, np.eye(nz), lower=True)
    D = L_inv @ H @ P @ H.T @ L_inv.T
    d, V = np.linalg.eigh(0.5 * (D + D.T))

    E = P @ H.T @ L_inv.T @ V  # (nx, nz)
    Ft = V.T @ L_inv @ H  # (nz, nx)
    z_t = V.T @ L_inv @ z_eff
    x_t = Ft @ x_bar

    alpha = 1.0 + lam_a * d
    gamma = 1.0 + lam_b * d
    sa = np.sqrt(alpha)
    sg = np.sqrt(gamma)
    dl = lam_b - lam_a

    # Rank-deficient L'Hôpital branch is significant when the ensemble drifts
    # so far that H P Hᵀ becomes near-singular along a sensor direction.
    rank_def = d < _THRESH

    Omega = np.where(rank_def, -0.5 * dl, (sa / sg - 1.0) / d)
    c = np.where(
        rank_def,
        dl * (z_t - 0.5 * x_t),
        (z_t * (sg - sa) + (x_t - z_t) * (1.0 / sg - 1.0 / sa)) / (d * sg),
    )

    Phi = np.eye(P.shape[0]) + (E * Omega) @ Ft
    return Phi @ X + (E @ c)[:, None]


def flow(
    x_bar,
    P,
    X,
    z,
    R,
    h,
    dh,
    n_steps: int,
    schedule=LINEAR,
    record: bool = False,
):
    """N-step NA-EDH-D flow update.

    Parameters
    ----------
    x_bar : (nx,)            prior (log-homotopy anchor) — fixed across substeps.
    P     : (nx, nx)         prior covariance.
    X     : (nx, Np)         prior particles.
    z     : (nz,)             measurement.
    R     : (nz, nz)         measurement-noise covariance.
    h, dh : callables         measurement function and Jacobian.
    n_steps : int             number of segments.
    schedule : callable       (n, eigvals) → (n+1,) cumulative lambda grid.
    record : bool             if True also return (lams, substep_means).

    Returns
    -------
    X (and lams, means if `record`).
    """
    # One upfront eigvalsh(D) at the prior particle mean — supplies the
    # spectrum-aware `schedule` builders without a duplicate decomposition.
    nz = R.shape[0]
    L = np.linalg.cholesky(R)
    L_inv = scipy.linalg.solve_triangular(L, np.eye(nz), lower=True)
    H0 = dh(X.mean(axis=1))
    D0 = L_inv @ H0 @ P @ H0.T @ L_inv.T
    eigvals0 = np.linalg.eigvalsh(0.5 * (D0 + D0.T))
    lams = np.asarray(schedule(n_steps, eigvals0), dtype=float)

    if record:
        means = np.empty((len(lams), x_bar.shape[0]))
        means[0] = X.mean(axis=1)

    for k in range(len(lams) - 1):
        x_mean = X.mean(axis=1)
        H_k = dh(x_mean)
        z_eff = z - h(x_mean) + H_k @ x_mean
        X = _substep(x_bar, P, X, z_eff, H_k, R, lams[k], lams[k + 1])
        if record:
            means[k + 1] = X.mean(axis=1)

    if record:
        return X, lams, means
    return X


# ----------------------------------------------------------------------------
# Uniform filter-module interface.
# ----------------------------------------------------------------------------


def init(rng, problem, x0_hat, P0, Np, **_):
    """Particles sampled from N(x0_hat, P0); parallel Kalman P seeded at P0."""
    X = x0_hat[:, None] + problem.L_P0 @ rng.standard_normal((problem.nx, Np))
    return {"X": X, "P": P0.copy()}


def predict(state, problem, rng):
    state["X"] = problem.F @ state["X"] + problem.Q_sqrt @ rng.standard_normal(
        state["X"].shape
    )
    P = problem.F @ state["P"] @ problem.F.T + problem.Q
    state["P"] = 0.5 * (P + P.T)
    return state


def update(state, problem, z, rng, n_steps, schedule=LINEAR, **_):
    """Flow particles via `flow`, then refresh parallel Kalman P."""
    X_pred = state["X"]
    P_pred = state["P"]
    x_pred = X_pred.mean(axis=1)

    X_new = flow(
        x_pred,
        P_pred,
        X_pred,
        z,
        problem.R,
        problem.h,
        problem.dh,
        n_steps=n_steps,
        schedule=schedule,
    )
    x_post = X_new.mean(axis=1)

    H = problem.dh(x_post)
    S = H @ P_pred @ H.T + problem.R
    K = np.linalg.solve(S.T, (P_pred @ H.T).T).T
    P_post = (np.eye(problem.nx) - K @ H) @ P_pred
    state["X"] = X_new
    state["P"] = 0.5 * (P_post + P_post.T)
    return state, x_post, n_steps
