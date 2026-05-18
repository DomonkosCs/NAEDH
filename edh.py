"""Euler-stepped Daum–Huang EDH baseline.

Integrates the affine flow ODE
    dx/dλ = A(λ) x + b(λ)
with
    A(λ) = -½ P Hᵀ (λ H P Hᵀ + R)⁻¹ H
    b(λ) = (I + 2λ A) [ (I + λ A) P Hᵀ R⁻¹ z_eff + A x_bar ]
via forward Euler, relinearizing at the particle mean at each substep.

Two integration modes:
  flow_fixed     — uniform [0, 1] grid of n_steps + 1 points
  flow_adaptive  — \cite{Mori2026Adaptive} Eq. 20: Δλ chosen so the largest
                   per-particle drift displacement is bounded by ΔL.
"""

import numpy as np


def _flow_terms(x_bar, P, H, R, z_eff, lam):
    """Return (A(λ), b(λ))."""
    nx = P.shape[0]
    PHT = P @ H.T
    HPHT = H @ PHT
    A = -0.5 * PHT @ np.linalg.solve(lam * HPHT + R, H)
    R_inv_z = np.linalg.solve(R, z_eff)
    I_nx = np.eye(nx)
    b = (I_nx + 2 * lam * A) @ ((I_nx + lam * A) @ PHT @ R_inv_z + A @ x_bar)
    return A, b


def flow_fixed(x_bar, P, X, z, R, h, dh, n_steps, record=False):
    """Forward Euler with uniform λ-schedule. Drift evaluated at λ_b."""
    lams = np.linspace(0.0, 1.0, n_steps + 1)
    if record:
        means = np.empty((n_steps + 1, x_bar.shape[0]))
        means[0] = X.mean(axis=1)
    for k in range(n_steps):
        x_mean = X.mean(axis=1)
        H = dh(x_mean)
        z_eff = z - h(x_mean) + H @ x_mean
        A, b = _flow_terms(x_bar, P, H, R, z_eff, lams[k + 1])  # right endpoint
        X = X + (lams[k + 1] - lams[k]) * (A @ X + b[:, None])
        if record:
            means[k + 1] = X.mean(axis=1)
    if record:
        return X, lams, means
    return X


def flow_adaptive(x_bar, P, X, z, R, h, dh, delta_L, max_steps=2000, record=False):
    """ΔL-adaptive forward Euler. Step-size at λ_a, then re-evaluate at λ_b.

    Provisional `(A, b)` at λ_a only sizes Δλ via max ‖f‖; the actual Euler
    update uses `(A, b)` at λ_b. This avoids implicit step-size + fixed-point.

    Returns
    -------
    X       : flowed particles.
    n_eval  : substeps taken between λ=0 and λ=1.
    (with `record=True`, also returns lams_visited and per-substep means.)
    """
    lam = 0.0
    lams_visited = [0.0]
    if record:
        means = [X.mean(axis=1)]
    n_eval = 0
    for _ in range(max_steps):
        if lam >= 1.0:
            break
        x_mean = X.mean(axis=1)
        H = dh(x_mean)
        z_eff = z - h(x_mean) + H @ x_mean

        A_a, b_a = _flow_terms(x_bar, P, H, R, z_eff, lam)
        f = A_a @ X + b_a[:, None]
        f_max = float(np.max(np.linalg.norm(f, axis=0)))
        d_lam = min(delta_L / f_max, 1.0 - lam)
        lam_b = lam + d_lam

        A_b, b_b = _flow_terms(x_bar, P, H, R, z_eff, lam_b)
        X = X + d_lam * (A_b @ X + b_b[:, None])
        lam = lam_b
        lams_visited.append(lam)
        n_eval += 1
        if record:
            means.append(X.mean(axis=1))

    if record:
        return X, np.array(lams_visited), np.array(means), n_eval
    return X, n_eval


# ----------------------------------------------------------------------------
# Uniform filter-module interface — fixed or adaptive selected by hyperparams.
# ----------------------------------------------------------------------------


def init(rng, problem, x0_hat, P0, Np, **_):
    X = x0_hat[:, None] + problem.L_P0 @ rng.standard_normal((problem.nx, Np))
    return {"X": X, "P": P0.copy()}


def predict(state, problem, rng):
    state["X"] = problem.F @ state["X"] + problem.Q_sqrt @ rng.standard_normal(
        state["X"].shape
    )
    P = problem.F @ state["P"] @ problem.F.T + problem.Q
    state["P"] = 0.5 * (P + P.T)
    return state


def update(state, problem, z, rng, n_steps, delta_L=None, **_):
    """Pick the integrator from hyperparameters: ΔL-adaptive if `delta_L` is set."""
    X_pred = state["X"]
    P_pred = state["P"]
    x_pred = X_pred.mean(axis=1)

    if delta_L is not None:
        X_new, n_eval = flow_adaptive(
            x_pred, P_pred, X_pred, z, problem.R, problem.h, problem.dh, delta_L=delta_L
        )
    else:
        X_new = flow_fixed(
            x_pred, P_pred, X_pred, z, problem.R, problem.h, problem.dh, n_steps=n_steps
        )
        n_eval = n_steps

    x_post = X_new.mean(axis=1)
    H = problem.dh(x_post)
    S = H @ P_pred @ H.T + problem.R
    K = np.linalg.solve(S.T, (P_pred @ H.T).T).T
    P_post = (np.eye(problem.nx) - K @ H) @ P_pred
    state["X"] = X_new
    state["P"] = 0.5 * (P_post + P_post.T)
    return state, x_post, n_eval
