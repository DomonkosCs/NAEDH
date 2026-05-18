"""Stochastic Gromov flow with optimised diffusion Q*(λ)
see \cite{Zhang2025Importance} and \cite{Dai2023On}

The flow SDE on pseudo-time λ ∈ [0, 1] is
    dx = ζ_s(x, λ) dλ + √Q*(λ) dW

with the linearised affine drift
    M(λ)   = P⁻¹ + λ Hᵀ R⁻¹ H
    A_dg   = -½ M⁻¹ Hᵀ R⁻¹ H                          (deterministic Gromov drift)
    Q_g    = M⁻¹ Hᵀ R⁻¹ H M⁻¹                          (natural Gromov diffusion)
    Q*(λ)  = c* M⁻¹                                    (optimised diffusion)
    A_s    = A_dg - ½ Q* M
    b_s    = M⁻¹ Hᵀ R⁻¹ z_eff
             + ½ (Q* - Q_g) (P⁻¹ μ₀ + λ Hᵀ R⁻¹ z_eff)
    c*     = max(√((|λ|_max - |λ|_min)/α) - |λ|_min, 0)

With nz = 2 < nx = 6, A_dg has rank ≤ 2 so 4 of its 6 eigenvalues are
exactly zero ⇒ |λ|_min = 0 ⇒ c* = √(|λ|_max / α). The zeros are kept (not
filtered) to match the paper's formula.

We use ΔL-adaptive Euler-Maruyama stepping: drift+diffusion at λ_a size
the step; the SDE step itself re-evaluates drift+diffusion at λ_b. Wiener
increment is √Δλ · chol(½(Q*+Q*ᵀ)) @ randn.
"""

import numpy as np


def _build_terms(x_bar, P_inv, H, R_inv, lam, alpha, z_eff):
    """All Gromov building blocks at pseudo-time λ.

    Returns (A_s, b_s, Q_opt, c_opt).
    """
    HtRinv = H.T @ R_inv
    HtRinvH = HtRinv @ H
    M = P_inv + lam * HtRinvH
    M_inv = np.linalg.inv(M)
    A_dg = -0.5 * M_inv @ HtRinvH
    Q_g = M_inv @ HtRinvH @ M_inv

    ev = np.linalg.eigvals(A_dg)
    abs_ev = np.abs(ev.real)
    lam_hi = abs_ev.max()
    lam_lo = abs_ev.min()
    c_opt = max(np.sqrt((lam_hi - lam_lo) / alpha) - lam_lo, 0.0)
    Q_opt = c_opt * M_inv

    A_s = A_dg - 0.5 * (Q_opt @ M)
    HtRinv_z = HtRinv @ z_eff
    b_s = M_inv @ HtRinv_z + 0.5 * (Q_opt - Q_g) @ (P_inv @ x_bar + lam * HtRinv_z)
    return A_s, b_s, Q_opt, c_opt


def flow_adaptive(
    x_bar, P, X, z, R, h, dh, rng, delta_L, alpha=0.1, max_steps=2000, record=False
):
    """ΔL-adaptive Euler-Maruyama through the Gromov SDE.

    Returns
    -------
    X       : flowed particles.
    n_eval  : substeps taken.
    (with `record=True`, also returns lams_visited and per-substep means.)
    """
    nx, Np = X.shape
    P_inv = np.linalg.inv(P)
    R_inv = np.linalg.inv(R)
    I_nx = np.eye(nx)

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

        # Step-sizing at λ_a, drift only, see \cite{Mori2016Adaptive}
        A_s_a, b_s_a, _, _ = _build_terms(x_bar, P_inv, H, R_inv, lam, alpha, z_eff)
        f_a = A_s_a @ X + b_s_a[:, None]
        f_max = float(np.max(np.linalg.norm(f_a, axis=0)))
        d_lam = min(delta_L / f_max, 1.0 - lam)
        lam_b = lam + d_lam

        # Re-evaluate at λ_b for the Euler-Maruyama step
        A_s_b, b_s_b, Q_opt_b, c_opt_b = _build_terms(
            x_bar,
            P_inv,
            H,
            R_inv,
            lam_b,
            alpha,
            z_eff,
        )
        drift = A_s_b @ X + b_s_b[:, None]

        if c_opt_b > 0.0:
            Q_sym = 0.5 * (Q_opt_b + Q_opt_b.T)
            try:
                L_chol = np.linalg.cholesky(Q_sym)
            except np.linalg.LinAlgError:
                L_chol = np.linalg.cholesky(Q_sym + 1e-12 * I_nx)
            noise = np.sqrt(d_lam) * (L_chol @ rng.standard_normal((nx, Np)))
        else:
            noise = 0.0

        X = X + drift * d_lam + noise
        lam = lam_b
        lams_visited.append(lam)
        n_eval += 1
        if record:
            means.append(X.mean(axis=1))

    if record:
        return X, np.array(lams_visited), np.array(means), n_eval
    return X, n_eval


# ----------------------------------------------------------------------------
# Uniform filter-module interface.
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


def update(state, problem, z, rng, delta_L, alpha=0.1, **_):
    X_pred = state["X"]
    P_pred = state["P"]
    x_pred = X_pred.mean(axis=1)
    X_new, n_eval = flow_adaptive(
        x_pred,
        P_pred,
        X_pred,
        z,
        problem.R,
        problem.h,
        problem.dh,
        rng,
        delta_L=delta_L,
        alpha=alpha,
    )
    x_post = X_new.mean(axis=1)

    H = problem.dh(x_post)
    S = H @ P_pred @ H.T + problem.R
    K = np.linalg.solve(S.T, (P_pred @ H.T).T).T
    P_post = (np.eye(problem.nx) - K @ H) @ P_pred
    state["X"] = X_new
    state["P"] = 0.5 * (P_post + P_post.T)
    return state, x_post, n_eval
