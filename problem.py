"""Modified version of Example 2 \cite{Dai2023On}: bearings-only target tracking,
with a deterministic sine-wave trajectory that violates the filter's constant-acceleration model
(our modification).

A 6-D constant-acceleration target observed by two ground sensors. The truth
deviates from the filter's motion model by following a deterministic sine
wave in y, so every F-step prediction is wrong by a bias.

A `Problem` instance is the single source of truth for every filter:
dynamics matrices, measurement model, sensor geometry, deterministic truth,
prior, and measurement-noise generation. Filters receive a `Problem` argument
and never see module-level globals.
"""

from dataclasses import dataclass

import numpy as np
import scipy.linalg


@dataclass(frozen=True)
class Problem:
    nx: int
    nz: int
    dt: float
    T_steps: int
    F: np.ndarray
    Q: np.ndarray
    Q_sqrt: np.ndarray  # symmetric PSD sqrt of Q (eigh-based)
    R: np.ndarray
    L_R: np.ndarray  # cholesky(R), perturbs the measurements
    x0_truth: np.ndarray
    prior_bias: np.ndarray
    x0_hat: np.ndarray  # x0_truth + prior_bias (deterministic)
    P0: np.ndarray
    L_P0: np.ndarray  # cholesky(P0), seeds particle clouds at init
    sensor_positions: np.ndarray  # (n_sensors, 2)
    truth: np.ndarray  # (T_steps + 1, nx) — deterministic trajectory

    def h(self, x: np.ndarray) -> np.ndarray:
        """Bearings to each sensor. Broadcasts: x is (nx,) → (nz,), or (nx, N) → (nz, N)."""
        shape = (self.nz,) if x.ndim == 1 else (self.nz, x.shape[1])
        z = np.empty(shape)
        for i, s in enumerate(self.sensor_positions):
            z[i] = np.arctan2(x[1] - s[1], x[0] - s[0])
        return z

    def dh(self, x: np.ndarray) -> np.ndarray:
        """Jacobian at scalar x. Shape (nz, nx); only first 2 columns are nonzero."""
        H = np.zeros((self.nz, self.nx))
        for i, s in enumerate(self.sensor_positions):
            dx = x[0] - s[0]
            dy = x[1] - s[1]
            r2 = dx * dx + dy * dy
            H[i, 0] = -dy / r2
            H[i, 1] = dx / r2
        return H

    def generate_Z(self, rng: np.random.Generator) -> np.ndarray:
        """Measurements at t = 1..T_steps: noise-free bearings of truth + L_R @ randn."""
        Z = np.empty((self.T_steps, self.nz))
        for t in range(self.T_steps):
            Z[t] = self.h(self.truth[t + 1]) + self.L_R @ rng.standard_normal(self.nz)
        return Z


def _dynamics_continuous():
    A = np.zeros((6, 6))
    A[0, 2] = A[1, 3] = A[2, 4] = A[3, 5] = 1.0
    L = np.zeros((6, 2))
    L[4, 0] = L[5, 1] = 1.0
    return A, L


def _van_loan_discretize(A, L, sigma_a_sq, dt):
    """Van Loan: dx/dt = A x + L w with Cov(w)=σ_a²·I → (F, Q) for one dt step."""
    n = A.shape[0]
    LSL = L @ L.T * sigma_a_sq
    upper = np.hstack([-A, LSL])
    lower = np.hstack([np.zeros((n, n)), A.T])
    M = np.vstack([upper, lower]) * dt
    eM = scipy.linalg.expm(M)
    F = eM[n:, n:].T
    Q = F @ eM[:n, n:]
    return F, 0.5 * (Q + Q.T)


def _psd_sqrt(M):
    """Symmetric square root of a PSD matrix via eigh; negative eigvals clipped to 0."""
    evals, evecs = np.linalg.eigh(0.5 * (M + M.T))
    evals = np.maximum(evals, 0.0)
    return evecs @ np.diag(np.sqrt(evals)) @ evecs.T


def _sine_truth(x0_truth, dt, T_steps, period, y_amp):
    """Sine-y / constant-accel-x trajectory. Pos / vel / acc set analytically."""
    nx = x0_truth.shape[0]
    truth = np.empty((T_steps + 1, nx))
    t = np.arange(T_steps + 1, dtype=float) * dt
    omega = 2.0 * np.pi / period
    truth[:, 0] = x0_truth[0] + x0_truth[2] * t + 0.5 * x0_truth[4] * t**2
    truth[:, 2] = x0_truth[2] + x0_truth[4] * t
    truth[:, 4] = x0_truth[4]
    truth[:, 1] = x0_truth[1] + y_amp * (1.0 - np.cos(omega * t))
    truth[:, 3] = y_amp * omega * np.sin(omega * t)
    truth[:, 5] = y_amp * omega**2 * np.cos(omega * t)
    return truth


def make_example(
    sigma_theta_deg: float,
    sigma_a: float,
    T_total: float,
    sensor_positions: np.ndarray,
    x0_truth: np.ndarray,
    prior_bias: np.ndarray,
    sine_period: float = 6.0,
    sine_y_amp: float = 2.0,
) -> Problem:
    """Build the Example-2 `Problem` at the requested measurement-noise level."""
    dt = 1.0

    A_c, L_c = _dynamics_continuous()
    F, Q = _van_loan_discretize(A_c, L_c, sigma_a_sq=sigma_a**2, dt=dt)
    Q_sqrt = _psd_sqrt(Q)

    sigma_theta = np.deg2rad(sigma_theta_deg)
    R = (sigma_theta**2) * np.eye(2)
    L_R = np.linalg.cholesky(R)

    P0 = scipy.linalg.block_diag(200.0 * np.eye(2), 10.0 * np.eye(2), 1.0 * np.eye(2))
    L_P0 = np.linalg.cholesky(P0)

    T_steps = int(round(T_total / dt))
    truth = _sine_truth(x0_truth, dt, T_steps, sine_period, sine_y_amp)

    return Problem(
        nx=6,
        nz=2,
        dt=dt,
        T_steps=T_steps,
        F=F,
        Q=Q,
        Q_sqrt=Q_sqrt,
        R=R,
        L_R=L_R,
        x0_truth=x0_truth,
        prior_bias=prior_bias,
        x0_hat=x0_truth + prior_bias,
        P0=P0,
        L_P0=L_P0,
        sensor_positions=sensor_positions,
        truth=truth,
    )
