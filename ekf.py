"""Extended Kalman Filter — Gaussian baseline.

Linearizes the measurement model at the prior mean, applies the standard
Kalman gain, returns the posterior mean and covariance.
"""

import numpy as np


def init(rng, problem, x0_hat, P0, **_):
    return {'x': x0_hat.copy(), 'P': P0.copy()}


def predict(state, problem, rng):
    state['x'] = problem.F @ state['x']
    P = problem.F @ state['P'] @ problem.F.T + problem.Q
    state['P'] = 0.5 * (P + P.T)
    return state


def update(state, problem, z, rng, **_):
    x = state['x']
    P = state['P']
    H = problem.dh(x)
    y = z - problem.h(x)
    S = H @ P @ H.T + problem.R
    K = np.linalg.solve(S.T, (P @ H.T).T).T
    x_post = x + K @ y
    P_post = (np.eye(problem.nx) - K @ H) @ P
    state['x'] = x_post
    state['P'] = 0.5 * (P_post + P_post.T)
    return state, x_post, 1
