"""
CVXPY layer that mirrors the continuous relaxation of the MIQP controller.

Given the binary activation patterns produced by a classifier (or enumerated),
this layer solves the convex quadratic program for the continuous states and
controls. It exposes a :class:`cvxpylayers.torch.CvxpyLayer` that can be used
inside PyTorch pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from Robots import obstacle


@dataclass(frozen=True)
class ObstacleInfo:
    """Lightweight container capturing obstacle geometry."""

    x_c: float
    y_c: float
    L: float  # already half-length in the controller
    W: float  # already half-width in the controller
    theta: float


def _prepare_obstacles(obstacles: Sequence) -> List[ObstacleInfo]:
    """Convert controller obstacle objects to a serializable structure."""
    prepared = []
    for ob in obstacles:
        prepared.append(
            ObstacleInfo(
                x_c=float(ob.x_c),
                y_c=float(ob.y_c),
                L=float(ob.L),
                W=float(ob.W),
                theta=float(ob.theta),
            )
        )
    return prepared


def build_qp_cvxpy_layer(
    T: float,
    H: int,
    M: int,
    bounds: dict,
    weights: Tuple[float, float, float],
    d_min: float,
    obstacles: Sequence,
    coupling_pairs: Iterable[Tuple[int, int]],
    big_m: float = 1e2,
) -> Tuple[CvxpyLayer, dict]:
    """
    Construct a CVXPYLayer that solves the continuous relaxation of the MIQP.

    Parameters
    ----------
    T : float
        Sampling period of the controller.
    H : int
        Prediction horizon.
    M : int
        Number of robots (each with x/y position and velocity components).
    bounds : dict
        Dictionary with keys ``x_min``, ``x_max``, ``y_min``, ``y_max``,
        ``v_min``, ``v_max``, ``u_min``, ``u_max`` following `controller.set_params`.
    weights : (float, float, float)
        Tuple ``(Wu, Wp, Wpt)`` weighting control effort, trajectory tracking,
        and terminal position.
    d_min : float
        Minimum separation distance wrt obstacles and robots.
    obstacles : Sequence
        Sequence of obstacle objects (matching those stored in `controller.obs`).
    coupling_pairs : Iterable[Tuple[int, int]]
        Iterable with unique robot index pairs (m, n) for collision avoidance.
    big_m : float, optional
        Big-M constant used in mixed-integer constraints.

    Returns
    -------
    layer : cvxpylayers.torch.CvxpyLayer
        Layer that maps parameters to optimal continuous trajectories. Returns
        the control/state trajectories along with slack variables that relax the
        integer-activated constraints.
    meta : dict
        Metadata describing parameter ordering and dimensions for future calls.
    """

    # Default parameters

    T = 0.25 if T is None else T
    H = 20 if H is None else H
    M = 1 if M is None else M
    bounds = {
        "x_max": 2.00,
        "x_min": -0.5,
        "y_max": 0.5,
        "y_min": -3.0,
        "v_max": 0.50,
        "v_min": -0.50,
        "u_max": 0.50,
        "u_min": -0.50,
    } if bounds is None else bounds
    weights = (1.0, 1.0, 10.0) if weights is None else weights  # (Wu, Wp, Wpt)
    d_min = 0.25

    # Obstacles exactly as in the simulator
    
    obstacles = [
        obstacle(1.0, 0.0, 0.4, 0.5, 0.0),
        obstacle(0.7, -1.1, 0.5, 0.4, 0.0),
        obstacle(0.40, -2.50, 0.4, 0.5, 0.0),
    ]

    M = 1
    p = np.zeros((2, M))  # stack of robot positions; replace with actual state
    d_prox = 2.0
    coupling_pairs = [
        (m, n)
        for m in range(M)
        for n in range(m + 1, M)
        if np.linalg.norm(p[:, m] - p[:, n]) <= d_prox
    ]

    Wu, Wp, Wpt = map(float, weights)

    num_states = 2 * M
    horizon_vars = H + 1
    prepared_obs = _prepare_obstacles(obstacles)
    num_obs = len(prepared_obs)

    unique_pairs = [(min(i, j), max(i, j)) for i, j in coupling_pairs if i < j]
    num_pairs = len(unique_pairs)

    eps_ridge = 1e-6     # try 1e-6 .. 1e-4
    eps_slack = 1e-8     # strict interior

    # Decision variables.
    P = cp.Variable((num_states, horizon_vars))
    V = cp.Variable((num_states, horizon_vars))
    U = cp.Variable((num_states, H))
    S_obs = cp.Variable((num_obs, M, H, 4), nonneg=True)

    # Parameters that will be passed in at solve time.
    p0 = cp.Parameter(num_states)
    v0 = cp.Parameter(num_states)
    goals = cp.Parameter(num_states)

    # Binary activation patterns (4 inequalities per obstacle) as parameters.
    obs_binary_param = cp.Parameter((num_obs, M, H, 4))
    if num_pairs > 0:
        bb_param = cp.Parameter((num_pairs, H))
        cc_param = cp.Parameter((num_pairs, H))
        S_pairs = cp.Variable((num_pairs, H, 4), nonneg=True)
    else:
        bb_param = None
        cc_param = None
        S_pairs = None

    constraints = []

    # Initial state constraints.
    constraints += [P[:, 0] == p0, V[:, 0] == v0]

    # Precompute constants used multiple times.
    xy_min = np.tile([bounds["x_min"], bounds["y_min"]], M)
    xy_max = np.tile([bounds["x_max"], bounds["y_max"]], M)
    v_min = np.full(num_states, bounds["v_min"])
    v_max = np.full(num_states, bounds["v_max"])
    u_min = np.full(num_states, bounds["u_min"])
    u_max = np.full(num_states, bounds["u_max"])

    goal_column = cp.reshape(goals, (num_states, 1))

    # Dynamics and box constraints.
    for k in range(H):
        constraints += [
            V[:, k + 1] == V[:, k] + T * U[:, k],
            P[:, k + 1] == P[:, k] + T * V[:, k] + 0.5 * (T**2) * U[:, k],
            P[:, k + 1] <= xy_max,
            P[:, k + 1] >= xy_min,
            V[:, k + 1] <= v_max,
            V[:, k + 1] >= v_min,
            U[:, k] <= u_max,
            U[:, k] >= u_min,
        ]

    # Obstacle avoidance constraints (relaxed with big-M).
    cos_th = np.array([np.cos(ob.theta) for ob in prepared_obs])
    sin_th = np.array([np.sin(ob.theta) for ob in prepared_obs])
    xo = np.array([ob.x_c for ob in prepared_obs])
    yo = np.array([ob.y_c for ob in prepared_obs])
    L0 = np.array([ob.L + d_min for ob in prepared_obs])
    W0 = np.array([ob.W + d_min for ob in prepared_obs])

    for obs_idx in range(num_obs):
        cth = cos_th[obs_idx]
        sth = sin_th[obs_idx]
        for m in range(M):
            px = P[2 * m, 1:]
            py = P[2 * m + 1, 1:]
            constraints += [
                cth * (px - xo[obs_idx]) + sth * (py - yo[obs_idx])
                + S_obs[obs_idx, m, :, 0]
                >= L0[obs_idx] - big_m * obs_binary_param[obs_idx, m, :, 0],
                -sth * (px - xo[obs_idx]) + cth * (py - yo[obs_idx])
                + S_obs[obs_idx, m, :, 1]
                >= W0[obs_idx] - big_m * obs_binary_param[obs_idx, m, :, 1],
                -cth * (px - xo[obs_idx]) - sth * (py - yo[obs_idx])
                + S_obs[obs_idx, m, :, 2]
                >= L0[obs_idx] - big_m * obs_binary_param[obs_idx, m, :, 2],
                sth * (px - xo[obs_idx]) - cth * (py - yo[obs_idx])
                + S_obs[obs_idx, m, :, 3]
                >= W0[obs_idx] - big_m * obs_binary_param[obs_idx, m, :, 3],
            ]

    # constraints += [S_obs >= eps_slack]
    # constraints += [S_obs <= 0.1]

    # Inter-robot collision avoidance constraints.
    if num_pairs > 0:
        for pair_idx, (m, n) in enumerate(unique_pairs):
            px_m = P[2 * m, 1:]
            py_m = P[2 * m + 1, 1:]
            px_n = P[2 * n, 1:]
            py_n = P[2 * n + 1, 1:]
            bb = bb_param[pair_idx, :]
            cc = cc_param[pair_idx, :]

            constraints += [
                px_m - px_n + S_pairs[pair_idx, :, 0]
                >= 2 * d_min - big_m * (bb + cc),
                px_n - px_m + S_pairs[pair_idx, :, 1]
                >= 2 * d_min - big_m * (1 - bb + cc),
                py_m - py_n + S_pairs[pair_idx, :, 2]
                >= 2 * d_min - big_m * (1 + bb - cc),
                py_n - py_m + S_pairs[pair_idx, :, 3]
                >= 2 * d_min - big_m * (2 - bb - cc),
            ]

    # Objective mirrors the original quadratic cost.
    slack_penalty = 1e4
    slack_cost = slack_penalty * cp.sum(S_obs)
    if S_pairs is not None:
        slack_cost += slack_penalty * cp.sum(S_pairs)

    objective = cp.Minimize(
        Wu * cp.sum_squares(U)
        + Wp * cp.sum_squares(P[:, :H] - goal_column)
        + Wpt * cp.sum_squares(P[:, -1] - goals)
        + slack_cost
        # --- ridge for strong convexity ---
        # + eps_ridge * (cp.sum_squares(U) + cp.sum_squares(P) + cp.sum_squares(V))
        )
    problem = cp.Problem(objective, constraints)

    parameters = [p0, v0, goals, obs_binary_param]
    if num_pairs > 0:
        parameters.extend([bb_param, cc_param])

    layer_variables = [U, P, V, S_obs]
    if S_pairs is not None:
        layer_variables.append(S_pairs)

    # solver_args_osqp = dict(
    #     solver=cp.OSQP,
    #     eps_abs=1e-3, eps_rel=1e-3,
    #     max_iter=4000,
    #     polish=False,
    #     verbose=False,
    #     adaptive_rho=True,
    # )

    layer = CvxpyLayer(problem, 
                       parameters=parameters, 
                       variables=layer_variables,
                    #    solver_args=solver_args_osqp
                       )

    meta = {
        "num_states": num_states,
        "horizon": H,
        "num_obs": num_obs,
        "num_pairs": num_pairs,
        "obs_bits_shape": (num_obs, M, H, 4),
        "slack_penalty": slack_penalty,
        "output_order": (
            ["U", "P", "V", "S_obs", "S_pairs"]
            if num_pairs > 0
            else ["U", "P", "V", "S_obs"]
        ),
        "parameter_order": (
            ["p0", "v0", "goals", "obs_bits", "bb", "cc"]
            if num_pairs > 0
            else ["p0", "v0", "goals", "obs_bits"]
        ),
    }

    layer.meta = meta
    layer.output_order = meta["output_order"]
    return layer, meta


__all__ = ["build_mpc_cvxpy_layer", "ObstacleInfo"]
