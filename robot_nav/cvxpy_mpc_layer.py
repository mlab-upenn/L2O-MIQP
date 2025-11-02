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


def build_mpc_cvxpy_layer(
    T: float,
    H: int,
    M: int,
    bounds: dict,
    weights: Tuple[float, float, float],
    d_min: float,
    obstacles: Sequence,
    coupling_pairs: Iterable[Tuple[int, int]],
    big_m: float = 1e3,
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
        Layer that maps parameters to optimal continuous trajectories.
    meta : dict
        Metadata describing parameter ordering and dimensions for future calls.
    """
    Wu, Wp, Wpt = map(float, weights)
    num_states = 2 * M
    horizon_vars = H + 1
    prepared_obs = _prepare_obstacles(obstacles)
    num_obs = len(prepared_obs)

    unique_pairs = [(min(i, j), max(i, j)) for i, j in coupling_pairs if i < j]
    num_pairs = len(unique_pairs)

    # Decision variables.
    P = cp.Variable((num_states, horizon_vars))
    V = cp.Variable((num_states, horizon_vars))
    U = cp.Variable((num_states, H))

    # Parameters that will be passed in at solve time.
    p0 = cp.Parameter(num_states)
    v0 = cp.Parameter(num_states)
    goals = cp.Parameter(num_states)

    # Binary activation patterns (4 inequalities per obstacle) as parameters.
    obs_binary_param = cp.Parameter((num_obs, M, H, 4))
    if num_pairs > 0:
        bb_param = cp.Parameter((num_pairs, H))
        cc_param = cp.Parameter((num_pairs, H))
    else:
        bb_param = None
        cc_param = None

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
                >= L0[obs_idx] - big_m * obs_binary_param[obs_idx, m, :, 0],
                -sth * (px - xo[obs_idx]) + cth * (py - yo[obs_idx])
                >= W0[obs_idx] - big_m * obs_binary_param[obs_idx, m, :, 1],
                -cth * (px - xo[obs_idx]) - sth * (py - yo[obs_idx])
                >= L0[obs_idx] - big_m * obs_binary_param[obs_idx, m, :, 2],
                sth * (px - xo[obs_idx]) - cth * (py - yo[obs_idx])
                >= W0[obs_idx] - big_m * obs_binary_param[obs_idx, m, :, 3],
            ]

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
                px_m - px_n >= 2 * d_min - big_m * (bb + cc),
                px_n - px_m >= 2 * d_min - big_m * (1 - bb + cc),
                py_m - py_n >= 2 * d_min - big_m * (1 + bb - cc),
                py_n - py_m >= 2 * d_min - big_m * (2 - bb - cc),
            ]

    # Objective mirrors the original quadratic cost.
    objective = cp.Minimize(
        Wu * cp.sum_squares(U)
        + Wp * cp.sum_squares(P[:, :H] - goal_column)
        + Wpt * cp.sum_squares(P[:, -1] - goals)
    )

    problem = cp.Problem(objective, constraints)

    parameters = [p0, v0, goals, obs_binary_param]
    if num_pairs > 0:
        parameters.extend([bb_param, cc_param])

    layer = CvxpyLayer(problem, parameters=parameters, variables=[U, P, V])

    meta = {
        "num_states": num_states,
        "horizon": H,
        "num_obs": num_obs,
        "num_pairs": num_pairs,
        "obs_bits_shape": (num_obs, M, H, 4),
        "parameter_order": (
            ["p0", "v0", "goals", "obs_bits", "bb", "cc"]
            if num_pairs > 0
            else ["p0", "v0", "goals", "obs_bits"]
        ),
    }

    return layer, meta


__all__ = ["build_mpc_cvxpy_layer", "ObstacleInfo"]
