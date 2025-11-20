from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch
from Robots import obstacle

class MIQP:

    def __init__(self, nn_model, device):

        self.device = device if device is not None else torch.device("cpu")
        self.nn_model = nn_model.to(self.device)
        self.cvx_layer, self.cvx_layer_meta = build_qp_cvxpy_layer()
        self.num_obstacles = self.cvx_layer_meta["num_obs"]

    def predict_y(self, theta: torch.Tensor) -> torch.Tensor:
        """Neural network forward pass on the configured device."""
        return self.nn_model(theta.to(self.device)).float()

    def solve_miqp(self, theta):

        device = self.device

        theta_device = theta.to(device)
        y_pred = self.predict_y(theta)

        # Reshape the inputs for the CVXPY layer, to match expected dimensions
        # CVXPY layer call, CVXPYLayer is CPU only
        solution = self.cvx_layer(
            theta_device[:, 0:2].cpu(),
            theta_device[:, 2:4].cpu(),
            theta_device[:, 4:6].cpu(),
            y_pred.reshape(-1, 3, 1, 20, 4).cpu(),
        )

        u_opt, p_opt, v_opt, s_opt = solution
        u_opt = u_opt.to(device)
        p_opt = p_opt.to(device)
        v_opt = v_opt.to(device)
        s_opt = s_opt.to(device)

        obj_val = obj_function(u_opt.to(device), p_opt, theta_device)
        sol_qp = (u_opt, p_opt, v_opt, s_opt)

        return sol_qp, y_pred, obj_val

    def eval_solution_train(self, theta, sol_qp, y_pred, gt, y_loss_func):
        device = self.device

        u_opt, p_opt, v_opt, s_opt = [t.to(device) for t in sol_qp]
        y_gt, p_gt, v_gt = gt
        theta = theta.to(device)

        obj_val = obj_function(u_opt, p_opt, theta).mean()
        supervised_loss = y_loss_func(y_pred.to(device), y_gt.to(device))
        slack_loss = s_opt.sum(dim=1).mean()
        y_transposed = NNoutput_reshape_torch(y_pred.to(device), self.num_obstacles)
        violation_loss = violation_metric(y_transposed, p_opt).mean()

        return obj_val, supervised_loss, slack_loss, violation_loss

    def eval_solution_test(self, theta, solution, y_pred, gt, y_loss_func):
        device = self.device

        u_opt, p_opt, v_opt, s_opt = [t.to(device) for t in solution]
        y_gt, pv_gt, u_gt = gt
        theta = theta.to(device)

        obj_val = obj_function(u_opt, p_opt, theta).mean()
        supervised_loss = y_loss_func(y_pred.to(device), y_gt.to(device))
        slack_loss = s_opt.sum(dim=1).mean()
        y_transposed = NNoutput_reshape_torch(y_pred.to(device), self.num_obstacles)
        violation_loss = violation_metric(y_transposed, p_opt).mean()

        opt_obj_val = obj_function(u_gt.to(device), pv_gt.to(device), theta).mean()
        violation_total, violation_percent = violation_count(
            y_transposed, p_opt, evaluate=True
        )

        return (
            obj_val,
            supervised_loss,
            slack_loss,
            violation_loss,
            opt_obj_val,
            violation_total,
            violation_percent,
        )

"""
## BUILD CVXPY LAYER
"""

@dataclass(frozen=True)
class ObstacleInfo:
    """Lightweight container capturing obstacle geometry."""

    x_c: float
    y_c: float
    L: float  # already half-length in the controller
    W: float  # already half-width in the controller
    theta: float


_DEFAULT_OBSTACLES = [
    obstacle(1.0, 0.0, 0.4, 0.5, 0.0),
    obstacle(0.7, -1.1, 0.5, 0.4, 0.0),
    obstacle(0.40, -2.50, 0.4, 0.5, 0.0),
]


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


def _default_bounds() -> dict:
    return {
        "x_max": 2.00,
        "x_min": -0.5,
        "y_max": 0.5,
        "y_min": -3.0,
        "v_max": 0.50,
        "v_min": -0.50,
        "u_max": 0.50,
        "u_min": -0.50,
    }


def _prepare_coupling_pairs(
    M: int, coupling_pairs: Iterable[Tuple[int, int]] = None
) -> List[Tuple[int, int]]:
    if M < 2:
        return []
    if coupling_pairs is None:
        return [(m, n) for m in range(M) for n in range(m + 1, M)]
    pairs = {(min(i, j), max(i, j)) for i, j in coupling_pairs if i != j}
    return sorted(pairs)


def build_qp_cvxpy_layer(
    T: float=None,
    H: int=None,
    M: int=None,
    bounds: dict=None,
    weights: Tuple[float, float, float]=None,
    d_min: float=None,
    obstacles: Sequence=None,
    coupling_pairs: Iterable[Tuple[int, int]]=None,
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
    bounds = _default_bounds() if bounds is None else dict(bounds)
    weights = (1.0, 1.0, 10.0) if weights is None else weights  # (Wu, Wp, Wpt)
    d_min = 0.25 if d_min is None else float(d_min)
    obstacles = _DEFAULT_OBSTACLES if obstacles is None else obstacles
    coupling_pairs = _prepare_coupling_pairs(M, coupling_pairs)
    big_m = float(big_m)

    Wu, Wp, Wpt = map(float, weights)

    num_states = 2 * M
    horizon_vars = H + 1
    prepared_obs = _prepare_obstacles(obstacles)
    num_obs = len(prepared_obs)

    unique_pairs = coupling_pairs
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

"""
### CONSTRAINTS and LOSS
"""

def obj_function(u_opt, p_opt, theta):
    """
    Example objective function computation.
    Args:
        u_opt: Optimal control inputs (B, nu, H)
        p_opt: Optimal positions (B, 2, H)
        x: Initial states (B, nx)
        meta: Dictionary containing objective parameters
    Returns:
        Objective value tensor of shape (B,).
    """
    Wu, Wp, Wpt = 1.0, 1.0, 10.0

    u_cost = torch.sum(u_opt ** 2, dim=(1, 2))
    target_pos = theta[:, 4:]
    pos_traj = p_opt[:, :2, :]
    tracking_target = target_pos.unsqueeze(-1).expand_as(pos_traj[:, :, :-1])
    tracking_cost = torch.sum((pos_traj[:, :, :-1] - tracking_target) ** 2, dim=(1, 2))
    terminal_cost = torch.sum((pos_traj[:, :, -1] - target_pos) ** 2, dim=1)

    objective_value = Wu * u_cost + Wp * tracking_cost + Wpt * terminal_cost
    return objective_value

# torch version of the function above
def NNoutput_reshape_torch(outputs: torch.Tensor, N_obs: int):
    """Reshape NN outputs to (N_obs, 4, H) or (B, N_obs, 4, H)."""
    if outputs.dim() == 1:
        total_dim = outputs.numel()
        if total_dim % (4 * N_obs) != 0:
            raise ValueError("Output length not divisible by expected obstacle/bin count")
        outputs = outputs.view(N_obs, -1)
        H = outputs.shape[1] // 4
        dis_traj = outputs.view(N_obs, H, 4).transpose(1, 2).contiguous()
        return dis_traj

    if outputs.dim() == 2:
        batch = outputs.size(0)
        total_dim = outputs.size(1)
        if total_dim % (4 * N_obs) != 0:
            raise ValueError("Output length not divisible by expected obstacle/bin count")
        outputs = outputs.view(batch, N_obs, -1)
        H = outputs.size(2) // 4
        dis_traj = outputs.view(batch, N_obs, H, 4).transpose(2, 3).contiguous()
        return dis_traj

    if outputs.dim() == 3 and outputs.size(1) == 4:
        return outputs

    if outputs.dim() == 4 and outputs.size(2) == 4:
        return outputs

    raise ValueError("Unexpected output shape for NNoutput_reshape_torch")

def _prepare_violation_inputs(dis_traj, cont_traj):
    squeeze_batch = False
    if dis_traj.dim() == 3:
        dis_traj = dis_traj.unsqueeze(0)
        cont_traj = cont_traj.unsqueeze(0)
        squeeze_batch = True
    if cont_traj.dim() == 2:
        cont_traj = cont_traj.unsqueeze(0)
    if dis_traj.dim() != 4 or cont_traj.dim() != 3:
        raise ValueError("Unexpected input shapes for constraint violation check")
    return dis_traj, cont_traj, squeeze_batch


def _get_obstacle_tensor(obs_info, device, dtype):
    if obs_info is None:
        base = torch.tensor(
            [
                [1.0, 0.0, 0.4, 0.5, 0.0],
                [0.7, -1.1, 0.5, 0.4, 0.0],
                [0.40, -2.50, 0.4, 0.5, 0.0],
            ],
            dtype=dtype,
            device=device,
        )
    else:
        base = torch.as_tensor(obs_info, dtype=dtype, device=device)
    return base


def _relative_obstacle_projections(dis_traj, cont_traj, obs_info, d_min=0.25):
    N_obs = dis_traj.size(1)
    if obs_info.size(0) != N_obs:
        raise ValueError("Mismatch between obstacle info and discrete trajectory shape")
    x = cont_traj[:, 0, 1:]
    y = cont_traj[:, 1, 1:]
    rel_x = x.unsqueeze(1) - obs_info[:, 0].view(1, -1, 1)
    rel_y = y.unsqueeze(1) - obs_info[:, 1].view(1, -1, 1)
    rel_coords = torch.stack((rel_x, rel_y), dim=2)
    if dis_traj.size(-1) != rel_coords.size(-1):
        raise ValueError("Mismatch between discrete and continuous horizons")
    th = obs_info[:, 4]
    cos_th = torch.cos(th)
    sin_th = torch.sin(th)
    rot_mat = torch.stack(
        (
            torch.stack((cos_th, sin_th), dim=1),
            torch.stack((-sin_th, cos_th), dim=1),
            torch.stack((-cos_th, -sin_th), dim=1),
            torch.stack((sin_th, -cos_th), dim=1),
        ),
        dim=1,
    ).unsqueeze(0)
    proj = torch.matmul(rot_mat, rel_coords)
    L0 = obs_info[:, 2] / 2 + d_min
    W0 = obs_info[:, 3] / 2 + d_min
    thresholds = torch.stack((L0, W0, L0, W0), dim=1).unsqueeze(0).unsqueeze(-1)
    return proj, thresholds


def violation_metric(dis_traj, cont_traj, Obs_info=None, evaluate=False):
    """
    dis_traj: (N_obs, 4, H) or (B, N_obs, 4, H)
    cont_traj: (2, H) or (B, 2, H)
    Returns: scalar tensor or per-sample tensor of constraint violations
    """
    dis_traj, cont_traj, squeeze_batch = _prepare_violation_inputs(dis_traj, cont_traj)
    device, dtype = cont_traj.device, cont_traj.dtype
    obs_info = _get_obstacle_tensor(Obs_info, device, dtype)
    proj, thresholds = _relative_obstacle_projections(dis_traj, cont_traj, obs_info)

    bigM = 1e3
    d = dis_traj.to(dtype)
    c = torch.relu(proj - thresholds - bigM * (1 - d))
    c5 = torch.relu(1 - d.sum(dim=2))
    con_violation = c.sum(dim=(1, 2, 3))
    dis_violation = c5.sum(dim=(1, 2))
    if evaluate:
        return con_violation, dis_violation
    violation = con_violation + dis_violation
    if squeeze_batch:
        return violation.squeeze(0)
    return violation


def violation_count(dis_traj, cont_traj, Obs_info=None, tolerance=1e-6, evaluate=False):
    """
    Count the number of constraint violations (not magnitude).
    """
    dis_traj, cont_traj, squeeze_batch = _prepare_violation_inputs(dis_traj, cont_traj)
    device, dtype = cont_traj.device, cont_traj.dtype
    obs_info = _get_obstacle_tensor(Obs_info, device, dtype)
    proj, thresholds = _relative_obstacle_projections(dis_traj, cont_traj, obs_info)

    bigM = 1e3
    d = dis_traj.to(dtype)
    c = proj - thresholds - bigM * (1 - d)
    c_violations = (c > tolerance).sum(dim=(1, 2, 3))
    shortfall = 1 - d.sum(dim=2)
    c5_violations = (shortfall > tolerance).sum(dim=(1, 2))
    violation_count = c_violations + c5_violations
    constraint_count = c.numel() + shortfall.numel()
    violation_rate = violation_count / constraint_count

    if squeeze_batch:
        violation_count = violation_count.squeeze(0)
        violation_rate = violation_rate.squeeze(0)
    if evaluate:
        return violation_count, violation_rate
    return violation_count, violation_rate
