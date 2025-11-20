from dataclasses import dataclass
from typing import Tuple

import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer


@dataclass(frozen=True)
class EnergyMeta:
    horizon: int
    state_dim: int
    input_dim: int
    slack_penalty: float
    parameter_order: Tuple[str, ...]
    output_order: Tuple[str, ...]


def build_qp_cvxpy_layer(N: int = 20, slack_penalty: float = 1e3) -> Tuple[CvxpyLayer, EnergyMeta]:
    """
    Build the CVXPY layer that mirrors the energy-system MPC relaxation.
    """
    # Fixed system matrices (from the paper)
    alpha1, alpha2, nu = 0.9983, 0.9966, 0.001
    b1 = b2 = 0.075
    b3 = 0.0825
    b4 = b5 = 0.0833

    A = cp.Constant([[alpha1, nu], [0.0, alpha2 - nu]])
    Bu = cp.Constant([[b1, 0.0], [0.0, b2]])
    B_delta = cp.Constant([b3, b3])
    E = cp.Constant([[-b4, 0.0], [0.0, -b5]])

    Q = cp.Constant([[1.0, 0.0], [0.0, 1.0]])
    R = cp.Constant([[0.5, 0.0], [0.0, 0.5]])
    P = cp.Constant([[1.0, 0.0], [0.0, 1.0]])
    rho = 0.1
    r = cp.Constant([4.2, 1.8])

    x_lo = cp.Constant([0.0, 0.0])
    x_hi = cp.Constant([8.4, 3.6])
    u_sum_hi = 8.0

    # Decision variables
    X = cp.Variable((N + 1, 2))
    U = cp.Variable((N, 2))
    slack_state = cp.Variable((N + 1, 4), nonneg=True)
    slack_delta = cp.Variable((max(N - 1, 1), 2), nonneg=True)

    # Parameters per instance
    x0_param = cp.Parameter(2)
    disturbance_param = cp.Parameter((N, 2))
    delta_param = cp.Parameter(N)

    constraints = [X[0, :] == x0_param]
    for k in range(N):
        constraints += [
            X[k + 1, :]
            == A @ X[k, :] + Bu @ U[k, :] + B_delta * delta_param[k] + E @ disturbance_param[k, :]
        ]
        constraints += [
            U[k, 0] >= 0.0,
            U[k, 1] >= 0.0,
            U[k, 0] + U[k, 1] <= u_sum_hi,
        ]
        if k < N - 1:
            constraints += [
                U[k + 1, 1] - U[k, 1] + 1 >= -slack_delta[k, 0],
                U[k + 1, 1] - U[k, 1] - 1 <= slack_delta[k, 1],
            ]

    for k in range(N + 1):
        constraints += [
            X[k, :] >= x_lo - slack_state[k, 0:2],
            X[k, :] <= x_hi + slack_state[k, 2:4],
        ]

    obj_terms = []
    for k in range(N):
        x_err = X[k, :] - r
        obj_terms += [cp.quad_form(x_err, Q)]
        obj_terms += [cp.quad_form(U[k, :], R)]
        obj_terms += [rho * cp.square(delta_param[k])]
    xN_err = X[N, :] - r
    obj_terms += [cp.quad_form(xN_err, P)]
    obj_terms += [slack_penalty * (cp.sum(slack_state) + cp.sum(slack_delta))]

    problem = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints)
    layer = CvxpyLayer(
        problem,
        parameters=[x0_param, disturbance_param, delta_param],
        variables=[U, X, slack_state, slack_delta],
    )

    meta = EnergyMeta(
        horizon=N,
        state_dim=2,
        input_dim=2,
        slack_penalty=slack_penalty,
        parameter_order=("x0", "disturbances", "delta"),
        output_order=("u", "x", "slack_state", "slack_delta"),
    )
    return layer, meta


class MIQP:
    """
    Mimics the robot navigation MIQP helper but tailored to the energy task.
    """

    def __init__(self, nn_model, device=None, horizon: int = 20, slack_penalty: float = 1e3):

        self.device = device if device is not None else torch.device("cpu")
        self.nn_model = nn_model.to(self.device)
        self.cvx_layer, self.meta = build_qp_cvxpy_layer(N=horizon, slack_penalty=slack_penalty)
        self.horizon = self.meta.horizon

    def predict_y(self, theta: torch.Tensor) -> torch.Tensor:
        """Forward pass through the NN that outputs delta sequences."""
        return self.nn_model(theta.to(self.device)).float()

    def _split_theta(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split feature tensor into initial state and disturbance profile."""
        x0 = theta[:, :2]
        disturbances = theta[:, 2:].reshape(-1, self.horizon, 2)
        return x0, disturbances

    def solve_miqp(self, theta: torch.Tensor):

        theta_device = theta.to(self.device)
        y_pred = self.predict_y(theta)
        x0, disturbances = self._split_theta(theta_device)

        u_opt, x_opt, slack_state, slack_delta = self.cvx_layer(
            x0.detach().cpu(),
            disturbances.detach().cpu(),
            y_pred.detach().cpu(),
        )

        u_opt = u_opt.to(self.device)
        x_opt = x_opt.to(self.device)
        slack_state = slack_state.to(self.device)
        slack_delta = slack_delta.to(self.device)

        obj_val = obj_function(u_opt, x_opt, y_pred.to(self.device))
        sol_qp = (u_opt, x_opt, slack_state, slack_delta)

        return sol_qp, y_pred, obj_val

    def eval_solution_train(self, theta, sol_qp, y_pred, gt, y_loss_func):
        y_gt, x_gt, u_gt = gt
        u_opt, x_opt, slack_state, slack_delta = [t.to(self.device) for t in sol_qp]
        y_pred = y_pred.to(self.device)

        obj_val = obj_function(u_opt, x_opt, y_pred).mean()
        supervised_loss = y_loss_func(y_pred, y_gt.to(self.device))
        slack_loss = (slack_state.reshape(slack_state.size(0), -1).sum(dim=1) +
                      slack_delta.reshape(slack_delta.size(0), -1).sum(dim=1)).mean()
        violation_loss = violation_metric(y_pred, x_opt).mean()

        return obj_val, supervised_loss, slack_loss, violation_loss

    def eval_solution_test(self, theta, sol_qp, y_pred, gt, y_loss_func):
        y_gt, x_gt, u_gt = gt
        u_opt, x_opt, slack_state, slack_delta = [t.to(self.device) for t in sol_qp]
        y_pred = y_pred.to(self.device)

        obj_val = obj_function(u_opt, x_opt, y_pred).mean()
        supervised_loss = y_loss_func(y_pred, y_gt.to(self.device))
        slack_loss = (slack_state.reshape(slack_state.size(0), -1).sum(dim=1) +
                      slack_delta.reshape(slack_delta.size(0), -1).sum(dim=1)).mean()
        violation_loss = violation_metric(y_pred, x_opt).mean()

        opt_obj_val = obj_function(u_gt.to(self.device), x_gt.to(self.device), y_gt.to(self.device)).mean()
        violation_total, violation_percent = violation_count(y_pred, x_opt, evaluate=True)

        return (
            obj_val,
            supervised_loss,
            slack_loss,
            violation_loss,
            opt_obj_val,
            violation_total,
            violation_percent,
        )


def obj_function(u_opt, x_opt, delta, weights=None):
    """
    Compute the quadratic objective used by the CVXPY layer.
    """
    device = u_opt.device
    horizon = u_opt.shape[1]

    if weights is None:
        Q = torch.eye(2, device=device)
        R = 0.5 * torch.eye(2, device=device)
        P = torch.eye(2, device=device)
        rho = torch.tensor(0.1, device=device)
        ref = torch.tensor([4.2, 1.8], device=device)
    else:
        Q = weights["Q"].to(device)
        R = weights["R"].to(device)
        P = weights["P"].to(device)
        rho = weights["rho"].to(device)
        ref = weights["ref"].to(device)

    state_error = x_opt[:, :horizon, :] - ref.view(1, 1, 2)
    stage_cost = torch.einsum("bhi,ij,bhj->b", state_error, Q, state_error)
    control_cost = torch.einsum("bhi,ij,bhj->b", u_opt, R, u_opt)

    delta = delta.view(delta.size(0), horizon, -1)[:, :, 0]
    integer_cost = rho * (delta**2).sum(dim=1)

    terminal_error = x_opt[:, -1, :] - ref.view(1, 2)
    terminal_cost = torch.einsum("bi,ij,bj->b", terminal_error, P, terminal_error)

    return stage_cost + control_cost + integer_cost + terminal_cost


def violation_metric(dis_traj, cont_traj, evaluate=False):
    """
    Measure soft constraint violations for the energy problem.
    """
    dis = dis_traj
    if dis.dim() == 3:
        dis = dis.squeeze(-1)
    if dis.dim() == 1:
        dis = dis.unsqueeze(0)
    if cont_traj.dim() == 2:
        cont = cont_traj.unsqueeze(0)
    elif cont_traj.dim() == 3 and cont_traj.size(1) == 2 and cont_traj.size(2) != 2:
        cont = cont_traj.transpose(1, 2)
    elif cont_traj.dim() == 3 and cont_traj.size(2) == 2:
        cont = cont_traj
    else:
        raise ValueError("Unexpected continuous trajectory shape")

    device = cont.device
    x_lo = torch.tensor([0.0, 0.0], device=device)
    x_hi = torch.tensor([8.4, 3.6], device=device)

    d_delta = dis[:, 1:] - dis[:, :-1]
    c1 = torch.relu(d_delta - 1)
    c2 = torch.relu(-d_delta - 1)
    dis_violation = c1.sum(dim=1) + c2.sum(dim=1)

    lower_violation = torch.relu(x_lo.view(1, 1, 2) - cont).sum(dim=(1, 2))
    upper_violation = torch.relu(cont - x_hi.view(1, 1, 2)).sum(dim=(1, 2))
    con_violation = lower_violation + upper_violation

    if evaluate:
        return con_violation, dis_violation
    return con_violation + dis_violation


def violation_count(dis_traj, cont_traj, tolerance=1e-6, evaluate=False):
    """
    Count the number of hard constraint violations.
    """
    dis = dis_traj
    if dis.dim() == 3:
        dis = dis.squeeze(-1)
    if dis.dim() == 1:
        dis = dis.unsqueeze(0)
    if cont_traj.dim() == 2:
        cont = cont_traj.unsqueeze(0)
    elif cont_traj.dim() == 3 and cont_traj.size(1) == 2 and cont_traj.size(2) != 2:
        cont = cont_traj.transpose(1, 2)
    elif cont_traj.dim() == 3 and cont_traj.size(2) == 2:
        cont = cont_traj
    else:
        raise ValueError("Unexpected continuous trajectory shape")

    device = cont.device
    x_lo = torch.tensor([0.0, 0.0], device=device)
    x_hi = torch.tensor([8.4, 3.6], device=device)

    d_delta = dis[:, 1:] - dis[:, :-1]
    delta_viol = (d_delta - 1 > tolerance).sum(dim=1) + (-d_delta - 1 > tolerance).sum(dim=1)

    lower = (x_lo.view(1, 1, 2) - cont > tolerance).sum(dim=(1, 2))
    upper = (cont - x_hi.view(1, 1, 2) > tolerance).sum(dim=(1, 2))
    cont_viol = lower + upper

    total = delta_viol + cont_viol
    total_constraints = float(2 * cont.size(1) + 2 * max(dis.size(1) - 1, 0))
    violation_rate = total.float() / max(total_constraints, 1.0)

    if evaluate:
        return total, violation_rate
    return total, violation_rate
