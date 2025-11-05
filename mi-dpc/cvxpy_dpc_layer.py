
import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer

def build_dpc_cvxpy_layer(N: int, use_soft: bool = False, slack_penalty: float = 1e3):
    # ---- Fixed system matrices (from the paper) ----
    alpha1, alpha2, nu = 0.9983, 0.9966, 0.001
    b1 = b2 = 0.075
    b3 = 0.0825
    b4 = b5 = 0.0833

    A  = torch.tensor([[alpha1, nu],
                       [0.0,    alpha2 - nu]], dtype=torch.float32)
    Bu = torch.tensor([[b1, 0.0],
                       [0.0, b2]], dtype=torch.float32)
    Bδ = torch.tensor([[0.0],
                       [b3]], dtype=torch.float32)
    E  = torch.tensor([[-b4, 0.0],
                       [0.0, -b5]], dtype=torch.float32)

    # Cost weights & reference (constants; edit if you want them as parameters)
    Q = torch.diag(torch.tensor([1.0, 1.0]))
    R = torch.diag(torch.tensor([0.5, 0.5]))
    P = torch.diag(torch.tensor([1.0, 1.0]))
    rho = 0.1
    r = torch.tensor([4.2, 1.8], dtype=torch.float32)

    # Bounds
    x_lo = torch.tensor([0.0, 0.0], dtype=torch.float32)
    x_hi = torch.tensor([8.4, 3.6], dtype=torch.float32)
    u_sum_hi = 8.0

    # Soft penalty weights (used only if use_soft=True)
    c_x = 25.0
    c_u = 25.0

    # ---- CVXPY variables (decision) ----
    x = cp.Variable((N + 1, 2))
    u = cp.Variable((N, 2))

    # ---- CVXPY parameters (given every call) ----
    x0_param    = cp.Parameter((2,))      # x0
    d_param     = cp.Parameter((N, 2))    # disturbances
    delta_param = cp.Parameter((N,))      # integer sequence (values in {0,1,2,3})

    # Wrap constants as CVXPY constants
    A_c  = cp.Constant(A.numpy())
    Bu_c = cp.Constant(Bu.numpy())
    Bδ_c = cp.Constant(Bδ.numpy())
    E_c  = cp.Constant(E.numpy())
    Q_c  = cp.Constant(Q.numpy())
    R_c  = cp.Constant(R.numpy())
    P_c  = cp.Constant(P.numpy())
    r_c  = cp.Constant(r.numpy())
    x_lo_c = cp.Constant(x_lo.numpy())
    x_hi_c = cp.Constant(x_hi.numpy())

    # ---- Constraints ----
    cons = []
    cons += [x[0, :] == x0_param]

    for k in range(N):
        # x_{k+1} = A x_k + Bu u_k + Bδ * delta_k + E d_k
        cons += [
            x[k + 1, :] ==
            A_c @ x[k, :] +
            Bu_c @ u[k, :] +
            (Bδ_c.flatten() * delta_param[k]) +   # (2,) * scalar
            E_c @ d_param[k, :]
        ]

    # Input bounds: u >= 0, u1+u2 <= u_sum_hi (hard or soft)
    obj_terms = []

    slack_vars = None
    slack_var = None
    if use_soft:
        # soft penalties: hinge^2 for state and input bounds
        def hinge_sq(z):  # elementwise pos(z)^2
            return cp.sum_squares(cp.pos(z))

        # state soft bounds for k=0..N (both sides)
        for k in range(N + 1):
            obj_terms += [c_x * hinge_sq(x_lo_c - x[k, :])]
            obj_terms += [c_x * hinge_sq(x[k, :] - x_hi_c)]

        # input soft bounds for k=0..N-1
        for k in range(N):
            obj_terms += [c_u * hinge_sq(-u[k, 0])]
            obj_terms += [c_u * hinge_sq(-u[k, 1])]
            obj_terms += [c_u * hinge_sq(u[k, 0] + u[k, 1] - u_sum_hi)]
    else:
        # compact slack tensor storing all violations
        slack_var = cp.Variable((N + 1 + N, 4), nonneg=True)
        x_lo_slack = slack_var[: N + 1, 0:2]
        x_hi_slack = slack_var[: N + 1, 2:4]
        u_lo_slack = slack_var[N + 1 :, 0:2]
        u_sum_slack = slack_var[N + 1 :, 2]

        # hard bounds as constraints
        for k in range(N + 1):
            cons += [
                x[k, :] >= x_lo_c - x_lo_slack[k, :],
                x[k, :] <= x_hi_c + x_hi_slack[k, :],
            ]
        for k in range(N):
            cons += [
                u[k, 0] >= -u_lo_slack[k, 0],
                u[k, 1] >= -u_lo_slack[k, 1],
                u[k, 0] + u[k, 1] <= u_sum_hi + u_sum_slack[k],
            ]

        obj_terms += [
            slack_penalty * cp.sum_squares(slack_var),
        ]

    # ---- Objective: quadratic tracking + control effort + rho*delta^2 + terminal
    # stage cost
    for k in range(N):
        x_err = x[k, :] - r_c
        obj_terms += [cp.quad_form(x_err, Q_c)]
        obj_terms += [cp.quad_form(u[k, :], R_c)]
        obj_terms += [rho * cp.square(delta_param[k])]  # δ is fixed parameter here

    # terminal cost
    xN_err = x[N, :] - r_c
    obj_terms += [cp.quad_form(xN_err, P_c)]

    obj = cp.sum(obj_terms)

    prob = cp.Problem(cp.Minimize(obj), cons)

    # Build layer: returns (x, u). You can also include obj.value via dual trick if needed.
    if slack_var is not None:
        layer_vars = [x, u, slack_var]
    else:
        layer_vars = [x, u]

    layer = CvxpyLayer(prob, parameters=[x0_param, d_param, delta_param], variables=layer_vars)
    return layer
