import torch

def obj_function(u_opt, p_opt, theta, meta):
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
    tracking_target = target_pos.unsqueeze(-1).expand_as(p_opt[:, :, :-1])
    tracking_cost = torch.sum((p_opt[:, :, :-1] - tracking_target) ** 2, dim=(1, 2))
    terminal_cost = torch.sum((p_opt[:, :, -1] - target_pos) ** 2, dim=1)

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

# torch version of the function above
def constraint_violation_torch(dis_traj, cont_traj, Obs_info=None, evaluate = False):
    """
    dis_traj: (N_obs, 4, H) or (B, N_obs, 4, H)
    cont_traj: (2, H) or (B, 2, H)
    Returns: scalar tensor or per-sample tensor of constraint violations
    """
    squeeze_batch = False
    if dis_traj.dim() == 3:
        dis_traj = dis_traj.unsqueeze(0)
        cont_traj = cont_traj.unsqueeze(0)
        squeeze_batch = True

    if cont_traj.dim() == 2:
        cont_traj = cont_traj.unsqueeze(0)

    if dis_traj.dim() != 4 or cont_traj.dim() != 3:
        raise ValueError("Unexpected input shapes for constraint_violation_torch")

    device = cont_traj.device
    dtype = cont_traj.dtype
    bigM = 1e3
    d_min = 0.25
    if Obs_info is not None:
        Obs_info = torch.tensor(Obs_info, device=device, dtype=dtype)
    else:
        Obs_info = torch.tensor([
            [1.0, 0.0, 0.4, 0.5, 0.0],
            [0.7, -1.1, 0.5, 0.4, 0.0],
            [0.40, -2.50, 0.4, 0.5, 0.0]
        ], device=device, dtype=dtype)

    N_obs = dis_traj.size(1)
    if Obs_info.size(0) != N_obs:
        raise ValueError("Mismatch between obstacle info and discrete trajectory shape")

    x = cont_traj[:, 0, 1:]
    y = cont_traj[:, 1, 1:]

    rel_x = x.unsqueeze(1) - Obs_info[:, 0].view(1, -1, 1)
    rel_y = y.unsqueeze(1) - Obs_info[:, 1].view(1, -1, 1)
    rel_coords = torch.stack((rel_x, rel_y), dim=2)  # (B, N_obs, 2, H)

    if dis_traj.size(-1) != rel_coords.size(-1):
        raise ValueError("Mismatch between discrete trajectory horizon and continuous trajectory horizon")

    th = Obs_info[:, 4]
    cos_th = torch.cos(th)
    sin_th = torch.sin(th)
    rot_mat = torch.stack(
        (
            torch.stack((cos_th,  sin_th), dim=1),
            torch.stack((-sin_th, cos_th), dim=1),
            torch.stack((-cos_th, -sin_th), dim=1),
            torch.stack((sin_th, -cos_th), dim=1),
        ),
        dim=1,
    ).unsqueeze(0)  # (1, N_obs, 4, 2)

    proj = torch.matmul(rot_mat, rel_coords)  # (B, N_obs, 4, H)

    L0 = Obs_info[:, 2] / 2 + d_min
    W0 = Obs_info[:, 3] / 2 + d_min
    thresholds = torch.stack((L0, W0, L0, W0), dim=1).unsqueeze(0).unsqueeze(-1)

    d = dis_traj.to(dtype)
    c = torch.relu(proj - thresholds - bigM * (1 - d))
    c5 = torch.relu(1 - d.sum(dim=2))

    con_violation = c.sum(dim=(1, 2, 3))
    dis_violation = c5.sum(dim=(1, 2))

    violation = con_violation + dis_violation
    if evaluate:
        return con_violation, dis_violation
    
    if squeeze_batch:
        return violation.squeeze(0)
    return violation

def constraint_violation_count_torch(dis_traj, cont_traj, Obs_info=None, tolerance=1e-6):
    """
    Count the number of constraint violations (not magnitude).
    
    dis_traj: (N_obs, 4, H) or (B, N_obs, 4, H)
    cont_traj: (2, H) or (B, 2, H)
    tolerance: Small value to determine if a constraint is violated
    
    Returns: integer tensor counting number of violated constraints
    """
    squeeze_batch = False
    if dis_traj.dim() == 3:
        dis_traj = dis_traj.unsqueeze(0)
        cont_traj = cont_traj.unsqueeze(0)
        squeeze_batch = True

    if cont_traj.dim() == 2:
        cont_traj = cont_traj.unsqueeze(0)

    if dis_traj.dim() != 4 or cont_traj.dim() != 3:
        raise ValueError("Unexpected input shapes for constraint_violation_count_torch")

    device = cont_traj.device
    dtype = cont_traj.dtype
    bigM = 1e3
    d_min = 0.25
    if Obs_info is not None:
        Obs_info = torch.tensor(Obs_info, device=device, dtype=dtype)
    else:
        Obs_info = torch.tensor([
            [1.0, 0.0, 0.4, 0.5, 0.0],
            [0.7, -1.1, 0.5, 0.4, 0.0],
            [0.40, -2.50, 0.4, 0.5, 0.0]
        ], device=device, dtype=dtype)

    N_obs = dis_traj.size(1)
    if Obs_info.size(0) != N_obs:
        raise ValueError("Mismatch between obstacle info and discrete trajectory shape")

    x = cont_traj[:, 0, 1:]
    y = cont_traj[:, 1, 1:]

    rel_x = x.unsqueeze(1) - Obs_info[:, 0].view(1, -1, 1)
    rel_y = y.unsqueeze(1) - Obs_info[:, 1].view(1, -1, 1)
    rel_coords = torch.stack((rel_x, rel_y), dim=2)  # (B, N_obs, 2, H)

    if dis_traj.size(-1) != rel_coords.size(-1):
        raise ValueError("Mismatch between discrete trajectory horizon and continuous trajectory horizon")

    th = Obs_info[:, 4]
    cos_th = torch.cos(th)
    sin_th = torch.sin(th)
    rot_mat = torch.stack(
        (
            torch.stack((cos_th,  sin_th), dim=1),
            torch.stack((-sin_th, cos_th), dim=1),
            torch.stack((-cos_th, -sin_th), dim=1),
            torch.stack((sin_th, -cos_th), dim=1),
        ),
        dim=1,
    ).unsqueeze(0)  # (1, N_obs, 4, 2)

    proj = torch.matmul(rot_mat, rel_coords)  # (B, N_obs, 4, H)

    L0 = Obs_info[:, 2] / 2 + d_min
    W0 = Obs_info[:, 3] / 2 + d_min
    thresholds = torch.stack((L0, W0, L0, W0), dim=1).unsqueeze(0).unsqueeze(-1)

    d = dis_traj.to(dtype)
    
    # Count projection constraint violations (c > tolerance)
    c = proj - thresholds - bigM * (1 - d)
    c_violations = (c > tolerance).sum(dim=(1, 2, 3))
    
    # Count binary constraint violations only when the selection shortfall is positive
    shortfall = 1 - d.sum(dim=2)
    c5_violations = (shortfall > tolerance).sum(dim=(1, 2))

    violation_count = c_violations + c5_violations
    constraint_count = c.numel() + shortfall.numel()
    violation_rate = violation_count/constraint_count

    if squeeze_batch:
        return violation_count.squeeze(0)
    return violation_count, violation_rate
