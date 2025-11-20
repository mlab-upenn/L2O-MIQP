import torch

def obj_function(u_opt, x_opt, y, meta=None):
    """
    Example objective function computation.
    Args:
        u_opt: Optimal control inputs (B, H, 2)
        x_opt: Optimal states (B, H+1, 2)
        y: integer variables (B, H) - delta values in {0,1,2,3}
        meta: Dictionary containing objective parameters
    Returns:
        Objective value tensor (B,).
    """
    device = u_opt.device
    B, H, _ = u_opt.shape
    
    # Default weights (from data_collection.py)
    if meta is None:
        Q = torch.eye(2, device=device)           # state cost
        R = torch.eye(2, device=device) * 0.5           # control cost
        P = torch.eye(2, device=device)           # terminal cost
        rho = torch.tensor(0.1, device=device)    # integer variable penalty
        x_ref = torch.tensor([4.2, 1.8], device=device).view(1, 1, 2).expand(B, H+1, 2)
    else:
        Q = meta.get('Q', torch.eye(2)).to(device)
        R = meta.get('R', torch.eye(2)).to(device)
        P = meta.get('P', torch.eye(2)).to(device)
        rho = meta.get('rho', torch.tensor(0.1)).to(device)
        x_ref = meta.get('x_ref', torch.tensor([4.2, 1.8], device=device).view(1, 1, 2).expand(B, H+1, 2)).to(device)
    
    # Stage costs over horizon (k = 0 to H-1)
    # State cost: sum_k (x_k - x_ref)^T Q (x_k - x_ref)
    state_error = x_opt[:, :-1, :] - x_ref[:, :-1, :]  # (B, H, 2) - exclude terminal state
    state_cost = torch.einsum('bhi,ij,bhj->b', state_error, Q, state_error)
    
    # Control cost: sum_k u_k^T R u_k
    control_cost = torch.einsum('bhi,ij,bhj->b', u_opt, R, u_opt)
    
    # Integer variable penalty: sum_k rho * delta_k^2
    if y.dim() == 2:  # (B, H)
        integer_cost = rho * (y ** 2).sum(dim=1)  # (B,)
    elif y.dim() == 3:  # (B, H, 1) or similar
        integer_cost = rho * (y ** 2).sum(dim=[1, 2])  # (B,)
    else:
        integer_cost = torch.zeros(B, device=device)
    
    # Terminal cost: (x_N - x_ref)^T P (x_N - x_ref)
    terminal_error = x_opt[:, -1, :] - x_ref[:, -1, :]  # (B, 2)
    terminal_cost = torch.einsum('bi,ij,bj->b', terminal_error, P, terminal_error)
    
    # Total objective
    obj = state_cost + control_cost + integer_cost + terminal_cost  # (B,)
    return obj

def constraint_violation_torch(dis_traj, cont_traj, evaluate = False):
    """
    dis_traj: (B, H)
    cont_traj: (B, 2, H+1)
    Returns: scalar tensor or per-sample tensor of constraint violations
    """
    device = cont_traj.device
    dtype = cont_traj.dtype
    # bounds on states
    x_lo = torch.tensor([0.0, 0.0], dtype=torch.float32).to(device)
    x_hi = torch.tensor([8.4, 3.6], dtype=torch.float32).to(device)

    delta = dis_traj.to(dtype)
    d_delta = delta[:, 1:] - delta[:, :-1]
    c1 = torch.relu(d_delta - 1)
    c2 = torch.relu(- d_delta - 1)
    dis_violation = c1.sum(dim=1) + c2.sum(dim=1)
    
    # x[k, :] >= x_lo and x[k, :] <= x_hi
    # Expand to broadcast properly
    x_lo = x_lo.view(1, 1, 2)
    x_hi = x_hi.view(1, 1, 2)
    # Compute ReLU penalties
    lower_violation = torch.relu(x_lo - cont_traj).sum(dim=(1,2))   # lower bound
    upper_violation = torch.relu(cont_traj - x_hi).sum(dim=(1,2))   # upper bound
    # Combine violations and sum over the horizon 
    con_violation = (lower_violation + upper_violation)
    if evaluate:
        return con_violation, dis_violation
    violation = dis_violation + con_violation
    return violation