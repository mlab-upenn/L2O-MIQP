import numpy as np
import numpy.linalg as LA
import torch
import torch.nn as nn
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple


# def _cvx_output_order(layer) -> Optional[Sequence[str]]:
#     meta = getattr(layer, "meta", None)
#     if isinstance(meta, dict):
#         order = meta.get("output_order")
#         if order:
#             return list(order)
#     order = getattr(layer, "output_order", None)
#     if order:
#         return list(order)
#     return None


# def extract_cvx_outputs(layer, outputs) -> Dict[str, Any]:
#     """
#     Normalize CVX layer outputs into a dictionary keyed by variable name.
#     Works with the MPC layer (rich output order) and the simple toy layers
#     that only expose x/s slacks.
#     """
#     if isinstance(outputs, dict):
#         return outputs

#     if not isinstance(outputs, (tuple, list)):
#         outputs = (outputs,)

#     mapping: Dict[str, Any] = {}
#     output_order = _cvx_output_order(layer)

#     if output_order and len(output_order) == len(outputs):
#         mapping.update(dict(zip(output_order, outputs)))
#     else:
#         default_orders = {
#             2: ("x_opt", "s_opt"),
#             4: ("U", "P", "V", "S_obs"),
#             5: ("U", "P", "V", "S_obs", "obj_value"),
#             6: ("U", "P", "V", "S_obs", "obj_value", "S_pairs"),
#         }
#         names = default_orders.get(len(outputs))
#         if names is not None:
#             mapping.update(dict(zip(names, outputs)))

#     # Populate common aliases where possible
#     if "P" in mapping and "x_opt" not in mapping:
#         mapping["x_opt"] = mapping["P"]
#     if "S_obs" in mapping and "s_opt" not in mapping:
#         mapping["s_opt"] = mapping["S_obs"]
#     if "obj_value" not in mapping and len(outputs) >= 5:
#         mapping["obj_value"] = outputs[4]
#     if "S_obs" not in mapping and len(outputs) >= 4:
#         mapping["S_obs"] = outputs[3]
#         mapping.setdefault("s_opt", outputs[3])
#     if "x_opt" not in mapping and len(outputs) >= 1:
#         mapping["x_opt"] = outputs[0]
#     if "s_opt" not in mapping and len(outputs) >= 2:
#         mapping["s_opt"] = outputs[1]

#     return mapping


# def reshape_obs_bits(layer, obs_bits: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
#     """
#     Reshape flattened binary activation tensors to the layout expected by the CVX layer.
#     Falls back to the original tensor when metadata is unavailable.
#     """
#     if obs_bits is None:
#         return None

#     obs_shape = None
#     meta = getattr(layer, "meta", None)
#     if isinstance(meta, dict):
#         obs_shape = meta.get("obs_bits_shape")

#     if obs_shape is None:
#         return obs_bits

#     obs_shape = tuple(int(dim) for dim in obs_shape)
#     target_dim = len(obs_shape)

#     if obs_bits.dim() == target_dim + 1:
#         return obs_bits

#     expected = int(np.prod(obs_shape))

#     if obs_bits.dim() == 2 and obs_bits.size(1) == expected:
#         return obs_bits.view(obs_bits.size(0), *obs_shape)

#     if obs_bits.dim() == 1 and obs_bits.numel() % expected == 0:
#         batch = obs_bits.numel() // expected
#         return obs_bits.view(batch, *obs_shape)

#     raise ValueError(
#         f"Cannot reshape obs_bits with shape {tuple(obs_bits.shape)} "
#         f"to expected layout (batch, {obs_shape})."
#     )


# def run_cvx_layer(layer, feature_block: torch.Tensor, obs_bits: torch.Tensor,
#                   splits: Tuple[int, int, int] = (2, 4, 6),
#                   return_dict: bool = False):
#     """
#     Convenience wrapper for batched calls to the MPC CVXPY layer.

#     Parameters
#     ----------
#     layer : cvxpylayers.torch.CvxpyLayer
#         Layer to evaluate.
#     feature_block : torch.Tensor
#         Tensor containing concatenated problem parameters; by default the first
#         two columns encode p0, the next two v0, and the final two the goal.
#     obs_bits : torch.Tensor
#         Flattened or pre-shaped binary activation patterns.
#     splits : (int, int, int), optional
#         Indices describing how to slice feature_block into (p0, v0, goals).
#     return_dict : bool, optional
#         When True, return the full mapping of outputs; otherwise unpack the
#         standard tuple (U, P, V, S_obs, obj_value).
#     """
#     p_end, v_end, g_end = splits
#     if feature_block.size(1) < g_end:
#         raise ValueError(
#             f"Expected feature block with at least {g_end} columns, got {feature_block.size(1)}"
#         )

#     p0 = feature_block[:, :p_end]
#     v0 = feature_block[:, p_end:v_end]
#     goals = feature_block[:, v_end:g_end]
#     obs_bits_reshaped = reshape_obs_bits(layer, obs_bits)

#     outputs = layer(p0, v0, goals, obs_bits_reshaped)
#     output_map = extract_cvx_outputs(layer, outputs)

#     if return_dict:
#         return output_map

#     u_opt = output_map.get("U", output_map.get("x_opt"))
#     p_opt = output_map.get("P")
#     v_opt = output_map.get("V")
#     s_opt = output_map.get("S_obs", output_map.get("s_opt"))
#     obj_value = output_map.get("obj_value")

#     return u_opt, p_opt, v_opt, s_opt, obj_value

def binary_to_decimal(binary_array):
    powers_of_two = 2**np.arange(len(binary_array))[::-1]  # Compute powers of 2
    integer_2 = np.sum(binary_array * powers_of_two)
    return int(integer_2)

def decimal_to_binary(n, bits=20):
    binary = bin(n)[2:]
    binary = binary.zfill(bits)
    return np.array([int(bit) for bit in binary], dtype=np.int8)

# Function to check whether initial locations have collisions
def check_inter_collision(s, d_min):
    M = s.shape[1]
    # Check for inter-robot collisions
    for i in range(M):
        for j in range(i + 1, M):
            if LA.norm(s[:, i] - s[:, j]) <= 2*d_min:
                return False
    return True

def check_obstacle_collision(s, obstacles, d_min):
    """
    Checks if the initial positions are collision-free.
    """
    M = s.shape[1]
    n_obs = len(obstacles)

    # Check for collisions with obstacles
    for i in range(M):
        for obs in obstacles:
            x0, y0, theta = obs.x_c, obs.y_c, obs.theta
            L0 = obs.L
            W0 = obs.W
            d1 = np.cos(theta)*s[0, i] + np.sin(theta)*s[1, i] - (d_min+L0 + x0*np.cos(theta) + y0*np.sin(theta))
            d2 = -np.cos(theta)*s[0, i] - np.sin(theta)*s[1, i] - (d_min+L0 - x0*np.cos(theta) - y0*np.sin(theta))
            d3 = np.sin(theta)*s[0, i] - np.cos(theta)*s[1, i] - (d_min+W0 + x0*np.sin(theta) - y0*np.cos(theta))
            d4 = -np.sin(theta)*s[0, i] + np.cos(theta)*s[1, i] - (d_min+W0 - x0*np.sin(theta) + y0*np.cos(theta))

            if (d1 <= 0) and (d2 <= 0) and (d3 <= 0) and (d4 <= 0):
                return False
    return True

# Function to create feasible goal positions
def goal_position(obstacles, bounds, d_min, M):
    while True:
        s = np.vstack((
            np.round(np.random.uniform(bounds["x_min"], bounds["x_max"], (1, M)), 2),
            np.round(np.random.uniform(bounds["y_min"], bounds["y_max"], (1, M)), 2)
        ))
        if check_obstacle_collision(s, obstacles, d_min) and check_inter_collision(s, d_min):
            return s
        
# Function to create feasible initial positions
def init_position(obstacles, goals, bounds, d_min, M, d_sep = 0.0, d_dec = 0.0/100):
    while True:
        s = np.vstack((
            np.round(np.random.uniform(bounds["x_min"], bounds["x_max"], (1, M)), 3),
            np.round(np.random.uniform(bounds["y_min"], bounds["y_max"], (1, M)), 3)
        ))
        if check_obstacle_collision(s, obstacles, d_min) \
                and check_inter_collision(np.hstack((s, goals)), d_min):
            if np.all(LA.norm(s - goals, axis=0) >= d_sep): # want to make the goal far enough
                return s
            else:
                d_sep -= d_dec 

def init_velocity(bounds, M):
    v = np.vstack((
        np.round(np.random.uniform(bounds["v_min"], bounds["v_max"], (1, M)), 3),
        np.round(np.random.uniform(bounds["v_min"], bounds["v_max"], (1, M)), 3)
    ))
    return v

# Convert an integer to 4 binaries (bits)
def int_to_four_bins(n):
    return np.array(list(np.binary_repr(int(n), width=4))).astype(int)

# This function computes the average accuracy per binary
def compute_bitwise_accuracy(preds, targets):
    return (preds == targets).float().mean().item()

# This function computes the strict accuracy: it counts how many entire binary vectors are predicted exactly right
def compute_exact_match_accuracy(preds, targets):
    return torch.all(preds == targets, dim=1).float().mean().item()

def NNoutput_reshape(outputs, N_obs):
    dis_traj_temp = outputs.reshape([N_obs,-1], order='C')
    dis_traj = np.array([dis_traj_temp[i].reshape([4,-1], order='F') for i in range(2)])
    return dis_traj

# torch version of the function above
# def NNoutput_reshape_torch(outputs: torch.Tensor, N_obs: int):
#     if outputs.dim() == 1:
#         outputs = outputs.view(N_obs, -1) # (N_obs, 4*H)
#     H = outputs.shape[1] // 4
#     # reshape to (N_obs, H, 4) in C-order
#     dis_traj = outputs.view(N_obs, H, 4)
#     # transpose to (N_obs, 4, H) to match NumPy's F-order reshape
#     dis_traj = dis_traj.transpose(1, 2).contiguous()
#     return dis_traj

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
