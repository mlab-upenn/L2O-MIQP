# dataset_two_tank_fixed_params.py
# Silent MIQP dataset generator with fixed problem params.
# X = [x0, d_seq], Y = delta trajectory, Z = {u_trj, x_trj, obj}
from __future__ import annotations
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from tqdm import tqdm

# ---------------- Fixed problem parameters (from the paper) ----------------
ALPHA1, ALPHA2, NU = 0.9983, 0.9966, 0.001
B1 = B2 = 0.075
B3 = 0.0825
B4 = B5 = 0.0833
# T = 300  # not used directly

A = np.array([[ALPHA1, NU],
              [0.0,     ALPHA2 - NU]])
BU = np.array([[B1, 0.0],
               [0.0, B2]])
BDELTA = np.array([[0.0],
                   [B3]])
E = np.array([[-B4, 0.0],
              [0.0, -B5]])

# Weights & bounds (fixed)
Q = np.diag([1.0, 1.0])
R = np.diag([0.5, 0.5])
P = np.diag([1.0, 1.0])
RHO = 0.1
REF = np.array([4.2, 1.8])
X_LO = np.array([0.0, 0.0])
X_HI = np.array([8.4, 3.6])
U_SUM_HI = 8.0
# soft penalties (set to None to enforce hard bounds)
C_X = 25.0
C_U = 25.0


@dataclass
class Spec:
    N: int
    # choose soft (paper-style) or hard bounds
    c_x: Optional[float] = C_X
    c_u: Optional[float] = C_U
    # time limit per solve (seconds)
    time_limit: Optional[float] = None
    # encode Y as one-hot (length 4) per step if True; otherwise int {0,1,2,3}
    y_one_hot: bool = False


# ---------------- Core solver (silent) ----------------
def solve_one(x0: np.ndarray, d_seq: np.ndarray, spec: Spec, env: gp.Env) -> Dict[str, Any]:
    N = spec.N
    assert x0.shape == (2,)
    assert d_seq.shape == (N, 2)

    m = gp.Model("two_tank", env=env)
    m.Params.OutputFlag = 0
    if spec.time_limit is not None:
        m.Params.TimeLimit = spec.time_limit

    # variables
    x = m.addVars(N+1, 2, vtype=GRB.CONTINUOUS, name="x")
    u = m.addVars(N,   2, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="u")
    delta = m.addVars(N, vtype=GRB.INTEGER, name="delta")
    for k in range(N):
        m.addConstr(delta[k] >= 0)
        m.addConstr(delta[k] <= 3)

    # init
    m.addConstr(x[0,0] == float(x0[0]))
    m.addConstr(x[0,1] == float(x0[1]))

    # dynamics
    for k in range(N):
        for i in range(2):
            m.addConstr(
                x[k+1, i] ==
                A[i,0]*x[k,0] + A[i,1]*x[k,1] +
                BU[i,0]*u[k,0] + BU[i,1]*u[k,1] +
                BDELTA[i,0]*delta[k] +
                E[i,0]*d_seq[k,0] + E[i,1]*d_seq[k,1]
            )

    # bounds (soft or hard)
    if spec.c_x is None:
        x_slacks = None
        for k in range(N+1):
            for i in range(2):
                m.addConstr(x[k,i] >= X_LO[i])
                m.addConstr(x[k,i] <= X_HI[i])
    else:
        x_v_lo = m.addVars(N+1, 2, lb=0.0, name="x_violate_lo")
        x_v_hi = m.addVars(N+1, 2, lb=0.0, name="x_violate_hi")
        for k in range(N+1):
            for i in range(2):
                m.addConstr(x_v_lo[k,i] >= X_LO[i] - x[k,i])
                m.addConstr(x_v_hi[k,i] >= x[k,i] - X_HI[i])
        x_slacks = (x_v_lo, x_v_hi)

    if spec.c_u is None:
        u_slacks = None
        for k in range(N):
            m.addConstr(u[k,0] >= 0.0)
            m.addConstr(u[k,1] >= 0.0)
            m.addConstr(u[k,0] + u[k,1] <= U_SUM_HI)
    else:
        v_u_lo1 = m.addVars(N, lb=0.0, name="u1_violate_lo")
        v_u_lo2 = m.addVars(N, lb=0.0, name="u2_violate_lo")
        v_usum  = m.addVars(N, lb=0.0, name="usum_violate_hi")
        for k in range(N):
            m.addConstr(v_u_lo1[k] >= -u[k,0])
            m.addConstr(v_u_lo2[k] >= -u[k,1])
            m.addConstr(v_usum[k]  >=  u[k,0] + u[k,1] - U_SUM_HI)
        u_slacks = (v_u_lo1, v_u_lo2, v_usum)

    # objective
    obj = gp.QuadExpr(0.0)
    for k in range(N):
        # (x_k - r)^T Q (x_k - r)
        for i in range(2):
            for j in range(2):
                obj += Q[i,j] * (x[k,i] - float(REF[i])) * (x[k,j] - float(REF[j]))
        # u_k^T R u_k
        for i in range(2):
            for j in range(2):
                obj += R[i,j] * u[k,i] * u[k,j]
        # rho * delta_k^2
        obj += float(RHO) * delta[k] * delta[k]
        # soft penalties
        if x_slacks is not None:
            xvlo, xvhi = x_slacks
            for i in range(2):
                obj += float(spec.c_x) * xvlo[k,i] * xvlo[k,i]
                obj += float(spec.c_x) * xvhi[k,i] * xvhi[k,i]
        if u_slacks is not None:
            v1, v2, vs = u_slacks
            obj += float(spec.c_u) * v1[k] * v1[k]
            obj += float(spec.c_u) * v2[k] * v2[k]
            obj += float(spec.c_u) * vs[k] * vs[k]

    # terminal
    for i in range(2):
        for j in range(2):
            obj += P[i,j] * (x[N,i] - float(REF[i])) * (x[N,j] - float(REF[j]))

    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()
    relaxed = False

    if m.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        relaxed = True
        m.feasRelaxS(1, False, True, True)
        m.optimize()

    # assemble outputs
    ok = (m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT)) and (m.SolCount > 0)
    if not ok:
        return {
            "ok": False, "status": int(m.Status), "relaxed": relaxed,
            "X": None, "Y": None, "Z": None
        }

    x_trj = np.array([[x[k,i].X for i in range(2)] for k in range(N+1)])  # (N+1,2)
    u_trj = np.array([[u[k,i].X for i in range(2)] for k in range(N)])    # (N,2)
    dlt   = np.array([int(round(delta[k].X)) for k in range(N)])          # (N,)

    # X = concat of x0 and d_seq -> shape (2 + 2N,)
    X_feat = np.concatenate([x0, d_seq.reshape(-1)], axis=0)

    # Y = integer solution (either ints or one-hot)
    if spec.y_one_hot:
        Y_lbl = np.zeros((N, 4), dtype=np.float32)
        Y_lbl[np.arange(N), dlt] = 1.0
    else:
        Y_lbl = dlt.astype(np.int64)

    # Z = continuous & other: pack u_trj, x_trj, objective
    Z_aux = {
        "u_trj": u_trj,           # (N,2)
        "x_trj": x_trj,           # (N+1,2)
        "obj": float(m.ObjVal),
    }
    return {
        "ok": True,
        "status": int(m.Status),
        "relaxed": relaxed,
        "X": X_feat,
        "Y": Y_lbl,
        "Z": Z_aux,
    }


# ---------------- Public API ----------------
def build_dataset(
    num_samples: int,
    N: int,
    *,
    seed: int = 0,
    c_x: Optional[float] = C_X,
    c_u: Optional[float] = C_U,
    time_limit: Optional[float] = None,
    y_one_hot: bool = False,
    save_npz: Optional[str] = None,
    return_torch: bool = False,
) -> Dict[str, Any]:
    """
    Returns a dict with:
      X: (S, 2 + 2N)  — features [x0, vec(d_seq)]
      Y: (S, N) int or (S, N, 4) one-hot — integer solution
      Z: dict of arrays/lists {u_trj: (S, N, 2), x_trj: (S, N+1, 2), obj: (S,)}
      status: (S,) Gurobi status codes
    """
    rng = np.random.default_rng(seed)
    spec = Spec(N=N, c_x=c_x, c_u=c_u, time_limit=time_limit, y_one_hot=y_one_hot)

    # silent Gurobi environment (shared)
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("LogToConsole", 0)
    env.start()

    # storage
    X_list = []
    Y_list = []
    U_list = []
    Xtraj_list = []
    OBJ_list = []
    STAT_list = []
    RELAX_list = []

    for _ in tqdm(range(num_samples)):
        # sample x0 ~ U(X_LO, X_HI), and d_seq (paper-like)
        x0 = np.array([rng.uniform(X_LO[0], X_HI[0]),
                       rng.uniform(X_LO[1], X_HI[1])], dtype=float)

        d_seq = np.zeros((N, 2), dtype=float)
        d_seq[:,0] = 7.0 * rng.beta(0.6, 1.4, size=N)  # scaled Beta for d1
        # bursty peaks for d2
        n_peaks = rng.integers(1, max(2, N // 6))
        for _p in range(n_peaks):
            amp = rng.uniform(1.0, 16.0)
            dur = int(rng.integers(2, 6))
            start = int(rng.integers(0, max(1, N - dur)))
            d_seq[start:start+dur, 1] += amp

        res = solve_one(x0, d_seq, spec, env)
        STAT_list.append(res["status"])
        RELAX_list.append(bool(res.get("relaxed", False)))
        if not res["ok"]:
            # store placeholders
            X_list.append(np.concatenate([x0, d_seq.reshape(-1)], 0))
            if y_one_hot:
                Y_list.append(np.zeros((N,4), dtype=np.float32))
            else:
                Y_list.append(np.full((N,), -1, dtype=np.int64))
            U_list.append(np.zeros((N,2), dtype=float))
            Xtraj_list.append(np.zeros((N+1,2), dtype=float))
            OBJ_list.append(np.nan)
            continue

        X_list.append(res["X"])
        Y_list.append(res["Y"])
        U_list.append(res["Z"]["u_trj"])
        Xtraj_list.append(res["Z"]["x_trj"])
        OBJ_list.append(res["Z"]["obj"])

    # stack
    X = np.stack(X_list, axis=0)                                 # (S, 2+2N)
    Y = (np.stack(Y_list, axis=0) if y_one_hot
         else np.asarray(Y_list, dtype=np.int64))                # (S,N,4) or (S,N)
    U = np.stack(U_list, axis=0)                                 # (S, N, 2)
    Xtraj = np.stack(Xtraj_list, axis=0)                         # (S, N+1, 2)
    OBJ = np.array(OBJ_list, dtype=float)                        # (S,)
    STAT = np.array(STAT_list, dtype=np.int32)                   # (S,)
    RELAX = np.array(RELAX_list, dtype=bool)                     # (S,)

    if save_npz is not None:
        # one silent compressed file
        np.savez_compressed(
            save_npz,
            X=X, Y=Y, U=U, Xtraj=Xtraj, OBJ=OBJ, STATUS=STAT, RELAX=RELAX,
            N=np.array([N], dtype=np.int32),
            REF=REF, Q=Q, R=R, P=P, RHO=np.array([RHO], dtype=float),
            X_LO=X_LO, X_HI=X_HI, U_SUM_HI=np.array([U_SUM_HI], dtype=float),
            C_X=np.array([-1.0 if c_x is None else c_x], dtype=float),
            C_U=np.array([-1.0 if c_u is None else c_u], dtype=float),
        )

    if return_torch:
        import torch
        X_t = torch.from_numpy(X).float()
        if y_one_hot:
            Y_t = torch.from_numpy(Y).float()
        else:
            Y_t = torch.from_numpy(Y).long()
        U_t = torch.from_numpy(U).float()
        Xtraj_t = torch.from_numpy(Xtraj).float()
        OBJ_t = torch.from_numpy(OBJ).float()
        STAT_t = torch.from_numpy(STAT).int()
        RELAX_t = torch.from_numpy(RELAX.astype(np.int32)).bool()
        return {
            "X": X_t,
            "Y": Y_t,
            "Z": {"U": U_t, "Xtraj": Xtraj_t, "OBJ": OBJ_t},
            "STATUS": STAT_t,
            "RELAX": RELAX_t,
        }

    return {"X": X, "Y": Y, "Z": {"U": U, "Xtraj": Xtraj, "OBJ": OBJ}, "STATUS": STAT, "RELAX": RELAX}


# ---------------- Example (kept silent) ----------------
if __name__ == "__main__":
    # Build a small silent dataset (no prints)
    data = build_dataset(
        num_samples=10000,
        N=20,
        seed=114514,
        c_x=None, c_u=None,         # None/None -> hard bounds
        time_limit=None,            # e.g., 5.0
        y_one_hot=False,            # True -> (S,N,4)
        save_npz="data/data.npz",
        return_torch=False,
    )
