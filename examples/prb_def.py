import torch
import torch.nn as nn
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import sys
import os
from src.cvxlayer import CVXLayer

nx = 2; ny = 2 

class example_QP(CVXLayer):

    def __init__(self, nx, ny, penalty="l1", rho1=1.0, bigM = 1e3, **kwargs):
        self.nx = nx
        self.ny = ny
        # Define CVXPY variables and parameters
        x = cp.Variable((nx,))  # continuous decision variables
        y = cp.Parameter((ny,))  # integer decision variables

        p = cp.Parameter((nx,))  # linear term in the objective
        b = cp.Parameter((nx,))  # RHS of the constraint x <= b
        a = cp.Parameter((1,))   # RHS of the constraint 1.T*x <= a
        s = cp.Variable((nx,), nonneg=True)  # slack variables
        
        # Define the QP problem
        if penalty == "l1": # default to l1 penalty
            objective = cp.Minimize(cp.quad_form(x, np.eye(nx)) + p.T @ x + rho1*cp.sum(s))
        elif penalty == "l2": 
            objective = cp.Minimize(cp.quad_form(x, np.eye(nx)) + p.T @ x + rho1*cp.quad_form(s, np.eye(nx)))
        constraints = [
            x <= b,
            sum(x) <= a,
            x <= bigM * y + s,
            s >= 0,
        ]
        problem = cp.Problem(objective, constraints)

        # Create CVXPY layer
        self.cvxpylayer = CvxpyLayer(problem, parameters=[p, b, a, y], variables=[x, s])        

    def solve(self, theta, y):
        """
        Run the CVXPYLayer in batch. All inputs are torch tensors.
        Returns x, s (each torch tensor with grad).
        """
        p = theta[:, :self.nx]
        b = theta[:, self.nx:2*self.nx]
        a = theta[:, -1]

        if a.ndim == 1:
            a = a.unsqueeze(-1)

        # Device management
        device = p.device
        self.cvxpylayer = self.cvxpylayer.to(device)
        p = p.to(device)
        b = b.to(device)
        a = a.to(device)
        y = y.to(device)

        x_opt, s_opt = self.cvxpylayer(p, b, a, y)
        return x_opt, s_opt

def _solve_single_miqp(args):
    """Helper function to solve a single MIQP problem."""
    # Redirect stdout and stderr to devnull at the start of each process
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    """Helper function to solve a single MIQP problem."""
    i, p_i, b_i, a_i, nx, ny = args
    
    # Variables
    x = cp.Variable(nx)
    y = cp.Variable(ny, boolean=True)
    
    # Objective and constraints
    objective = cp.Minimize(cp.sum_squares(x) + p_i @ x)
    constraints = [
        x <= b_i,
        cp.sum(x) <= a_i,
        cp.sum(y) <= 1,
        x <= 1e3 * y
    ]
    
    # Problem definition
    prob = cp.Problem(objective, constraints)
    
    try:
        prob.solve(solver=cp.GUROBI, verbose=False, OutputFlag=0)
        
        if x.value is not None and y.value is not None:
            return i, x.value, y.value
        else:
            return i, np.zeros(nx), np.zeros(ny)
    except Exception as e:
        print(f"Error solving sample {i}: {e}")
        return i, np.zeros(nx), np.zeros(ny)

@torch.no_grad()
def GUROBI_solve_parallel(p: torch.Tensor, b: torch.Tensor, a: torch.Tensor, max_workers=None):
    """
    Solve MIQP for each sample in the batch using parallel processing.
    """
    device = p.device
    p_np = p.detach().cpu().numpy()
    b_np = b.detach().cpu().numpy()
    a_np = a.detach().cpu().numpy()
    
    if a_np.ndim == 2:
        a_np = a_np.squeeze(-1)
    
    B = p_np.shape[0]
    
    # Prepare arguments for parallel execution
    args_list = [(i, p_np[i], b_np[i], a_np[i], nx, ny) for i in range(B)]
    
    # Preallocate result arrays
    x_results = np.zeros((B, nx))
    y_results = np.zeros((B, ny))
    
    # Use ProcessPoolExecutor for parallel solving
    if max_workers is None:
        max_workers = min(B, mp.cpu_count())
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_solve_single_miqp, args): args[0] for args in args_list}
        
        for future in as_completed(futures):
            i, x_sol, y_sol = future.result()
            x_results[i] = x_sol
            y_results[i] = y_sol
    
    return torch.tensor(x_results).float().to(device), torch.tensor(y_results).float().to(device)

def ground_truth_solver(theta: torch.Tensor):
    """
    Solve the MIQP problem using Gurobi for ground truth labels.
    Args:
        theta: Tensor of shape (B, 2n+1) where each row is [p, b, a].
    Returns:
        x_solver: Tensor of shape (B, n) with optimal continuous variables. 
    """
    p = theta[:, :nx]
    b = theta[:, nx:2*nx]
    a = theta[:, 2*nx].unsqueeze(-1)
    x_solver, y_solver = GUROBI_solve_parallel(p, b, a)
    return x_solver, y_solver

# --- Custom dataset class ---
class SampleDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
def create_data_loaders(ntrain, ntest, batch_size = 64):
    # Problem setup
    data_seed = 18
    np.random.seed(data_seed)
    torch.manual_seed(data_seed)

    p_low, p_high = -30.0, 5.0   # linear term in objective
    b_low, b_high = 5.0, 25.0    # RHS of constraint x <= b
    a_low, a_high = 10.0, 30.0   # RHS of constraint 1^T x <= a

    # Generate samples
    samples_train = {
        "p": torch.FloatTensor(ntrain, nx).uniform_(p_low, p_high),
        "b": torch.FloatTensor(ntrain, nx).uniform_(b_low, b_high),
        "a": torch.FloatTensor(ntrain, 1).uniform_(a_low, a_high),
    }
    samples_train = torch.concat([samples_train['p'], samples_train['b'], samples_train['a']], dim=-1)

    samples_test = {
        "p": torch.FloatTensor(ntest, nx).uniform_(p_low, p_high),
        "b": torch.FloatTensor(ntest, nx).uniform_(b_low, b_high),
        "a": torch.FloatTensor(ntest, 1).uniform_(a_low, a_high),
    }
    samples_test = torch.concat([samples_test['p'], samples_test['b'], samples_test['a']], dim=-1)

    # Create datasets
    train_dataset = SampleDataset(samples_train)
    test_dataset = SampleDataset(samples_test)
    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader