import numpy as np
from typing import Sequence


class robot:
    """Second-order integrator robot with position/velocity history."""

    def __init__(self, T: float, p0: Sequence[float], v0: Sequence[float]) -> None:
        self.T = float(T)
        # Store copies so external mutations on inputs do not alter state.
        self.p = np.array(p0, dtype=float, copy=True)
        self.v = np.array(v0, dtype=float, copy=True)
        self.st = np.hstack((self.p, self.v))
        self.X_hist = self.st.reshape((4, 1))
        self.U_hist = np.empty((0,), dtype=float)
        self.T_hist = np.empty((0,), dtype=float)

    def run(self, u: Sequence[float], t_now: float) -> None:
        """Advance the robot one step with control input `u` at time `t_now`."""
        control = np.array(u, dtype=float, copy=True)
        v_new = self.v + self.T * control
        p_new = self.p + self.T * self.v + 0.5 * self.T**2 * control
        self.v = v_new
        self.p = p_new
        self.st = np.hstack((self.p, self.v))
        self.u = control
        # Save data history of the vehicle.
        self.X_hist = np.hstack([self.X_hist, self.st.reshape((-1, 1))])
        self.U_hist = np.hstack([self.U_hist, self.u])
        self.T_hist = np.hstack([self.T_hist, float(t_now)])


class obstacle:
    """Axis-aligned rectangular obstacle with orientation `theta`."""

    def __init__(self, x_c: float, y_c: float, L: float, W: float, theta: float) -> None:
        self.x_c = float(x_c)
        self.y_c = float(y_c)
        self.L = L / 2
        self.W = W / 2
        self.theta = float(theta)
        # Compute auxiliary values.
        alpha = np.arctan(W / L)
        r = np.sqrt(L**2 + W**2) / 2
        # Define the corners.
        self.p1 = np.array([x_c - r * np.cos(alpha + theta), y_c - r * np.sin(alpha + theta)])
        self.p2 = np.array([x_c + r * np.cos(alpha - theta), y_c - r * np.sin(alpha - theta)])
        self.p3 = np.array([x_c + r * np.cos(alpha + theta), y_c + r * np.sin(alpha + theta)])
        self.p4 = np.array([x_c - r * np.cos(alpha - theta), y_c + r * np.sin(alpha - theta)])
