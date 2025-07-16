import numpy as np

class robot:

    def __init__(self, T, p0, v0):
        self.T = T
        self.p = p0; self.v = v0
        self.st = np.hstack((p0, v0))
        self.X_hist = self.st.reshape([4,1])
        self.U_hist = np.empty([0], dtype=float)
        self.T_hist = np.empty([0], dtype=float)
        
    def run(self, u, t_now):
        v_new = self.v + self.T*u   
        p_new = self.p + self.T*self.v + 0.5*self.T**2*u
        self.v = v_new; self.p = p_new
        self.st = np.hstack((self.p, self.v))
        self.u = u
        ### Save data history of the vehicle
        self.X_hist = np.hstack([self.X_hist, self.st.reshape([-1,1])])
        self.U_hist = np.hstack([self.U_hist, self.u])
        self.T_hist = np.hstack([self.T_hist, t_now])

class obstacle:
    
    def __init__(self, x_c, y_c, L, W, theta):
        self.x_c = x_c
        self.y_c = y_c
        self.L = L / 2
        self.W = W / 2
        self.theta = theta
        # Compute auxiliary values
        alpha = np.arctan(W / L)
        r = np.sqrt(L**2 + W**2) / 2
        # Define the corners
        self.p1 = [x_c - r * np.cos(alpha + theta), y_c - r * np.sin(alpha + theta)]
        self.p2 = [x_c + r * np.cos(alpha - theta), y_c - r * np.sin(alpha - theta)]
        self.p3 = [x_c + r * np.cos(alpha + theta), y_c + r * np.sin(alpha + theta)]
        self.p4 = [x_c - r * np.cos(alpha - theta), y_c + r * np.sin(alpha - theta)]
