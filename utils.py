import numpy as np
import numpy.linalg as LA

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