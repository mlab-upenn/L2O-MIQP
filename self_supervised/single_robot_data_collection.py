import numpy as np
import numpy.linalg as LA
from matplotlib import pyplot as plt
from scipy.integrate import odeint
import pickle as pk
import random
import argparse
import sys, os 
sys.path.append(os.path.abspath("..")) # Adds the parent folder to sys.path
from utils import *
from Robots import *
from Controller import *

def save_data(data_list, shuffle = False):  
    # post-processing + write out
    num_data = len(data_list)
    print("Auto-save with ", num_data, " data points")
    if shuffle:
        # Shuffle the data
        arr = np.arange(num_data)
        np.random.shuffle(arr)
        shf_data_list = [data_list[i] for i in arr]
        data_list = shf_data_list

    with open("single.p", "wb") as file:
        pk.dump(data_list, file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sim", type=int, default=int(1e4))
    parser.add_argument("--autosave", type=int, default=int(1e2))
    parser.add_argument("--shuffle", action="store_true", help="default: False")
    input_parser = parser.parse_args()
    N_sim = input_parser.n_sim
    autosave = input_parser.autosave
    shuffle = input_parser.shuffle

    # Define all the constants first
    T = 0.25; H = 20
    bounds = {
        "x_max": 4.0, "x_min": 0.0, 
        "y_max": 4.0, "y_min": 0.0,
        "v_max": 0.5, "v_min": -0.5, 
        "u_max": 0.5, "u_min": -0.5, 
    }
    d_min = 0.25

    # List of obstacles, fixed for now
    with_obs = True
    if with_obs:
        Obs = [
            obstacle(1.0, 1.5, 0.8, 1.6, 0.),
            obstacle(3.0, 2.0, 1.8, 0.6, np.pi/2)
        ]
    else:
        Obs = []
    n_obs = len(Obs)
    Obs_info = np.array([[o.x_c, o.y_c, o.theta, o.L, o.W] for o in Obs]).T

    #create numpy containers for data: (params, x, u, y, J*, solve_time)
    sampled_params = ['x0', 'xg'] # 'obstacles'

    data_list = []

    # np.random.seed(7)
    for ss in range(N_sim):
        M = 1 # random.randint(2, 5)  
        print("Simulation #", ss, "with", M, "robots")
        # generate the goal positions
        goals = goal_position(Obs, bounds, d_min+0.1, M)
        # generate the initial position so that no collision avoidance constraint is violated
        p_init = init_position(Obs, goals, bounds, d_min+0.1, M)

        # Initialization
        Robots = [robot(T, p_init[:,i], np.zeros(2)) for i in range(M)]
        Ctrl = controller(T, H, M)
        Ctrl.set_params(bounds, d_min, Wu=1e0, Wp=1e0, Wpt=1e1)
        Ctrl.set_goals_obstacles(goals, Obs)
        all_p = np.hstack([rob.p for rob in Robots])
        all_v = np.hstack([rob.v for rob in Robots])
        Ctrl.set_state(all_p, all_v)

        # The main loop
        N_step = 100
        for t in range(N_step):
            # print("Time step:", t)
            all_p = np.hstack([rob.p for rob in Robots])
            all_v = np.hstack([rob.v for rob in Robots])
            Ctrl.set_state(all_p, all_v)
            Ctrl.find_cpl_rbs()
            Ctrl.solve_MIQP()
            # print(Ctrl.SOL_disc["BB"], Ctrl.SOL_disc["CC"])
            
            # Expand the data if the problem is feasible
            if not Ctrl.INFEASIBLE:
                new_x0 = np.vstack([rob.st for rob in Robots]).T
                new_xg = np.vstack([goals[:,n] for n in range(M)]).T                
                new_data = {}
                if 'x0' in sampled_params: new_data["x0"] = new_x0
                if 'xg' in sampled_params: new_data["xg"] = new_xg
                if 'obstacles' in sampled_params: new_data["obstacles"] = Obs_info
                new_data["nodes"] = np.arange(0,M)
                new_data["edges"] = np.array(Ctrl.cpl_agt).T
                new_data["XX"] = Ctrl.SOL_cont["X"]
                new_data["UU"] = Ctrl.SOL_cont["U"]
                SOL_disc_loc = Ctrl.SOL_disc['BO'].reshape([n_obs, M, H])
                # SOL_disc_cpl = Ctrl.SOL_disc['BB']
                new_data["YY"] = SOL_disc_loc
                # new_data["ZZ"] = SOL_disc_cpl
                new_data["cost"] = Ctrl.cost_value
                data_list.append(new_data)

            # Now move the robots
            U_ctrl = Ctrl.SOL_cont["U"][:, 0].reshape([2,-1], order = 'F')
            for i in range(M):
                Robots[i].run(U_ctrl[:,i], t*T)

            # Check and break if reaching goals
            if Ctrl.reaching_goals():
                break

        if (ss+1) % autosave == 0:
            save_data(data_list, shuffle)

    # Save data at the end
    save_data(data_list, shuffle)

if __name__ == "__main__":
    main()