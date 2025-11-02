import numpy as np
import numpy.linalg as LA
import time
import gurobipy as grb
from gurobipy import GRB
import torch
from torch.autograd import Variable
from torch.nn import Sigmoid

def obs_edge_index(N_obs):
    i, j = np.indices((N_obs, N_obs))
    mask = i.ravel() != j.ravel()
    return np.vstack((i.ravel()[mask], j.ravel()[mask]))

def map_binary2activecontraint(bin):
    bin = bin.astype(int)
    map_dict = {
        (0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3
    }
    return np.array([map_dict[(bin[0,i], bin[1,i])] for i in range(bin.shape[1])]) 

def map_activecontraint2binary(con):
    map_dict = {
        0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)
    }
    return np.array([map_dict[con[i]] for i in range(con.shape[0])]).T

class controller:
    bigM = 1e3

    def __init__(self, T, H, M):
        self.T = T
        self.H = H
        self.M = M

    def set_params(self, bounds, d_min, Wu, Wp, Wpt):
        self.x_min = bounds["x_min"]; self.x_max = bounds["x_max"]
        self.y_min = bounds["y_min"]; self.y_max = bounds["y_max"]
        self.v_min = bounds["v_min"]; self.v_max = bounds["v_max"]
        self.u_min = bounds["u_min"]; self.u_max = bounds["u_max"]
        self.Wu = Wu; self.Wp = Wp; self.Wpt = Wpt 
        self.d_min = d_min

    def set_goals_obstacles(self, goals, obstacles):
        self.goals = goals.ravel(order = 'F')
        self.obs = obstacles

    def set_state(self, all_p, all_v):
        self.p = all_p
        self.v = all_v

    def find_cpl_rbs(self, d_prox = 2.0):
        self.cpl_agt = []
        self.adj_mat = np.zeros([self.M, self.M])
        for m in range(self.M):
            for n in range(m+1, self.M):
                if LA.norm(self.p[2*m:2*(m+1)] - self.p[2*n:2*(n+1)]) <= d_prox:
                    self.cpl_agt.append([m, n])
                    self.cpl_agt.append([n, m])
                    self.adj_mat[m, n] = self.adj_mat[n, m] = 1

    def check_collision(self, epsi=1e-1):
        # add epsi to avoid numerical issue
        # Check collision with obstacles
        for idx, ob in enumerate(self.obs):
            th = ob.theta; xo = ob.x_c; yo = ob.y_c; L0 = ob.L; W0 = ob.W
            for m in range(self.M):
                xm, ym = self.p[2*m], self.p[2*m+1]
                # 4 constraints
                if (np.cos(th)*(xm-xo) + np.sin(th)*(ym-yo) < L0) \
                        and (-np.sin(th)*(xm-xo) + np.cos(th)*(ym-yo) < W0) \
                        and (-np.cos(th)*(xm-xo) - np.sin(th)*(ym-yo) < L0) \
                        and (np.sin(th)*(xm-xo) - np.cos(th)*(ym-yo) < W0):
                    print("Collision with obstacles")
                    return True

        # Check collision among robots
        for m in range(self.M):
            xm, ym = self.p[2*m], self.p[2*m+1]
            for n in range(m+1, self.M):
                xn, yn = self.p[2*n], self.p[2*n+1]
                if abs(xm - xn) < self.d_min and abs(ym - yn) < self.d_min:
                    print("Collision among robots")
                    return True
        return False


    def reaching_goals(self, thres = 0.2):
        if LA.norm(self.p - self.goals) < thres:
            return True
        else:
            return False

    def solve_MIQP(self, solve_relax=True):
        self.INFEASIBLE = False
        with grb.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            with grb.Model(env=env) as self.opt:    
                self.opt.setParam("Seed", 12)  # Any fixed integer
                # self.opt.setParam("Heuristics", 0)  # Disable primal heuristics
                # self.opt.setParam("Cuts", 0)  # Disable cutting planes
                # self.opt.setParam("Presolve", 0)  # Disable presolving
                self.opt.setParam('TimeLimit', 5.0)  # Set a time limit

                # Create variables 
                self.PP = self.opt.addMVar(shape=(2*self.M, self.H+1), lb=-GRB.INFINITY, ub=GRB.INFINITY,  name="P")        
                self.VV = self.opt.addMVar(shape=(2*self.M, self.H+1), lb=-GRB.INFINITY, ub=GRB.INFINITY,  name="V")        
                self.UU = self.opt.addMVar(shape=(2*self.M, self.H), lb=-GRB.INFINITY, ub=GRB.INFINITY,  name="U")        

                # Initial conditions and dynamics
                self.opt.addConstr(self.VV[:, 0] == self.v)
                self.opt.addConstr(self.PP[:, 0] == self.p)
                for k in range(self.H):
                    self.opt.addConstr(self.VV[:, k+1] == self.VV[:, k] + self.T*self.UU[:, k])
                    self.opt.addConstr(self.PP[:, k+1] == self.PP[:, k] + self.T*self.VV[:, k] + self.T**2/2*self.UU[:, k])
                    # Bound constraints
                    self.opt.addConstr(self.PP[:, k+1] <= np.tile([self.x_max, self.y_max], self.M))
                    self.opt.addConstr(self.PP[:, k+1] >= np.tile([self.x_min, self.y_min], self.M))
                    self.opt.addConstr(self.VV[:, k+1] <= self.v_max)
                    self.opt.addConstr(self.VV[:, k+1] >= self.v_min)
                    self.opt.addConstr(self.UU[:, k] <= self.u_max)
                    self.opt.addConstr(self.UU[:, k] >= self.u_min)

                # Objective function 
                obj = self.Wu*sum(sum(self.UU[:, k]*self.UU[:, k]) for k in range(self.H)) \
                    + self.Wp*sum(sum((self.PP[:, k]-self.goals)*(self.PP[:, k]-self.goals)) for k in range(self.H)) \
                    + self.Wpt*sum((self.PP[:, -1]-self.goals)*(self.PP[:, -1]-self.goals))
                
                self.opt.setObjective(obj, GRB.MINIMIZE)

                # Collision avoidance with static obstacles
                N_obs = len(self.obs)
                self.BO = [self.opt.addMVar(shape = (self.M, self.H), vtype=GRB.BINARY) for _ in range(N_obs)]
                self.CO = [self.opt.addMVar(shape = (self.M, self.H), vtype=GRB.BINARY) for _ in range(N_obs)]
                for idx, ob in enumerate(self.obs):
                    th = ob.theta; xo = ob.x_c; yo = ob.y_c; L0 = ob.L + self.d_min; W0 = ob.W + self.d_min
                    for m in range(self.M):
                        # 4 constraints
                        self.opt.addConstr(np.cos(th)*(self.PP[2*m, 1:]-xo) + np.sin(th)*(self.PP[2*m+1, 1:]-yo) >= L0 - self.bigM*(self.BO[idx][m,:]+self.CO[idx][m,:]))
                        self.opt.addConstr(-np.sin(th)*(self.PP[2*m, 1:]-xo) + np.cos(th)*(self.PP[2*m+1, 1:]-yo) >= W0 - self.bigM*(1-self.BO[idx][m,:]+self.CO[idx][m,:]))
                        self.opt.addConstr(-np.cos(th)*(self.PP[2*m, 1:]-xo) - np.sin(th)*(self.PP[2*m+1, 1:]-yo) >= L0 - self.bigM*(1+self.BO[idx][m,:]-self.CO[idx][m,:]))
                        self.opt.addConstr(np.sin(th)*(self.PP[2*m, 1:]-xo) - np.cos(th)*(self.PP[2*m+1, 1:]-yo) >= W0 - self.bigM*(2-self.BO[idx][m,:]-self.CO[idx][m,:]))

                # Collision avoidance between robots
                N_cpl_cons = int(len(self.cpl_agt)/2)
                self.BB = [self.opt.addMVar(shape = (self.H), vtype=GRB.BINARY) for _ in range(N_cpl_cons)]
                self.CC = [self.opt.addMVar(shape = (self.H), vtype=GRB.BINARY) for _ in range(N_cpl_cons)]
                idx = 0
                for pair in self.cpl_agt:
                    m, n = pair
                    if m < n:
                        # 4 constraints
                        self.opt.addConstr(self.PP[2*m, 1:] - self.PP[2*n, 1:] >= 2*self.d_min - self.bigM*(self.BB[idx]+self.CC[idx]))
                        self.opt.addConstr(self.PP[2*n, 1:] - self.PP[2*m, 1:] >= 2*self.d_min - self.bigM*(1-self.BB[idx]+self.CC[idx]))
                        self.opt.addConstr(self.PP[2*m+1, 1:] - self.PP[2*n+1, 1:] >= 2*self.d_min - self.bigM*(1+self.BB[idx]-self.CC[idx]))
                        self.opt.addConstr(self.PP[2*n+1, 1:] - self.PP[2*m+1, 1:] >= 2*self.d_min - self.bigM*(2-self.BB[idx]-self.CC[idx]))
                        idx += 1

                # Check the status
                self.opt.optimize()

                if self.opt.status == grb.GRB.OPTIMAL:
                    self.SOL_cont = {"U": self.UU.X, 
                            "X": np.vstack([np.vstack([self.PP.X[2*m:2*m+2,:], self.VV.X[2*m:2*m+2,:]]) for m in range(self.M)])
                            }
                    self.SOL_disc = {}
                    self.SOL_disc["BO"], self.SOL_disc["BB"] = self.refine_binary(self.PP.X)
                    self.cost_value = self.opt.ObjVal
                elif self.opt.status == grb.GRB.TIME_LIMIT:
                    self.SOL_cont = {"U": self.UU.X, 
                            "X": np.vstack([np.vstack([self.PP.X[2*m:2*m+2,:], self.VV.X[2*m:2*m+2,:]]) for m in range(self.M)])
                            }
                    self.SOL_disc = {}
                    self.SOL_disc["BO"], self.SOL_disc["BB"] = self.refine_binary(self.PP.X)
                    self.cost_value = self.opt.ObjVal  
                elif self.opt.status == GRB.INFEASIBLE or self.opt.status == GRB.INF_OR_UNBD:
                    self.INFEASIBLE = True
                    if solve_relax:
                        print("THE CENTRALIZED PROBLEM IS INFEASIBLE!!!")
                        # https://www.gurobi.com/documentation/current/refman/py_model_feasrelaxs.html#pythonmethod:Model.feasRelaxS
                        self.opt.feasRelaxS(0, False, True, True)
                        self.opt.optimize()     
                        self.SOL_cont = {"U": self.UU.X, 
                                "X": np.vstack([np.vstack([self.PP.X[2*m:2*m+2,:], self.VV.X[2*m:2*m+2,:]]) for m in range(self.M)])
                                }
                        self.SOL_disc = {}
                        self.SOL_disc["BO"], self.SOL_disc["BB"] = self.refine_binary(self.PP.X)
                        self.cost_value = self.opt.ObjVal

    def refine_binary(self, PP):
        N_obs = len(self.obs)
        BO = [np.zeros((self.M, self.H)) for _ in range(N_obs)]
        for idx, ob in enumerate(self.obs):
            th = ob.theta; xo = ob.x_c; yo = ob.y_c; L0 = ob.L + self.d_min; W0 = ob.W + self.d_min
            for m in range(self.M):
                # 4 constraints
                c1 = (np.cos(th)*(PP[2*m, 1:]-xo) + np.sin(th)*(PP[2*m+1, 1:]-yo) - L0 < 0).astype(int)
                c2 = (-np.sin(th)*(PP[2*m, 1:]-xo) + np.cos(th)*(PP[2*m+1, 1:]-yo) - W0 < 0).astype(int)
                c3 = (-np.cos(th)*(PP[2*m, 1:]-xo) - np.sin(th)*(PP[2*m+1, 1:]-yo) - L0 < 0).astype(int)
                c4 = (np.sin(th)*(PP[2*m, 1:]-xo) - np.cos(th)*(PP[2*m+1, 1:]-yo) - W0 < 0).astype(int)
                c5 = np.vstack([c1, c2, c3, c4])
                for k in range(self.H):
                    BO[idx][m,k] = c5[:,k].dot(2 ** np.arange(c5[:,k].size)[::-1])
        BO = np.stack(BO)

        N_cpl_cons = len(self.cpl_agt)
        if N_cpl_cons > 0:
            BB = [np.zeros(self.H) for _ in range(N_cpl_cons)]
            idx = 0
            for pair in self.cpl_agt:
                m, n = pair
                # 4 constraints
                c1 = (PP[2*m, 1:] - PP[2*n, 1:] - 2*self.d_min < 0).astype(int)
                c2 = (PP[2*n, 1:] - PP[2*m, 1:] - 2*self.d_min < 0).astype(int)
                c3 = (PP[2*m+1, 1:] - PP[2*n+1, 1:] - 2*self.d_min < 0).astype(int)
                c4 = (PP[2*n+1, 1:] - PP[2*m+1, 1:] - 2*self.d_min < 0).astype(int)
                c5 = np.vstack([c1, c2, c3, c4])
                for k in range(self.H):
                    BB[idx][k] = c5[:,k].dot(2 ** np.arange(c5[:,k].size)[::-1])
                idx += 1
            BB = np.vstack(BB)
        else:
            BB = np.zeros((0, self.H)) 

        return BO, BB

class fast_controller(controller):

    def __init__(self, T, H, M, model, model_name):
        super().__init__(T, H, M)
        self.model = model # the trained NN model
        self.model_name = model_name

    def predict_binary(self, prob_params):
        model = self.model
        if self.model_name == "MLOpt": # MLOpt
            features = model.construct_features(prob_params)
            inpt = Variable(torch.from_numpy(features)).float().to(device=model.device)
            scores = model.model(inpt).cpu().detach().numpy()[:]
            torch.cuda.synchronize()
            ind_max = np.argsort(scores)[-model.n_evals:][::-1]

            y_guesses = np.zeros((model.n_evals, model.n_y), dtype=int)
            for ii, idx in enumerate(ind_max):
                for jj in range(model.num_train):
                    label = model.labels[jj]
                    if label[0] == idx:
                        y_guesses[ii] = label[1:]
                        break
            self.l_guess = y_guesses[0]
            self.l_guess = np.reshape(self.l_guess, (-1,2*self.H))
            # self.l_guess = np.vstack([self.l_guess[i::2,:] for i in range(len(self.obs))])

        elif self.model_name == "LSTM":
            features = model.construct_features(prob_params)
            features = np.tile(features, (1, self.H, 1))
            inpt = Variable(torch.from_numpy(features)).float().to(device=model.device)
            scores = model.model(inpt).cpu().detach().numpy()[0]
            torch.cuda.synchronize()
            ind_max = np.argsort(scores)[:,-model.n_evals:][:,::-1]

            y_guesses = np.zeros((model.n_evals, model.H, model.n_y), dtype=int)

            for hh in range(model.H):
                for ii, idx in enumerate(ind_max[hh]):
                    for val in model.strategy_dict.values():
                        label = val
                        if label[0] == idx:
                            y_guesses[ii][hh] = label[1:]
                            break

            self.l_guess = y_guesses[0].T
            self.l_guess = np.vstack([self.l_guess[i::2,:] for i in range(len(self.obs))])
        else:
            print("Unsupported model!")

    def predict_node_binary(self, prob_params):   
        node_model = self.model[0]
        # Make the node prediction for local binaries    
        x = Variable(torch.from_numpy(prob_params)).float().to(device=node_model.device)
        if len(self.cpl_agt) > 0:
            edge_index = Variable(torch.tensor(self.cpl_agt).T).to(device=node_model.device)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long).to(device=node_model.device)
        scores = node_model.model(x, edge_index).cpu().detach().numpy()[:]
        torch.cuda.synchronize()
        if self.model_name == "GNNet":
            ind_max = np.argmax(scores, axis=1)
            n_nodes = x.shape[0]
            
            y_guesses = np.zeros((n_nodes, node_model.N_targs), dtype=int)
            for ii, idx in enumerate(ind_max):
                y_guesses[ii] = node_model.reverse_strategy_dict[idx]
            # print(y_guesses)
            self.l_guess = y_guesses

    def predict_edge_binary(self, prob_params):
        edge_model = self.model[1]
        # Make the edge prediction for coupling binaries 
        x = Variable(torch.from_numpy(prob_params)).float().to(device=edge_model.device)
        if len(self.cpl_agt) > 0:
            edge_index = Variable(torch.tensor(self.cpl_agt).T).to(device=edge_model.device)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long).to(device=edge_model.device)
        scores = edge_model.model(x, edge_index).cpu().detach().numpy()[:]
        torch.cuda.synchronize()
        if self.model_name == "GNNet":
            ind_max = np.argmax(scores, axis=1)
            n_edges = edge_index.shape[1]
            y_guesses = np.zeros((n_edges, edge_model.N_targs), dtype=int)
            for ii, idx in enumerate(ind_max):
                y_guesses[ii] = edge_model.reverse_strategy_dict[idx]
            self.le_guess = y_guesses

    def predict_heterognn_binary(self, prob_params, Obs_info):
        model = self.model
        # Make the node prediction for local binaries    
        x = {
            'robot': Variable(torch.from_numpy(prob_params)).float().to(device=model.device),
            'obs': Variable(torch.from_numpy(Obs_info)).float().to(device=model.device)
        }
        N_obs = x['obs'].shape[0]
        N_rob = x['robot'].shape[0]
        if len(self.cpl_agt) > 0:
            RR_edge_index = Variable(torch.tensor(self.cpl_agt).T)
        else:
            RR_edge_index = torch.empty((2, 0), dtype=torch.long)
        OR_edge_index = torch.from_numpy(np.indices((N_obs, N_rob)).reshape(2, -1)).long()
        RO_edge_index = torch.from_numpy(np.indices((N_rob, N_obs)).reshape(2, -1)).long()
        OO_edge_index = torch.from_numpy(obs_edge_index(N_obs).reshape(2, -1)).long()
        edge_index = {
            ('robot', 'rr-link', 'robot'): RR_edge_index.to(device=model.device),  
            ('obs', 'or-link', 'robot'): OR_edge_index.to(device=model.device),
            ('robot', 'ro-link', 'obs'): RO_edge_index.to(device=model.device), 
            ('obs', 'ro-link', 'obs'): OO_edge_index.to(device=model.device) 
        }

        scores = model.model(x, edge_index)
        node_scores = scores['OR'].cpu().detach().numpy()[:]
        edge_scores = scores['RR'].cpu().detach().numpy()[:]

        n_RO_edges = OR_edge_index.shape[1]
        n_RR_edges = RR_edge_index.shape[1]
        node_y_guesses = np.zeros((n_RO_edges, model.N_targs_RO), dtype=int)
        for ii in range(n_RO_edges):
            ind_max = np.argmax(node_scores, axis=1)
            for ii, idx in enumerate(ind_max):
                node_y_guesses[ii] = model.RO_reverse_strategy_dict[idx]
        self.l_guess = np.hstack([node_y_guesses[kk*N_rob:(kk+1)*N_rob, :] for kk in range(N_obs)])

        edge_y_guesses = np.zeros((n_RR_edges, model.N_targs_RR), dtype=int)
        for ii in range(n_RR_edges):
            ind_max = np.argmax(edge_scores, axis=1)
            for ii, idx in enumerate(ind_max):
                edge_y_guesses[ii] = model.RR_reverse_strategy_dict[idx]
        self.le_guess = edge_y_guesses


    def predict_joint_binary(self, prob_params):
        model = self.model
        # Make the node prediction for local binaries    
        x = Variable(torch.from_numpy(prob_params)).float().to(device=model.device)
        if len(self.cpl_agt) > 0:
            edge_index = Variable(torch.tensor(self.cpl_agt).T).to(device=model.device)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long).to(device=model.device)
        node_scores, edge_scores = model.model(x, edge_index)
        node_scores = node_scores.cpu().detach().numpy()[:]
        edge_scores = edge_scores.cpu().detach().numpy()[:]
        torch.cuda.synchronize()                  
        n_nodes = x.shape[0]
        n_edges = edge_index.shape[1]
        node_y_guesses = np.zeros((n_nodes, model.N_node_targs), dtype=int)
        for ii in range(n_nodes):
            ind_max = np.argmax(node_scores, axis=1)
            for ii, idx in enumerate(ind_max):
                node_y_guesses[ii] = model.node_reverse_strategy_dict[idx]
        self.l_guess = node_y_guesses

        edge_y_guesses = np.zeros((n_edges, model.N_edge_targs), dtype=int)
        for ii in range(n_edges):
            ind_max = np.argmax(edge_scores, axis=1)
            for ii, idx in enumerate(ind_max):
                edge_y_guesses[ii] = model.edge_reverse_strategy_dict[idx]
        self.le_guess = edge_y_guesses

    def NNoutput_to_binary(self):
        N_obs = len(self.obs)
        self.BO = np.vstack([map_activecontraint2binary(self.l_guess[i,:]) for i in range(self.l_guess.shape[0])])
        if self.M == 1:
            self.BO = np.vstack((self.BO[:,:self.H], self.BO[:,self.H:]))
        else:
            self.BO = np.stack([self.BO[:,n*self.H:(n+1)*self.H] for n in range(N_obs)]) # (self.BO[:,:self.H], self.BO[:,self.H:])

        N_cpl_cons = len(self.cpl_agt)
        if N_cpl_cons > 0:
            self.BB = np.vstack([map_activecontraint2binary(self.le_guess[i,:]) for i in range(self.le_guess.shape[0])])

    
    def solve_QP(self):
        W_pen = 1e6
        self.INFEASIBLE = False
        with grb.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            with grb.Model(env=env) as self.opt:        
                # Create variables 
                self.PP = self.opt.addMVar(shape=(2*self.M, self.H+1), lb=-GRB.INFINITY, ub=GRB.INFINITY,  name="P")        
                self.VV = self.opt.addMVar(shape=(2*self.M, self.H+1), lb=-GRB.INFINITY, ub=GRB.INFINITY,  name="V")        
                self.UU = self.opt.addMVar(shape=(2*self.M, self.H), lb=-GRB.INFINITY, ub=GRB.INFINITY,  name="U")        

                # Initial conditions and dynamics
                self.opt.addConstr(self.VV[:, 0] == self.v)
                self.opt.addConstr(self.PP[:, 0] == self.p)
                for k in range(self.H):
                    self.opt.addConstr(self.VV[:, k+1] == self.VV[:, k] + self.T*self.UU[:, k])
                    self.opt.addConstr(self.PP[:, k+1] == self.PP[:, k] + self.T*self.VV[:, k] + self.T**2/2*self.UU[:, k])
                    # Bound constraints
                    self.opt.addConstr(self.PP[:, k+1] <= np.tile([self.x_max, self.y_max], self.M))
                    self.opt.addConstr(self.PP[:, k+1] >= np.tile([self.x_min, self.y_min], self.M))
                    self.opt.addConstr(self.VV[:, k+1] <= self.v_max)
                    self.opt.addConstr(self.VV[:, k+1] >= self.v_min)
                    self.opt.addConstr(self.UU[:, k] <= self.u_max)
                    self.opt.addConstr(self.UU[:, k] >= self.u_min)

                # Objective function 
                obj = self.Wu*sum(sum(self.UU[:, k]*self.UU[:, k]) for k in range(self.H)) \
                    + self.Wp*sum(sum((self.PP[:, k]-self.goals)*(self.PP[:, k]-self.goals)) for k in range(self.H)) \
                    + self.Wpt*sum((self.PP[:, -1]-self.goals)*(self.PP[:, -1]-self.goals))
                
                # Collision avoidance with static obstacles
                # Slack variables for all constraints
                N_obs = len(self.obs)
                SLACK = []
                for idx, ob in enumerate(self.obs):
                    th = ob.theta; xo = ob.x_c; yo = ob.y_c; L0 = ob.L + self.d_min; W0 = ob.W + self.d_min
                    for m in range(self.M):
                        SLACK.append(self.opt.addMVar(shape=(4, self.H), lb=0.0, ub=GRB.INFINITY))
                        obj += W_pen*sum(sum(SLACK[-1]))
                        if self.M == 1:
                            BO = self.BO[2*idx+m:2*(idx+1)+m,:]
                        else:
                            BO = self.BO[idx][2*m:2*(m+1),:]
                        # 4 constraints
                        self.opt.addConstr(SLACK[-1][0,:] + np.cos(th)*(self.PP[2*m, 1:]-xo) + np.sin(th)*(self.PP[2*m+1, 1:]-yo) >= L0 - self.bigM*(BO[0,:]+BO[1,:]))
                        self.opt.addConstr(SLACK[-1][1,:] - np.sin(th)*(self.PP[2*m, 1:]-xo) + np.cos(th)*(self.PP[2*m+1, 1:]-yo) >= W0 - self.bigM*(1-BO[0,:]+BO[1,:]))
                        self.opt.addConstr(SLACK[-1][2,:] - np.cos(th)*(self.PP[2*m, 1:]-xo) - np.sin(th)*(self.PP[2*m+1, 1:]-yo) >= L0 - self.bigM*(1+BO[0,:]-BO[1,:]))
                        self.opt.addConstr(SLACK[-1][3,:] + np.sin(th)*(self.PP[2*m, 1:]-xo) - np.cos(th)*(self.PP[2*m+1, 1:]-yo) >= W0 - self.bigM*(2-BO[0,:]-BO[1,:]))

                # Collision avoidance between robots
                N_cpl_cons = len(self.cpl_agt)
                if N_cpl_cons > 0:
                    for idx, pair in enumerate(self.cpl_agt):
                        m, n = pair
                        SLACK.append(self.opt.addMVar(shape=(4, self.H), lb=0.0, ub=GRB.INFINITY))
                        obj += W_pen*sum(sum(SLACK[-1]))                            
                        # 4 constraints
                        BB = self.BB[2*idx:2*(idx+1),:]
                        self.opt.addConstr(SLACK[-1][0,:] + self.PP[2*m, 1:] - self.PP[2*n, 1:] >= 2*self.d_min - self.bigM*(BB[0,:]+BB[1,:]))
                        self.opt.addConstr(SLACK[-1][1,:] + self.PP[2*n, 1:] - self.PP[2*m, 1:] >= 2*self.d_min - self.bigM*(1-BB[0,:]+BB[1,:]))
                        self.opt.addConstr(SLACK[-1][2,:] + self.PP[2*m+1, 1:] - self.PP[2*n+1, 1:] >= 2*self.d_min - self.bigM*(1+BB[0,:]-BB[1,:]))
                        self.opt.addConstr(SLACK[-1][3,:] + self.PP[2*n+1, 1:] - self.PP[2*m+1, 1:] >= 2*self.d_min - self.bigM*(2-BB[0,:]-BB[1,:]))
                    
                self.opt.setObjective(obj, GRB.MINIMIZE)
                        
                try:
                    self.opt.optimize()
                    self.SOL_cont = {"U": self.UU.X, "X": np.vstack([self.PP.X, self.VV.X])}
                    self.cost_value = self.opt.ObjVal

                except grb.GurobiError: 
                    # Most likely infeasible issue, Gurobi can handle that
                    if self.opt.status == GRB.INFEASIBLE or self.opt.status == GRB.INF_OR_UNBD:
                        self.INFEASIBLE = True
                        print("THE CENTRALIZED PROBLEM IS INFEASIBLE!!!")
                        # https://www.gurobi.com/documentation/current/refman/py_model_feasrelaxs.html#pythonmethod:Model.feasRelaxS
                        self.opt.feasRelaxS(1, False, True, True)
                        self.opt.optimize()     
                        self.SOL_cont = {"U": self.UU.X, "X": np.vstack([self.PP.X, self.VV.X])}
                        self.cost_value = self.opt.ObjVal