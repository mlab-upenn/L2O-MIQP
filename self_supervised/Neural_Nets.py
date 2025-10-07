import numpy as np
import pickle, os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, random_split
from datetime import datetime
import wandb

from utils import *
from model import FFNet

"""
Some functions
"""
def constraint_violation(dis_traj, cont_traj, Obs_info):
    """
    dis_traj: N_obs x 4 x H array
    cont_traj: 2 x H array only for positions
    some other parameters such as obstacle info are hard-coded for now
    """
    bigM = 1e3
    d_min = 0.25
    N_obs = Obs_info.shape[0]
    violation = 0.0
    for o in range(N_obs):
        xo, yo, L0, W0, th = Obs_info[o,:]
        L0 = L0/2 + d_min; W0 = W0/2 + d_min
        # 4 constraints
        c1 = np.maximum(0, np.cos(th)*(cont_traj[0, 1:]-xo) + np.sin(th)*(cont_traj[1, 1:]-yo) - L0 - bigM*(1-dis_traj[o,0,:]))
        c2 = np.maximum(0, -np.sin(th)*(cont_traj[0, 1:]-xo) + np.cos(th)*(cont_traj[1, 1:]-yo) - W0 - bigM*(1-dis_traj[o,1,:]))
        c3 = np.maximum(0, -np.cos(th)*(cont_traj[0, 1:]-xo) - np.sin(th)*(cont_traj[1, 1:]-yo) - L0 - bigM*(1-dis_traj[o,2,:]))
        c4 = np.maximum(0, np.sin(th)*(cont_traj[0, 1:]-xo) - np.cos(th)*(cont_traj[1, 1:]-yo) - W0 - bigM*(1-dis_traj[o,3,:]))
        violation += np.sum(c1 + c2 + c3 + c4)
    return violation

# torch version of the function above
def constraint_violation_torch(dis_traj, cont_traj, Obs_info):
    """
    dis_traj: (N_obs, 4, H) tensor of discrete decision variables
    cont_traj: (2, H) tensor of continuous variables (positions)
    Returns: scalar tensor representing negative constraint violation
    """
    device = cont_traj.device
    bigM = 1e3; d_min = 0.25
    Obs_info = torch.from_numpy(Obs_info).to(device)

    violation = 0.0
    for o in range(Obs_info.size(0)):
        xo, yo, L0, W0, th = Obs_info[o]
        L0 = L0 / 2 + d_min; W0 = W0 / 2 + d_min
        x = cont_traj[0, 1:]; y = cont_traj[1, 1:]
        cos_th = torch.cos(th); sin_th = torch.sin(th)
        d = dis_traj[o] # (4, H)
        c1 = torch.relu(cos_th * (x - xo) + sin_th * (y - yo) - L0 - bigM * (1 - d[0, :]))
        c2 = torch.relu(-sin_th * (x - xo) + cos_th * (y - yo) - W0 - bigM * (1 - d[1, :]))
        c3 = torch.relu(-cos_th * (x - xo) - sin_th * (y - yo) - L0 - bigM * (1 - d[2, :]))
        c4 = torch.relu(sin_th * (x - xo) - cos_th * (y - yo) - W0 - bigM * (1 - d[3, :]))
        c5 = torch.relu(1 - d.sum(dim=0))

        violation += torch.sum(c1 + c2 + c3 + c4 + c5)

    return violation

"""
Main classes
"""
class Regression:

    def __init__(self, prob_features):
        """
        Constructor for Regression class.
        """
        self.prob_features = prob_features
        self.num_train, self.num_test = 0, 0
        self.model, self.model_fn = None, None
        self.n_bin = 4 # number of binaries used in the collision avoidance constraint

    def construct_features(self, params):
        prob_features = self.prob_features
        feature_vec = np.array([])
        for feature in prob_features:
            if feature == "x0":
                x0 = params['x0']
                feature_vec = np.hstack((feature_vec, x0))
            elif feature == "xg":
                xg = params['xg'] 
                feature_vec = np.hstack((feature_vec, xg))
            elif feature == "obstacles":
                obstacles = params['obstacles']
                feature_vec = np.hstack((feature_vec, np.reshape(obstacles, (4*self.n_obs))))
            elif feature == "obstacles_map":
                continue
            else:
                print('Feature {} is unknown'.format(feature))
        return feature_vec

    def setup_data(self, n_features, train_data, Obs_info):
        """
        Reads in data and constructs strategy dictionary
        """
        self.Obs_info = Obs_info
        self.n_obs = Obs_info.shape[0] # number of obstacles
        self.n_features = n_features

        self.X_train = train_data[0] # Problem parameters, will be inputs of the NNs
        self.Y_train = train_data[2] # Discrete solutions, will be outputs of the NNs
        self.P_train = train_data[1] # Continuous trajectories, will be used as parameters in training
        self.n_y = self.Y_train[0].size # will be the dimension of the output
        self.y_shape = self.Y_train[0].shape
        self.num_train = self.Y_train.shape[0]        

        # Create features and labels based on raw data
        self.features = np.zeros((self.num_train, self.n_features))
        self.labels = np.zeros((self.num_train, self.n_y))
        self.outputs = np.zeros((self.num_train, self.n_y*self.n_bin))        
        for ii in range(self.num_train):
            self.labels[ii] = np.reshape(self.Y_train[ii,:,:], (self.n_y))
            self.outputs[ii] = np.hstack([int_to_four_bins(val) for val in (self.labels[ii])])
            prob_params = {}
            for k in self.X_train:
                prob_params[k] = self.X_train[k][ii]
            self.features[ii] = self.construct_features(prob_params)

    def setup_network(self, depth=3, neurons=32, device_id=0):
        self.device = torch.device('cuda:{}'.format(device_id))
        ff_shape = [self.n_features]
        for ii in range(depth):
            ff_shape.append(neurons)
        ff_shape.append(self.n_y*self.n_bin)

        self.model = FFNet(ff_shape, activation=nn.ReLU()).to(device=self.device)

        # file names for PyTorch models
        now = datetime.now().strftime('%Y%m%d_%H%M')
        model_fn = 'regression_{}.pt'
        model_fn = os.path.join(os.getcwd(), model_fn)
        self.model_fn = model_fn.format(now)

    def load_network(self, fn_classifier_model):
        if os.path.exists(fn_classifier_model):
            print('Loading presaved Hetero GNN classifier model from {}'.format(fn_classifier_model))
            self.model.load_state_dict(torch.load(fn_classifier_model))
            self.model_fn = fn_classifier_model

    def train(self, training_params, verbose=True):
        BATCH_SIZE = training_params['BATCH_SIZE']
        TEST_BATCH_SIZE = training_params['TEST_BATCH_SIZE']
        TRAINING_EPOCHS = training_params['TRAINING_EPOCHS']
        CHECKPOINT_AFTER = training_params['CHECKPOINT_AFTER']
        SAVEPOINT_AFTER = training_params['SAVEPOINT_AFTER']
        LEARNING_RATE = training_params['LEARNING_RATE']
        WEIGHT_DECAY = training_params['WEIGHT_DECAY']
        EARLY_STOPPING_PATIENCE = training_params['EARLY_STOPPING_PATIENCE']

        model = self.model
        device = self.device

        # Initialize wandb
        wandb.init(project="Learning_MICP", config=training_params)        

        # Prepare dataset
        X_tensor = torch.from_numpy(self.features).float()
        Y_tensor = torch.from_numpy(self.outputs).float()
        P_tensor = torch.from_numpy(self.P_train['XX'][:,:2,:]).float()
        full_dataset = TensorDataset(X_tensor, Y_tensor, P_tensor)

        # Split into train/validation
        num_total = len(full_dataset)
        num_train = int(0.9*num_total)
        train_dataset = TensorDataset(X_tensor[:num_train], Y_tensor[:num_train], P_tensor[:num_train])
        val_dataset   = TensorDataset(X_tensor[num_train:], Y_tensor[num_train:], P_tensor[num_train:])
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

        # Loss and optimizer
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        best_val_loss = float('inf')
        epochs_since_improvement = 0

        itr = 1
        for epoch in range(TRAINING_EPOCHS):
            model.train()
            running_loss = 0.0
            for inputs, y_true, _ in train_loader:
                inputs = inputs.to(device)
                y_true = y_true.to(device)
                optimizer.zero_grad()
                logits = model(inputs)
                loss = loss_fn(logits, y_true)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                wandb.log({"train_loss": loss.item(), "iteration": itr})
                itr += 1

            avg_train_loss = running_loss / len(train_loader)
            # Log to wandb
            wandb.log({"avg_train_loss": avg_train_loss, "epoch": epoch})
            
            if epoch % SAVEPOINT_AFTER == 0:
                torch.save(model.state_dict(), self.model_fn)
                if verbose:
                    print(f"[Epoch {epoch}], [Iter {itr}] Saved model at {self.model_fn}")

            if epoch % CHECKPOINT_AFTER == 0:
                # Evaluate on validation set
                model.eval()
                with torch.no_grad():
                    val_loss_total = 0
                    val_cons_violation = []
                    bitwise_accs = []

                    for val_inputs, val_targets, val_params in val_loader:
                        # Get the loss values
                        val_inputs = val_inputs.to(device)
                        val_targets = val_targets.to(device)
                        val_params = val_params.to(device)
                        val_logits = model(val_inputs)
                        val_loss = loss_fn(val_logits, val_targets)
                        val_loss_total += val_loss.item()

                        val_preds = val_logits.int() # Already rounded by STE_Round
                        # Compare accuracy
                        bitwise_accs.append(compute_bitwise_accuracy(val_preds, val_targets.int()))

                        # Evaluate constraint violation
                        constraint_loss = self.batch_constraint_violation_loss(val_preds, val_params).item()
                        val_cons_violation.append(constraint_loss)

                    avg_val_loss = val_loss_total/len(val_loader)
                    avg_bitwise_acc = np.mean(bitwise_accs)
                    avg_val_cons_violation = np.mean(val_cons_violation)

                    if verbose:
                        print(f"[Epoch {epoch}], [Iter {itr}] Validation loss: {avg_val_loss:.4f} | "
                            f"Validation accuracy (bitwise): {avg_bitwise_acc:.4f} | "
                            f"Constraint violation: {avg_val_cons_violation:.4f}")
                    
                    # Log to wandb
                    wandb.log({"val/loss": avg_val_loss,
                        "val/bitwise_acc": avg_bitwise_acc,
                        "val/constraint_violation": avg_val_cons_violation,
                        "epoch": epoch})

                    # Check for early stopping
                    if avg_val_loss < best_val_loss - 1e-3:
                        best_val_loss = avg_val_loss
                        epochs_since_improvement = 0
                    else:
                        epochs_since_improvement += 1
                        if epochs_since_improvement >= EARLY_STOPPING_PATIENCE:
                            print(f"Early stopping: no improvement for {EARLY_STOPPING_PATIENCE} epochs")
                            torch.save(model.state_dict(), self.model_fn)
                            wandb.save(self.model_fn)
                            print(f"Final model saved at {self.model_fn}")
                            wandb.finish()
                            return  # Exit training early

        # Save final model
        torch.save(model.state_dict(), self.model_fn)
        wandb.save(self.model_fn)
        print(f"Final model saved at {self.model_fn}")
        print("Done training.")
        wandb.finish()

    def batch_constraint_violation_loss(self, dis_traj_pred, cont_traj_pred, lambda_penalty=1.0):
        """
        Compute the constraint violation as loss function for a batch data
        """
        batch_size = dis_traj_pred.size(0)
        total_violation = 0.0
        for b in range(batch_size):
            dis_traj = NNoutput_reshape_torch(dis_traj_pred[b], self.n_obs)
            cont_traj = cont_traj_pred[b]
            total_violation += constraint_violation_torch(dis_traj, cont_traj, self.Obs_info)
        return lambda_penalty * total_violation / batch_size

    # Train with self-supervised loss function
    def SS_train(self, training_params, verbose=True, penalty_weight = 1.0):
        """
        Implement self-supervised learning with constraint violation based loss
        penalty_weight: the penalty weight for constraint violation if linearly combine with supervised loss 
        """
        BATCH_SIZE = training_params['BATCH_SIZE']
        TEST_BATCH_SIZE = training_params['TEST_BATCH_SIZE']
        TRAINING_EPOCHS = training_params['TRAINING_EPOCHS']
        CHECKPOINT_AFTER = training_params['CHECKPOINT_AFTER']
        SAVEPOINT_AFTER = training_params['SAVEPOINT_AFTER']
        LEARNING_RATE = training_params['LEARNING_RATE']
        WEIGHT_DECAY = training_params['WEIGHT_DECAY']
        EARLY_STOPPING_PATIENCE = training_params['EARLY_STOPPING_PATIENCE']

        model = self.model
        device = self.device

        wandb.init(project="Learning_MICP", config=training_params)        

        # Prepare dataset
        X_tensor = torch.from_numpy(self.features).float()
        Y_tensor = torch.from_numpy(self.outputs).float()
        P_tensor = torch.from_numpy(self.P_train['XX'][:,:2,:]).float()
        full_dataset = TensorDataset(X_tensor, Y_tensor, P_tensor)

        # Split into train/val
        num_total = len(full_dataset)
        num_train = int(0.9*num_total)
        train_dataset = TensorDataset(X_tensor[:num_train], Y_tensor[:num_train], P_tensor[:num_train])
        val_dataset = TensorDataset(X_tensor[num_train:], Y_tensor[num_train:], P_tensor[num_train:])
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True)

        # supervised loss and optimizer
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)        
        best_val_loss = float('inf')
        epochs_since_improvement = 0

        itr = 1
        for epoch in range(TRAINING_EPOCHS):
            model.train()
            running_loss = 0.0
            for inputs, y_true, params in train_loader:
                inputs = inputs.to(device)
                y_true = y_true.to(device)
                params = params.to(device)
                optimizer.zero_grad()
                logits = model(inputs)  # shape: (B, N_obs*4*H)
                loss = self.batch_constraint_violation_loss(logits, params, lambda_penalty=penalty_weight) # + loss_fn(logits, y_true)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                wandb.log({"train_loss": loss.item(), "iteration": itr})
                itr += 1

            avg_train_loss = running_loss / len(train_loader)
            # Log to wandb
            wandb.log({"avg_train_loss": avg_train_loss, "epoch": epoch})

            if epoch % SAVEPOINT_AFTER == 0:
                torch.save(model.state_dict(), self.model_fn)
                if verbose:
                    print(f"[Epoch {epoch}], [Iter {itr}] Saved model at {self.model_fn}")

            if epoch % CHECKPOINT_AFTER == 0:
                # Evaluate on validation set
                model.eval()
                with torch.no_grad():
                    val_loss_total = 0
                    val_cons_violation = []
                    bitwise_accs = []

                    for val_inputs, val_targets, val_params in val_loader:
                        val_inputs = val_inputs.to(device)
                        val_targets = val_targets.to(device)
                        val_params = val_params.to(device)

                        val_logits = model(val_inputs)
                        val_loss = loss_fn(val_logits, val_targets) + self.batch_constraint_violation_loss(val_logits, val_params, lambda_penalty=penalty_weight)
                        val_loss_total += val_loss.item()
                        val_preds = val_logits.int() # Already rounded by STE_Round

                        # Compare accuracy
                        bitwise_accs.append(compute_bitwise_accuracy(val_preds, val_targets.int()))
                        # Evaluate constraint violation
                        constraint_loss = self.batch_constraint_violation_loss(val_preds, val_params).item()
                        val_cons_violation.append(constraint_loss)

                    avg_val_loss = val_loss_total/len(val_loader)
                    avg_bitwise_acc = np.mean(bitwise_accs)
                    avg_val_cons_violation = np.mean(val_cons_violation)

                    if verbose:
                        print(f"[Epoch {epoch}], [Iter {itr}] Validation loss: {avg_val_loss:.4f} | "
                            f"Validation accuracy (bitwise): {avg_bitwise_acc:.4f} | "
                            f"Constraint violation: {avg_val_cons_violation:.4f}")
                    # Log to wandb
                    wandb.log({"val/loss": avg_val_loss,
                        "val/bitwise_acc": avg_bitwise_acc,
                        "val/constraint_violation": avg_val_cons_violation,
                        "epoch": epoch})
                    
                    # Check for early stopping
                    if avg_val_loss < best_val_loss - 1e-3:
                        best_val_loss = avg_val_loss
                        epochs_since_improvement = 0
                    else:
                        epochs_since_improvement += 1
                        if epochs_since_improvement >= EARLY_STOPPING_PATIENCE:
                            print(f"Early stopping: no improvement for {EARLY_STOPPING_PATIENCE} epochs")
                            torch.save(model.state_dict(), self.model_fn)
                            wandb.save(self.model_fn)
                            print(f"Final model saved at {self.model_fn}")
                            wandb.finish()
                            return  # Exit training early

        # Save final model
        torch.save(model.state_dict(), self.model_fn)
        wandb.save(self.model_fn)
        print(f"Final model saved at {self.model_fn}")
        print("Done training.")
        wandb.finish()