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
# def constraint_violation(dis_traj, cont_traj, Obs_info):
#     """
#     dis_traj: N_obs x 4 x H array
#     cont_traj: 2 x H array only for positions
#     some other parameters such as obstacle info are hard-coded for now
#     """
#     bigM = 1e3
#     d_min = 0.25
#     N_obs = Obs_info.shape[0]
#     violation = 0.0
#     for o in range(N_obs):
#         xo, yo, L0, W0, th = Obs_info[o,:]
#         L0 = L0/2 + d_min; W0 = W0/2 + d_min
#         # 4 constraints
#         c1 = np.maximum(0, np.cos(th)*(cont_traj[0, 1:]-xo) + np.sin(th)*(cont_traj[1, 1:]-yo) - L0 - bigM*(1-dis_traj[o,0,:]))
#         c2 = np.maximum(0, -np.sin(th)*(cont_traj[0, 1:]-xo) + np.cos(th)*(cont_traj[1, 1:]-yo) - W0 - bigM*(1-dis_traj[o,1,:]))
#         c3 = np.maximum(0, -np.cos(th)*(cont_traj[0, 1:]-xo) - np.sin(th)*(cont_traj[1, 1:]-yo) - L0 - bigM*(1-dis_traj[o,2,:]))
#         c4 = np.maximum(0, np.sin(th)*(cont_traj[0, 1:]-xo) - np.cos(th)*(cont_traj[1, 1:]-yo) - W0 - bigM*(1-dis_traj[o,3,:]))
#         violation += np.sum(c1 + c2 + c3 + c4)
#     return violation

# torch version of the function above
# def constraint_violation_torch(dis_traj, cont_traj, Obs_info):
#     """
#     dis_traj: (N_obs, 4, H) tensor of discrete decision variables
#     cont_traj: (2, H) tensor of continuous variables (positions)
#     Returns: scalar tensor representing negative constraint violation
#     """
#     device = cont_traj.device
#     bigM = 1e3; d_min = 0.25
#     Obs_info = torch.from_numpy(Obs_info).to(device)

#     violation = 0.0
#     for o in range(Obs_info.size(0)):
#         xo, yo, L0, W0, th = Obs_info[o]
#         L0 = L0 / 2 + d_min; W0 = W0 / 2 + d_min
#         x = cont_traj[0, 1:]; y = cont_traj[1, 1:]
#         cos_th = torch.cos(th); sin_th = torch.sin(th)
#         d = dis_traj[o] # (4, H)
#         c1 = torch.relu(cos_th * (x - xo) + sin_th * (y - yo) - L0 - bigM * (1 - d[0, :]))
#         c2 = torch.relu(-sin_th * (x - xo) + cos_th * (y - yo) - W0 - bigM * (1 - d[1, :]))
#         c3 = torch.relu(-cos_th * (x - xo) - sin_th * (y - yo) - L0 - bigM * (1 - d[2, :]))
#         c4 = torch.relu(sin_th * (x - xo) - cos_th * (y - yo) - W0 - bigM * (1 - d[3, :]))
#         c5 = torch.relu(1 - d.sum(dim=0))

#         violation += torch.sum(c1 + c2 + c3 + c4 + c5)

#     return violation

# torch version of the function above
def constraint_violation_torch(dis_traj, cont_traj, Obs_info=None):
    """
    dis_traj: (N_obs, 4, H) or (B, N_obs, 4, H)
    cont_traj: (2, H) or (B, 2, H)
    Returns: scalar tensor or per-sample tensor of constraint violations
    """
    squeeze_batch = False
    if dis_traj.dim() == 3:
        dis_traj = dis_traj.unsqueeze(0)
        cont_traj = cont_traj.unsqueeze(0)
        squeeze_batch = True

    if cont_traj.dim() == 2:
        cont_traj = cont_traj.unsqueeze(0)

    if dis_traj.dim() != 4 or cont_traj.dim() != 3:
        raise ValueError("Unexpected input shapes for constraint_violation_torch")

    device = cont_traj.device
    dtype = cont_traj.dtype
    bigM = 1e3
    d_min = 0.25
    if Obs_info is None:
        # Hard-coded obstacle info for now
        Obs_info = np.array([[1.0, 0.0, 0.4, 0.5, 0.0],
            [0.7, -1.1, 0.5, 0.4, 0.0],
            [0.40, -2.50, 0.4, 0.5, 0.0]])
    Obs_info = torch.tensor(Obs_info, device=device, dtype=dtype)

    N_obs = dis_traj.size(1)
    if Obs_info.size(0) != N_obs:
        raise ValueError("Mismatch between obstacle info and discrete trajectory shape")

    x = cont_traj[:, 0, 1:]
    y = cont_traj[:, 1, 1:]

    rel_x = x.unsqueeze(1) - Obs_info[:, 0].view(1, -1, 1)
    rel_y = y.unsqueeze(1) - Obs_info[:, 1].view(1, -1, 1)
    rel_coords = torch.stack((rel_x, rel_y), dim=2)  # (B, N_obs, 2, H)

    if dis_traj.size(-1) != rel_coords.size(-1):
        raise ValueError("Mismatch between discrete trajectory horizon and continuous trajectory horizon")

    th = Obs_info[:, 4]
    cos_th = torch.cos(th)
    sin_th = torch.sin(th)
    rot_mat = torch.stack(
        (
            torch.stack((cos_th,  sin_th), dim=1),
            torch.stack((-sin_th, cos_th), dim=1),
            torch.stack((-cos_th, -sin_th), dim=1),
            torch.stack((sin_th, -cos_th), dim=1),
        ),
        dim=1,
    ).unsqueeze(0)  # (1, N_obs, 4, 2)

    proj = torch.matmul(rot_mat, rel_coords)  # (B, N_obs, 4, H)

    L0 = Obs_info[:, 2] / 2 + d_min
    W0 = Obs_info[:, 3] / 2 + d_min
    thresholds = torch.stack((L0, W0, L0, W0), dim=1).unsqueeze(0).unsqueeze(-1)

    d = dis_traj.to(dtype)
    c = torch.relu(proj - thresholds - bigM * (1 - d))
    c5 = torch.relu(1 - d.sum(dim=2))

    violation = c.sum(dim=(1, 2, 3)) + c5.sum(dim=(1, 2))

    if squeeze_batch:
        return violation.squeeze(0)
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

    def setup_data(self, n_features: int, train_data: np.ndarray, Obs_info: np.ndarray=None):
        self.n_features = n_features
        self.Obs_info = Obs_info if Obs_info is not None else np.zeros((3,4))
        self.n_obs = self.Obs_info.shape[0] if Obs_info is not None else 3

        self.X_train = train_data[0]  # Problem parameters, will be inputs of the NNs
        self.Y_train = train_data[2]  # Discrete solutions, will be outputs of the NNs
        self.P_train = train_data[1]  # Continuous trajectories, will be used as parameters in training

        self.num_train = self.Y_train.shape[0]
        self.y_shape = self.Y_train.shape[1:]
        self.n_y = int(np.prod(self.y_shape))

        feature_blocks = []
        for feature in self.prob_features:
            if feature == "obstacles_map":
                continue

            values = self.X_train.get(feature)
            if values is None:
                print('Feature {} is unknown or missing'.format(feature))
                continue

            values = np.asarray(values)
            if values.shape[0] != self.num_train:
                raise ValueError(
                    f"Feature '{feature}' has {values.shape[0]} samples, expected {self.num_train}"
                )

            feature_blocks.append(values.reshape(self.num_train, -1))

        if feature_blocks:
            self.features = np.concatenate(feature_blocks, axis=1)
        else:
            self.features = np.zeros((self.num_train, 0))

        if self.features.shape[1] != self.n_features:
            self.n_features = self.features.shape[1]

        self.labels = self.Y_train.reshape(self.num_train, self.n_y)
        labels_int = self.labels.astype(np.int64, copy=False)
        bit_shifts = np.arange(self.n_bin - 1, -1, -1, dtype=np.int64)
        outputs_bits = (labels_int[..., None] >> bit_shifts) & 1
        self.outputs = outputs_bits.reshape(self.num_train, -1)

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
            for inputs, y_true, params in train_loader:
                inputs = inputs.to(device)
                y_true = y_true.to(device)
                params = params.to(device)
                optimizer.zero_grad()
                logits = model(inputs)
                loss = loss_fn(logits, y_true)  # + 0.001 * self.batch_constraint_violation_loss(logits, params, lambda_penalty=1.0)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

    # def batch_constraint_violation_loss(self, dis_traj_pred, cont_traj_pred, lambda_penalty=1.0):
    #     """
    #     Compute the constraint violation as loss function for a batch data
    #     """
    #     batch_size = dis_traj_pred.size(0)
    #     total_violation = 0.0
    #     for b in range(batch_size):
    #         dis_traj = NNoutput_reshape_torch(dis_traj_pred[b], self.n_obs)
    #         cont_traj = cont_traj_pred[b]
    #         total_violation += constraint_violation_torch(dis_traj, cont_traj, self.Obs_info)
    #     return lambda_penalty * total_violation / batch_size

    def batch_constraint_violation_loss(self, dis_traj_pred, cont_traj_pred, lambda_penalty=1.0):
        dis_traj = NNoutput_reshape_torch(dis_traj_pred, self.n_obs)
        violations = constraint_violation_torch(dis_traj, cont_traj_pred, self.Obs_info)
        mean_violation = violations.mean()
        return lambda_penalty * mean_violation

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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

                    # itr = 0
                    for val_inputs, val_targets, val_params in val_loader:
                        val_inputs = val_inputs.to(device)
                        val_targets = val_targets.to(device)
                        val_params = val_params.to(device)

                        val_logits = model(val_inputs)
                        val_loss = loss_fn(val_logits, val_targets) + self.batch_constraint_violation_loss(val_logits, val_params, lambda_penalty=penalty_weight)
                        val_loss_total += val_loss.item()
                        # val_preds = val_logits.int() # Already rounded by STE_Round
                        val_preds = val_logits

                        # Compare accuracy
                        bitwise_accs.append(compute_bitwise_accuracy(val_preds, val_targets.int()))
                        # Evaluate constraint violation
                        constraint_loss = self.batch_constraint_violation_loss(val_preds, val_params, 1.0).item()
                        val_cons_violation.append(constraint_loss)
                        # itr += 1
                        # if itr > 5:
                        #     break

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