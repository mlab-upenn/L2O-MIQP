import os
import torch
import torch.nn as nn
import numpy as np
import cvxpy as cp
import wandb
import random
from robot_nav.backups.cons_utils import *

# Define some penalty functions that we may use
l1_penalty = lambda s: s.sum(dim=1)
l2_penalty = lambda s: (s**2).sum(dim=1)

def combined_loss_fcn(loss_components, weights):
    """
    Combine multiple loss components with given weights.
    """
    assert len(loss_components) == len(weights), "Number of loss components must match number of weights."
    combined_loss = sum(w * lc for w, lc in zip(weights, loss_components))
    return combined_loss

class SSL_MIQP_incorporated:

    def __init__(self, nn_model, cvx_layer, nx, ny, device=None):
        """
        Initialize the SSL_MIQP model.
        Args:
            nn_model: The neural network model.
            cvx_layer: The CVXPY layer for optimization.
            nx: Number of input features.
            ny: Number of output features.
            device: The device (CPU or GPU).    
        """
        self.nn_model = nn_model
        self.cvx_layer = cvx_layer
        self.nx, self.ny = nx, ny
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_SL(self, train_loader, test_loader, training_params, wandb_log = False):
        """
        Train the neural network model using supervised learning.
        Args:
            ground_truth_solver: Function to get ground truth solutions.
            train_loader: DataLoader for training data.
            test_loader: DataLoader for testing data.
            training_params: Dictionary containing training parameters.
            wandb_log: Boolean indicating whether to log to Weights & Biases.
        """
        # Hyperparameters
        TRAINING_EPOCHS = training_params['TRAINING_EPOCHS']
        CHECKPOINT_AFTER = training_params['CHECKPOINT_AFTER']
        LEARNING_RATE = training_params['LEARNING_RATE']
        WEIGHT_DECAY = training_params['WEIGHT_DECAY']
        PATIENCE = training_params['PATIENCE']
        
        # Put all layers in device
        device = self.device
        self.nn_model.to(device)

        # Initialize training components
        global_step = 0
        optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=int(PATIENCE/2))
        supervised_loss_fn = nn.HuberLoss() # supervised loss function 
        best_val_loss = float("inf") # Store best validation 
        epochs_no_improve = 0  # Count epochs with no improvement  
        if wandb_log: wandb.init(
            project=training_params.get("WANDB_PROJECT", "supervised_learning"),
            name=training_params.get("RUN_NAME", None),
            config=training_params
        )    

        for epoch in range(1, TRAINING_EPOCHS+1):
            self.nn_model.train()
            for theta_batch, y_gt_batch, _, _ in train_loader:
                theta_batch = theta_batch.to(device)
                y_gt_batch = y_gt_batch.to(device)
                
                # B = theta_batch.shape[0] # batch size
                # ---- Predict y from theta ----
                y_pred = self.nn_model(theta_batch).float() # (B, ny), hard {0,1}
                supervised_loss = supervised_loss_fn(y_pred, y_gt_batch.float())
                loss = supervised_loss
                if wandb_log: wandb.log({
                    "train/loss": loss.item(),
                    "step": global_step})
                
                # ---- Backprop ----
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.nn_model.parameters(), max_norm=1e1)
                optimizer.step()
                global_step += 1
                
                # ---- Logging ----
                if global_step == 1 or (global_step % CHECKPOINT_AFTER) == 0:
                    training_loss = loss.item()
                    val_loss_total = 0.0
                    self.nn_model.eval()
                    with torch.no_grad():
                        for val_theta_batch, val_y_gt_batch, _, _ in test_loader:
                            val_theta_batch = val_theta_batch.to(device)
                            val_y_gt_batch = val_y_gt_batch.to(device)
                            # ---- Predict y from theta ----
                            y_pred_test = self.nn_model(val_theta_batch).float() # (B, ny), hard {0,1}
                            val_loss_total += supervised_loss_fn(y_pred_test, val_y_gt_batch.float()).item()                           
                        avg_val_loss = val_loss_total / len(test_loader)

                    print(f"[epoch {epoch} | step {global_step}] "
                        f"training loss = {training_loss:.4f}, "
                        f"validation loss = {avg_val_loss:.4f}")
                    # --- Log losses to wandb ---
                    if wandb_log: wandb.log({
                        "val/loss": avg_val_loss,
                        "epoch": epoch})

                    # Check if need to update the learning rates
                    last_lr = optimizer.param_groups[0]['lr']
                    scheduler.step(loss.item())
                    current_lr = optimizer.param_groups[0]['lr']
                    if current_lr != last_lr:
                        print(f"Learning rate updated: {last_lr:.6f} -> {current_lr:.6f}")
                        last_lr = current_lr

                    # if avg_val_loss < best_val_loss:
                    #     best_val_loss = avg_val_loss
                    #     epochs_no_improve = 0
                    # else:
                    #     epochs_no_improve += 1
                    #     if epochs_no_improve >= PATIENCE:
                    #         print("Early stopping triggered!")
                    #         return

        if wandb_log: wandb.finish()
                    
    def train_SSL(self, train_loader, test_loader, training_params, loss_weights, 
            loss_scale = 1.0, wandb_log = False):
        """
        Train the neural network model using self-supervised learning.
        Args:
            ground_truth_solver: Function to get ground truth solutions.
            train_loader: DataLoader for training data.
            test_loader: DataLoader for testing data.
            training_params: Dictionary containing training parameters.
            loss_weights: List of weights for different loss components.
            obj_fcn: Function to compute objective value given (x, y).
            y_cons: List of functions to compute constraint violations only on y (integer variables).
            slack_penalty: Function to compute penalty on slack variables.
            constraint_penalty: Function to compute penalty on constraint violations.
            wandb_log: Boolean indicating whether to log to Weights & Biases.
        """
        # Hyperparameters
        TRAINING_EPOCHS = training_params['TRAINING_EPOCHS']
        CHECKPOINT_AFTER = training_params['CHECKPOINT_AFTER']
        LEARNING_RATE = training_params['LEARNING_RATE']
        WEIGHT_DECAY = training_params['WEIGHT_DECAY']
        PATIENCE = training_params['PATIENCE']
        
        # Put all layers in device
        device = self.device
        self.nn_model.to(device)
        # self.cvx_layer.to(device)

        # Define weights for loss components
        weights = torch.tensor(loss_weights, device=device)
        weights = loss_scale*weights / weights.sum() # may scale up the weights a bit
        global_step = 0
        optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=int(PATIENCE/2))
        supervised_loss_fn = nn.HuberLoss() 
        best_val_loss = float("inf") # Store best validation 
        epochs_no_improve = 0  # Count epochs with no improvement 

        started_wandb_run = False
        if wandb_log and wandb.run is None:
            wandb.init(
                project=training_params.get("WANDB_PROJECT", "self_supervised_learning"),
                name=training_params.get("RUN_NAME", None),
                config=training_params
            )
            started_wandb_run = True     
    
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(1, TRAINING_EPOCHS+1):
            self.nn_model.train()
            for theta, y_gt, _, _ in train_loader:
                theta = theta.to(device)
                y_gt = y_gt.to(device)
                
                # warped forward pass
                y_pred, u_opt, p_opt, s_opt, obj_val = self.forward(theta)

                obj_val = obj_val.mean()

                # Supervised loss from dataset
                supervised_loss = supervised_loss_fn(y_pred, y_gt.float())
                slack_pen = s_opt.sum(dim=1).mean()
                y_transposed = NNoutput_reshape_torch(y_pred, N_obs=3) # (B, 3, 4, H)
                y_penalty = constraint_violation_torch(y_transposed, p_opt[:, :2, :]).mean()
                
                # Total loss with balanced weights
                # loss = supervised_loss
                loss = combined_loss_fcn([obj_val, slack_pen, y_penalty, supervised_loss], weights)
                if wandb_log: wandb.log({
                    "train/combined_loss": loss.item(),
                    "train/obj_val": obj_val.item(),
                    "train/slack_pen": slack_pen.item(),
                    "train/y_penalty": y_penalty.item(),
                    "train/supervised_loss": supervised_loss.item(),
                    "step": global_step})
                
                # print(f"[epoch {epoch} | step {global_step}] "
                #     f"train: loss = {loss.item():.4f}, "
                #     f"obj_val = {obj_val.item():.4f}, "
                #     f"slack_pen = {slack_pen.item():.4f}, "
                #     f"y_penalty = {y_penalty.item():.4f}, "
                #     f"supervised_loss = {supervised_loss.item():.4f}")

                # ---- Backprop ----
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.nn_model.parameters(), max_norm=1e1)
                optimizer.step()         
                global_step += 1

                # ---- Logging ----
                if global_step == 1 or (global_step % CHECKPOINT_AFTER) == 0:
                    self.nn_model.eval()
                    val_loss_total = []; obj_val_total = []; opt_obj_val_total = []
                    slack_pen_total = []; y_penalty_total = []
                    supervised_loss_total = []

                
                    with torch.no_grad():
                        # for theta, y_gt, pv_gt, u_gt in test_loader:
                        theta, y_gt, pv_gt, u_gt = next(iter(test_loader))                     
                        theta = theta.to(device)
                        y_gt = y_gt.to(device)
                        pv_gt = pv_gt.to(device)
                        u_gt = u_gt.to(device)

                        # ---- Predict y from theta ----
                        y_pred, u_opt, p_opt, s_opt, obj_val = self.forward(theta)    
                        obj_val = obj_val.mean()

                        supervised_loss = supervised_loss_fn(y_pred, y_gt.float())

                        # Slack penalty included inside cvx_layer.solve
                        slack_pen = s_opt.sum(dim=1).mean()
                        y_transposed = NNoutput_reshape_torch(y_pred, N_obs=3) # (B, 3, 4, H)
                        y_penalty = constraint_violation_torch(y_transposed, p_opt[:, :2, :]).mean()
            
                        # Total loss with balanced weights
                        loss = combined_loss_fcn([obj_val, slack_pen, y_penalty, supervised_loss], weights)
                        opt_obj_val = obj_function(u_gt, pv_gt[:, :2, :], theta, meta=None).mean()

                        # Collect loss values
                        val_loss_total.append(loss.item())       
                        obj_val_total.append(obj_val.item())
                        opt_obj_val_total.append(opt_obj_val.item())
                        slack_pen_total.append(slack_pen.item())
                        y_penalty_total.append(y_penalty.item())
                        supervised_loss_total.append(supervised_loss.item())

                        # Compute the averages
                        avg_val_loss = torch.mean(torch.tensor(val_loss_total))
                        avg_obj_val = torch.mean(torch.tensor(obj_val_total))
                        avg_opt_obj_val = torch.mean(torch.tensor(opt_obj_val_total))
                        # Added a constant to make the opt gap not crazy
                        opt_gap = 100*(torch.tensor(obj_val_total) - 
                            torch.tensor(opt_obj_val_total))/(1e2 + torch.tensor(opt_obj_val_total).abs())
                        avg_opt_gap = opt_gap.mean()
                        avg_slack_pen = torch.mean(torch.tensor(slack_pen_total))
                        avg_y_penalty = torch.mean(torch.tensor(y_penalty_total))
                        avg_supervised_loss = torch.mean(torch.tensor(supervised_loss_total))

                    print(f"[epoch {epoch} | step {global_step}] "
                        f"validation: loss = {avg_val_loss:.4f}, "
                        f"obj_val = {avg_obj_val:.4f}, "
                        f"opt_gap = {avg_opt_gap:.4f} %, "
                        f"slack_pen = {avg_slack_pen:.4f}, "
                        f"y_penalty = {avg_y_penalty:.4f}, "
                        f"supervised_loss = {avg_supervised_loss:.4f}, ")
                    # --- Log losses to wandb ---
                    if wandb_log: wandb.log({
                        # "val/loss": avg_val_loss,
                        "val/avg_loss": avg_val_loss,
                        "val/obj_val": avg_obj_val,
                        "val/slack_pen": avg_slack_pen,
                        "val/y_penalty": avg_y_penalty,
                        "val/supervised_loss": avg_supervised_loss,
                        "epoch": epoch})       

                    # Check if need to update the learning rates
                    last_lr = optimizer.param_groups[0]['lr']
                    scheduler.step(loss.item())
                    current_lr = optimizer.param_groups[0]['lr']
                    if current_lr != last_lr:
                        print(f"Learning rate updated: {last_lr:.6f} -> {current_lr:.6f}")
                        last_lr = current_lr
                    # Early stopping check
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= PATIENCE:
                            print("Early stopping triggered!")
                            return          
                    # # Stop if constraint violations are small enough
                    # if slack_pen.item() < 1e-4 and y_penalty < 1e-4:
                    #     print("Slack penalty below threshold, stopping training.")
                    #     return

        if wandb_log and started_wandb_run: 
            wandb.finish()          

    def evaluate(self, data_loader, ground_truth_solver = None, save_path = "eval_results.pt"):
        """
        Evaluate the neural network model on given data.
        Args:
            data_loader: DataLoader for evaluation data.
            ground_truth_solver: Function to get ground truth solutions. (probably not needed here)
        Returns:
            Dictionary containing evaluation metrics.
        """
        device = self.device
        self.nn_model.to(device)
        self.nn_model.eval()
        supervised_loss_fn = nn.HuberLoss(reduction='none') 

        obj_val_total = []
        opt_obj_val_total = []
        slack_pen_total = []
        constraint_violation_total = []
        violation_count_total = []
        violation_percentage_total = []
        supervised_loss_total = []
        bit_accuracy_total = []

        with torch.no_grad():
            for theta, y_gt, pv_gt, u_gt in data_loader:
                theta = theta.to(device)
                y_gt = y_gt.to(device)
                pv_gt = pv_gt.to(device)
                u_gt = u_gt.to(device)
                B = theta.shape[0]

                y_pred, u_opt, p_opt, s_opt, obj_val = self.forward(theta)
                supervised_loss = supervised_loss_fn(y_pred, y_gt.float())
                supervised_loss = supervised_loss.view(B, -1).mean(dim=1)
                y_transposed = NNoutput_reshape_torch(y_pred, N_obs=3) # (B, 3, 4, H)
                slack_pen, y_penalty = constraint_violation_torch(y_transposed, p_opt[:, :2, :], evaluate=True)
                violation_count, violation_fraction = constraint_violation_count_torch(y_transposed, p_opt[:, :2, :])
                violation_percentage = violation_fraction * 100

                binary_pred = (y_pred >= 0.5).float()
                bit_accuracy = (binary_pred == y_gt.float()).float().view(B, -1).mean(dim=1)
                
                # Total loss with balanced weights
                # loss = combined_loss_fcn([obj_val, slack_pen, y_penalty, supervised_loss], weights)
                opt_obj_val = obj_function(u_gt, pv_gt[:, :2, :], theta, meta=None)

                obj_val_total.append(obj_val.detach().cpu())
                opt_obj_val_total.append(opt_obj_val.detach().cpu())
                slack_pen_total.append(slack_pen.detach().cpu())
                constraint_violation_total.append(y_penalty.detach().cpu())
                violation_count_total.append(violation_count.float().detach().cpu())
                violation_percentage_total.append(violation_percentage.detach().cpu())
                supervised_loss_total.append(supervised_loss.detach().cpu())
                bit_accuracy_total.append(bit_accuracy.detach().cpu())

        if not obj_val_total:
            raise ValueError("No evaluation data provided.")

        obj_vals = torch.cat(obj_val_total)
        opt_obj_vals = torch.cat(opt_obj_val_total)
        slack_penalties = torch.cat(slack_pen_total)
        y_penalties = torch.cat(constraint_violation_total)
        violation_counts = torch.cat(violation_count_total)
        violation_percentages = torch.cat(violation_percentage_total)
        supervised_losses = torch.cat(supervised_loss_total)
        bit_accuracies = torch.cat(bit_accuracy_total)
        # Added a constant to make the opt gap not crazy
        optimality_gap = 100 * (obj_vals - opt_obj_vals) / (1e2 + opt_obj_vals.abs())

        results = {
            "obj_val": obj_vals,
            "opt_obj_val": opt_obj_vals,
            "slack_penalty": slack_penalties,
            "y_penalty": y_penalties,
            "constraint_violation_magnitude": y_penalties,
            "constraint_violation_count": violation_counts,
            "constraint_violation_percentage": violation_percentages,
            "supervised_loss": supervised_losses,
            "bit_accuracy": bit_accuracies,
            "optimality_gap": optimality_gap,
        }

        print("=== Evaluation Summary ===")
        for key, value in results.items():
            if hasattr(value, "mean"):  # e.g. tensor or array
                print(f"{key:20s}: mean = {value.mean():.4f}, std = {value.std():.4f}")
            else:
                print(f"{key:20s}: {value}")


        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            torch.save(results, save_path)
            print(f"Saved evaluation results to {save_path}")

        return results

    def forward(self, theta):
        """
        Forward pass through the neural network model and CVXPY layer.
        Args:
            theta: Input tensor of problem parameters.
        Returns:
            y_pred: Predicted integer variables from the neural network.
            p_opt: Optimal positions from the CVXPY layer.
            u_opt: Optimal control inputs from the CVXPY layer.
            s_opt: Slack variables from the CVXPY layer.
            obj_val: Objective value computed using the optimal solutions.
        """
        device = self.device

        theta = theta.to(device)
        y_pred = self.nn_model(theta).float()

        theta = theta.cpu()
        y_pred = y_pred.cpu()

        u_opt, p_opt, _, s_opt = self.cvx_layer( theta[:, 0:2], 
                                             theta[:, 2:4], 
                                             theta[:, 4:6], 
                                             y_pred.reshape(-1, 3, 1, 20, 4))

        theta = theta.to(device)
        y_pred = y_pred.to(device)
        u_opt = u_opt.to(device)
        p_opt = p_opt.to(device)

        obj_val = obj_function(u_opt, p_opt, theta.to(device), meta=None)
        # obj_val = obj_val + 1e4 * s_opt.sum(dim=1)  # include slack penalty per-sample

        return y_pred, u_opt, p_opt, s_opt, obj_val

