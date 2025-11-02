
import torch
import torch.nn as nn
import wandb

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

def extract_cvx_outputs(cvx_layer, outputs):
    """
    Normalise CVX layer outputs into a dictionary keyed by variable name.
    Falls back to reasonable defaults when metadata is unavailable.
    """
    if isinstance(outputs, dict):
        return outputs

    if not isinstance(outputs, (tuple, list)):
        outputs = (outputs,)

    mapping = {}
    output_order = None

    if hasattr(cvx_layer, "meta"):
        output_order = cvx_layer.meta.get("output_order")
    if output_order is None and hasattr(cvx_layer, "output_order"):
        output_order = getattr(cvx_layer, "output_order")

    if output_order is not None and len(output_order) == len(outputs):
        mapping.update(dict(zip(output_order, outputs)))
    else:
        default_orders = {
            2: ["x_opt", "s_opt"],
            4: ["U", "P", "V", "S_obs"],
            5: ["U", "P", "V", "S_obs", "obj_value"],
            6: ["U", "P", "V", "S_obs", "obj_value", "S_pairs"],
        }
        names = default_orders.get(len(outputs))
        if names is not None:
            mapping.update(dict(zip(names, outputs)))

    # Ensure common aliases are populated
    if "P" in mapping and "x_opt" not in mapping:
        mapping["x_opt"] = mapping["P"]
    if "S_obs" in mapping and "s_opt" not in mapping:
        mapping["s_opt"] = mapping["S_obs"]
    if "obj_value" not in mapping and len(outputs) >= 5:
        mapping["obj_value"] = outputs[4]
    if "S_obs" not in mapping and len(outputs) >= 4:
        mapping["S_obs"] = outputs[3]
        mapping.setdefault("s_opt", outputs[3])
    if "x_opt" not in mapping and len(outputs) >= 1:
        mapping["x_opt"] = outputs[0]
    if "s_opt" not in mapping and len(outputs) >= 2:
        mapping["s_opt"] = outputs[1]

    return mapping

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
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def train_SL(self, ground_truth_solver, train_loader, test_loader, training_params, wandb_log = False):
        """
        Train the neural network model using supervised learning.
        Args:
            ground_truth_solver: Function to get ground truth solutions.
            train_loader: DataLoader for training data.
            test_loader: DataLoader for testing data.
            training_params: Dictionary containing training parameters.
            wandb_log: Boolean indicating whether to log to Weights & Biases.
        """
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
            for batch in train_loader:
                theta = batch.to(device)
                B = theta.shape[0] # batch size
                # ---- Predict y from theta ----
                y_pred = self.nn_model(theta).float() # (B, ny), hard {0,1}
                # May need it to include a supervised loss function 
                x_solver, y_solver = ground_truth_solver(theta)
                supervised_loss = supervised_loss_fn(y_pred, y_solver.float())
                loss = supervised_loss
                if wandb_log: wandb.log({
                    "train/loss": loss.item(),
                    "step": global_step})
                # ---- Backprop ----
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1e1)
                optimizer.step()
                global_step += 1
                
                # ---- Logging ----
                if global_step == 1 or (global_step % CHECKPOINT_AFTER) == 0:
                    training_loss = loss.item()
                    val_loss_total = 0.0
                    with torch.no_grad():
                        for val_batch in test_loader:
                            theta = val_batch.to(device)
                            # ---- Predict y from theta ----
                            y_pred_test = self.nn_model(theta).float() # (B, ny), hard {0,1}
                            _, y_solver_test = ground_truth_solver(theta)
                            val_loss_total += supervised_loss_fn(y_pred_test, y_solver_test).item()                           
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
                    scheduler.step(avg_val_loss)
                    current_lr = optimizer.param_groups[0]['lr']
                    if current_lr != last_lr:
                        print(f"Learning rate updated: {last_lr:.6f} -> {current_lr:.6f}")
                        last_lr = current_lr

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= PATIENCE:
                            print("Early stopping triggered!")
                            return
        if wandb_log: wandb.finish()
                    
    def train_SSL(self, ground_truth_solver, train_loader, test_loader, training_params, loss_weights,
            loss_scale = 1.0, obj_fcn = None, y_cons = None,
            slack_penalty = l1_penalty, constraint_penalty = torch.relu, 
            wandb_log = False):
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
        TRAINING_EPOCHS = training_params['TRAINING_EPOCHS']
        CHECKPOINT_AFTER = training_params['CHECKPOINT_AFTER']
        LEARNING_RATE = training_params['LEARNING_RATE']
        WEIGHT_DECAY = training_params['WEIGHT_DECAY']
        PATIENCE = training_params['PATIENCE']

        # Put all layers in device
        device = self.device
        self.nn_model.to(device)
        self.cvx_layer.cvxpylayer.to(device)

        if obj_fcn is None:
            def obj_fcn(_, __, theta):
                # default objective is zero; ensure correct device/dtype
                return theta.new_zeros(theta.shape[0])

        if y_cons is None:
            y_cons = []

        # Define weights for loss components
        weights = torch.tensor(loss_weights, device=device)
        weights = loss_scale*weights / weights.sum() # may scale up the weights a bit
        global_step = 0
        optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=int(PATIENCE/2))
        supervised_loss_fn = nn.HuberLoss() 
        best_val_loss = float("inf") # Store best validation 
        epochs_no_improve = 0  # Count epochs with no improvement 

        if wandb_log: wandb.init(
            project=training_params.get("WANDB_PROJECT", "self_supervised_learning"),
            name=training_params.get("RUN_NAME", None),
            config=training_params
        )     
               
        for epoch in range(1, TRAINING_EPOCHS+1):
            self.nn_model.train()
            for batch in train_loader:
                theta = batch.to(device)
                B = theta.shape[0] # batch size
                # ---- Predict y from theta ----
                y_pred = self.nn_model(theta).float() # (B, ny), hard {0,1}
                # ---- Solve convex subproblem given y ----
                # CVXPYLayer supports autograd; keep inputs requiring grad if needed
                cvx_outputs = self.cvx_layer.solve(theta, y_pred)
                cvx_dict = extract_cvx_outputs(self.cvx_layer, cvx_outputs)
                # x_opt = cvx_dict.get("x_opt")
                s_opt = cvx_dict.get("s_opt")
                # May need it to include a supervised loss function 
                x_solver, y_solver = ground_truth_solver(theta)
                supervised_loss = supervised_loss_fn(y_pred, y_solver.float())
                obj_value_tensor = cvx_dict.get("obj_value")
                if obj_value_tensor is None:
                    raise ValueError("CVX layer output missing objective value required for loss computation.")
                obj_val = obj_value_tensor.mean()
                # Slack penalty, for constraint violation of the ones involving continuous decision variables
                if s_opt is None:
                    raise ValueError("CVX layer output missing slack variables required for penalty computation.")
                slack_pen = slack_penalty(s_opt).mean()
                # Violation penalty for constraint violation only on the integer decision variables
                if y_cons:
                    y_constraint = torch.stack([f(y_pred, theta) for f in y_cons]) # shape: (num_constraints, batch_size)
                    y_penalty = constraint_penalty(y_constraint).mean()
                else:
                    y_penalty = theta.new_tensor(0.0)
                # Total loss with balanced weights
                loss = combined_loss_fcn([obj_val, slack_pen, y_penalty, supervised_loss], weights)
                if wandb_log: wandb.log({
                    "train/combined_loss": loss.item(),
                    "train/obj_val": obj_val.item(),
                    "train/slack_pen": slack_pen.item(),
                    "train/y_penalty": y_penalty.item(),
                    "train/supervised_loss": supervised_loss.item(),
                    "step": global_step})

                # ---- Backprop ----
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1e1)
                optimizer.step()         
                global_step += 1

                # ---- Logging ----
                if global_step == 1 or (global_step % CHECKPOINT_AFTER) == 0:
                    val_loss_total = []; obj_val_total = []; opt_obj_val_total = []
                    slack_pen_total = []; y_penalty_total = []
                    supervised_loss_total = []
                    with torch.no_grad():
                        for val_batch in test_loader:
                            theta = val_batch.to(device)
                            # ---- Predict y from theta ----
                            y_pred = self.nn_model(theta).float() # (B, ny), hard {0,1}
                            # ---- Solve convex subproblem given y ----
                            cvx_outputs = self.cvx_layer.solve(theta, y_pred)
                            cvx_dict = extract_cvx_outputs(self.cvx_layer, cvx_outputs)
                            # x_opt = cvx_dict.get("x_opt")
                            s_opt = cvx_dict.get("s_opt")
                            obj_value_tensor = cvx_dict.get("obj_value")
                            # May need it to include a supervised loss function 
                            x_solver, y_solver = ground_truth_solver(theta)
                            supervised_loss = supervised_loss_fn(y_pred, y_solver.float())
                            if obj_value_tensor is None:
                                raise ValueError("CVX layer output missing objective value required for loss computation.")
                            obj_val = obj_value_tensor.mean()
                            opt_obj_val = obj_fcn(x_solver, y_solver, theta).mean()
                            # Slack penalty, for constraint violation of the ones involving continuous decision variables
                            if s_opt is None:
                                raise ValueError("CVX layer output missing slack variables required for penalty computation.")
                            slack_pen = slack_penalty(s_opt).mean()
                            # Violation penalty for constraint violation only on the integer decision variables
                            if y_cons:
                                y_constraint = torch.stack([f(y_pred, theta) for f in y_cons]) # shape: (num_constraints, batch_size, ...)
                                y_penalty = constraint_penalty(y_constraint).mean()
                            else:
                                y_penalty = theta.new_tensor(0.0)
                            # Total loss with balanced weights
                            loss = combined_loss_fcn([obj_val, slack_pen, y_penalty, supervised_loss], weights)

                            # Collect loss values
                            val_loss_total.append(loss.item())       
                            obj_val_total.append(obj_val.item())
                            opt_obj_val_total.append(opt_obj_val.item())
                            slack_pen_total.append(slack_pen.item())
                            y_penalty_total.append(y_penalty.item())
                            supervised_loss_total.append(supervised_loss.item())

                        # Compute the averages
                        val_loss_tensor = torch.tensor(val_loss_total)
                        obj_val_tensor = torch.tensor(obj_val_total)
                        opt_obj_val_tensor = torch.tensor(opt_obj_val_total)
                        slack_pen_tensor = torch.tensor(slack_pen_total)
                        y_penalty_tensor = torch.tensor(y_penalty_total)
                        supervised_loss_tensor = torch.tensor(supervised_loss_total)

                        avg_val_loss = val_loss_tensor.mean().item()
                        avg_obj_val = obj_val_tensor.mean().item()
                        denom = opt_obj_val_tensor.abs().clamp_min(1e-6)
                        opt_gap = 100 * (obj_val_tensor - opt_obj_val_tensor) / denom
                        avg_opt_gap = opt_gap.mean().item()
                        avg_slack_pen = slack_pen_tensor.mean().item()
                        avg_y_penalty = y_penalty_tensor.mean().item()
                        avg_supervised_loss = supervised_loss_tensor.mean().item()

                    print(f"[epoch {epoch} | step {global_step}] "
                        f"validation: loss = {avg_val_loss:.4f}, "
                        f"obj_val = {avg_obj_val:.4f}, "
                        f"opt_gap = {avg_opt_gap:.4f} %, "
                        f"slack_pen = {avg_slack_pen:.4f}, "
                        f"y_penalty = {avg_y_penalty:.4f}, "
                        f"supervised_loss = {avg_supervised_loss:.4f}, ")
                    # --- Log losses to wandb ---
                    if wandb_log: wandb.log({
                        "val/avg_loss": avg_val_loss,
                        "val/obj_val": avg_obj_val,
                        "val/slack_pen": avg_slack_pen,
                        "val/y_penalty": avg_y_penalty,
                        "val/supervised_loss": avg_supervised_loss,
                        "epoch": epoch})       

                    # Check if need to update the learning rates
                    last_lr = optimizer.param_groups[0]['lr']
                    scheduler.step(avg_val_loss)
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

        if wandb_log: wandb.finish()                        


class SSL_MIQP_corrected:

    def __init__(self, nn_model, sl_model, cvx_layer, nx, ny, device=None):
        """
        Initialize the SSL_MIQP model.
        Args:
            nn_model: The neural network model for correction.
            sl_model: The pre-trained neural network.
            cvx_layer: The CVXPY layer for optimization.
            nx: Number of input features.
            ny: Number of output features.
            device: The device (CPU or GPU).    
        """
        self.nn_model = nn_model
        self.sl_model = sl_model
        self.cvx_layer = cvx_layer
        self.nx, self.ny = nx, ny
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
    def train_SL(self, ground_truth_solver, train_loader, test_loader, training_params, wandb_log = False):
        """
        Train the neural network model using supervised learning.
        Args:
            ground_truth_solver: Function to get ground truth solutions.
            train_loader: DataLoader for training data.
            test_loader: DataLoader for testing data.
            training_params: Dictionary containing training parameters.
            wandb_log: Boolean indicating whether to log to Weights & Biases.
        """
        TRAINING_EPOCHS = training_params['TRAINING_EPOCHS']
        CHECKPOINT_AFTER = training_params['CHECKPOINT_AFTER']
        LEARNING_RATE = training_params['LEARNING_RATE']
        WEIGHT_DECAY = training_params['WEIGHT_DECAY']
        PATIENCE = training_params['PATIENCE']
        # Put all layers in device
        device = self.device
        self.sl_model.to(device)

        # Initialize training components
        global_step = 0
        optimizer = torch.optim.Adam(self.sl_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
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
            self.sl_model.train()
            for batch in train_loader:
                theta = batch.to(device)
                B = theta.shape[0] # batch size
                # ---- Predict y from theta ----
                y_pred = self.sl_model(theta).float() # (B, ny), hard {0,1}
                # May need it to include a supervised loss function 
                x_solver, y_solver = ground_truth_solver(theta)
                supervised_loss = supervised_loss_fn(y_pred, y_solver.float())
                loss = supervised_loss
                if wandb_log: wandb.log({
                    "train/loss": loss.item(),
                    "step": global_step})
                # ---- Backprop ----
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1e1)
                optimizer.step()
                global_step += 1
                
                # ---- Logging ----
                if global_step == 1 or (global_step % CHECKPOINT_AFTER) == 0:
                    training_loss = loss.item()
                    val_loss_total = 0.0
                    with torch.no_grad():
                        for val_batch in test_loader:
                            theta = val_batch.to(device)
                            # ---- Predict y from theta ----
                            y_pred_test = self.sl_model(theta).float() # (B, ny), hard {0,1}
                            _, y_solver_test = ground_truth_solver(theta)
                            val_loss_total += supervised_loss_fn(y_pred_test, y_solver_test).item()                           
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
                    scheduler.step(avg_val_loss)
                    current_lr = optimizer.param_groups[0]['lr']
                    if current_lr != last_lr:
                        print(f"Learning rate updated: {last_lr:.6f} -> {current_lr:.6f}")
                        last_lr = current_lr

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= PATIENCE:
                            print("Early stopping triggered!")
                            return
        if wandb_log: wandb.finish()

    def train_SSL(self, ground_truth_solver, train_loader, test_loader, training_params, loss_weights,
            loss_scale = 1.0, obj_fcn = None, y_cons = None,
            slack_penalty = l1_penalty, constraint_penalty = torch.relu,
            wandb_log = False):
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
        TRAINING_EPOCHS = training_params['TRAINING_EPOCHS']
        CHECKPOINT_AFTER = training_params['CHECKPOINT_AFTER']
        LEARNING_RATE = training_params['LEARNING_RATE']
        WEIGHT_DECAY = training_params['WEIGHT_DECAY']
        PATIENCE = training_params['PATIENCE']

        # Put all layers in device
        device = self.device
        self.nn_model.to(device)
        self.sl_model.to(device)
        self.cvx_layer.cvxpylayer.to(device)
        self.sl_model.eval()

        if obj_fcn is None:
            def obj_fcn(_, __, theta):
                return theta.new_zeros(theta.shape[0])

        if y_cons is None:
            y_cons = []

        # Define weights for loss components
        weights = torch.tensor(loss_weights, device=device)
        weights = loss_scale*weights / weights.sum() # may scale up the weights a bit
        global_step = 0
        optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=int(PATIENCE/2))
        supervised_loss_fn = nn.HuberLoss() 
        best_val_loss = float("inf") # Store best validation 
        epochs_no_improve = 0  # Count epochs with no improvement 

        if wandb_log: wandb.init(
            project=training_params.get("WANDB_PROJECT", "self_supervised_learning"),
            name=training_params.get("RUN_NAME", None),
            config=training_params
        )

        # Validation for the supervised learning model
        print("Validation for the supervised learning model: ")
        val_loss_total = []; obj_val_total = []; opt_obj_val_total = []
        slack_pen_total = []; y_penalty_total = []; supervised_loss_total = []
        with torch.no_grad():
            for val_batch in test_loader:
                theta = val_batch.to(device)
                # ---- Predict y from theta ----
                y_pred = self.sl_model(theta).float() # (B, ny), hard {0,1}
                # ---- Solve convex subproblem given y ----
                # CVXPYLayer supports autograd; keep inputs requiring grad if needed
                cvx_outputs = self.cvx_layer.solve(theta, y_pred)
                cvx_dict = extract_cvx_outputs(self.cvx_layer, cvx_outputs)
                # x_opt = cvx_dict.get("x_opt")
                s_opt = cvx_dict.get("s_opt")
                obj_value_tensor = cvx_dict.get("obj_value")
                # May need it to include a supervised loss function 
                x_solver, y_solver = ground_truth_solver(theta)
                supervised_loss = supervised_loss_fn(y_pred, y_solver.float())
                if obj_value_tensor is None:
                    raise ValueError("CVX layer output missing objective value required for loss computation.")
                obj_val = obj_value_tensor.mean()
                opt_obj_val = obj_fcn(x_solver, y_solver, theta).mean()
                # Slack penalty, for constraint violation of the ones involving continuous decision variables
                if s_opt is None:
                    raise ValueError("CVX layer output missing slack variables required for penalty computation.")
                slack_pen = slack_penalty(s_opt).mean()
                # Violation penalty for constraint violation only on the integer decision variables
                if y_cons:
                    y_constraint = torch.stack([f(y_pred, theta) for f in y_cons]) # shape: (num_constraints, batch_size, ...)
                    y_penalty = constraint_penalty(y_constraint).mean()
                else:
                    y_penalty = theta.new_tensor(0.0)
                # Total loss with balanced weights
                loss = combined_loss_fcn([obj_val, slack_pen, y_penalty, supervised_loss], weights)

                # Collect loss values
                val_loss_total.append(loss.item())       
                obj_val_total.append(obj_val.item())
                opt_obj_val_total.append(opt_obj_val.item())
                slack_pen_total.append(slack_pen.item())
                y_penalty_total.append(y_penalty.item())
                supervised_loss_total.append(supervised_loss.item())

        # Compute the averages
        val_loss_tensor = torch.tensor(val_loss_total)
        obj_val_tensor = torch.tensor(obj_val_total)
        opt_obj_val_tensor = torch.tensor(opt_obj_val_total)
        slack_pen_tensor = torch.tensor(slack_pen_total)
        y_penalty_tensor = torch.tensor(y_penalty_total)
        supervised_loss_tensor = torch.tensor(supervised_loss_total)

        avg_val_loss = val_loss_tensor.mean().item()
        avg_obj_val = obj_val_tensor.mean().item()
        denom = opt_obj_val_tensor.abs().clamp_min(1e-6)
        opt_gap = 100 * (obj_val_tensor - opt_obj_val_tensor) / denom
        avg_opt_gap = opt_gap.mean().item()
        avg_slack_pen = slack_pen_tensor.mean().item()
        avg_y_penalty = y_penalty_tensor.mean().item()
        avg_supervised_loss = supervised_loss_tensor.mean().item()

        print(
            f"val_loss = {avg_val_loss:.4f}, "
            f"obj_val = {avg_obj_val:.4f}, "
            f"opt_gap = {avg_opt_gap:.4f} %, "
            f"slack_pen = {avg_slack_pen:.4f}, "
            f"y_penalty = {avg_y_penalty:.4f}, "
            f"supervised_loss = {avg_supervised_loss:.4f}, ")
        print("_"*50)                 
               
        for epoch in range(1, TRAINING_EPOCHS+1):
            self.nn_model.train()
            for batch in train_loader:
                theta = batch.to(device)
                B = theta.shape[0] # batch size
                # ---- Predict y from theta ----
                y_pred_hat = self.sl_model(theta).float() # (B, ny), hard {0,1}
                # construct the concatenated input
                concat_input = torch.cat([theta, y_pred_hat], dim=-1)
                y_pred = self.nn_model(concat_input).float() # (B, ny), hard {0,1}
                # ---- Solve convex subproblem given y ----
                # CVXPYLayer supports autograd; keep inputs requiring grad if needed
                cvx_outputs = self.cvx_layer.solve(theta, y_pred)
                cvx_dict = extract_cvx_outputs(self.cvx_layer, cvx_outputs)
                # x_opt = cvx_dict.get("x_opt")
                s_opt = cvx_dict.get("s_opt")
                obj_value_tensor = cvx_dict.get("obj_value")
                # May need it to include a supervised loss function 
                x_solver, y_solver = ground_truth_solver(theta)
                # supervised learning loss for deviation from SL model prediction
                supervised_loss = supervised_loss_fn(y_pred, y_pred_hat.float())
                if obj_value_tensor is None:
                    raise ValueError("CVX layer output missing objective value required for loss computation.")
                obj_val = obj_value_tensor.mean()
                # Slack penalty, for constraint violation of the ones involving continuous decision variables
                if s_opt is None:
                    raise ValueError("CVX layer output missing slack variables required for penalty computation.")
                slack_pen = slack_penalty(s_opt).mean()
                # Violation penalty for constraint violation only on the integer decision variables
                if y_cons:
                    y_constraint = torch.stack([f(y_pred, theta) for f in y_cons]) # shape: (num_constraints, batch_size)
                    y_penalty = constraint_penalty(y_constraint).mean()
                else:
                    y_penalty = theta.new_tensor(0.0)
                # Total loss with balanced weights
                loss = combined_loss_fcn([obj_val, slack_pen, y_penalty, supervised_loss], weights)
                if wandb_log: wandb.log({
                    "train/combined_loss": loss.item(),
                    "train/obj_val": obj_val.item(),
                    "train/slack_pen": slack_pen.item(),
                    "train/y_penalty": y_penalty.item(),
                    "train/supervised_loss": supervised_loss.item(),
                    "step": global_step})

                # ---- Backprop ----
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1e1)
                optimizer.step()         
                global_step += 1

                # ---- Logging ----
                if global_step == 1 or (global_step % CHECKPOINT_AFTER) == 0:
                    val_loss_total = []; obj_val_total = []; opt_obj_val_total = []
                    slack_pen_total = []; y_penalty_total = []
                    supervised_loss_total = []
                    with torch.no_grad():
                        for val_batch in test_loader:
                            theta = val_batch.to(device)
                            # ---- Predict y from theta ----
                            y_pred_hat = self.sl_model(theta).float() # (B, ny), hard {0,1}
                            # construct the concatenated input
                            concat_input = torch.cat([theta, y_pred_hat], dim=-1)
                            y_pred = self.nn_model(concat_input).float() # (B, ny), hard {0,1}
                            # ---- Solve convex subproblem given y ----
                            cvx_outputs = self.cvx_layer.solve(theta, y_pred)
                            cvx_dict = extract_cvx_outputs(self.cvx_layer, cvx_outputs)
                            # x_opt = cvx_dict.get("x_opt")
                            s_opt = cvx_dict.get("s_opt")
                            obj_value_tensor = cvx_dict.get("obj_value")
                            # May need it to include a supervised loss function 
                            x_solver, y_solver = ground_truth_solver(theta)
                            supervised_loss = supervised_loss_fn(y_pred, y_solver.float())
                            if obj_value_tensor is None:
                                raise ValueError("CVX layer output missing objective value required for loss computation.")
                            obj_val = obj_value_tensor.mean()
                            opt_obj_val = obj_fcn(x_solver, y_solver, theta).mean()
                            # Slack penalty, for constraint violation of the ones involving continuous decision variables
                            if s_opt is None:
                                raise ValueError("CVX layer output missing slack variables required for penalty computation.")
                            slack_pen = slack_penalty(s_opt).mean()
                            # Violation penalty for constraint violation only on the integer decision variables
                            if y_cons:
                                y_constraint = torch.stack([f(y_pred, theta) for f in y_cons]) # shape: (num_constraints, batch_size, ...)
                                y_penalty = constraint_penalty(y_constraint).mean()
                            else:
                                y_penalty = theta.new_tensor(0.0)
                            # Total loss with balanced weights
                            loss = combined_loss_fcn([obj_val, slack_pen, y_penalty, supervised_loss], weights)

                            # Collect loss values
                            val_loss_total.append(loss.item())       
                            obj_val_total.append(obj_val.item())
                            opt_obj_val_total.append(opt_obj_val.item())
                            slack_pen_total.append(slack_pen.item())
                            y_penalty_total.append(y_penalty.item())
                            supervised_loss_total.append(supervised_loss.item())

                        val_loss_tensor = torch.tensor(val_loss_total)
                        obj_val_tensor = torch.tensor(obj_val_total)
                        opt_obj_val_tensor = torch.tensor(opt_obj_val_total)
                        slack_pen_tensor = torch.tensor(slack_pen_total)
                        y_penalty_tensor = torch.tensor(y_penalty_total)
                        supervised_loss_tensor = torch.tensor(supervised_loss_total)

                        avg_val_loss = val_loss_tensor.mean().item()
                        avg_obj_val = obj_val_tensor.mean().item()
                        denom = opt_obj_val_tensor.abs().clamp_min(1e-6)
                        opt_gap = 100 * (obj_val_tensor - opt_obj_val_tensor) / denom
                        avg_opt_gap = opt_gap.mean().item()
                        avg_slack_pen = slack_pen_tensor.mean().item()
                        avg_y_penalty = y_penalty_tensor.mean().item()
                        avg_supervised_loss = supervised_loss_tensor.mean().item()

                    print(f"[epoch {epoch} | step {global_step}] "
                        f"validation: loss = {avg_val_loss:.4f}, "
                        f"obj_val = {avg_obj_val:.4f}, "
                        f"opt_gap = {avg_opt_gap:.4f} %, "
                        f"slack_pen = {avg_slack_pen:.4f}, "
                        f"y_penalty = {avg_y_penalty:.4f}, "
                        f"supervised_loss = {avg_supervised_loss:.4f}, ")
                    # --- Log losses to wandb ---
                    if wandb_log: wandb.log({
                        "val/avg_loss": avg_val_loss,
                        "val/obj_val": avg_obj_val,
                        "val/slack_pen": avg_slack_pen,
                        "val/y_penalty": avg_y_penalty,
                        "val/supervised_loss": avg_supervised_loss,
                        "epoch": epoch})       

                    # Check if need to update the learning rates
                    last_lr = optimizer.param_groups[0]['lr']
                    scheduler.step(avg_val_loss)
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

        if wandb_log: wandb.finish()                                    
