import numpy as np
import torch 
import datetime
import os
import yaml
from sklearn.model_selection import train_test_split

from src.neural_net import *
from cvxpy_dpc_layer import *
from trainer import *

def prepare_data():
    relative_path = os.getcwd()
    dataset_fn = relative_path + '/data' + '/data.npz'
    bundle = np.load(dataset_fn)
    X = bundle["X"]; Y = bundle["Y"]; U = bundle["U"]; Xtraj = bundle["Xtraj"]
    OBJ = bundle["OBJ"]; status = bundle["STATUS"]

    X_train, X_test, Y_train, Y_test, U_train, U_test, Xtraj_train, Xtraj_test, OBJ_train, OBJ_test = train_test_split(
        X, Y, U, Xtraj, OBJ, test_size=0.2, random_state=42
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)
    U_train_tensor = torch.tensor(U_train, dtype=torch.float32)
    U_test_tensor = torch.tensor(U_test, dtype=torch.float32)
    Xtraj_train_tensor = torch.tensor(Xtraj_train, dtype=torch.float32)
    Xtraj_test_tensor = torch.tensor(Xtraj_test, dtype=torch.float32)
    # OBJ_train_tensor = torch.tensor(OBJ_train, dtype=torch.float32)
    # OBJ_test_tensor = torch.tensor(OBJ_test, dtype=torch.float32)

    train_dataset = torch.utils.data.TensorDataset(
        X_train_tensor, Y_train_tensor, Xtraj_train_tensor, U_train_tensor
    )
    test_dataset = torch.utils.data.TensorDataset(
        X_test_tensor, Y_test_tensor, Xtraj_test_tensor, U_test_tensor
    )

    batch_size = 256
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader

def main():
    cfg = yaml.safe_load(open("train_config.yaml"))
    weights = cfg["weights"]
    obj_weight = float(weights["w_obj"])
    slack_weight = float(weights["w_slack"])
    constraint_weight = float(weights["w_con"])
    supervised_weight = float(weights["w_sup"])
    print(f"Using loss weights - Obj: {obj_weight}, Slack: {slack_weight}, Constraint: {constraint_weight}, Supervised: {supervised_weight}")
    loss_weights = [obj_weight, slack_weight, constraint_weight, supervised_weight]

    training_params = cfg["training"]
    training_params['TRAINING_EPOCHS'] = int(training_params['TRAINING_EPOCHS'])
    training_params['CHECKPOINT_AFTER'] = int(training_params['CHECKPOINT_AFTER'])
    training_params['LEARNING_RATE'] = float(training_params['LEARNING_RATE'])
    training_params['WEIGHT_DECAY'] =  float(training_params['WEIGHT_DECAY'])
    training_params['PATIENCE'] = int(training_params['TRAINING_EPOCHS'])
    now = datetime.datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    training_params['RUN_NAME'] = "ssl_" + dt_string

    train_loader, test_loader = prepare_data()
    cp_layer = build_dpc_cvxpy_layer(N = 20)
    nn_model = MLPWithSoftmaxSTE(
        insize=42,
        outsize=20,
        integer_choices=[0, 1, 2, 3],
        hsizes=[128] * 4
    )

    Trainer_SSL = SSL_MIQPP_Trainer(
        nn_model=nn_model,
        cvx_layer=cp_layer,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    Trainer_SSL.train_SSL(
        train_loader=train_loader,
        test_loader=test_loader,
        training_params=training_params,
        loss_weights=loss_weights,
        loss_scale=10.0,
        wandb_log=True
    )

    Trainer_SSL.evaluate(test_loader)

if __name__ == "__main__":
    main()