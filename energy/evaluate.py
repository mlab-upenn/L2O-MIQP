import argparse
import numpy as np
import torch 
import os
from pathlib import Path
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

    batch_size = 128
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def resolve_paths(model_arg, stats_override=None, default_dir="checkpoints"):
    """
    Resolve model (.pth) and stats (.pt) paths based on user-provided arguments.
    Paths without a directory are placed inside `default_dir`, and stats files
    get a `stats_` prefix unless explicitly overridden.
    """
    model_path = Path(model_arg)
    if model_path.suffix != ".pth":
        model_path = model_path.with_suffix(".pth")
    if model_path.parent == Path("."):
        model_path = Path(default_dir) / model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if stats_override:
        stats_path = Path(stats_override)
        if stats_path.suffix != ".pt":
            stats_path = stats_path.with_suffix(".pt")
    else:
        stats_name = f"stats_{model_path.stem}"
        stats_path = model_path.parent / f"{stats_name}.pt"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    return str(model_path), str(stats_path)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained energy-system model.")
    parser.add_argument(
        "--model", required=True,
        help="Path to the saved model checkpoint (.pth or base name)."
    )
    parser.add_argument(
        "--stats_out", default=None,
        help="Optional path for evaluation stats (.pt). Defaults to stats_<model>.pt."
    )
    args = parser.parse_args()

    model_path, stats_path = resolve_paths(args.model, args.stats_out)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        device=device)

    Trainer_SSL.nn_model.load_state_dict(torch.load(model_path))
    Trainer_SSL.nn_model.to(device)
    Trainer_SSL.evaluate(test_loader, save_path = stats_path)

    print("\033[31;42m FINISHED \033[0m")

if __name__ == "__main__":
    main()
