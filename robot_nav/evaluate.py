import argparse
import torch
import numpy as np
import pickle
import os
from pathlib import Path

from src.neural_net import *
from src.cfg_utils import *
from src.trainer import MIQP_Trainer
from robot_nav.miqp import MIQP

def prepare_data(seed=42):
    relative_path = os.getcwd()
    relative_path = os.path.abspath("..")
    dataset_fn = relative_path + '/robot_nav/data' + '/single.p'
    prob_features = ['x0', 'xg']

    data_file = open(dataset_fn,'rb')
    all_data = pickle.load(data_file) # [:100000] use only part of the dataset for quick testing
    data_file.close()
    num_train = len(all_data)
    # print(f"Number of training samples: {num_train}")

    X0 = np.vstack([all_data[ii]['x0'].T for ii in range(num_train)])  
    XG = np.vstack([all_data[ii]['xg'].T for ii in range(num_train)])  
    OBS = np.vstack([all_data[ii]['xg'].T for ii in range(num_train)])  
    XX = np.array([all_data[ii]['XX'] for ii in range(num_train)])
    UU = np.array([all_data[ii]['UU'] for ii in range(num_train)])
    YY = np.concatenate([all_data[ii]['YY'].astype(int) for ii in range(num_train)], axis=1).transpose(1,0,2)
    train_data = [{'x0': X0, 'xg': XG}, {'XX': XX, 'UU' : UU}, YY]

    # Obs info
    Obs_info = np.array([[1.0,  0.0, 0.4, 0.5, 0.0],
                        [0.7, -1.1, 0.5, 0.4, 0.0],
                        [0.4, -2.5, 0.4, 0.5, 0.0]])
    n_obs = 3 

    # Dataset
    n_features = 6

    X_train = train_data[0]  # Problem parameters, will be inputs of the NNs
    Y_train = train_data[2]  # Discrete solutions, will be outputs of the NNs
    P_train = train_data[1]  # Continuous trajectories, will be used as parameters in training
    num_train = Y_train.shape[0]
    y_shape = Y_train.shape[1:]
    n_y = int(np.prod(y_shape))

    feature_blocks = []
    for feature in prob_features:
        if feature == "obstacles_map":
            continue
        values = X_train.get(feature)
        if values is None:
            print('Feature {} is unknown or missing'.format(feature))
            continue
        values = np.asarray(values)
        if values.shape[0] != num_train:
            raise ValueError(
                f"Feature '{feature}' has {values.shape[0]} samples, expected {num_train}"
            )
        feature_blocks.append(values.reshape(num_train, -1))
    if feature_blocks:
        features = np.concatenate(feature_blocks, axis=1)
    else:
        features = np.zeros((num_train, 0))
    if features.shape[1] != n_features:
        n_features = features.shape[1]
    labels = Y_train.reshape(num_train, n_y)
    labels_int = labels.astype(np.int64, copy=False)
    bit_shifts = np.arange(4 - 1, -1, -1, dtype=np.int64)
    outputs_bits = (labels_int[..., None] >> bit_shifts) & 1
    outputs = outputs_bits.reshape(num_train, -1)

    X_arr = features
    Y_arr = outputs
    P_arr = P_train['XX'][:, :, :]
    Pu_arr = P_train['UU'][:, :, :]
    # P_arr = np.concatenate([P_train['XX'][:, :, 1:], P_train['UU']], axis=1)

    X_tensor = torch.from_numpy(X_arr).float()
    Y_tensor = torch.from_numpy(Y_arr).float()
    P_tensor = torch.from_numpy(P_arr).float()
    Pu_tensor = torch.from_numpy(Pu_arr).float()

    # DataLoader
    batch_size = 128
    dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor, P_tensor, Pu_tensor)

    from torch.utils.data import random_split
    torch.manual_seed(seed)

    n_train = int(0.9 * len(dataset))
    n_test = len(dataset) - n_train
    train_set, test_set = random_split(dataset, [n_train, n_test])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, n_features, n_y, n_obs, Obs_info


def resolve_paths(model_arg, stats_override=None, default_dir="checkpoints"):
    """
    Resolve model (.pth) and stats (.pt) paths based on the provided arguments.
    Paths without directories are placed under `default_dir`, and stats files
    automatically get a `stats_` prefix unless overridden.
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
    parser = argparse.ArgumentParser(description="Evaluate a trained robot-navigation model.")
    parser.add_argument(
        "--model", required=True,
        help="Path to the saved model checkpoint (.pth or base name)."
    )
    parser.add_argument(
        "--stats_out", default=None,
        help="Optional output path for evaluation stats (.pt). Defaults to stats_<model>.pt."
    )
    args = parser.parse_args()

    _, test_loader, _, _, _, _ = prepare_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_input = 6
    n_output = 240

    nn_model_1 = MLPWithSTE(insize=n_input, outsize=n_output,
                    bias=True,
                    linear_map=torch.nn.Linear,
                    nonlin=nn.ReLU,
                    hsizes=[128] * 4)

    robot_miqp = MIQP(nn_model=nn_model_1, device=device)
    Model_SSL = MIQP_Trainer(robot_miqp)
    model_path, stats_path = resolve_paths(args.model, args.stats_out)
    Model_SSL.nn_model.load_state_dict(torch.load(model_path))
    Model_SSL.evaluate(test_loader, save_path=stats_path)

    print("\033[31;42m FINISHED \033[0m")


if __name__ == "__main__":
    main()
