import torch
import numpy as np
import pickle
import os
from src.neural_net import *
from src.cfg_utils import *
from trainer import *
from cvxpy_mpc_layer import *
from Robots import obstacle

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

def make_cplayer(weights=None, M=None, H=None, bounds=None, T=None, obstacles=None, coupling_pairs=None):
    T = 0.25 if T is None else T
    H = 20 if H is None else H
    M = 1 if M is None else M
    bounds = {
        "x_max": 2.00,
        "x_min": -0.5,
        "y_max": 0.5,
        "y_min": -3.0,
        "v_max": 0.50,
        "v_min": -0.50,
        "u_max": 0.50,
        "u_min": -0.50,
    } if bounds is None else bounds
    weights = (1.0, 1.0, 10.0) if weights is None else weights  # (Wu, Wp, Wpt)
    d_min = 0.25

    # Obstacles exactly as in the simulator
    
    obstacles = [
        obstacle(1.0, 0.0, 0.4, 0.5, 0.0),
        obstacle(0.7, -1.1, 0.5, 0.4, 0.0),
        obstacle(0.40, -2.50, 0.4, 0.5, 0.0),
    ]

    M = 1
    p = np.zeros((2, M))  # stack of robot positions; replace with actual state
    d_prox = 2.0
    coupling_pairs = [
        (m, n)
        for m in range(M)
        for n in range(m + 1, M)
        if np.linalg.norm(p[:, m] - p[:, n]) <= d_prox
    ]

    cplayer, meta = build_mpc_cvxpy_layer(
            T=.25,
            H=20,
            M=1,
            bounds=bounds,
            weights=weights,
            d_min=d_min,
            obstacles=obstacles,
            coupling_pairs=coupling_pairs
        )

    return cplayer.to(torch.device("cpu")), meta

def main():
    # ---------------------------------------------- #
    # Either of the following ways to load config
    # ---------------------------------------------- #
    # This one is useful for running the script once
    # filename, loss_weights, training_params = load_yaml_config("train_config.yaml")
    # This one is useful for running the script multiple times sequentially using .sh
    filenames, loss_weights, training_params = load_argparse_config()

    train_loader, test_loader, n_features, n_y, n_obs, Obs_info = prepare_data()

    cplayer, meta = make_cplayer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_input = 6
    n_output = 240

    nn_model_1 = MLPWithSTE(insize=n_input, outsize=n_output,
                    bias=True,
                    linear_map=torch.nn.Linear,
                    nonlin=nn.ReLU,
                    hsizes=[128] * 4)

    Model_SSL = SSL_MIQP_incorporated(nn_model_1, cplayer, 6, 4, device=device)
    # Load model from file
    Model_SSL.nn_model.load_state_dict(torch.load(filenames[1]))    
    Model_SSL.evaluate(test_loader, save_path = filenames[0])

    print("\033[31;42m FINISHED \033[0m")


if __name__ == "__main__":
    main()