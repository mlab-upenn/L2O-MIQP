import numpy as np
import pickle, os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
import time
import sys
import os

sys.path.append(os.path.abspath("..")) # Adds the parent folder to sys.path
from utils import *
from model import *
from Neural_Nets import *

def main():
    relative_path = os.getcwd()
    relative_path = os.path.abspath("..")
    dataset_fn = relative_path + '/data' + '/single.p'
    prob_features = ['x0', 'xg']

    data_file = open(dataset_fn,'rb')
    all_data = pickle.load(data_file)
    data_file.close()
    num_train = len(all_data)
    print(f"Number of training samples: {num_train}")

    X0 = np.vstack([all_data[ii]['x0'].T for ii in range(num_train)])  
    XG = np.vstack([all_data[ii]['xg'].T for ii in range(num_train)])  
    OBS = np.vstack([all_data[ii]['xg'].T for ii in range(num_train)])  
    XX = np.array([all_data[ii]['XX'] for ii in range(num_train)])
    UU = np.array([all_data[ii]['UU'] for ii in range(num_train)])
    YY = np.concatenate([all_data[ii]['YY'].astype(int) for ii in range(num_train)], axis=1).transpose(1,0,2)
    train_data = [{'x0': X0, 'xg': XG}, {'XX': XX, 'UU' : UU}, YY]

    # Build the FFNet model
    FFNet_reg = Regression(prob_features)
    n_features = 6 # the dimension of feature (input vector)
    FFNet_reg.setup_data(n_features, train_data)
    FFNet_reg.setup_network(depth=4, neurons=1024)
    print(FFNet_reg.model)

    # Phase 1: Regression with Supervised Learning
    print("\033[1;31;42m>>> Phase 1: Regression with Supervised Learning <<<\033[0m")
    training_params = {}
    training_params['TRAINING_EPOCHS'] = int(1000)
    training_params['BATCH_SIZE'] = 200
    training_params['CHECKPOINT_AFTER'] = int(1)
    training_params['SAVEPOINT_AFTER'] = int(10)
    training_params['TEST_BATCH_SIZE'] = 100
    training_params['LEARNING_RATE'] = 1e-3
    training_params['WEIGHT_DECAY'] = 1e-4
    training_params['EARLY_STOPPING_PATIENCE'] = 3

    FFNet_reg.train(training_params, verbose=True)

    # Phase 2: Refine with Self-Supervised Learning
    print("\033[1;31;42m>>> Phase 2: Refine with Self-Supervised Learning <<<\033[0m")
    training_params = {}
    training_params['TRAINING_EPOCHS'] = int(1000)
    training_params['BATCH_SIZE'] = 200
    training_params['CHECKPOINT_AFTER'] = int(1)
    training_params['SAVEPOINT_AFTER'] = int(10)
    training_params['TEST_BATCH_SIZE'] = 100
    training_params['LEARNING_RATE'] = 1e-3
    training_params['WEIGHT_DECAY'] = 1e-4
    training_params['EARLY_STOPPING_PATIENCE'] = 3

    FFNet_reg.SS_train(training_params, verbose=True, penalty_weight=1e0)

if __name__ == "__main__":
    main()    