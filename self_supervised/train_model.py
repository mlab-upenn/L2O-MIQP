import numpy as np
import pickle as pk
import argparse
import sys
import os
sys.path.append(os.path.abspath("..")) # Adds the parent folder to sys.path

from Neural_Nets import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=int, default=1)
    parser.add_argument("--n_iter", type=float, default=1000)
    input_parser = parser.parse_args()
    which_model = input_parser.model

    relative_path = os.getcwd()
    relative_path = os.path.abspath("..")
    dataset_fn = relative_path + '/data' + '/comb_single.p'
    prob_features = ['x0', 'xg']

    data_file = open(dataset_fn,'rb')
    all_data = pickle.load(data_file)
    data_file.close()
    data_list = []
    num_train = len(all_data)

    X0 = np.vstack([all_data[ii]['x0'].T for ii in range(num_train)])  
    XG = np.vstack([all_data[ii]['xg'].T for ii in range(num_train)])  
    OBS = np.vstack([all_data[ii]['xg'].T for ii in range(num_train)])  
    YY = np.concatenate([all_data[ii]['YY'].astype(int) for ii in range(num_train)], axis=1).transpose(1,0,2)

    train_data = [{'x0': X0, 'xg': XG}, None, None, YY, None]

    training_params = {}
    training_params['TRAINING_EPOCHS'] = int(input_parser.n_iter)
    training_params['BATCH_SIZE'] = 128
    training_params['CHECKPOINT_AFTER'] = int(1e3)
    training_params['SAVEPOINT_AFTER'] = int(1e4)
    training_params['TEST_BATCH_SIZE'] = 512
    training_params['LEARNING_RATE'] = 1e-3
    training_params['WEIGHT_DECAY'] = 1e-5

    if which_model == 1:
        # Build the MLOPT classifier model (in CoCo)
        print("Train the MLOPT classifier model")
        mlopt = CoCo(prob_features)
        n_features = 6 # the dimension of feature (input vector)
        mlopt.construct_strategies(n_features, train_data)
        mlopt.setup_network(depth=3, neurons=256)
        print(mlopt.model)
        mlopt.train(training_params, verbose=True)
    elif which_model == 2:
        # Build the LSTM (in PRISM) regression model
        print("Train the LSTM classifier model")
        lstm = PRISM(prob_features)
        n_features = 6 # the dimension of feature (input vector)
        lstm.construct_strategies(n_features, train_data)
        ff_neurons=256; lstm_neurons=128; lstm_lay = 3
        lstm.setup_network(ff_neurons, lstm_neurons, lstm_lay)
        print(lstm.model)
        lstm.train(training_params, verbose=True) 
            
    # elif which_model == 2:
    #     # Build the FFNet regression model
    #     print("Train the FFNet regression model")
    #     FFNet_reg = Regression(prob_features)
    #     n_features = 8 # the dimension of feature (input vector)
    #     FFNet_reg.construct_strategies(n_features, train_data)
    #     FFNet_reg.setup_network()
    #     FFNet_reg.model
    #     FFNet_reg.train(training_params, verbose=True)
    else:
        print("UNKNOWN MODEL !!!")

if __name__ == "__main__":
    main()    