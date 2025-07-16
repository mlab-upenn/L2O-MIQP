import numpy as np
import random
import pickle, os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Sigmoid
from datetime import datetime
import time

from utils import *
from model import FFNet, LSTMNet

class Regression:

    def __init__(self, prob_features):
        """
        Constructor for Regression class.
        """
        self.prob_features = prob_features

        self.num_train, self.num_test = 0, 0
        self.model, self.model_fn = None, None

    def construct_features(self, params, ii_obs=None):
        prob_features = self.prob_features
        feature_vec = np.array([])

        x0, xg = params['x0'], params['xg'] 
        obstacles = params['obstacles']

        for feature in prob_features:
            if feature == "x0":
                feature_vec = np.hstack((feature_vec, x0))
            elif feature == "xg":
                feature_vec = np.hstack((feature_vec, xg))
            elif feature == "obstacles":
                feature_vec = np.hstack((feature_vec, np.reshape(obstacles, (4*self.n_obs))))
            elif feature == "obstacles_map":
                continue
            else:
                print('Feature {} is unknown'.format(feature))

        # Append one-hot encoding to end
        if ii_obs is not None:
            one_hot = np.zeros(self.n_obs)
            one_hot[ii_obs] = 1.
            feature_vec = np.hstack((feature_vec, one_hot))

        return feature_vec

    def construct_strategies(self, n_features, train_data):
        """
        Reads in data and constructs strategy dictionary
        """
        self.n_features = n_features

        self.X_train = train_data[0]
        self.Y_train = train_data[3]
        self.n_y = self.Y_train[0].size # will be the dimension of the output
        self.y_shape = self.Y_train[0].shape
        self.num_train = self.Y_train.shape[0]        

        self.features = np.zeros((self.num_train, self.n_features))
        self.labels = np.zeros((self.num_train, self.n_y))

        for ii in range(self.num_train):
            self.labels[ii] = np.reshape(self.Y_train[ii,:,:].T, (self.n_y))
            prob_params = {}
            for k in self.X_train:
                prob_params[k] = self.X_train[k][ii]
            self.features[ii] = self.construct_features(prob_params)

    def setup_network(self, depth=3, neurons=32, device_id=0):
        self.device = torch.device('cuda:{}'.format(device_id))
        
        ff_shape = [self.n_features]
        for ii in range(depth):
            ff_shape.append(neurons)
        ff_shape.append(self.n_y)

        self.model = FFNet(ff_shape, activation=torch.nn.ReLU()).to(device=self.device)

        # file names for PyTorch models
        now = datetime.now().strftime('%Y%m%d_%H%M')
        model_fn = 'regression_{}.pt'
        model_fn = os.path.join(os.getcwd(), model_fn)
        self.model_fn = model_fn.format(now)

    def load_network(self, fn_regressor_model):
        if os.path.exists(fn_regressor_model):
            print('Loading presaved regression model from {}'.format(fn_regressor_model))
            self.model.load_state_dict(torch.load(fn_regressor_model))
            self.model_fn = fn_regressor_model

    def train(self, training_params, verbose=True):
        # grab training params
        BATCH_SIZE = training_params['BATCH_SIZE']
        TRAINING_EPOCHS = training_params['TRAINING_EPOCHS']
        BATCH_SIZE = training_params['BATCH_SIZE']
        CHECKPOINT_AFTER = training_params['CHECKPOINT_AFTER']
        SAVEPOINT_AFTER = training_params['SAVEPOINT_AFTER']
        TEST_BATCH_SIZE = training_params['TEST_BATCH_SIZE']

        model = self.model
        X_train = self.features
        Y_train = self.labels

        # Define loss and optimizer
        training_loss = torch.nn.BCEWithLogitsLoss()
        opt = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)

        itr = 1
        for epoch in range(TRAINING_EPOCHS):  # loop over the dataset multiple times
            t0 = time.time()
            running_loss = 0.0
            rand_idx = list(np.arange(0,X_train.shape[0]-1))
            random.shuffle(rand_idx)

            # Sample all data points
            indices = [rand_idx[ii * BATCH_SIZE:(ii + 1) * BATCH_SIZE] for ii in range((len(rand_idx) + BATCH_SIZE - 1) // BATCH_SIZE)]

            for ii, idx in enumerate(indices):
                # zero the parameter gradients
                opt.zero_grad()

                inputs = Variable(torch.from_numpy(X_train[idx,:])).float().to(device=self.device)
                y_true = Variable(torch.from_numpy(Y_train[idx,:])).float().to(device=self.device)

                # forward + backward + optimize
                outputs = model(inputs)
                loss = training_loss(outputs, y_true).float().to(device=self.device)
                loss.backward()
                opt.step()

                # print statistics\n",
                running_loss += loss.item()
                if itr % CHECKPOINT_AFTER == 0:
                    rand_idx = list(np.arange(0,X_train.shape[0]-1))
                    random.shuffle(rand_idx)
                    test_inds = rand_idx[:TEST_BATCH_SIZE]
                    inputs = Variable(torch.from_numpy(X_train[test_inds,:])).float().to(device=self.device)
                    y_out = Variable(torch.from_numpy(Y_train[test_inds])).float().to(device=self.device)

                    # forward + backward + optimize
                    outputs = model(inputs)
                    loss = training_loss(outputs, y_out).float().to(device=self.device)
                    outputs = Sigmoid()(outputs).round()
                    accuracy = [float(all(torch.eq(outputs[ii],y_out[ii]))) for ii in range(TEST_BATCH_SIZE)]
                    accuracy = np.mean(accuracy)
                    verbose and print("loss:   " + str(loss.item()) + " , acc: " + str(accuracy))

                if itr % SAVEPOINT_AFTER == 0:
                    torch.save(model.state_dict(), self.model_fn)
                    verbose and print('Saved model at {}'.format(self.model_fn))
                    # writer.add_scalar('Loss/train', running_loss, epoch)

                itr += 1
            # verbose and print('Done with epoch {} in {}s'.format(epoch, time.time()-t0))

        # Save final model
        torch.save(model.state_dict(), self.model_fn)
        print('Saved model at {}'.format(self.model_fn))
        print('Done training')

"""
Class to reproduce the model in
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9653847
"""
class CoCo:
    
    def __init__(self, prob_features, n_evals=10):
        """
        Constructor for CoCo class.
        """
        self.prob_features = prob_features
        self.n_evals = n_evals

        self.num_train, self.num_test = 0, 0
        self.model, self.model_fn = None, None

    def construct_features(self, params, ii_obs=None):
        prob_features = self.prob_features
        feature_vec = np.array([])
        x0, xg = params['x0'], params['xg'] 

        for feature in prob_features:
            if feature == "x0":
                feature_vec = np.hstack((feature_vec, x0))
            elif feature == "xg":
                feature_vec = np.hstack((feature_vec, xg[:2]))
            elif feature == "obstacles":
                obstacles = params['obstacles']
                feature_vec = np.hstack((feature_vec, np.reshape(obstacles, (4*self.n_obs))))
            elif feature == "obstacles_map":
                continue
            else:
                print('Feature {} is unknown'.format(feature))

        # Append one-hot encoding to end
        if ii_obs is not None:
            one_hot = np.zeros(self.n_obs)
            one_hot[ii_obs] = 1.
            feature_vec = np.hstack((feature_vec, one_hot))

        return feature_vec
    
    def construct_strategies(self, n_features, train_data):
        """
        Reads in data and constructs strategy dictionary
        """
        self.n_features = n_features
        self.strategy_dict = {}

        self.X_train = train_data[0]
        self.Y_train = train_data[3]
        self.num_train = self.Y_train.shape[0]        

        self.n_y = self.Y_train[0].size # will be the dimension of the output
        self.y_shape = self.Y_train[0].shape
        self.features = np.zeros((self.num_train, self.n_features))
        self.labels = np.zeros((self.num_train, 1+self.n_y))
        self.n_strategies = 0

        for ii in range(self.num_train):
            y_true = np.reshape(self.Y_train[ii,:,:], (self.n_y))

            if tuple(y_true) not in self.strategy_dict.keys():
                self.strategy_dict[tuple(y_true)] = np.hstack((self.n_strategies,np.copy(y_true)))
                self.n_strategies += 1
            self.labels[ii] = self.strategy_dict[tuple(y_true)]

            prob_params = {}
            for k in self.X_train:
                prob_params[k] = self.X_train[k][ii]

            self.features[ii] = self.construct_features(prob_params)

    def setup_network(self, depth=3, neurons=32, device_id=0):
        self.device = torch.device('cuda:{}'.format(device_id))
        
        ff_shape = [self.n_features]
        for ii in range(depth):
            ff_shape.append(neurons)

        ff_shape.append(self.n_strategies)
        self.model = FFNet(ff_shape, activation=torch.nn.ReLU()).to(device=self.device)

        # file names for PyTorch models
        now = datetime.now().strftime('%Y%m%d_%H%M')
        model_fn = 'CoCo_{}.pt'
        model_fn = os.path.join(os.getcwd(), model_fn)
        self.model_fn = model_fn.format(now)

    def load_network(self, fn_classifier_model):
        if os.path.exists(fn_classifier_model):
            print('Loading presaved classifier model from {}'.format(fn_classifier_model))
            self.model.load_state_dict(torch.load(fn_classifier_model))
            self.model_fn = fn_classifier_model

    def train(self, training_params, verbose=True):
        # grab training params
        TRAINING_EPOCHS = training_params['TRAINING_EPOCHS']
        BATCH_SIZE = training_params['BATCH_SIZE']
        CHECKPOINT_AFTER = training_params['CHECKPOINT_AFTER']
        SAVEPOINT_AFTER = training_params['SAVEPOINT_AFTER']
        TEST_BATCH_SIZE = training_params['TEST_BATCH_SIZE']
        LEARNING_RATE = training_params['LEARNING_RATE']
        WEIGHT_DECAY = training_params['WEIGHT_DECAY']

        model = self.model
        X = self.features; Y = self.labels[:,0]
        nn = int(0.8*len(Y))
        X_train = X[:nn, :]; Y_train = Y[:nn] 
        X_valid = X[nn:, :]; Y_valid = Y[nn:] 

        training_loss = torch.nn.CrossEntropyLoss()
        opt = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        itr = 1
        for epoch in range(TRAINING_EPOCHS):  # loop over the dataset multiple times
            t0 = time.time()
            running_loss = 0.0
            rand_idx = list(np.arange(X_train.shape[0]))
            random.shuffle(rand_idx)

            # Sample all data points
            indices = [rand_idx[ii * BATCH_SIZE:(ii + 1) * BATCH_SIZE] for ii in range((len(rand_idx) + BATCH_SIZE - 1) // BATCH_SIZE)]

            for ii,idx in enumerate(indices):
                # zero the parameter gradients
                opt.zero_grad()

                inputs = Variable(torch.from_numpy(X_train[idx,:])).float().to(device=self.device)
                labels = Variable(torch.from_numpy(Y_train[idx])).long().to(device=self.device)

                # forward + backward + optimize
                outputs = model(inputs)
                loss = training_loss(outputs, labels).float().to(device=self.device)
                class_guesses = torch.argmax(outputs,1)
                accuracy = torch.mean(torch.eq(class_guesses,labels).float())
                loss.backward()
                #torch.nn.utils.clip_grad_norm(model.parameters(),0.1)
                opt.step()

                # print statistics\n",
                running_loss += loss.item()
                if itr % CHECKPOINT_AFTER == 0:
                    verbose and print("Training: loss:   "+str(loss.item())+",   acc:  "+str(accuracy.item()))
                    
                    # Validate on different data (not used in training)
                    rand_idx = list(np.arange(X_valid.shape[0]))
                    random.shuffle(rand_idx)
                    test_inds = rand_idx[:TEST_BATCH_SIZE]
                    inputs = Variable(torch.from_numpy(X_valid[test_inds,:])).float().to(device=self.device)
                    labels = Variable(torch.from_numpy(Y_valid[test_inds])).long().to(device=self.device)
                    # forward + backward + optimize
                    outputs = model(inputs)
                    loss = training_loss(outputs, labels).float().to(device=self.device)
                    class_guesses = torch.argmax(outputs,1)
                    accuracy = torch.mean(torch.eq(class_guesses,labels).float())
                    verbose and print("Validation: loss:   "+str(loss.item())+",   acc:  "+str(accuracy.item()))

                if itr % SAVEPOINT_AFTER == 0:
                    torch.save(model.state_dict(), self.model_fn)
                    verbose and print('Saved model at {}'.format(self.model_fn))
                    # writer.add_scalar('Loss/train', running_loss, epoch)

                itr += 1
            # verbose and print('Done with epoch {} in {}s'.format(epoch, time.time()-t0))

        torch.save(model.state_dict(), self.model_fn)
        print('Saved model at {}'.format(self.model_fn))

        print('Done training')

"""
Class to reproduce the LSTM model in
https://proceedings.mlr.press/v168/cauligi22a/cauligi22a.pdf
"""
class PRISM:

    def __init__(self, prob_features, n_evals=10):
        """
        Constructor for PRISM class.
        """
        self.prob_features = prob_features
        self.n_evals = n_evals

        self.num_train, self.num_test = 0, 0
        self.model, self.model_fn = None, None
        
    def construct_features(self, params, ii_obs=None):
        prob_features = self.prob_features
        feature_vec = np.array([])
        x0, xg = params['x0'], params['xg'] 

        for feature in prob_features:
            if feature == "x0":
                feature_vec = np.hstack((feature_vec, x0))
            elif feature == "xg":
                feature_vec = np.hstack((feature_vec, xg[:2]))
            elif feature == "obstacles":
                obstacles = params['obstacles']
                feature_vec = np.hstack((feature_vec, np.reshape(obstacles, (4*self.n_obs))))
            elif feature == "obstacles_map":
                continue
            else:
                print('Feature {} is unknown'.format(feature))

        # Append one-hot encoding to end
        if ii_obs is not None:
            one_hot = np.zeros(self.n_obs)
            one_hot[ii_obs] = 1.
            feature_vec = np.hstack((feature_vec, one_hot))

        return feature_vec   
    
    def construct_strategies(self, n_features, train_data):
        """
        Reads in data and constructs strategy dictionary
        """
        self.n_features = n_features
        self.strategy_dict = {}

        self.X_train = train_data[0]
        self.Y_train = train_data[3]
        self.num_train = self.Y_train.shape[0]        

        self.n_y = self.Y_train[0].shape[0] # will be the dimension of the output
        self.H = self.Y_train[0].shape[1]
        self.features = np.zeros((self.num_train, self.n_features))
        self.labels = np.zeros((self.num_train, self.H, 1+self.n_y))
        self.n_strategies = 0
        n_str_max = 2**self.n_y # maximum number of strategies

        for ii in range(self.num_train):
            for hh in range(self.H):
                y_true = self.Y_train[ii,:,hh]
                if not self.n_strategies >= n_str_max:
                    # check if y_true is not in strategy_dict
                    if tuple(y_true) not in self.strategy_dict.keys():
                        self.strategy_dict[tuple(y_true)] = np.hstack((self.n_strategies, np.copy(y_true)))
                        self.n_strategies += 1
            
                self.labels[ii][hh] = self.strategy_dict[tuple(y_true)]

            prob_params = {}
            for k in self.X_train:
                prob_params[k] = self.X_train[k][ii]

            self.features[ii] = self.construct_features(prob_params)    

    def setup_network(self, ff_neurons=256, lstm_neurons=128, lstm_lay = 3, device_id=0):
        self.device = torch.device('cuda:{}'.format(device_id))

        net_shape = [self.n_features, ff_neurons, lstm_neurons, self.n_strategies]
        self.model = LSTMNet(net_shape, lstm_lay, activation=torch.nn.ReLU()).to(device=self.device)

        # file names for PyTorch models
        now = datetime.now().strftime('%Y%m%d_%H%M')
        model_fn = 'PRISM_{}.pt'
        model_fn = os.path.join(os.getcwd(), model_fn)
        self.model_fn = model_fn.format(now)

    def load_network(self, fn_classifier_model):
        if os.path.exists(fn_classifier_model):
            print('Loading presaved classifier model from {}'.format(fn_classifier_model))
            self.model.load_state_dict(torch.load(fn_classifier_model))
            self.model_fn = fn_classifier_model

    def train(self, training_params, verbose=True):
        # grab training params
        TRAINING_EPOCHS = training_params['TRAINING_EPOCHS']
        BATCH_SIZE = training_params['BATCH_SIZE']
        CHECKPOINT_AFTER = training_params['CHECKPOINT_AFTER']
        SAVEPOINT_AFTER = training_params['SAVEPOINT_AFTER']
        TEST_BATCH_SIZE = training_params['TEST_BATCH_SIZE']
        LEARNING_RATE = training_params['LEARNING_RATE']
        WEIGHT_DECAY = training_params['WEIGHT_DECAY']

        model = self.model
        X = self.features; Y = self.labels[:,:,0]
        nn = int(0.8*len(Y))
        X_train = X[:nn, :]; Y_train = Y[:nn, :] 
        X_valid = X[nn:, :]; Y_valid = Y[nn:, :] 

        training_loss = torch.nn.CrossEntropyLoss()
        opt = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        itr = 1
        for epoch in range(TRAINING_EPOCHS):  # loop over the dataset multiple times
            running_loss = 0.0
            rand_idx = list(np.arange(X_train.shape[0]))
            random.shuffle(rand_idx)

            # Sample all data points
            indices = [rand_idx[ii * BATCH_SIZE:(ii + 1) * BATCH_SIZE] for ii in range((len(rand_idx) + BATCH_SIZE - 1) // BATCH_SIZE)]
            for ii,idx in enumerate(indices):
                # zero the parameter gradients
                opt.zero_grad()

                inputs = Variable(torch.from_numpy(X_train[idx,:])).float().to(device=self.device)
                ip_shape = inputs.shape
                inputs = inputs.unsqueeze(1).expand(ip_shape[0], self.H, self.n_features)
                labels = Variable(torch.from_numpy(Y_train[idx,:])).long().to(device=self.device)

                # forward + backward + optimize
                outputs = model(inputs)
                loss = training_loss(outputs.permute(0, 2, 1), labels).float().to(device=self.device)
                class_guesses = torch.argmax(outputs, 2)
                accuracy = torch.mean(torch.all(torch.eq(class_guesses, labels), axis=1).float())
                loss.backward()
                #torch.nn.utils.clip_grad_norm(model.parameters(),0.1)
                opt.step()

                # print statistics\n",
                running_loss += loss.item()
                if itr % CHECKPOINT_AFTER == 0:
                    verbose and print("Training: loss:   "+str(loss.item())+",   acc:  "+str(accuracy.item()))

                    # Validate on different data (not used in training)
                    rand_idx = list(np.arange(X_valid.shape[0]))
                    random.shuffle(rand_idx)
                    test_inds = rand_idx[:TEST_BATCH_SIZE]
                    inputs = Variable(torch.from_numpy(X_valid[test_inds,:])).float().to(device=self.device)
                    inputs = inputs.unsqueeze(1).expand(TEST_BATCH_SIZE, self.H, self.n_features)
                    labels = Variable(torch.from_numpy(Y_valid[test_inds])).long().to(device=self.device)

                    # forward + backward + optimize
                    outputs = model(inputs)
                    loss = training_loss(outputs.permute(0, 2, 1), labels).float().to(device=self.device)
                    class_guesses = torch.argmax(outputs,2)
                    accuracy = torch.mean(torch.all(torch.eq(class_guesses, labels), axis=1).float())
                    verbose and print("Validation: loss:   "+str(loss.item())+",   acc:  "+str(accuracy.item()))

                if itr % SAVEPOINT_AFTER == 0:
                    torch.save(model.state_dict(), self.model_fn)
                    verbose and print('Saved model at {}'.format(self.model_fn))
                    # writer.add_scalar('Loss/train', running_loss, epoch)

                itr += 1
            # verbose and print('Done with epoch {} in {}s'.format(epoch, time.time()-t0))

        torch.save(model.state_dict(), self.model_fn)
        print('Saved model at {}'.format(self.model_fn))

        print('Done training')