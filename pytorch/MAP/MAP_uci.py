# implement MAP
import numpy as np
from numpy import linalg as LA
import torch
from tqdm import tqdm
from tqdm import trange
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad

import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import cProfile

import scipy
from scipy.stats import multivariate_normal
from sklearn.utils.extmath import cartesian

from copy import deepcopy

class MLP(nn.Module):
    def __init__(self, noise_variance, hidden_sizes, omega, activation=torch.tanh, learned_noise_var=False, input_dim=None, noise_param_init=None, standard_normal_prior=None):
        super(MLP, self).__init__()
        self.standard_normal_prior = standard_normal_prior
        self.dim_input = input_dim 
        self.activation = activation
        self.omega = float(omega)
        self.learned_noise_var = learned_noise_var
        if learned_noise_var == False:
            self.noise_variance = torch.cuda.DoubleTensor([noise_variance])
        else:
            # self.noise_var_param = nn.Parameter(torch.Tensor([-5]).cuda()) # this seemed to work OK
            self.noise_var_param = nn.Parameter(torch.cuda.DoubleTensor([noise_param_init]))
            self.noise_variance = self.get_noise_var(self.noise_var_param)
        self.hidden_sizes = hidden_sizes
        self.linears = nn.ModuleList([nn.Linear(input_dim, self.hidden_sizes[0]).double()])
        self.linears.extend([nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i+1]).double() for i in range(0, len(self.hidden_sizes)-1)])
        self.linears.append(nn.Linear(self.hidden_sizes[-1], 1).double())
        print(self.linears)

        # calculate number of parameters in network
        no_params = input_dim*self.hidden_sizes[0] # first weight matrix
        for i in range(len(self.hidden_sizes)-1):
            no_params = no_params + self.hidden_sizes[i] + self.hidden_sizes[i]*self.hidden_sizes[i+1]
        no_params = no_params + self.hidden_sizes[-1] + self.hidden_sizes[-1]*1 + 1 # final weight matrix and last 2 biases
        self.no_params = no_params

    def get_noise_var(self, noise_var_param):
        return torch.exp(noise_var_param) # try just a log representation
        #return torch.log(1 + torch.exp(noise_var_param)) + 1e-5 # softplus representation

    def forward(self, x):
        for i, l in enumerate(self.linears):
            x = l(x)
            if i < len(self.linears) - 1:
                x = self.activation(x) 
        return x

    def get_U(self, inputs, labels, trainset_size):
        minibatch_size = labels.size()[0]

        if self.learned_noise_var == True:
            self.noise_variance = self.get_noise_var(self.noise_var_param)
        outputs = self.forward(inputs)
        labels = labels.reshape(labels.size()[0], 1)
        L2_term = 0
        for _, l in enumerate(self.linears): # Neal's prior (bias has variance 1)
            n_inputs = l.weight.size()[0]
            if self.standard_normal_prior == True:
                single_layer_L2 = 0.5*(1/(self.omega**2))*(torch.sum(l.weight**2) + torch.sum(l.bias**2))
            else:
                single_layer_L2 = 0.5*(n_inputs/(self.omega**2))*torch.sum(l.weight**2) + 0.5*torch.sum(l.bias**2)
            L2_term = L2_term + single_layer_L2
  
        if self.learned_noise_var == True:
            error = (trainset_size/minibatch_size)*(1/(2*self.get_noise_var(self.noise_var_param)))*torch.sum((labels - outputs)**2)
        else:
            error = (trainset_size/minibatch_size)*(1/(2*self.noise_variance))*torch.sum((labels - outputs)**2)

        if self.learned_noise_var == True:
            noise_term = (trainset_size/2)*torch.log(2*3.1415926536*self.get_noise_var(self.noise_var_param))
        else:
            noise_term = 0 # this is a constant

        U = error + L2_term + noise_term
        return U

def evaluate(model, x_test, y_test, train_mean, train_sd, x_train_normalised=None, validation=None, optimizer=None, directory=None, name=None):
    
    # evaluate the model on the test/validation set
    if validation == True: # the validation set was normalised, so need to unnormalise it when evaluating
        x_test = x_test*train_sd[:-1]
        x_test =  x_test + train_mean[:-1]
        y_test = y_test*train_sd[-1]
        y_test = y_test + train_mean[-1]

    eval_batch_size = 100 # feed in 100 points at a time
    testset_size = y_test.size()[0]
    num_batches = int(np.ceil(testset_size/eval_batch_size))

    sum_squared_error = 0
    sum_log_likelihood = 0
    # for plotting
    predictive_sds = np.zeros(testset_size)
    abs_errors = np.zeros(testset_size)
       
    for i in range(num_batches):
        if i != num_batches - 1: # this isn't the final batch
            # fetch a batch of test inputs
            inputs = x_test[i*eval_batch_size:(i+1)*eval_batch_size,:]
            labels = y_test[i*eval_batch_size:(i+1)*eval_batch_size]
        else:
            # fetch the rest of the test inputs
            inputs = x_test[i*eval_batch_size:,:]
            labels = y_test[i*eval_batch_size:]
        actual_batch_size = labels.shape[0]

        # scale the inputs because of the normalisation        
        inputs = inputs - train_mean[:-1] # broadcast
        inputs = inputs/train_sd[:-1] # broadcast
        outputs = model(inputs)
        # scale the outputs because of the normalisation
        outputs = outputs*train_sd[-1]
        outputs = outputs + train_mean[-1]
        outputs = torch.squeeze(outputs)

        # calculate sum squared errors for this batch
        squared_error = torch.sum((outputs - labels)**2)
        sum_squared_error = sum_squared_error + squared_error

        # calculate log likelihood for this batch      
        noise_var = model.get_noise_var(model.noise_var_param)
        # scale the noise var because of the normalisation
        noise_var = noise_var*(train_sd[-1]**2)            
        log_likelihood = -0.5*actual_batch_size*torch.log(2*3.1415926536*noise_var) - (1/(2*noise_var))*squared_error

        if i != num_batches - 1: # this isn't the final batch
            predictive_sds[i*eval_batch_size:(i+1)*eval_batch_size] = np.sqrt(noise_var.data.cpu().numpy())
            abs_errors[i*eval_batch_size:(i+1)*eval_batch_size] = np.abs((outputs - labels).data.cpu().numpy())
        else:
            predictive_sds[i*eval_batch_size:] = np.sqrt(noise_var.data.cpu().numpy())
            abs_errors[i*eval_batch_size:] = np.abs((outputs - labels).data.cpu().numpy())
                    
        sum_log_likelihood = sum_log_likelihood + log_likelihood

    if directory is not None:
        # plot calibration curve
        fig, ax = plt.subplots()
        plt.scatter(predictive_sds, abs_errors)
        plt.axvline(x= train_sd[-1].data.cpu().numpy() * np.sqrt(model.get_noise_var(model.noise_var_param).data.cpu().numpy()), color='k', linestyle='--')
        plt.xlabel('predictive standard deviation')
        plt.ylabel('error magnitude')
        plt.gca().set_xlim(left=0)
        plt.gca().set_ylim(bottom=0)
        ax.set_aspect('equal')
        filepath = directory + '//calibration_' + name + '.pdf'
        fig.savefig(filepath)
        plt.close()

    mean_squared_error = sum_squared_error.detach()/testset_size # detach everything
    mean_ll = sum_log_likelihood.detach()/testset_size

    return mean_squared_error, mean_ll.item()

def train(model, train_x, train_y, eval_x, eval_y, train_mean, train_sd, validation=False, minibatch_size=None, no_epochs=None, optimizer=None, early_stopping=False): 
    # if validation is true, expect eval_x and eval_y to be normalised as well
    """if early_stopping is True, expect no_epochs to be a list"""
    if early_stopping == True:
        no_epochs_range = deepcopy(no_epochs)
        no_epochs = max(no_epochs_range)

    results_dict_list = []    

    trainset_size = train_y.size()[0]
    # train MAP network
    no_epochs = int(no_epochs)
    with trange(no_epochs) as epochs:
        for epoch in epochs: # loop over epochs
            # calculate the number of batches in this epoch
            no_batches = int(np.floor(trainset_size/minibatch_size))
            # loop over trainset
            # shuffle the dataset - this samples minibatches WITHOUT REPLACEMENT
            idx = torch.randperm(trainset_size)
            x_train_normalised = train_x[idx,:] 
            y_train_normalised = train_y[idx] 
            for i in range(no_batches):
                # clear previous gradients
                optimizer.zero_grad()

                # # shuffle the dataset - bug: this samples minibatches WITH REPLACEMENT
                # idx = torch.randperm(trainset_size)
                # x_train_normalised = train_x[idx,:] 
                # y_train_normalised = train_y[idx] 
                
                # fetch the batch, but only if there are enough datapoints left
                if (i+1)*minibatch_size <= trainset_size - 1:
                    x_train_batch = x_train_normalised[i*minibatch_size:(i+1)*minibatch_size,:]
                    y_train_batch = y_train_normalised[i*minibatch_size:(i+1)*minibatch_size]
                
                # forward pass and calculate loss
                loss = model.get_U(x_train_batch, y_train_batch, trainset_size=trainset_size)
            
                # compute gradients of all variables wrt loss               
                loss.backward()

                # perform updates using calculated gradients
                optimizer.step()

            if early_stopping == True:
                if (epoch + 1) in no_epochs_range:
                    MAP_MSE, MAP_LL = evaluate(model, eval_x, eval_y, train_mean, train_sd, validation=True, optimizer=optimizer) 
                    results_dict = {'MAP_MSE':MAP_MSE, 'MAP_LL':MAP_LL, 'no_epochs':epoch + 1}
                    results_dict_list.append(results_dict)

    if early_stopping == True:
        return results_dict_list 

def individual_train(data_location, test, noise_variance, hidden_sizes, omega, activation_function, \
    learned_noise_var, input_dim, noise_param_init, learning_rate, no_epochs, standard_normal_prior, \
        minibatch_size, results_dir=None, split=None, early_stopping=False):
    """if early_stopping == True, expect no_epochs to be a list. Else it should be an int"""
    # reset seed for reproducibility
    # np.random.seed(seed) 
    # torch.manual_seed(seed) 
    # model
    model = MLP(noise_variance, hidden_sizes, omega, activation=activation_function, learned_noise_var=learned_noise_var, input_dim=input_dim, noise_param_init=noise_param_init, standard_normal_prior=standard_normal_prior)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)
    model.to(device)
    for param in model.parameters():
        print(type(param.data), param.size())

    # get dataset
    with open(data_location, 'rb') as f:
        train_set, train_set_normalised, val_set_normalised, test_set, train_mean, train_sd = pickle.load(f)

    train_mean = torch.cuda.DoubleTensor(train_mean)
    train_sd = torch.cuda.DoubleTensor(train_sd)

    x_train_normalised = torch.cuda.DoubleTensor(train_set_normalised[:,:-1])
    y_train_normalised = torch.cuda.DoubleTensor(train_set_normalised[:,-1])

    x_val_normalised = torch.cuda.DoubleTensor(val_set_normalised[:,:-1])
    y_val_normalised = torch.cuda.DoubleTensor(val_set_normalised[:,-1])

    if test == True: # combine train and val sets
        x_train_normalised = torch.cat((x_train_normalised, x_val_normalised), 0)
        y_train_normalised = torch.cat((y_train_normalised, y_val_normalised), 0)

    x_test = torch.cuda.DoubleTensor(test_set[:,:-1])
    y_test = torch.cuda.DoubleTensor(test_set[:,-1])

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # train the model, and print out the validation set log likelihood when training

    if test == True: # this is test time, no early stopping needed
        train(model, x_train_normalised, y_train_normalised, x_test, y_test, train_mean, train_sd, validation=False, minibatch_size=minibatch_size, no_epochs=no_epochs, optimizer=optimizer)
        MAP_MSE, MAP_LL = evaluate(model, x_test, y_test, train_mean, train_sd, validation=False, optimizer=optimizer, directory=results_dir, name=str(split))        
        return MAP_MSE, MAP_LL

    else: # this is validation time, do early stopping for hyperparam search
        results_dict_list = train(model, x_train_normalised, y_train_normalised, x_val_normalised, y_val_normalised, train_mean, train_sd, validation=True, minibatch_size=minibatch_size, no_epochs=no_epochs, optimizer=optimizer, early_stopping=True)
        return results_dict_list

def individual_tune_train(results_dir, standard_normal_prior, activation_function, hidden_sizes, omega_range, learning_rate_range, minibatch_size_range, no_epochs_range, input_dim, noise_param_init, dataset):
    # do a grid search on each split separately, then evaluate on the test set
    # this grids over omega, minibatch_size and no_epochs
    
    # create array of values to grid search over - but don't repeat searches when doing early stopping
    list_hypers = [omega_range, learning_rate_range, minibatch_size_range]
    hyperparams = cartesian(list_hypers)
    MAP_RMSEs = np.zeros(no_splits)
    MAP_LLs = np.zeros(no_splits)

    for split in range(no_splits): 
        # find data location
        if gap == True:
            data_location = '..//vision//data//' + dataset + '_gap//' + dataset + str(split) + '.pkl'
        else:
            data_location = '..//vision//data//' + dataset + '_yarin//' + dataset + str(split) + '.pkl'
        test = False # do hyperparam grid search on validation set
        for i in range(hyperparams.shape[0]):
            # get hyperparams
            copy_hyperparams = deepcopy(hyperparams)
            omega = copy_hyperparams[i,0]
            learning_rate = copy_hyperparams[i,1]
            minibatch_size = int(copy_hyperparams[i,2])

            # train on one split, and validate            
            noise_variance = 0 # not using this parameter
            learned_noise_var = True # always true for UCI regression
            results_dict_list = individual_train(data_location, test, noise_variance, hidden_sizes, omega, \
                activation_function, learned_noise_var, input_dim, noise_param_init, learning_rate, no_epochs_range, \
                    standard_normal_prior, minibatch_size, early_stopping=True)
            
            # save text file with results
            for results_dict in results_dict_list:
                file = open(results_dir + '/results' + str(split) + '.txt','a') 
                file.write('omega, learning_rate, minibatch_size: {} \n'.format(hyperparams[i,:]))
                file.write('no_epochs: {} \n'.format(results_dict['no_epochs']))
                file.write('MAP_RMSE: {} \n'.format(torch.sqrt(results_dict['MAP_MSE'])))
                file.write('MAP_LL: {} \n'.format(results_dict['MAP_LL']))
                file.close() 
    
            # record the hyperparams that maximise validation set MAP_LL
            if i == 0: # first hyperparam setting
                # find the best MAP_LL in results_dict_list
                for k, results_dict in enumerate(results_dict_list):
                    if k == 0:
                        max_LL = results_dict['MAP_LL']
                        best_no_epochs = results_dict['no_epochs']
                    else:
                        if float(results_dict['MAP_LL']) > float(max_LL):
                            max_LL = results_dict['MAP_LL']
                            best_no_epochs = results_dict['no_epochs']
                best_hyperparams = copy_hyperparams[i,:]
            else:
                for results_dict in results_dict_list:
                    if float(results_dict['MAP_LL']) > float(max_LL):
                        max_LL = results_dict['MAP_LL']
                        best_no_epochs = results_dict['no_epochs']
                        best_hyperparams = copy_hyperparams[i,:]
        
        # use the best hyperparams found to retrain on all the train data, and evaluate on the test set
        test = True # this is test time
        omega = best_hyperparams[0]
        learning_rate = best_hyperparams[1]
        minibatch_size = int(best_hyperparams[2])
        no_epochs = best_no_epochs

        MAP_MSE, MAP_LL = individual_train(data_location, test, noise_variance, hidden_sizes,\
             omega, activation_function, learned_noise_var, input_dim, noise_param_init, learning_rate, no_epochs,\
                  standard_normal_prior, minibatch_size, results_dir, split)
        MAP_RMSEs[split] = torch.sqrt(MAP_MSE).data.cpu().numpy()

        MAP_LLs[split] = MAP_LL
        
        # record best hyperparams
        file = open(results_dir + '/best_hypers.txt','a') 
        file.write('split: {} \n'.format(str(split)))
        file.write('omega: {} \n'.format(omega))
        file.write('learning_rate: {} \n'.format(learning_rate))
        file.write('no_epochs: {} \n'.format(no_epochs))
        file.write('minibatch_size: {} \n'.format(minibatch_size))
        file.write('test_MAP_RMSE: {} \n'.format(MAP_RMSEs[split]))
        file.write('test_MAP_LL: {} \n'.format(MAP_LLs[split]))
        file.close() 

    # find the mean and std error of the RMSEs and LLs
    mean_MAP_RMSE = np.mean(MAP_RMSEs)
    sd_MAP_RMSE = np.std(MAP_RMSEs)
    mean_MAP_LL = np.mean(MAP_LLs)
    sd_MAP_LL = np.std(MAP_LLs)

    # save the answer
    file = open(results_dir + '/test_results.txt','w') 
    file.write('MAP_RMSEs: {} \n'.format(MAP_RMSEs))
    file.write('MAP_LLs: {} \n'.format(MAP_LLs))
    file.write('mean_MAP_RMSE: {} \n'.format(mean_MAP_RMSE))
    file.write('sd_MAP_RMSE: {} \n'.format(sd_MAP_RMSE))
    file.write('mean_MAP_LL: {} \n'.format(mean_MAP_LL))
    file.write('sd_MAP_LL: {} \n'.format(sd_MAP_LL))
    file.close() 

if __name__ == "__main__":

    # set RNG
    seed = 0
    np.random.seed(seed) 
    torch.manual_seed(seed) 

    input_dims = {'boston_housing': 13, 'concrete': 8, 'energy': 8, 'kin8nm': 8, 'power': 4, 'protein': 9, 'wine': 11, 'yacht': 6, 'naval': 16}
    datasets = ['energy', 'naval', 'boston_housing', 'concrete', 'kin8nm', 'power', 'protein', 'wine', 'yacht']
    #datasets = ['protein', 'wine', 'yacht']

    # hyperparameters
    standard_normal_prior = True
    activation_function = F.relu
    hidden_sizes = [50, 50]
    learned_noise_var = True
    noise_param_init = -1
    gap = True

    for dataset in datasets: 
        if gap == True:
            no_splits = input_dims[dataset]
        else:
            if dataset == 'protein':
                no_splits = 5
            else:
                no_splits = 20

        directory = './/experiments//gap_no_replacement//' + dataset + '//2HL'
        os.mkdir(directory)
        input_dim = input_dims[dataset]
        omega_range = [1.0, 2.0]
        minibatch_size_range = [100]
        learning_rate_range = [0.01, 0.001]
        no_epochs_range = [20, 40, 100]

        # save text file with hyperparameters
        file = open(directory + '/hyperparameters.txt','w') 
        file.write('standard_normal_prior: {} \n'.format(standard_normal_prior))
        file.write('activation_function: {} \n'.format(activation_function.__name__))
        file.write('seed: {} \n'.format(seed))
        file.write('hidden_sizes: {} \n'.format(hidden_sizes))
        file.write('learned_noise_var: {} \n'.format(learned_noise_var))
        file.write('minibatch_size_range: {} \n'.format(minibatch_size_range))
        file.write('noise_param_init: {} \n'.format(noise_param_init))
        file.write('omega_range: {} \n'.format(omega_range))
        file.write('learning_rate_range: {} \n'.format(learning_rate_range))
        file.write('no_epochs_range: {} \n'.format(no_epochs_range))
        file.close() 

        individual_tune_train(directory, standard_normal_prior, activation_function, hidden_sizes, omega_range, learning_rate_range, minibatch_size_range, no_epochs_range, input_dim, noise_param_init, dataset)  

    


    

    




       



