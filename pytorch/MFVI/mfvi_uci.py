# run MFVI on UCI benchmarks

import numpy as np
from numpy import linalg as LA
import torch
from tqdm import tqdm
from tqdm import trange
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
from torch.autograd import Variable

import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import cProfile

import scipy
from scipy.stats import multivariate_normal
from sklearn.utils.extmath import cartesian

from copy import deepcopy
import gc

class MFVI_Linear_Layer(nn.Module):
    def __init__(self, n_input, n_output, omega):
        super(MFVI_Linear_Layer, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        # omega is the prior standard deviation (assume isotropic prior)
        prior_logvar = 2*np.log(omega) 
        self.prior_logvar = prior_logvar

        """initialise parameters and priors following 'Neural network ensembles and variational inference revisited', Appendix A"""
        # weight parameters
        self.W_mean = nn.Parameter(torch.Tensor(n_input, n_output).normal_(0, 1/np.sqrt(4*n_output))) # initialisation of weight means
        self.W_logvar = nn.Parameter(torch.Tensor(n_input, n_output).normal_(-11.5, 1e-10)) # initialisation of weight logvariances 
        # bias parameters
        self.b_mean = nn.Parameter(torch.Tensor(n_output).normal_(0, 1e-10)) # initialisation of bias means
        self.b_logvar = nn.Parameter(torch.Tensor(n_output).normal_(-11.5, 1e-10)) # initialisation of bias logvariances (why uniform?)
        
        # prior parameters 
        self.W_prior_mean = Variable(torch.zeros(n_input, n_output).cuda())
        self.W_prior_logvar = Variable((prior_logvar*torch.ones(n_input, n_output)).cuda())
        self.b_prior_mean = Variable(torch.zeros(n_output).cuda()) 
        self.b_prior_logvar = Variable((prior_logvar*torch.ones(n_output)).cuda())

        self.num_weights = n_input*n_output + n_output # number of weights and biases
 
    def forward(self, x, no_samples): # number of samples per forward pass
        """
        input is either (batch_size x no_input), if this is the first layer of the network, or (no_samples x batch_size x no_input), 
        and output is (no_samples x batch_size x no_output)
        """
        # local reparameterisation trick
        # find out if this is the first layer of the network. if it is, perform an expansion to no_samples
        if len(x.shape) == 2: 
            batch_size = x.size()[0]
            z = self.get_random(no_samples, batch_size)
            gamma = torch.mm(x, self.W_mean) + self.b_mean.expand(batch_size, -1)
            W_var = torch.exp(self.W_logvar) 
            b_var = torch.exp(self.b_logvar)
            delta = torch.mm(x**2, W_var) + b_var.expand(batch_size, -1)
            sqrt_delta = torch.sqrt(delta) 
            samples_gamma = gamma.expand(no_samples, -1, -1)
            samples_sqrt_delta = sqrt_delta.expand(no_samples, -1, -1)
            samples_activations = samples_gamma + torch.mul(samples_sqrt_delta, z)

        elif len(x.shape) == 3:
            batch_size = x.size()[1]
            z = self.get_random(no_samples, batch_size)
            # samples_gamma has different values for each sample, so has dimensions (no_samples x batch_size x no_outputs)
            samples_gamma = torch.matmul(x, self.W_mean) + self.b_mean.expand(no_samples, batch_size, -1)
            W_var = torch.exp(self.W_logvar)
            b_var = torch.exp(self.b_logvar)
            # delta has different values for each sample, so has dimensions (no_samples x batch_size x no_outputs)
            delta = torch.matmul(x**2, W_var) + b_var.expand(no_samples, batch_size, -1)
            samples_sqrt_delta = torch.sqrt(delta)
            samples_activations = samples_gamma + torch.mul(samples_sqrt_delta, z)
       
        return samples_activations

    def get_random(self, no_samples, batch_size):
        return Variable(torch.cuda.FloatTensor(no_samples, batch_size, self.n_output).normal_(0, 1)) # standard normal noise matrix

    def KL(self): # get KL between q and prior for this layer
        # W_KL = 0.5*(- self.W_logvar + torch.exp(self.W_logvar) + (self.W_mean)**2)
        # b_KL = 0.5*(- self.b_logvar + torch.exp(self.b_logvar) + (self.b_mean)**2)
        W_KL = 0.5*(self.W_prior_logvar - self.W_logvar + (torch.exp(self.W_logvar) + (self.W_mean - self.W_prior_mean)**2)/torch.exp(self.W_prior_logvar))
        b_KL = 0.5*(self.b_prior_logvar - self.b_logvar + (torch.exp(self.b_logvar) + (self.b_mean - self.b_prior_mean)**2)/torch.exp(self.b_prior_logvar))
        return W_KL.sum() + b_KL.sum() - 0.5*self.num_weights

class MFVI_Net(nn.Module):
    def __init__(self, noise_variance, hidden_sizes, omega, activation, learned_noise_var=False, input_dim=None, noise_param_init=None, standard_normal_prior=None):
        super(MFVI_Net, self).__init__()
        self.train_samples = 32 # following Marcin's work means 1 sample
        self.test_samples = 100
        self.omega = float(omega)
        self.dim_input = input_dim 
        self.activation = activation

        self.noise_var_param = nn.Parameter(torch.cuda.FloatTensor([noise_param_init]))
        self.noise_variance = self.get_noise_var(self.noise_var_param)

        # create the layers in the network based on params
        self.hidden_sizes = hidden_sizes
        self.linears = nn.ModuleList([MFVI_Linear_Layer(self.dim_input, self.hidden_sizes[0], self.omega)])
        self.linears.extend([MFVI_Linear_Layer(self.hidden_sizes[i], self.hidden_sizes[i+1], self.omega) for i in range(0, len(self.hidden_sizes)-1)])
        self.linears.append(MFVI_Linear_Layer(self.hidden_sizes[-1], 1, self.omega))

    def get_noise_var(self, noise_var_param):
        return torch.exp(noise_var_param) # try just a log representation
    
    def get_KL_term(self):
        # calculate KL divergence between q and the prior for the entire network
        KL_term = 0
        for _, l in enumerate(self.linears):
            KL_term = KL_term + l.KL()
        return KL_term

    def get_U(self, inputs, labels, trainset_size):
        # calculate stochastic estimate of the per-datapoint ELBO
        self.noise_variance = self.get_noise_var(self.noise_var_param)
        outputs = self.forward(inputs, self.train_samples)

        no_samples = outputs.size()[0]
        labels = labels.expand(no_samples, -1)
        const_term = 0.5*torch.log(2*3.141592654*self.get_noise_var(self.noise_var_param))
        reconstruction_loss = (trainset_size)*(const_term + (1/(2*self.get_noise_var(self.noise_var_param)))*torch.mean((labels - outputs)**2))
        KL_term = self.get_KL_term()

        U = (reconstruction_loss + KL_term)/trainset_size # per-datapoint ELBO
        return U

    def forward(self, s, no_samples):
        s = s.view(-1, self.dim_input)
        for i, l in enumerate(self.linears):
            s = l(s, no_samples = no_samples)
            if i < len(self.linears) - 1:
                s = self.activation(s) 
        # s has dimension (no_samples x batch_size x no_output=1)
        s = s.view(no_samples, -1) # (no_samples x batch_size)
        return s

def sample_mfvi(model, inputs, labels, train_mean, train_sd, no_samples):
    # inputs and labels assumed NOT normalised
    # sample from the MFVI, then return the function evaluations
    # scale the inputs
    inputs = inputs - train_mean[:-1] 
    inputs = inputs/train_sd[:-1]
    
    # sample forward passes from the MFVI net and evaluate
    all_outputs = model(inputs, no_samples) 
    all_outputs = torch.t(all_outputs) # batch_size x no_samples
    # scale the outputs
    all_outputs = all_outputs*train_sd[-1]
    all_outputs = all_outputs + train_mean[-1] # should be broadcast

    #########################################################
    # # fill the model with the sampled parameters and evaluate
    # all_outputs = torch.cuda.DoubleTensor(inputs.shape[0], no_samples)
    # for i, sample in enumerate(samples):
    #     j = 0 
    #     for name, param in model.named_parameters():
    #         if name != 'noise_var_param': # dont do laplace for noise variance
    #             param.data = sample[j] # fill in the model with these weights
    #             j=j+1
    #     # forward pass through the sampled model
    #     outputs = torch.squeeze(model(inputs))
    #     # scale the outputs
    #     outputs = outputs*train_sd[-1]
    #     outputs = outputs + train_mean[-1]
    #     all_outputs[:,i] = outputs
    #########################################################
    
    # calculate summed log likelihoods for this batch of inputs
    noise_var = model.get_noise_var(model.noise_var_param)
    # scale the noise var because of the normalisation
    noise_var = noise_var*(train_sd[-1]**2) 
    error_term = ((all_outputs - torch.unsqueeze(labels,1))**2)/(2*noise_var) 
    exponents = -torch.log(torch.cuda.FloatTensor([no_samples])) - 0.5*torch.log(2*3.1415926536*noise_var) - error_term
    LLs = torch.logsumexp(exponents, 1)
    sum_LLs = torch.sum(LLs)

    # calculate the quantities needed to plot calibration curves and get RMSEs
    mean_prediction = torch.mean(all_outputs, 1)
    squared_error = torch.sum((labels - mean_prediction)**2)
    abs_errors = torch.abs(labels - mean_prediction)
    variances = noise_var + torch.mean(all_outputs**2, 1) - mean_prediction**2

    return sum_LLs.detach(), squared_error.detach(), abs_errors.detach(), variances.detach() # detach everything

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

        no_samples = 100 # no of test samples 
        log_likelihood, squared_error, errors, variances = sample_mfvi(model, inputs, labels, train_mean, train_sd, no_samples)
        sum_squared_error = sum_squared_error + squared_error

        if i != num_batches - 1: # this isn't the final batch
            predictive_sds[i*eval_batch_size:(i+1)*eval_batch_size] = np.sqrt(variances.data.cpu().numpy())
            abs_errors[i*eval_batch_size:(i+1)*eval_batch_size] = errors.data.cpu().numpy()
        else:
            predictive_sds[i*eval_batch_size:] = np.sqrt(variances.data.cpu().numpy())
            abs_errors[i*eval_batch_size:] = errors.data.cpu().numpy()
                    
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
        ########################################################################

    mean_squared_error = sum_squared_error/testset_size
    mean_ll = sum_log_likelihood/testset_size

    return mean_squared_error, mean_ll.item()

def train(model, train_x, train_y, eval_x, eval_y, train_mean, train_sd, validation=False, minibatch_size=None, no_epochs=None, subsampling=None, optimizer=None, early_stopping=False): 
    # if validation is true, expect eval_x and eval_y to be normalised as well
    """if early_stopping is True, expect no_epochs to be a list"""
    if early_stopping == True:
        no_epochs_range = deepcopy(no_epochs)
        no_epochs = max(no_epochs_range)

    results_dict_list = []    

    trainset_size = train_y.size()[0]
    # train network
    no_epochs = int(no_epochs)
    with trange(no_epochs) as epochs:
        for epoch in epochs: # loop over epochs
            # calculate the number of batches in this epoch
            no_batches = int(np.floor(trainset_size/minibatch_size))
            #print('Beginning epoch {}'.format(epoch))
            # loop over trainset
            for i in range(no_batches):
                # clear previous gradients
                optimizer.zero_grad()

                # shuffle the dataset
                idx = torch.randperm(trainset_size)
                x_train_normalised = train_x[idx,:] 
                y_train_normalised = train_y[idx] 
                
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
                    MFVI_MSE, MFVI_LL = evaluate(model, eval_x, eval_y, train_mean, train_sd, validation=True, optimizer=optimizer) 
                    results_dict = {'MFVI_MSE':MFVI_MSE, 'MFVI_LL':MFVI_LL, 'no_epochs':epoch + 1}
                    results_dict_list.append(results_dict)

    if early_stopping == True:
        return results_dict_list 

def individual_train(data_location, test, noise_variance, hidden_sizes, omega, activation_function, \
    learned_noise_var, input_dim, noise_param_init, learning_rate, no_epochs, standard_normal_prior, \
        minibatch_size, results_dir=None, split=None, early_stopping=False):
    """if early_stopping == True, expect no_epochs to be a list. Else it should be an int"""
    model = MFVI_Net(noise_variance, hidden_sizes, omega, activation=activation_function, learned_noise_var=learned_noise_var, input_dim=input_dim, noise_param_init=noise_param_init, standard_normal_prior=standard_normal_prior)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)
    model.to(device)
    for param in model.parameters():
        print(type(param.data), param.size())

    # get dataset
    with open(data_location, 'rb') as f:
        train_set, train_set_normalised, val_set_normalised, test_set, train_mean, train_sd = pickle.load(f)

    train_mean = torch.cuda.FloatTensor(train_mean)
    train_sd = torch.cuda.FloatTensor(train_sd)

    x_train_normalised = torch.cuda.FloatTensor(train_set_normalised[:,:-1])
    y_train_normalised = torch.cuda.FloatTensor(train_set_normalised[:,-1])

    x_val_normalised = torch.cuda.FloatTensor(val_set_normalised[:,:-1])
    y_val_normalised = torch.cuda.FloatTensor(val_set_normalised[:,-1])

    if test == True: # combine train and val sets
        x_train_normalised = torch.cat((x_train_normalised, x_val_normalised), 0)
        y_train_normalised = torch.cat((y_train_normalised, y_val_normalised), 0)

    x_test = torch.cuda.FloatTensor(test_set[:,:-1])
    y_test = torch.cuda.FloatTensor(test_set[:,-1])

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # train the model, and print out the validation set log likelihood when training

    if test == True: # this is test time, no early stopping needed
        train(model, x_train_normalised, y_train_normalised, x_test, y_test, train_mean, train_sd, validation=False, minibatch_size=minibatch_size, no_epochs=no_epochs, optimizer=optimizer)
        MFVI_MSE, MFVI_LL = evaluate(model, x_test, y_test, train_mean, train_sd, validation=False, optimizer=optimizer, directory=results_dir, name=str(split)) 
        return MFVI_MSE, MFVI_LL

    else: # this is validation time, do early stopping for hyperparam search
        results_dict_list = train(model, x_train_normalised, y_train_normalised, x_val_normalised, y_val_normalised, train_mean, train_sd, validation=True, minibatch_size=minibatch_size, no_epochs=no_epochs, optimizer=optimizer, early_stopping=True)
        return results_dict_list

def individual_tune_train(results_dir, standard_normal_prior, activation_function, hidden_sizes, omega_range, learning_rate_range, minibatch_size_range, no_epochs_range, input_dim, noise_param_init, dataset):
    # do a grid search on each split separately, then evaluate on the test set
    # this grids over omega, minibatch_size and no_epochs
    
    # create array of values to grid search over - but don't repeat searches when doing early stopping
    list_hypers = [omega_range, learning_rate_range, minibatch_size_range]
    hyperparams = cartesian(list_hypers)
    MFVI_RMSEs = np.zeros(no_splits)
    MFVI_LLs = np.zeros(no_splits)

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
                file.write('MFVI_RMSE: {} \n'.format(torch.sqrt(results_dict['MFVI_MSE'])))
                file.write('MFVI_LL: {} \n'.format(results_dict['MFVI_LL']))
                file.close() 
    
            # record the hyperparams that maximise validation set MFVI_LL
            if i == 0: # first hyperparam setting
                # find the best MFVI_LL in results_dict_list
                for k, results_dict in enumerate(results_dict_list):
                    if k == 0:
                        max_LL = results_dict['MFVI_LL']
                        best_no_epochs = results_dict['no_epochs']
                    else:
                        if float(results_dict['MFVI_LL']) > float(max_LL):
                            max_LL = results_dict['MFVI_LL']
                            best_no_epochs = results_dict['no_epochs']
                best_hyperparams = copy_hyperparams[i,:]
            else:
                for results_dict in results_dict_list:
                    if float(results_dict['MFVI_LL']) > float(max_LL):
                        max_LL = results_dict['MFVI_LL']
                        best_no_epochs = results_dict['no_epochs']
                        best_hyperparams = copy_hyperparams[i,:]
        
        # use the best hyperparams found to retrain on all the train data, and evaluate on the test set
        test = True # this is test time
        omega = best_hyperparams[0]
        learning_rate = best_hyperparams[1]
        minibatch_size = int(best_hyperparams[2])
        no_epochs = best_no_epochs

        MFVI_MSE, MFVI_LL = individual_train(data_location, test, noise_variance, hidden_sizes,\
             omega, activation_function, learned_noise_var, input_dim, noise_param_init, learning_rate, no_epochs,\
                  standard_normal_prior, minibatch_size, results_dir, split)
        MFVI_RMSEs[split] = torch.sqrt(MFVI_MSE).data.cpu().numpy()

        MFVI_LLs[split] = MFVI_LL
        
        # record best hyperparams
        file = open(results_dir + '/best_hypers.txt','a') 
        file.write('split: {} \n'.format(str(split)))
        file.write('omega: {} \n'.format(omega))
        file.write('learning_rate: {} \n'.format(learning_rate))
        file.write('no_epochs: {} \n'.format(no_epochs))
        file.write('minibatch_size: {} \n'.format(minibatch_size))
        file.write('test_MFVI_RMSE: {} \n'.format(MFVI_RMSEs[split]))
        file.write('test_MFVI_LL: {} \n'.format(MFVI_LLs[split]))
        file.close() 

    # find the mean and std error of the RMSEs and LLs
    mean_MFVI_RMSE = np.mean(MFVI_RMSEs)
    sd_MFVI_RMSE = np.std(MFVI_RMSEs)
    mean_MFVI_LL = np.mean(MFVI_LLs)
    sd_MFVI_LL = np.std(MFVI_LLs)

    # save the answer
    file = open(results_dir + '/test_results.txt','w') 
    file.write('MFVI_RMSEs: {} \n'.format(MFVI_RMSEs))
    file.write('MFVI_LLs: {} \n'.format(MFVI_LLs))
    file.write('mean_MFVI_RMSE: {} \n'.format(mean_MFVI_RMSE))
    file.write('sd_MFVI_RMSE: {} \n'.format(sd_MFVI_RMSE))
    file.write('mean_MFVI_LL: {} \n'.format(mean_MFVI_LL))
    file.write('sd_MFVI_LL: {} \n'.format(sd_MFVI_LL))

    file.close() 

if __name__ == "__main__":

    # set RNG
    seed = 0
    np.random.seed(seed) 
    torch.manual_seed(seed) 

    input_dims = {'boston_housing': 13, 'concrete': 8, 'energy': 8, 'kin8nm': 8, 'power': 4, 'protein': 9, 'wine': 11, 'yacht': 6, 'naval': 16}
    #datasets = ['boston_housing', 'concrete', 'energy', 'kin8nm', 'naval','power', 'protein', 'wine', 'yacht']
    datasets = ['kin8nm', 'naval','power', 'protein']

    # hyperparameters
    standard_normal_prior = True
    activation_function = torch.tanh
    hidden_sizes = [50]
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

        directory = './/experiments//gap//' + dataset + '//1HL_tanh'
        os.mkdir(directory)
        input_dim = input_dims[dataset]
        omega_range = [1.0]
        minibatch_size_range = [32, 100]
        learning_rate_range = [0.01, 0.001]
        no_epochs_range = [50, 100, 200]

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