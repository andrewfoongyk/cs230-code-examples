# implement bayesian ensembles paper
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

def plot_regression(models, directory):
    plt.figure()
    plt.rcParams.update({'font.size': 14})
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    axes = plt.gca()
    axes.set_xlim([-2, 2])
    axes.set_ylim([-3, 3])
    plt.plot(data_load[:,0], data_load[:,1], '+k', markersize=16)
    N = 1000
    x_lower = -6
    x_upper = 8
    x_values = np.linspace(x_lower, x_upper, N)[:, None]
    test_inputs = torch.cuda.FloatTensor(x_values)

    # plot all the samples
    no_samp = len(models)
    all_test_outputs = np.zeros((no_samp, N))
    for model_no, model in enumerate(models):
        # plot regression
        test_outputs = model(test_inputs)
        # convert back to np array
        test_outputs = test_outputs.data.cpu().numpy()
        test_outputs = test_outputs.reshape(N)
        plt.plot(x_values, test_outputs, linewidth=1)
        # save data for ensemble mean and s.d. calculation
        all_test_outputs[model_no,:] = test_outputs

    # calculate mean and variance
    mean = np.mean(all_test_outputs, 0)
    variance = models[0].noise_variance.detach().cpu().numpy() + np.mean(all_test_outputs**2, 0) - mean**2 ####### THIS ASSUMES FIXED NOISE VARIANCE

    plt.plot(x_values, mean, color='b')
    plt.fill_between(x_values.squeeze(), mean + 2 * np.sqrt(variance), mean - 2 * np.sqrt(variance), color='b', alpha=0.3)

    filepath = os.path.join(directory, 'BE_regression.pdf')
    plt.savefig(filepath)
    plt.close()


class MLP(nn.Module):
    def __init__(self, noise_variance, hidden_sizes, omega, activation=torch.tanh, learned_noise_var=False, input_dim=None, noise_param_init=None, standard_normal_prior=None, random_prior=False):
        super(MLP, self).__init__()
        self.standard_normal_prior = standard_normal_prior
        self.dim_input = input_dim 
        self.activation = activation
        self.omega = float(omega)
        self.learned_noise_var = learned_noise_var
        if learned_noise_var == False:
            self.noise_variance = torch.cuda.FloatTensor([noise_variance])
        else:
            # self.noise_var_param = nn.Parameter(torch.Tensor([-5]).cuda()) # this seemed to work OK
            self.noise_var_param = nn.Parameter(torch.cuda.FloatTensor([noise_param_init]))
            self.noise_variance = self.get_noise_var(self.noise_var_param)
        self.hidden_sizes = hidden_sizes
        self.linears = nn.ModuleList([nn.Linear(input_dim, self.hidden_sizes[0])])
        self.linears.extend([nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i+1]) for i in range(0, len(self.hidden_sizes)-1)])
        self.linears.append(nn.Linear(self.hidden_sizes[-1], 1))

        # calculate number of parameters in network
        no_params = input_dim*self.hidden_sizes[0] # first weight matrix
        for i in range(len(self.hidden_sizes)-1):
            no_params = no_params + self.hidden_sizes[i] + self.hidden_sizes[i]*self.hidden_sizes[i+1]
        no_params = no_params + self.hidden_sizes[-1] + self.hidden_sizes[-1]*1 + 1 # final weight matrix and last 2 biases
        self.no_params = no_params

        self.random_prior = random_prior
        self.init_prior()

    def init_prior(self):
        # prior mean for Bayesian ensembling
        self.prior_mean = []
        for layer in self.linears:
            weight_shape = layer.weight.shape
            bias_shape = layer.bias.shape
            if self.random_prior:
                if self.standard_normal_prior:
                    weight_prior_mean = np.random.randn(weight_shape[0], weight_shape[1]) * self.omega
                    bias_prior_mean = np.random.randn(bias_shape[0]) * self.omega
                else: # use Neal's prior
                    n_inputs = weight_shape[0]
                    weight_prior_mean = np.random.randn(weight_shape[0], weight_shape[1]) * self.omega/n_inputs
                    bias_prior_mean = np.random.randn(bias_shape[0])   
            else: # just use zero for prior mean
                weight_prior_mean = np.zeros((weight_shape[0], weight_shape[1]))
                bias_prior_mean = np.zeros(bias_shape[0])          
            weight_prior_mean = torch.cuda.FloatTensor(weight_prior_mean)
            bias_prior_mean = torch.cuda.FloatTensor(bias_prior_mean)

            prior_mean_dict = {'weight': weight_prior_mean, 'bias': bias_prior_mean}
            self.prior_mean.append(prior_mean_dict)
            
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

        for layer_no, l in enumerate(self.linears): # Neal's prior (bias has variance 1)
            # get prior means
            weight_mean = self.prior_mean[layer_no]['weight']
            bias_mean = self.prior_mean[layer_no]['bias']
            n_inputs = l.weight.size()[0]
            if self.standard_normal_prior == True:
                single_layer_L2 = 0.5*(1/(self.omega**2))*(torch.sum((l.weight - weight_mean)**2) + torch.sum((l.bias - bias_mean)**2))
            else: # Neal's prior
                single_layer_L2 = 0.5*(n_inputs/(self.omega**2))*torch.sum((l.weight - weight_mean)**2) + 0.5*torch.sum((l.bias - bias_mean)**2)
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


if __name__ == "__main__":

    # set RNG
    np.random.seed(0) # 0
    torch.manual_seed(231) #  230

    # hyperparameters
    noise_variance = 0.01 # 0.01
    hidden_sizes = [50, 50]
    omega = 1
    ensemble_size = 10
    activation = F.relu
    no_epochs = 2000
    learning_rate = 1e-3

    directory = './/experiments//1D_cosine_separated'
    data_location = '..//vision//data//1D_COSINE//1d_cosine_separated.pkl'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)
    
    # initialise models
    models = []
    for ensemble_member in range(ensemble_size):
        model = MLP(noise_variance, hidden_sizes, omega, activation=F.relu, input_dim=1, standard_normal_prior=True, random_prior=False)
        model.to(device)
        models.append(model)
        if ensemble_member == 0:
            for param in model.parameters():
                print(type(param.data), param.size())

    # get dataset
    with open(data_location, 'rb') as f:
            data_load = pickle.load(f)

    x_train = torch.Tensor(data_load[:,0])[:, None].cuda()
    y_train = torch.Tensor(data_load[:,1]).cuda()
    trainset_size = x_train.shape[0]

    # train
    for ensemble_member in range(ensemble_size):
        optimizer = optim.Adam(models[ensemble_member].parameters(), lr=learning_rate) # init a new optimizer
        with trange(no_epochs) as epochs:
            for epoch in epochs: # loop over epochs
                loss = models[ensemble_member].get_U(x_train, y_train, trainset_size=trainset_size)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    # plot
    plot_regression(models, directory)

