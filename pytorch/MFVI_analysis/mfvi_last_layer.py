# test the effects of being Bayesian only on the last layer
 
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

# plot the output of regression network
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np

def plot_reg(model, model_type, fig_directory, title='', train_x=None, train_y=None):
    # get training data
    X = data_load[:,0]
    Y = data_load[:,1]
    X = np.float32(X)
    Y = np.float32(Y)

    # evaluate model on test points
    N = 1000 # number of test points
    x_lower = -3 #-1.5
    x_upper = 3 #1.5
    
    test_inputs = np.linspace(x_lower, x_upper, N)
    # move to GPU if available
    test_inputs = torch.FloatTensor(test_inputs)
    test_inputs = test_inputs.cuda() 
    plt.figure()

    # sample from the network then plot +/- 1 standard deviation range
    no_samp = 100   
    plt.plot(X, Y, '+k')

    if model_type == 'VI':
        all_test_outputs = np.zeros((no_samp, N))
        for i in range(no_samp):
            test_outputs = model(test_inputs, no_samples = 1, shared_weights = True) # make all the datapoints in a batch use the same network weights
            # convert back to np array
            test_x = test_inputs.data.cpu().numpy()
            all_test_outputs[i,:] = test_outputs.data.cpu().numpy()
            plt.plot(test_x, all_test_outputs[i,:], linewidth=0.3)
        # calculate mean and variance
        mean = np.mean(all_test_outputs, 0)
        variance = np.mean(all_test_outputs**2, 0) - mean**2
    elif model_type == 'NeuralLinear':
        mean = np.zeros(N)
        variance = np.zeros(N)
        for test_point in range(N):
            single_mean, single_variance = model.prediction(train_x, train_y, test_inputs[test_point])
            mean[test_point], variance[test_point] = single_mean.detach().cpu().numpy(), single_variance.detach().cpu().numpy()
        test_x = test_inputs.data.cpu().numpy()
    else:
        raise ValueError('model_type must be either VI or NeuralLinear')
    # variance = model.noise_variance.detach().cpu().numpy() + np.mean(all_test_outputs**2, 0) - mean**2
    plt.plot(test_x, mean, color='b')
    plt.fill_between(test_x, mean + 2 * np.sqrt(variance), 
            mean - 2 * np.sqrt(variance), color='b', alpha=0.3)
    
    filename = 'regression_' + title + '.pdf'
    filename = os.path.join(fig_directory, filename)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.ylim([-30, 30])
    plt.xlim([-3, 3])
    # plt.ylim([-1, 1])
    # plt.xlim([-1.5, 1.5])
    plt.savefig(filename)
    plt.close() # get rid of this if you want to animate

    # plot just the variance
    plt.figure()
    plt.plot(test_x, variance)
    plt.xlabel('$x$')
    plt.ylabel('predictive variance')
    filename = 'variance_' + title + '.pdf'
    filename = os.path.join(fig_directory, filename)
    plt.savefig(filename)
    plt.close()

    # pickle for posterity
    inputs = test_x
    mean = mean
    sd = np.sqrt(variance)

    pickle_location = os.path.join(fig_directory, title)
    outfile = open(pickle_location, 'wb')
    pickle.dump(inputs, outfile)
    pickle.dump(mean, outfile)
    pickle.dump(sd, outfile)
    outfile.close()

    # if neural linear, plot the correlation and covariance matrices
    if model_type == 'NeuralLinear':
        posterior_covar = model.posterior_covar.data.cpu().numpy()
        fig, ax = plt.subplots()
        im = ax.imshow(np.abs(posterior_covar) , interpolation='nearest', cmap=cm.Greys_r)
        filepath = os.path.join(fig_directory, 'neural_linear_covariance.pdf')
        fig.savefig(filepath)
        plt.close()

        # plot correlation matrix using cov matrix estimate
        variance_vector = np.diag(posterior_covar)
        sd_vector = np.sqrt(variance_vector)
        outer_prod = np.outer(sd_vector, sd_vector)
        correlations = posterior_covar/outer_prod

        fig, ax = plt.subplots()
        im = ax.imshow(correlations , interpolation='nearest')
        fig.colorbar(im)
        filepath = os.path.join(fig_directory, 'neural_linear_correlation.pdf')
        fig.savefig(filepath)
        plt.close()

class MAP_Linear_Layer(nn.Module):
    def __init__(self, n_input, n_output, omega):
        super(MAP_Linear_Layer, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.omega = omega
        # omega is the prior standard deviation (assume isotropic prior)
        prior_logvar = 2*np.log(omega) 
        self.prior_logvar = prior_logvar

        """initialise parameters and priors following 'Neural network ensembles and variational inference revisited', Appendix A"""
        # weight parameters
        self.W = nn.Parameter(torch.cuda.FloatTensor(n_input, n_output).normal_(0, 1/np.sqrt(4*n_output))) # initialisation of weight means
        # bias parameters
        self.b = nn.Parameter(torch.cuda.FloatTensor(n_output).normal_(0, 1e-10)) # initialisation of bias means

    def forward(self, x, no_samples=None, shared_weights=None): # number of samples per forward pass
        """
        input is either (batch_size x no_input), if this is the first layer of the network, or (no_samples x batch_size x no_input), 
        and output is (no_samples x batch_size x no_output)
        """
        batch_size = x.size()[0]
        # sample just one weight matrix and just one bias vector
        b = self.b.expand(batch_size, -1)
        samples_activations = torch.mm(x, self.W) + b
        return samples_activations

    def KL(self): # get L2 regularisation term
        return 0 
        # return 0.5*(1/(self.omega**2))*(torch.sum(self.W**2) + torch.sum(self.b**2)) # assume prior means are all zero!

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
        self.W_mean = nn.Parameter(torch.cuda.FloatTensor(n_input, n_output).normal_(0, 1/np.sqrt(4*n_output))) # initialisation of weight means
        self.W_logvar = nn.Parameter(torch.cuda.FloatTensor(n_input, n_output).normal_(-11.5, 1e-10)) # initialisation of weight logvariances 
        # bias parameters
        self.b_mean = nn.Parameter(torch.cuda.FloatTensor(n_output).normal_(0, 1e-10)) # initialisation of bias means
        self.b_logvar = nn.Parameter(torch.cuda.FloatTensor(n_output).normal_(-11.5, 1e-10)) # initialisation of bias logvariances (why uniform?)
        
        # prior parameters 
        self.W_prior_mean = Variable(torch.zeros(n_input, n_output).cuda())
        self.W_prior_logvar = Variable((prior_logvar*torch.ones(n_input, n_output)).cuda())
        self.b_prior_mean = Variable(torch.zeros(n_output).cuda()) 
        self.b_prior_logvar = Variable((prior_logvar*torch.ones(n_output)).cuda())

        self.num_weights = n_input*n_output + n_output # number of weights and biases
 
    def forward(self, x, no_samples, shared_weights=False): # number of samples per forward pass
        """
        input is either (batch_size x no_input), if this is the first layer of the network, or (no_samples x batch_size x no_input), 
        and output is (no_samples x batch_size x no_output)
        """
        # local reparameterisation trick

        if shared_weights == True: # can't use local reparam trick if we want to sample functions from the network. assume we will only do one test sample at a time
            batch_size = x.size()[0]
            # sample just one weight matrix and just one bias vector
            W_var = torch.exp(self.W_logvar)
            b_var = torch.exp(self.b_logvar)
            z_W = Variable(torch.cuda.FloatTensor(self.n_input, self.n_output).normal_(0, 1))
            z_b = Variable(torch.cuda.FloatTensor(self.n_output).normal_(0, 1))
            W = self.W_mean + torch.mul(torch.sqrt(W_var), z_W)
            b = self.b_mean + torch.mul(torch.sqrt(b_var), z_b)
            b = b.expand(batch_size, -1)
            samples_activations = torch.mm(x, W) + b

        else:
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

class Neural_Linear_Net(nn.Module):
    def __init__(self, noise_variance, hidden_sizes, omega, activation, learned_noise_var=False, input_dim=None, noise_param_init=None):
        super(Neural_Linear_Net, self).__init__()
        self.omega = float(omega)
        self.dim_input = input_dim 
        self.activation = activation
        self.learned_noise_var = learned_noise_var

        if self.learned_noise_var:
            self.noise_var_param = nn.Parameter(torch.cuda.FloatTensor([noise_param_init]))
            self.noise_variance = self.get_noise_var()
        else:
            self.noise_variance = torch.cuda.FloatTensor([float(noise_variance)])

        # create the layers in the network based on params
        self.hidden_sizes = hidden_sizes
        self.linears = nn.ModuleList([MAP_Linear_Layer(self.dim_input, self.hidden_sizes[0], self.omega)])
        self.linears.extend([MAP_Linear_Layer(self.hidden_sizes[i], self.hidden_sizes[i+1], self.omega) for i in range(0, len(self.hidden_sizes)-1)])
        self.linears.append(MAP_Linear_Layer(self.hidden_sizes[-1], 1, self.omega))

    def get_noise_var(self):
        if self.learned_noise_var:
            return torch.exp(self.noise_var_param) # try just a log representation
        else: 
            return self.noise_variance

    def get_KL_term(self):
        # calculate KL divergence between q and the prior for the entire network
        KL_term = 0
        for _, l in enumerate(self.linears):
            KL_term = KL_term + l.KL()
        return KL_term

    def get_U(self, inputs, labels, trainset_size):
        # calculate L2 regularised loss
        self.noise_variance = self.get_noise_var()
        outputs = self.forward(inputs)

        no_samples = outputs.size()[0]
        labels = labels.expand(no_samples, -1)
        const_term = 0.5*torch.log(2*3.141592654*self.get_noise_var())
        reconstruction_loss = (trainset_size)*(const_term + (1/(2*self.get_noise_var()))*torch.mean((labels - outputs)**2))
        # KL_term = self.get_KL_term()

        # U = (reconstruction_loss + KL_term)/trainset_size # per-datapoint ELBO
        U = reconstruction_loss # try MAXIMUM LIKELIHOOD
        return U

    def forward(self, s, shared_weights=False):
        for i, l in enumerate(self.linears):
            s = l(s)
            if i < len(self.linears) - 1:
                s = self.activation(s) 
        # s has dimension (no_samples x batch_size x no_output=1)
        s = s.view(1, -1) # (no_samples x batch_size)
        return s

    def get_last_feature(self, s):
        for i, l in enumerate(self.linears):           
            if i < len(self.linears) - 1: # up to right before the last linear layer
                s = l(s)
                s = self.activation(s)
        return s.squeeze() # might have to squeeze

    def prediction(self, train_x, train_y, test_x): # do Bayesian linear regression on the last layer
        num_data = train_x.shape[0]
        num_features = self.linears[-1].n_input + 1 # 1 for bias parameter

        # construct design matrix
        X = torch.cuda.FloatTensor(num_data, num_features).fill_(0)
        for datapoint in range(train_x.shape[0]):
            X[datapoint, :-1] = self.get_last_feature(train_x[datapoint][None, :]) # hidden features
            X[datapoint, -1] = 1 # bias feature is always 1

        # get test features
        test_feature = torch.cat((self.get_last_feature(test_x[None][None, :]), torch.cuda.FloatTensor([1.0])), dim=0) # add the constant bias feature --- CHECK THIS
        test_feature = test_feature.unsqueeze(1)
        self.posterior_covar = self.get_noise_var() * torch.inverse(self.get_noise_var() / (self.omega**2) * torch.eye(num_features).cuda() + torch.t(X) @ X)
        
        ############## diagonalise the posterior covariance to see if it leads to convexity
        #self.posterior_covar = torch.diag(torch.diag(self.posterior_covar))
        
        predictive_var = torch.t(test_feature) @ self.posterior_covar @ test_feature # this doesn't include observation noise

        # forward pass for the mean
        mean = self.forward(test_x[None][None, :])

        return mean, predictive_var

class MFVI_Net(nn.Module):
    def __init__(self, noise_variance, hidden_sizes, omega, activation, learned_noise_var=False, input_dim=None, noise_param_init=None, last_layer=False):
        super(MFVI_Net, self).__init__()
        self.train_samples = 32 # following Marcin's work means 1 sample
        self.test_samples = 100
        self.omega = float(omega)
        self.dim_input = input_dim 
        self.activation = activation
        self.learned_noise_var = learned_noise_var

        if self.learned_noise_var:
            self.noise_var_param = nn.Parameter(torch.cuda.FloatTensor([noise_param_init]))
            self.noise_variance = self.get_noise_var()
        else:
            self.noise_variance = torch.cuda.FloatTensor([float(noise_variance)])

        # create the layers in the network based on params
        self.hidden_sizes = hidden_sizes
        self.last_layer = last_layer
        if self.last_layer: # only do mean field for the last linear layer
            self.linears = nn.ModuleList([MAP_Linear_Layer(self.dim_input, self.hidden_sizes[0], self.omega)])
            self.linears.extend([MAP_Linear_Layer(self.hidden_sizes[i], self.hidden_sizes[i+1], self.omega) for i in range(0, len(self.hidden_sizes)-1)])
            self.linears.append(MFVI_Linear_Layer(self.hidden_sizes[-1], 1, self.omega))
        else: 
            self.linears = nn.ModuleList([MFVI_Linear_Layer(self.dim_input, self.hidden_sizes[0], self.omega)])
            self.linears.extend([MFVI_Linear_Layer(self.hidden_sizes[i], self.hidden_sizes[i+1], self.omega) for i in range(0, len(self.hidden_sizes)-1)])
            self.linears.append(MFVI_Linear_Layer(self.hidden_sizes[-1], 1, self.omega))

    def get_noise_var(self):
        if self.learned_noise_var:
            return torch.exp(self.noise_var_param) # try just a log representation
        else: 
            return self.noise_variance
    
    def get_KL_term(self):
        # calculate KL divergence between q and the prior for the entire network
        KL_term = 0
        for _, l in enumerate(self.linears):
            KL_term = KL_term + l.KL()
        return KL_term

    def get_U(self, inputs, labels, trainset_size):
        # calculate stochastic estimate of the per-datapoint ELBO
        self.noise_variance = self.get_noise_var()
        outputs = self.forward(inputs, self.train_samples)

        no_samples = outputs.size()[0]
        labels = labels.expand(no_samples, -1)
        const_term = 0.5*torch.log(2*3.141592654*self.get_noise_var())
        reconstruction_loss = (trainset_size)*(const_term + (1/(2*self.get_noise_var()))*torch.mean((labels - outputs)**2))
        KL_term = self.get_KL_term()

        U = (reconstruction_loss + KL_term)/trainset_size # per-datapoint ELBO
        return U

    def forward(self, s, no_samples, shared_weights=False):
        s = s.view(-1, self.dim_input)
        for i, l in enumerate(self.linears):
            s = l(s, no_samples = no_samples, shared_weights=shared_weights)
            if i < len(self.linears) - 1:
                s = self.activation(s) 
        # s has dimension (no_samples x batch_size x no_output=1)
        s = s.view(no_samples, -1) # (no_samples x batch_size)
        return s

if __name__ == "__main__":

    # set RNG
    np.random.seed(0) # 0
    torch.manual_seed(231) #  230

    # hyperparameters
    noise_variance = 0.01 #* (0.25)**2 
    hidden_sizes = [50]
    omega = 4
    activation = F.relu
    no_epochs = 10000
    learning_rate = 1e-3

    directory = './/experiments//last_layer'
    data_location = '..//vision//data//1D_COSINE//1d_cosine_separated.pkl'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)
    
    # initialise models
    # model = MFVI_Net(noise_variance, hidden_sizes, omega, activation, input_dim=1, last_layer=True)
    model = Neural_Linear_Net(noise_variance, hidden_sizes, omega, activation, input_dim=1)
    for param in model.parameters():
        print(type(param.data), param.size())

    # get dataset
    with open(data_location, 'rb') as f:
            data_load = pickle.load(f)

    ##### scaling the data
    # data_load[:,0] *= 0.5
    # data_load[:,1] *= 0.25

    x_train = torch.Tensor(data_load[:,0])[:, None].cuda()
    y_train = torch.Tensor(data_load[:,1]).cuda()
    trainset_size = x_train.shape[0]

    # train
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    with trange(no_epochs) as epochs:
        for epoch in epochs: # loop over epochs
            loss = model.get_U(x_train, y_train, trainset_size=trainset_size)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # plot
    # plot_reg(model, 'VI', directory, title='mfvi_last_layer_3HL')
    plot_reg(model, 'NeuralLinear', directory, title='neural_linear_21_6_19', train_x=x_train, train_y=y_train)