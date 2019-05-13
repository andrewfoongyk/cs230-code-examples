"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import collections
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math

# activation functions
def bump(x):
    return torch.exp(-x**2)*F.relu(2-torch.abs(x)) # truncated bump (untrainable?)

def retanh(x):
    return torch.tanh(x)*F.relu(x)

class FCVI_Net(nn.Module):
    def __init__(self, params, prior_init=False): 
        super(FCVI_Net, self).__init__() 
        self.train_samples = params.train_samples
        self.test_samples = params.test_samples
        self.dataset = params.dataset
        self.activation_name = params.activation
        self.hidden_sizes = params.hidden_sizes
        self.omega = params.omega
        if params.activation == 'relu':
            self.activation = F.relu
        elif params.activation == 'tanh':
            self.activation = torch.tanh
        elif params.activation == 'bump':
            self.activation = bump
        elif params.activation == 'prelu':
            self.prelu_weight = nn.Parameter(torch.Tensor([0.25]))
            self.activation = F.prelu
        elif params.activation == 'sine':
            self.activation = torch.sin
        elif params.activation == 'sigmoid':
            self.activation = F.sigmoid

        # adjust input and output size depending on dataset used and read output noise
        if params.dataset == 'mnist':
            self.input_channels = 1
            self.input_size = 28*28
            self.output_size = 10
        elif params.dataset == '1d_cosine' or params.dataset == 'prior_dataset':
            self.input_channels = 1
            self.input_size = 1
            self.output_size = 1
            self.noise_variance = params.noise_variance
        elif params.dataset == 'signs':
            self.input_channels = 3
            self.input_size = 64*64
            self.output_size = 6
        
        # count the number of parameters (weights and biases)
        self.no_params = self.count_params()
        # initialise the prior
        self.init_prior()
        # initialise covariance (in vector form) and means
        self.init_params(prior_init)

    def forward(self, x, no_samples, shared_weights=False): 
        batch_size = x.size()[0] # x has dimensions (batch_size x no_inputs)
        # initialise empty cholesky matrix
        L_init = torch.cuda.FloatTensor(self.no_params, self.no_params).fill_(0)
        # use cov_vector to fill in L
        L_init[np.tril_indices(self.no_params)] = self.cov_vector
        # exponentiate the diagonal
        L_diag = torch.diag(L_init)
        L_diag_mat = torch.diag(L_diag)
        self.L = L_init - L_diag_mat + torch.diag(torch.exp(L_diag)) # cholesky decomposition
        self.Sigma = torch.matmul(self.L, torch.t(self.L)) # covariance matrix
        # sample all parameters
        samples = self.get_samples(self.L, no_samples, batch_size, shared_weights)
        # unpack weights and biases
        weights, biases = self.unpack_samples(samples, no_samples, batch_size)
        # self.weights = weights #### for printing
        # self.biases = biases

        # forward propagate
        activations = x.expand(self.input_channels*self.input_size, -1) # expand in first layer
        for i in range(len(weights)-1): # all but the last weight matrix and bias
            if i == 0: # first layer 
                activities = torch.einsum('ib,iosb->osb', (activations, weights[i])) + biases[i]
            else: 
                activities = torch.einsum('isb,iosb->osb', (activations, weights[i])) + biases[i]
            activations = self.activation(activities) # apply nonlinearity
        output = torch.einsum('isb,iosb->osb', (activations, weights[-1])) + biases[-1] # final linear layer and bias
        output = output.view(no_samples, batch_size)
        return output

    def get_KL_term(self):
        # calculate KL divergence between q and the prior for the entire network
        const_term = -self.no_params
        logdet_prior = torch.sum(torch.log(self.prior_variance_vector))
        logdet_q = 2*torch.sum(torch.log(torch.diag(self.L)))
        logdet_term = logdet_prior - logdet_q
        prior_cov_inv = torch.diag(1/self.prior_variance_vector)
        trace_term = torch.trace(torch.matmul(prior_cov_inv, self.Sigma)) # is there a more efficient way to do this?
        mu_diff = self.prior_mean - self.mean
        quad_term = torch.matmul(mu_diff, torch.matmul(prior_cov_inv, mu_diff)) # and this?
        kl = 0.5*(const_term + logdet_term + trace_term + quad_term)
        return kl

    def return_weights(self):
        """return the marginal statistics of the weights and biases of the network"""
        # return a list of lists - the first list goes through layers, and each item is a list that has means in 0th place and S.D.'s in 1st place
        weights = []
        biases = []
        priors = []

        start_index = 0
        end_index = 0  

        diag_variances = torch.diag(self.Sigma)
        diag_sd = torch.sqrt(diag_variances)

        # unpack first weight matrix and bias
        layer_weights = []
        layer_biases = []
        end_index = end_index + self.input_channels*self.input_size*self.hidden_sizes[0]
        weight_mean_vector = self.mean[start_index:end_index]
        weight_mean_matrix = weight_mean_vector.view(self.input_channels*self.input_size, self.hidden_sizes[0])
        weight_sd_vector = diag_sd[start_index:end_index]
        weight_sd_matrix = weight_sd_vector.view(self.input_channels*self.input_size, self.hidden_sizes[0])

        start_index = start_index + self.input_channels*self.input_size*self.hidden_sizes[0]
        end_index = end_index + self.hidden_sizes[0]

        bias_mean_vector = self.mean[start_index:end_index]
        bias_sd_vector = diag_sd[start_index:end_index]

        layer_weights.append(weight_mean_matrix.cpu().detach().numpy())
        layer_biases.append(bias_mean_vector.cpu().detach().numpy())
        layer_weights.append(weight_sd_matrix.cpu().detach().numpy())
        layer_biases.append(bias_sd_vector.cpu().detach().numpy())
        weights.append(layer_weights)
        biases.append(layer_biases)
        n_input = self.input_channels*self.input_size
        priors.append(self.omega/np.sqrt(n_input))

        start_index = start_index + self.hidden_sizes[0]

        for i in range(len(self.hidden_sizes)-1):
            layer_weights = []
            layer_biases = [] 
            end_index = end_index + self.hidden_sizes[i]*self.hidden_sizes[i+1]

            weight_mean_vector = self.mean[start_index:end_index]
            weight_mean_matrix = weight_mean_vector.view(self.hidden_sizes[i], self.hidden_sizes[i+1])
            weight_sd_vector = diag_sd[start_index:end_index]
            weight_sd_matrix = weight_sd_vector.view(self.hidden_sizes[i], self.hidden_sizes[i+1])

            start_index = start_index + self.hidden_sizes[i]*self.hidden_sizes[i+1]
            end_index = end_index + self.hidden_sizes[i+1]

            bias_mean_vector = self.mean[start_index:end_index]
            bias_sd_vector = diag_sd[start_index:end_index]

            layer_weights.append(weight_mean_matrix.cpu().detach().numpy())
            layer_biases.append(bias_mean_vector.cpu().detach().numpy())
            layer_weights.append(weight_sd_matrix.cpu().detach().numpy())
            layer_biases.append(bias_sd_vector.cpu().detach().numpy())
            weights.append(layer_weights)
            biases.append(layer_biases)
            n_input = self.hidden_sizes[i]
            priors.append(self.omega/np.sqrt(n_input))

            start_index = start_index + self.hidden_sizes[i+1]

        # unpack output weight matrix and bias
        layer_weights = []
        layer_biases = []
        end_index = end_index + self.hidden_sizes[-1]*self.output_size
        
        weight_mean_vector = self.mean[start_index:end_index]
        weight_mean_matrix = weight_mean_vector.view(self.hidden_sizes[-1], self.output_size)
        weight_sd_vector = diag_sd[start_index:end_index]
        weight_sd_matrix = weight_sd_vector.view(self.hidden_sizes[-1], self.output_size)

        start_index = start_index + self.hidden_sizes[-1]*self.output_size

        bias_mean_vector = self.mean[start_index:] # should reach the end of the vector
        bias_sd_vector = diag_sd[start_index:]

        layer_weights.append(weight_mean_matrix.cpu().detach().numpy())
        layer_biases.append(bias_mean_vector.cpu().detach().numpy())
        layer_weights.append(weight_sd_matrix.cpu().detach().numpy())
        layer_biases.append(bias_sd_vector.cpu().detach().numpy())
        weights.append(layer_weights)
        biases.append(layer_biases)
        n_input = self.hidden_sizes[-1]
        priors.append(self.omega/np.sqrt(n_input))

        return weights, biases, priors

    def unpack_samples(self, samples, no_samples, batch_size):
        start_index = 0
        end_index = 0    
        weights = []
        biases = []
        # unpack first weight matrix and bias
        end_index = end_index + self.input_channels*self.input_size*self.hidden_sizes[0]
        weight_vector = samples[start_index:end_index, :, :]
        weight_matrix = weight_vector.view(self.input_channels*self.input_size, self.hidden_sizes[0], no_samples, batch_size)
        weights.append(weight_matrix)
        start_index = start_index + self.input_channels*self.input_size*self.hidden_sizes[0]
        end_index = end_index + self.hidden_sizes[0]
        biases_vector = samples[start_index:end_index, :, :]
        biases.append(biases_vector)
        start_index = start_index + self.hidden_sizes[0]
        for i in range(len(self.hidden_sizes)-1): 
            end_index = end_index + self.hidden_sizes[i]*self.hidden_sizes[i+1]
            weight_vector = samples[start_index:end_index, :, :]
            weight_matrix = weight_vector.view(self.hidden_sizes[i], self.hidden_sizes[i+1], no_samples, batch_size)
            weights.append(weight_matrix)
            start_index = start_index + self.hidden_sizes[i]*self.hidden_sizes[i+1]
            end_index = end_index + self.hidden_sizes[i+1]
            biases_vector = samples[start_index:end_index, :, :]
            biases.append(biases_vector)
            start_index = start_index + self.hidden_sizes[i+1]
        # unpack output weight matrix and bias
        end_index = end_index + self.hidden_sizes[-1]*self.output_size
        weight_vector = samples[start_index:end_index, :, :]
        weight_matrix = weight_vector.view(self.hidden_sizes[-1], self.output_size, no_samples, batch_size)
        weights.append(weight_matrix)
        start_index = start_index + self.hidden_sizes[-1]*self.output_size
        biases_vector = samples[start_index:, :, :] # should reach the end of the parameters vector at this point
        biases.append(biases_vector)
        return weights, biases

    def get_samples(self, L, no_samples, batch_size, shared_weights=False): # return samples of all the parameters, (no_params x no_samples x batch_size)
        if shared_weights == True:
            z = Variable(torch.cuda.FloatTensor(self.no_params, no_samples).normal_(0, 1))
            z = z.expand(batch_size, -1, -1)
            z = z.permute(1,2,0)
        else:
            z = Variable(torch.cuda.FloatTensor(self.no_params, no_samples, batch_size).normal_(0, 1))
        means = self.mean.expand(no_samples, batch_size, -1)
        means = means.permute(2,0,1)
        params_samples = means + torch.einsum('ab,bcd->acd', (L, z)) ######## check this
        return params_samples

    def init_params(self, prior_init=False): 
        no_cov_params = self.no_params*(self.no_params + 1)/2        
        vec = np.zeros(int(no_cov_params))
        ind = 1
        for i in range(1, self.no_params + 1): 
            vec[ind-1] = -3 # initialise diagonals to tiny variance - the diagonals in logspace, the off diagonals in linear space
            ind += (i+1)
        self.cov_vector = nn.Parameter(torch.Tensor(vec).cuda()) 
        self.mean = nn.Parameter(torch.Tensor(self.no_params).normal_(0, 1e-1).cuda())

    def init_prior(self):
        self.prior_mean = Variable(torch.zeros(self.no_params).cuda()) # zero mean for all weights and biases
        prior_variance_vector = np.ones(self.no_params)
        # apply omega scaling to the weight matrices only
        start_index = 0
        end_index = 0
        end_index = end_index + self.input_channels*self.input_size*self.hidden_sizes[0]
        prior_variance_vector[start_index: end_index] = self.omega**2/(self.input_channels*self.input_size)
        start_index = start_index + self.input_channels*self.input_size*self.hidden_sizes[0]
        end_index = end_index + self.hidden_sizes[0]
        start_index = start_index + self.hidden_sizes[0]
        for i in range(len(self.hidden_sizes)-1): 
            end_index = end_index + self.hidden_sizes[i]*self.hidden_sizes[i+1]
            prior_variance_vector[start_index: end_index] = self.omega**2/self.hidden_sizes[i]
            start_index = start_index + self.hidden_sizes[i]*self.hidden_sizes[i+1]
            end_index = end_index + self.hidden_sizes[i+1]
            start_index = start_index + self.hidden_sizes[i+1]        
        end_index = end_index + self.hidden_sizes[-1]*self.output_size
        prior_variance_vector[start_index: end_index] = self.omega**2/self.hidden_sizes[-1]
        self.prior_variance_vector = Variable(torch.Tensor(prior_variance_vector).cuda()) # vector of prior variances

    def count_params(self):
        no_params = self.input_channels*self.input_size*self.hidden_sizes[0] # first weight matrix
        for i in range(len(self.hidden_sizes)-1):
            no_params = no_params + self.hidden_sizes[i] + self.hidden_sizes[i]*self.hidden_sizes[i+1]
        no_params = no_params + self.hidden_sizes[-1] + self.hidden_sizes[-1]*self.output_size + self.output_size # final weight matrix and last 2 biases
        return no_params

# class Fixed_Mean_VI_Linear_Layer(nn.Module):
#     def __init__(self, n_input, n_output, omega, fixed_W, fixed_b, prior_init=False):
#         super(Fixed_Mean_VI_Linear_Layer, self).__init__()
#         self.n_input = n_input
#         self.n_output = n_output
#         # scale the prior with no. of hidden units, corresponds to radford neal's omega
#         prior_logvar = 2*np.log(omega) - np.log(n_input)
#         self.prior_logvar = prior_logvar

#         """initialise variance parameters following 'Neural network ensembles and variational inference revisited', Appendix A; 
#         keep the means fixed - use weights of pretrained network"""
#         # weight parameters
#         self.W_mean = fixed_W # initialisation of weight means
#         self.W_mean.requires_grad = False # keep these fixed
#         self.W_logvar = nn.Parameter(torch.Tensor(n_input, n_output).normal_(-11.5 + prior_logvar, 1e-10)) # initialisation of weight logvariances 
#         # bias parameters
#         self.b_mean = fixed_b # initialisation of bias means
#         self.b_mean.requires_grad = False # keep these fixed
#         self.b_logvar = nn.Parameter(torch.Tensor(n_output).normal_(-11.5, 1e-10)) # initialisation of bias logvariances 
        
#         # prior parameters 
#         self.W_prior_mean = Variable(torch.zeros(n_input, n_output).cuda())
#         self.W_prior_logvar = Variable((prior_logvar*torch.ones(n_input, n_output)).cuda())
#         self.b_prior_mean = Variable(torch.zeros(n_output).cuda())
#         self.b_prior_logvar = Variable((prior_logvar*torch.ones(n_output)).cuda())

#         if prior_init == True: # initialise parameters to their prior values
#             self.W_mean = nn.Parameter(torch.zeros(n_input, n_output))
#             self.W_logvar = nn.Parameter(prior_logvar*torch.ones(n_input, n_output))
#             self.b_mean = nn.Parameter(torch.zeros(n_output))
#             self.b_logvar = nn.Parameter(prior_logvar*torch.ones(n_output))

#         self.num_weights = n_input*n_output + n_output # number of weights and biases
 
#     def forward(self, x, no_samples, shared_weights): # number of samples per forward pass
#         """
#         input is either (batch_size x no_input), if this is the first layer of the network, or (no_samples x batch_size x no_input), 
#         and output is (no_samples x batch_size x no_output)
#         """
#         # local reparameterisation trick

#         if shared_weights == True: # can't use local reparam trick if we want to sample functions from the network. assume we will only do one test sample at a time
#             batch_size = x.size()[0]
#             # sample just one weight matrix and just one bias vector
#             W_var = torch.exp(self.W_logvar)
#             b_var = torch.exp(self.b_logvar)
#             z_W = Variable(torch.Tensor(self.n_input, self.n_output).normal_(0, 1).cuda())
#             z_b = Variable(torch.Tensor(self.n_output).normal_(0, 1).cuda())
#             W = self.W_mean + torch.mul(torch.sqrt(W_var), z_W)
#             b = self.b_mean + torch.mul(torch.sqrt(b_var), z_b)
#             b = b.expand(batch_size, -1)
#             samples_activations = torch.mm(x, W) + b

#         else:
#             # find out if this is the first layer of the network. if it is, perform an expansion to no_samples
#             if len(x.shape) == 2: 
#                 batch_size = x.size()[0]
#                 z = self.get_random(no_samples, batch_size)
#                 gamma = torch.mm(x, self.W_mean) + self.b_mean.expand(batch_size, -1)
#                 W_var = torch.exp(self.W_logvar) 
#                 b_var = torch.exp(self.b_logvar)
#                 delta = torch.mm(x**2, W_var) + b_var.expand(batch_size, -1)
#                 sqrt_delta = torch.sqrt(delta) 
#                 samples_gamma = gamma.expand(no_samples, -1, -1)
#                 samples_sqrt_delta = sqrt_delta.expand(no_samples, -1, -1)
#                 samples_activations = samples_gamma + torch.mul(samples_sqrt_delta, z)

#             elif len(x.shape) == 3:
#                 batch_size = x.size()[1]
#                 z = self.get_random(no_samples, batch_size)
#                 # samples_gamma has different values for each sample, so has dimensions (no_samples x batch_size x no_outputs)
#                 samples_gamma = torch.matmul(x, self.W_mean) + self.b_mean.expand(no_samples, batch_size, -1)
#                 W_var = torch.exp(self.W_logvar)
#                 b_var = torch.exp(self.b_logvar)
#                 # delta has different values for each sample, so has dimensions (no_samples x batch_size x no_outputs)
#                 delta = torch.matmul(x**2, W_var) + b_var.expand(no_samples, batch_size, -1)
#                 samples_sqrt_delta = torch.sqrt(delta)
#                 samples_activations = samples_gamma + torch.mul(samples_sqrt_delta, z)
       
#         return samples_activations

#     def get_shared_random(self, no_samples, batch_size):
#         z = Variable(torch.Tensor(no_samples, self.n_output).normal_(0, 1).cuda())
#         z = z.expand(batch_size, -1, -1)
#         return torch.transpose(z, 0, 1)

#     def get_random(self, no_samples, batch_size):
#         return Variable(torch.Tensor(no_samples, batch_size, self.n_output).normal_(0, 1).cuda()) # standard normal noise matrix

#     def KL(self): # get KL between q and prior for this layer
#         # W_KL = 0.5*(- self.W_logvar + torch.exp(self.W_logvar) + (self.W_mean)**2)
#         # b_KL = 0.5*(- self.b_logvar + torch.exp(self.b_logvar) + (self.b_mean)**2)
#         W_KL = 0.5*(self.W_prior_logvar - self.W_logvar + (torch.exp(self.W_logvar) + (self.W_mean - self.W_prior_mean)**2)/torch.exp(self.W_prior_logvar))
#         b_KL = 0.5*(self.b_prior_logvar - self.b_logvar + (torch.exp(self.b_logvar) + (self.b_mean - self.b_prior_mean)**2)/torch.exp(self.b_prior_logvar))
#         return W_KL.sum() + b_KL.sum() - 0.5*self.num_weights

# class Fixed_Mean_VI_Net(nn.Module):
#     def __init__(self, params, map_model, prior_init=False):
#         super(Fixed_Mean_VI_Net, self).__init__()
#         self.train_samples = params.train_samples
#         self.test_samples = params.test_samples
#         self.dataset = params.dataset
#         # self.prior_logvar = params.prior_logvar
#         self.omega = params.omega
#         self.activation_name = params.activation
#         if params.activation == 'relu':
#             self.activation = F.relu
#         elif params.activation == 'tanh':
#             self.activation = torch.tanh
#         elif params.activation == 'bump':
#             self.activation = bump
#         elif params.activation == 'prelu':
#             self.prelu_weight = nn.Parameter(torch.Tensor([0.25]))
#             self.activation = F.prelu
#         elif params.activation == 'sine':
#             self.activation = torch.sin
        
#          # adjust input and output size depending on dataset used and read output noise
#         if params.dataset == 'mnist':
#             self.input_channels = 1
#             self.input_size = 28*28
#             self.output_size = 10
#         elif params.dataset == '1d_cosine' or params.dataset == 'prior_dataset':
#             self.input_channels = 1
#             self.input_size = 1
#             self.output_size = 1
#             self.noise_variance = params.noise_variance
#         elif params.dataset == 'signs':
#             self.input_channels = 3
#             self.input_size = 64*64
#             self.output_size = 6

#         # extract weights from MAP model
#         map_weights = []
#         map_biases = []
#         for i, l in enumerate(map_model.linears):
#             map_weights.append(l.weight)
#             map_biases.append(l.bias)

#         # create the layers in the network based on params
#         self.hidden_sizes = params.hidden_sizes
#         self.linears = nn.ModuleList([Fixed_Mean_VI_Linear_Layer(self.input_size*self.input_channels, self.hidden_sizes[0], self.omega, map_weights[0], map_biases[0], prior_init)])
#         self.linears.extend([Fixed_Mean_VI_Linear_Layer(self.hidden_sizes[i], self.hidden_sizes[i+1], self.omega, map_weights[i+1], map_biases[i+1], prior_init) for i in range(0, lpip3 fren(self.hidden_sizes)-1)])
#         self.linears.append(Fixed_Mean_VI_Linear_Layer(self.hidden_sizes[-1], self.output_size, self.omega, map_weights[-1], map_biases[-1], prior_init))

#     def get_KL_term(self):
#         # calculate KL divergence between q and the prior for the entire network
#         KL_term = 0
#         for _, l in enumerate(self.linears):
#             KL_term = KL_term + l.KL()
#         return KL_term

#     def forward(self, s, no_samples, shared_weights=False):
#         s = s.view(-1, self.input_size*self.input_channels)
#         for i, l in enumerate(self.linears):
#             s = l(s, no_samples = no_samples, shared_weights = shared_weights)
#             if i < len(self.linears) - 1:
#                 if self.activation_name == 'prelu':
#                     s = self.activation(s, self.prelu_weight)
#                 else:
#                     s = self.activation(s) 
#         if self.dataset == '1d_cosine' or self.dataset == 'prior_dataset': # make this more flexible
#             # s has dimension (no_samples x batch_size x no_output=1)
#             s = s.view(no_samples, -1) # (no_samples x batch_size)
#             return s
#         else:
#             s = F.log_softmax(s, dim=2) # dimension (no_samples x batch_size x no_output)       
#             return torch.mean(s, 0) # taking the expectation, dimension (batch_size x no_output)

#     def return_weights(self):
#         """return the weights and biases of the network"""
#         # return a list of lists - the first list goes through layers, and each item is a list that has means in 0th place and S.D.'s in 1st place
#         weights = []
#         biases = []
#         priors = []
#         for _, l in enumerate(self.linears):
#             layer_weights = []
#             layer_biases = []
#             layer_weights.append(l.W_mean.cpu().detach().numpy())
#             layer_biases.append(l.b_mean.cpu().detach().numpy())
#             layer_weights.append(np.sqrt(np.exp(l.W_logvar.cpu().detach().numpy())))
#             layer_biases.append(np.sqrt(np.exp(l.b_logvar.cpu().detach().numpy())))
#             weights.append(layer_weights)
#             biases.append(layer_biases)
#             n_input = l.W_mean.size()[0]
#             priors.append(self.omega/np.sqrt(n_input))
#         return weights, biases, priors

# class Weight_Noise_Layer(nn.Module):
#     def __init__(self, n_input, n_output, prior_logvar, prior_init=False):
#         super(Weight_Noise_Layer, self).__init__()
#         self.n_input = n_input
#         self.n_output = n_output
#         # scale the prior with no. of hidden units #####################
#         omega = 2 # prior scale factor - corresponds to radford neal's omega
#         prior_logvar = 2*np.log(omega) -np.log(n_input)
#         self.prior_logvar = prior_logvar

#         """initialise parameters and priors following 'Neural network ensembles and variational inference revisited', Appendix A"""
#         # weight parameters
#         self.W_mean = nn.Parameter(torch.Tensor(n_input, n_output).normal_(0, 1/np.sqrt(4*n_output))) # initialisation of weight means
#         self.W_logvar = nn.Parameter(torch.Tensor(1).normal_(-11.5, 1e-10)) # initialisation of weight logvariance - tied 
#         # bias parameters
#         self.b_mean = nn.Parameter(torch.Tensor(n_output).normal_(0, 1e-10)) # initialisation of bias means
#         self.b_logvar = nn.Parameter(torch.Tensor(1).normal_(-11.5, 1e-10)) # initialisation of bias logvariance - tied 
        
#         # prior parameters 
#         self.W_prior_mean = Variable(torch.zeros(n_input, n_output).cuda())
#         self.W_prior_logvar = Variable((prior_logvar*torch.ones(n_input, n_output)).cuda())
#         self.b_prior_mean = Variable(torch.zeros(n_output).cuda())
#         self.b_prior_logvar = Variable((prior_logvar*torch.ones(n_output)).cuda())

#         if prior_init == True: # initialise parameters to their prior values
#             self.W_mean = nn.Parameter(torch.zeros(n_input, n_output))
#             self.W_logvar = nn.Parameter(prior_logvar*torch.ones(1))
#             self.b_mean = nn.Parameter(torch.zeros(n_output))
#             self.b_logvar = nn.Parameter(prior_logvar*torch.ones(1))

#         self.num_weights = n_input*n_output + n_output # number of weights and biases
 
#     def forward(self, x, no_samples, shared_weights): # number of samples per forward pass
#         """
#         input is either (batch_size x no_input), if this is the first layer of the network, or (no_samples x batch_size x no_input), 
#         and output is (no_samples x batch_size x no_output)
#         """
#         # local reparameterisation trick

#         if shared_weights == True: # can't use local reparam trick if we want to sample functions from the network. assume we will only do one test sample at a time
#             batch_size = x.size()[0]
#             # sample just one weight matrix and just one bias vector
#             W_var = torch.exp(self.W_logvar).expand(self.n_input, self.n_output)
#             b_var = torch.exp(self.b_logvar).expand(self.n_output)
#             z_W = Variable(torch.Tensor(self.n_input, self.n_output).normal_(0, 1).cuda())
#             z_b = Variable(torch.Tensor(self.n_output).normal_(0, 1).cuda())
#             W = self.W_mean + torch.mul(torch.sqrt(W_var), z_W)
#             b = self.b_mean + torch.mul(torch.sqrt(b_var), z_b)
#             b = b.expand(batch_size, -1)
#             samples_activations = torch.mm(x, W) + b

#         else:
#             # find out if this is the first layer of the network. if it is, perform an expansion to no_samples
#             if len(x.shape) == 2: 
#                 batch_size = x.size()[0]
#                 z = self.get_random(no_samples, batch_size)
#                 gamma = torch.mm(x, self.W_mean) + self.b_mean.expand(batch_size, -1)
#                 W_var = torch.exp(self.W_logvar).expand(self.n_input, self.n_output)
#                 b_var = torch.exp(self.b_logvar).expand(self.n_output)
#                 delta = torch.mm(x**2, W_var) + b_var.expand(batch_size, -1)
#                 sqrt_delta = torch.sqrt(delta) 
#                 samples_gamma = gamma.expand(no_samples, -1, -1)
#                 samples_sqrt_delta = sqrt_delta.expand(no_samples, -1, -1)
#                 samples_activations = samples_gamma + torch.mul(samples_sqrt_delta, z)

#             elif len(x.shape) == 3:
#                 batch_size = x.size()[1]
#                 z = self.get_random(no_samples, batch_size)
#                 # samples_gamma has different values for each sample, so has dimensions (no_samples x batch_size x no_outputs)
#                 samples_gamma = torch.matmul(x, self.W_mean) + self.b_mean.expand(no_samples, batch_size, -1)
#                 W_var = torch.exp(self.W_logvar).expand(self.n_input, self.n_output)
#                 b_var = torch.exp(self.b_logvar).expand(self.n_output)
#                 # delta has different values for each sample, so has dimensions (no_samples x batch_size x no_outputs)
#                 delta = torch.matmul(x**2, W_var) + b_var.expand(no_samples, batch_size, -1)
#                 samples_sqrt_delta = torch.sqrt(delta)
#                 samples_activations = samples_gamma + torch.mul(samples_sqrt_delta, z)
       
#         return samples_activations

#     def get_shared_random(self, no_samples, batch_size):
#         z = Variable(torch.Tensor(no_samples, self.n_output).normal_(0, 1).cuda())
#         z = z.expand(batch_size, -1, -1)
#         return torch.transpose(z, 0, 1)

#     def get_random(self, no_samples, batch_size):
#         return Variable(torch.Tensor(no_samples, batch_size, self.n_output).normal_(0, 1).cuda()) # standard normal noise matrix

#     def KL(self): # get KL between q and prior for this layer
#         # W_KL = 0.5*(- self.W_logvar + torch.exp(self.W_logvar) + (self.W_mean)**2)
#         # b_KL = 0.5*(- self.b_logvar + torch.exp(self.b_logvar) + (self.b_mean)**2)
#         W_KL = 0.5*(self.W_prior_logvar - self.W_logvar + (torch.exp(self.W_logvar) + (self.W_mean - self.W_prior_mean)**2)/torch.exp(self.W_prior_logvar))
#         b_KL = 0.5*(self.b_prior_logvar - self.b_logvar + (torch.exp(self.b_logvar) + (self.b_mean - self.b_prior_mean)**2)/torch.exp(self.b_prior_logvar))
#         return W_KL.sum() + b_KL.sum() - 0.5*self.num_weights

# class Weight_Noise_Net(nn.Module):
#     def __init__(self, params, prior_init=False):
#         super(Weight_Noise_Net, self).__init__()
#         self.train_samples = params.train_samples
#         self.test_samples = params.test_samples
#         self.dataset = params.dataset
#         self.prior_logvar = params.prior_logvar
#         self.activation_name = params.activation
#         if params.activation == 'relu':
#             self.activation = F.relu
#         elif params.activation == 'tanh':
#             self.activation = torch.tanh
#         elif params.activation == 'bump':
#             self.activation = bump
#         elif params.activation == 'prelu':
#             self.prelu_weight = nn.Parameter(torch.Tensor([0.25]))
#             self.activation = F.prelu
#         elif params.activation == 'sine':
#             self.activation = torch.sin
        
#          # adjust input and output size depending on dataset used and read output noise
#         if params.dataset == 'mnist':
#             self.input_channels = 1
#             self.input_size = 28*28
#             self.output_size = 10
#         elif params.dataset == '1d_cosine' or params.dataset == 'prior_dataset':
#             self.input_channels = 1
#             self.input_size = 1
#             self.output_size = 1
#             self.noise_variance = params.noise_variance
#         elif params.dataset == 'signs':
#             self.input_channels = 3
#             self.input_size = 64*64
#             self.output_size = 6

#         # create the layers in the network based on params
#         self.hidden_sizes = params.hidden_sizes
#         self.linears = nn.ModuleList([Weight_Noise_Layer(self.input_size*self.input_channels, self.hidden_sizes[0], self.prior_logvar, prior_init)])
#         self.linears.extend([Weight_Noise_Layer(self.hidden_sizes[i], self.hidden_sizes[i+1], self.prior_logvar, prior_init) for i in range(0, len(self.hidden_sizes)-1)])
#         self.linears.append(Weight_Noise_Layer(self.hidden_sizes[-1], self.output_size, self.prior_logvar, prior_init))

#     def get_KL_term(self):
#         # calculate KL divergence between q and the prior for the entire network
#         KL_term = 0
#         for _, l in enumerate(self.linears):
#             KL_term = KL_term + l.KL()
#         return KL_term

#     def forward(self, s, no_samples, shared_weights=False):
#         s = s.view(-1, self.input_size*self.input_channels)
#         for i, l in enumerate(self.linears):
#             s = l(s, no_samples = no_samples, shared_weights = shared_weights)
#             if i < len(self.linears) - 1:
#                 if self.activation_name == 'prelu':
#                     s = self.activation(s, self.prelu_weight)
#                 else:
#                     s = self.activation(s) 
#         if self.dataset == '1d_cosine' or self.dataset == 'prior_dataset': # make this more flexible
#             # s has dimension (no_samples x batch_size x no_output=1)
#             s = s.view(no_samples, -1) # (no_samples x batch_size)
#             return s
#         else:
#             s = F.log_softmax(s, dim=2) # dimension (no_samples x batch_size x no_output)       
#             return torch.mean(s, 0) # taking the expectation, dimension (batch_size x no_output)

#     def return_weights(self):
#         """return the weights and biases of the network"""
#         # return a list of lists - the first list goes through layers, and each item is a list that has means in 0th place and S.D.'s in 1st place
#         weights = []
#         biases = []
#         priors = []
#         for _, l in enumerate(self.linears):
#             layer_weights = []
#             layer_biases = []
#             layer_weights.append(l.W_mean.cpu().detach().numpy())
#             layer_biases.append(l.b_mean.cpu().detach().numpy())

#             # read and expand the S.D.'s
#             weight_sd = np.sqrt(np.exp(l.W_logvar.cpu().detach().numpy()))
#             weight_sd = np.tile(weight_sd, (l.n_input, l.n_output))
#             layer_weights.append(weight_sd)
#             bias_sd = np.sqrt(np.exp(l.b_logvar.cpu().detach().numpy()))
#             bias_sd = np.tile(bias_sd, (l.n_output))
#             layer_biases.append(bias_sd)

#             weights.append(layer_weights)
#             biases.append(layer_biases)
#             priors.append(np.exp(l.prior_logvar/2))
#         return weights, biases, priors

# class MFVI_Prebias_Layer(nn.Module):
#     def __init__(self, n_input, n_output, prior_logvar, prior_init=False):
#         super(MFVI_Prebias_Layer, self).__init__()
#         self.n_input = n_input
#         self.n_output = n_output
#         self.prior_logvar = prior_logvar

#         """initialise parameters and priors following 'Neural network ensembles and variational inference revisited', Appendix A"""
#         # weight parameters
#         self.W_mean = nn.Parameter(torch.Tensor(n_input, n_output).normal_(0, 1/np.sqrt(4*n_output))) # initialisation of weight means
#         self.W_logvar = nn.Parameter(torch.Tensor(n_input, n_output).normal_(-11.5, 1e-10)) # initialisation of weight logvariances # -11.5 usually
#         # bias parameters
#         self.b_mean = nn.Parameter(torch.Tensor(n_output).normal_(0, 1e-10)) # initialisation of bias means
#         self.b_logvar = nn.Parameter(torch.Tensor(n_output).normal_(-11.5, 1e-10)) # initialisation of bias logvariances # -11.5 usually
        
#         # prior parameters 
#         self.W_prior_mean = Variable(torch.zeros(n_input, n_output).cuda())
#         self.W_prior_logvar = Variable(prior_logvar*torch.ones(n_input, n_output).cuda())
#         self.b_prior_mean = Variable(torch.zeros(n_output).cuda())
#         self.b_prior_logvar = Variable(prior_logvar*torch.ones(n_output).cuda())

#         if prior_init == True: # initialise parameters to their prior values
#             self.W_mean = nn.Parameter(torch.zeros(n_input, n_output))
#             self.W_logvar = nn.Parameter(prior_logvar*torch.ones(n_input, n_output))
#             self.b_mean = nn.Parameter(torch.zeros(n_output)) #################################################
#             self.b_logvar = nn.Parameter(prior_logvar*torch.ones(n_output))

#         self.num_weights = n_input*n_output + n_output # number of weights and biases
 
#     def forward(self, x, no_samples, shared_weights): # number of samples per forward pass
#         """
#         input is either (batch_size x no_input), if this is the first layer of the network, or (no_samples x batch_size x no_input), 
#         and output is (no_samples x batch_size x no_output)
#         """

#         if shared_weights == True: # can't use local reparam trick if we want to sample functions from the network. assume we will only do one test sample at a time
#             batch_size = x.size()[0]
#             # sample just one weight matrix and just one bias vector
#             W_var = torch.exp(self.W_logvar)
#             b_var = torch.exp(self.b_logvar)
#             z_W = Variable(torch.Tensor(self.n_input, self.n_output).normal_(0, 1).cuda())
#             z_b = Variable(torch.Tensor(self.n_output).normal_(0, 1).cuda())
#             W = self.W_mean + torch.mul(torch.sqrt(W_var), z_W)
#             W_summed = torch.sum(W, dim=0)
#             b = self.b_mean + torch.mul(torch.sqrt(b_var), z_b)
#             b = torch.mul(W_summed, b)
#             b = b.expand(batch_size, -1)
#             samples_activations = torch.mm(x, W) + b

#         else: # can't use local reparam for prebiasing apparently
#             # find out if this is the first layer of the network. if it is, perform an expansion to no_samples
#             if len(x.shape) == 2:
#                 batch_size = x.size()[0] 
#                 W_sigma = torch.exp(self.W_logvar/2)
#                 b_sigma = torch.exp(self.b_logvar/2)
#                 z_W = Variable(torch.Tensor(no_samples, batch_size, self.n_input, self.n_output).normal_(0, 1).cuda())
#                 W = self.W_mean.expand(no_samples, batch_size, -1, -1) + torch.mul(W_sigma.expand(no_samples, batch_size, -1, -1), z_W)
#                 W_summed = torch.sum(W, dim=2)

#                 z_b = Variable(torch.Tensor(no_samples, batch_size, self.n_output).normal_(0, 1).cuda())
#                 b = self.b_mean.expand(no_samples, batch_size, -1) + torch.mul(b_sigma.expand(no_samples, batch_size, -1), z_b)

#                 Wx = torch.einsum('sbio,bi->sbo', (W, x))
#                 Wb = torch.mul(W_summed, b)

#                 samples_activations = Wx + Wb 

#             elif len(x.shape) == 3:
#                 batch_size = x.size()[1]
#                 W_sigma = torch.exp(self.W_logvar/2)
#                 b_sigma = torch.exp(self.b_logvar/2)
#                 z_W = Variable(torch.Tensor(no_samples, batch_size, self.n_input, self.n_output).normal_(0, 1).cuda())
#                 W = self.W_mean.expand(no_samples, batch_size, -1, -1) + torch.mul(W_sigma.expand(no_samples, batch_size, -1, -1), z_W)
#                 W_summed = torch.sum(W, dim=2)

#                 z_b = Variable(torch.Tensor(no_samples, batch_size, self.n_output).normal_(0, 1).cuda())
#                 b = self.b_mean.expand(no_samples, batch_size, -1) + torch.mul(b_sigma.expand(no_samples, batch_size, -1), z_b)

#                 Wx = torch.einsum('sbio,sbi->sbo', (W, x))
#                 Wb = torch.mul(W_summed, b)

#                 samples_activations = Wx + Wb

#         return samples_activations

#     def get_shared_random(self, no_samples, batch_size):
#         z = Variable(torch.Tensor(no_samples, self.n_output).normal_(0, 1).cuda())
#         z = z.expand(batch_size, -1, -1)
#         return torch.transpose(z, 0, 1)

#     def get_random(self, no_samples, batch_size):
#         return Variable(torch.Tensor(no_samples, batch_size, self.n_output).normal_(0, 1).cuda()) # standard normal noise matrix

#     def KL(self): # get KL between q and prior for this layer
#         # W_KL = 0.5*(- self.W_logvar + torch.exp(self.W_logvar) + (self.W_mean)**2)
#         # b_KL = 0.5*(- self.b_logvar + torch.exp(self.b_logvar) + (self.b_mean)**2)
#         W_KL = 0.5*(self.W_prior_logvar - self.W_logvar + (torch.exp(self.W_logvar) + (self.W_mean - self.W_prior_mean)**2)/torch.exp(self.W_prior_logvar))
#         b_KL = 0.5*(self.b_prior_logvar - self.b_logvar + (torch.exp(self.b_logvar) + (self.b_mean - self.b_prior_mean)**2)/torch.exp(self.b_prior_logvar))
#         return W_KL.sum() + b_KL.sum() - 0.5*self.num_weights

# class MFVI_Prebias_Net(nn.Module):
    # def __init__(self, params, prior_init=False):
    #     super(MFVI_Prebias_Net, self).__init__()
    #     self.train_samples = params.train_samples
    #     self.test_samples = params.test_samples
    #     self.dataset = params.dataset
    #     self.prior_logvar = params.prior_logvar
    #     self.activation_name = params.activation
    #     if params.activation == 'relu':
    #         self.activation = F.relu
    #     elif params.activation == 'tanh':
    #         self.activation = torch.tanh
    #     elif params.activation == 'bump':
    #         self.activation = bump
    #     elif params.activation == 'prelu':
    #         self.prelu_weight = nn.Parameter(torch.Tensor([0.25]))
    #         self.activation = F.prelu
        
    #      # adjust input and output size depending on dataset used and read output noise
    #     if params.dataset == 'mnist':
    #         self.input_channels = 1
    #         self.input_size = 28*28
    #         self.output_size = 10
    #     elif params.dataset == '1d_cosine' or params.dataset == 'prior_dataset':
    #         self.input_channels = 1
    #         self.input_size = 1
    #         self.output_size = 1
    #         self.noise_variance = params.noise_variance
    #     elif params.dataset == 'signs':
    #         self.input_channels = 3
    #         self.input_size = 64*64
    #         self.output_size = 6

    #     # create the layers in the network based on params
    #     self.hidden_sizes = params.hidden_sizes
    #     self.linears = nn.ModuleList([MFVI_Prebias_Layer(self.input_size*self.input_channels, self.hidden_sizes[0], self.prior_logvar, prior_init)])
    #     self.linears.extend([MFVI_Prebias_Layer(self.hidden_sizes[i], self.hidden_sizes[i+1], self.prior_logvar, prior_init) for i in range(0, len(self.hidden_sizes)-1)])
    #     self.linears.append(MFVI_Prebias_Layer(self.hidden_sizes[-1], self.output_size, self.prior_logvar, prior_init))

    # def get_KL_term(self):
    #     # calculate KL divergence between q and the prior for the entire network
    #     KL_term = 0
    #     for _, l in enumerate(self.linears):
    #         KL_term = KL_term + l.KL()
    #     return KL_term

    # def forward(self, s, no_samples, shared_weights=False):
    #     #print('forward net class')
    #     s = s.view(-1, self.input_size*self.input_channels)
    #     for i, l in enumerate(self.linears):
    #         s = l(s, no_samples = no_samples, shared_weights = shared_weights)
    #         if i < len(self.linears) - 1:
    #             if self.activation_name == 'prelu':
    #                 s = self.activation(s, self.prelu_weight)
    #             else:
    #                 s = self.activation(s) 
    #     if self.dataset == '1d_cosine' or self.dataset == 'prior_dataset': # make this more flexible
    #         # s has dimension (no_samples x batch_size x no_output=1)
    #         s = s.view(no_samples, -1) # (no_samples x batch_size)
    #         return s
    #     else:
    #         s = F.log_softmax(s, dim=2) # dimension (no_samples x batch_size x no_output)       
    #         return torch.mean(s, 0) # taking the expectation, dimension (batch_size x no_output)

    # def return_weights(self):
    #     """return the weights and biases of the network"""
    #     # return a list of lists - the first list goes through layers, and each item is a list that has means in 0th place and S.D.'s in 1st place
    #     weights = []
    #     biases = []
    #     for _, l in enumerate(self.linears):
    #         layer_weights = []
    #         layer_biases = []
    #         layer_weights.append(l.W_mean.cpu().detach().numpy())
    #         layer_biases.append(l.b_mean.cpu().detach().numpy())
    #         layer_weights.append(np.sqrt(np.exp(l.W_logvar.cpu().detach().numpy())))
    #         layer_biases.append(np.sqrt(np.exp(l.b_logvar.cpu().detach().numpy())))
    #         weights.append(layer_weights)
    #         biases.append(layer_biases)
    #     return weights, biases

class MAP_Linear_Layer(nn.Module):
    def __init__(self, n_input, n_output):
        super(MAP_Linear_Layer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_input, n_output).normal_(0, 1/np.sqrt(4*n_output)))
        self.bias = nn.Parameter(torch.Tensor(n_output).normal_(0, 1e-10))

    def forward(self, x):
        return torch.matmul(x, self.weight) + self.bias

class MAP_Net(nn.Module):
    def __init__(self, params):
        super(MAP_Net, self).__init__()
        self.dataset = params.dataset
        self.activation_name = params.activation
        self.omega = params.omega # radford neal's prior scaling factor
        if params.activation == 'relu':
            self.activation = F.relu
        elif params.activation == 'tanh':
            self.activation = torch.tanh
        elif params.activation == 'bump':
            self.activation = bump
        elif params.activation == 'prelu':
            self.prelu_weight = nn.Parameter(torch.Tensor([0.25]))
            self.activation = F.prelu
        elif params.activation == 'sine':
            self.activation = torch.sin
        
         # adjust input and output size depending on dataset used and read output noise
        if params.dataset == 'mnist':
            self.input_channels = 1
            self.input_size = 28*28
            self.output_size = 10
        elif params.dataset == '1d_cosine' or params.dataset == 'prior_dataset':
            self.input_channels = 1
            self.input_size = 1
            self.output_size = 1
            self.noise_variance = params.noise_variance
        elif params.dataset == 'signs':
            self.input_channels = 3
            self.input_size = 64*64
            self.output_size = 6

        # create the layers in the network based on params
        self.hidden_sizes = params.hidden_sizes
        self.linears = nn.ModuleList([MAP_Linear_Layer(self.input_size*self.input_channels, self.hidden_sizes[0])])
        self.linears.extend([MAP_Linear_Layer(self.hidden_sizes[i], self.hidden_sizes[i+1]) for i in range(0, len(self.hidden_sizes)-1)])
        self.linears.append(MAP_Linear_Layer(self.hidden_sizes[-1], self.output_size))

    def get_L2_term(self):
        # calculate L2 term for regularisation, using Radford Neal's omega scaling for the prior 
        L2_term = 0
        for i, l in enumerate(self.linears):
            if i == 0: # first layer
                single_layer_L2 = torch.sum(l.weight**2)*self.input_size*self.input_channels/(2*(self.omega**2)) + torch.sum(l.bias**2)/2
            else: 
                single_layer_L2 = torch.sum(l.weight**2)*self.hidden_sizes[i-1]/(2*(self.omega**2)) + torch.sum(l.bias**2)/2
            L2_term = L2_term + single_layer_L2
        return L2_term

    def forward(self, *args, **kwargs): 
        # flatten the input then forward propagate
        s = args[0]
        s = s.view(-1, self.input_size*self.input_channels)
        for i, l in enumerate(self.linears):
            s = l(s)
            if i < len(self.linears) - 1:
                if self.activation_name == 'prelu':
                    s = self.activation(s, self.prelu_weight)
                else:
                    s = self.activation(s) 
        # output type depends on regression/classification task
        if self.dataset == '1d_cosine' or self.dataset == 'prior_dataset': # should make this more general
            batch_size = s.size()[0]
            s = s.view(batch_size)
            return s # dimensions (batch_size)
        else:
            return F.log_softmax(s, dim=1)

    def return_weights(self):
        """return the weights and biases of the network"""
        # return a list of lists - the first list goes through layers, and each item is a list that has means in 0th place and S.D.'s in 1st place
        weights = []
        biases = []
        priors = []
        for _, l in enumerate(self.linears):
            layer_weights = []
            layer_biases = []
            layer_weights.append(l.weight.cpu().detach().numpy())
            layer_biases.append(l.bias.cpu().detach().numpy())
            n_input = l.weight.size()[0]
            n_output = l.weight.size()[1]
            layer_weights.append(5e-2*(self.omega/np.sqrt(n_input))*np.ones((n_input, n_output)))
            layer_biases.append(1e-2*np.ones(n_output))
            weights.append(layer_weights)
            biases.append(layer_biases)
            priors.append(self.omega/np.sqrt(n_input))
        return weights, biases, priors

class fully_connected_Net(nn.Module):
    def __init__(self, params):
        super(fully_connected_Net, self).__init__()
        self.dataset = params.dataset

         # adjust input and output size depending on dataset used
        if params.dataset == 'mnist':
            self.input_channels = 1
            self.input_size = 28*28
            self.output_size = 10
        elif params.dataset == '1d_cosine' or params.dataset == 'prior_dataset':
            self.input_channels = 1
            self.input_size = 1
            self.output_size = 1
            self.noise_variance = params.noise_variance
        elif params.dataset == 'signs':
            self.input_channels = 3
            self.input_size = 64*64
            self.output_size = 6

        # create the layers in the network based on params
        self.hidden_sizes = params.hidden_sizes
        self.linears = nn.ModuleList([nn.Linear(self.input_size*self.input_channels, self.hidden_sizes[0])])
        self.linears.extend([nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i+1]) for i in range(0, len(self.hidden_sizes)-1)])
        self.linears.append(nn.Linear(self.hidden_sizes[-1], self.output_size))

    # def forward(self, s):
    def forward(self, *args, **kwargs): 
        # flatten the input then forward propagate
        s = args[0]
        s = s.view(-1, self.input_size*self.input_channels)
        for i, l in enumerate(self.linears):
            s = l(s)
            if i < len(self.linears) - 1:
                s = F.relu(s) 
        # output type depends on regression/classification task
        if self.dataset == '1d_cosine' or self.dataset == 'prior_dataset': # should make this more general
            batch_size = s.size()[0]
            s = s.view(batch_size)
            return s # dimensions (batch_size)
        else:
            return F.log_softmax(s, dim=1)

class MFVI_Linear_Layer(nn.Module):
    def __init__(self, n_input, n_output, omega, prior_init=False):
        super(MFVI_Linear_Layer, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        # scale the prior with no. of hidden units, corresponds to radford neal's omega
        prior_logvar = 2*np.log(omega) - np.log(n_input)
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
        self.b_prior_logvar = Variable((torch.zeros(n_output)).cuda())# prior logvar on biases unaffected by Neal's scaling

        if prior_init == True: # initialise parameters to their prior values
            self.W_mean = nn.Parameter(torch.zeros(n_input, n_output))
            self.W_logvar = nn.Parameter(prior_logvar*torch.ones(n_input, n_output))
            self.b_mean = nn.Parameter(torch.zeros(n_output)) 
            self.b_logvar = nn.Parameter(torch.zeros(n_output)) # prior logvar on biases unaffected by Neal's scaling

        self.num_weights = n_input*n_output + n_output # number of weights and biases
 
    def forward(self, x, no_samples, shared_weights): # number of samples per forward pass
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

    def get_shared_random(self, no_samples, batch_size):
        z = Variable(torch.cuda.FloatTensor(no_samples, self.n_output).normal_(0, 1))
        z = z.expand(batch_size, -1, -1)
        return torch.transpose(z, 0, 1)

    def get_random(self, no_samples, batch_size):
        return Variable(torch.cuda.FloatTensor(no_samples, batch_size, self.n_output).normal_(0, 1)) # standard normal noise matrix

    def KL(self): # get KL between q and prior for this layer
        # W_KL = 0.5*(- self.W_logvar + torch.exp(self.W_logvar) + (self.W_mean)**2)
        # b_KL = 0.5*(- self.b_logvar + torch.exp(self.b_logvar) + (self.b_mean)**2)
        W_KL = 0.5*(self.W_prior_logvar - self.W_logvar + (torch.exp(self.W_logvar) + (self.W_mean - self.W_prior_mean)**2)/torch.exp(self.W_prior_logvar))
        b_KL = 0.5*(self.b_prior_logvar - self.b_logvar + (torch.exp(self.b_logvar) + (self.b_mean - self.b_prior_mean)**2)/torch.exp(self.b_prior_logvar))
        return W_KL.sum() + b_KL.sum() - 0.5*self.num_weights

class MFVI_Net(nn.Module):
    def __init__(self, params, prior_init=False):
        super(MFVI_Net, self).__init__()
        self.train_samples = params.train_samples
        self.test_samples = params.test_samples
        self.dataset = params.dataset
        # self.prior_logvar = params.prior_logvar
        self.omega = params.omega
        self.activation_name = params.activation
        if params.activation == 'relu':
            self.activation = F.relu
        elif params.activation == 'tanh':
            self.activation = torch.tanh
        elif params.activation == 'bump':
            self.activation = bump
        elif params.activation == 'prelu':
            self.prelu_weight = nn.Parameter(torch.Tensor([0.25]))
            self.activation = F.prelu
        elif params.activation == 'sine':
            self.activation = torch.sin
        elif params.activation == 'sigmoid':
            self.activation = F.sigmoid
        
         # adjust input and output size depending on dataset used and read output noise
        if params.dataset == 'mnist':
            self.input_channels = 1
            self.input_size = 28*28
            self.output_size = 10
        elif params.dataset == '1d_cosine' or params.dataset == 'prior_dataset':
            self.input_channels = 1
            self.input_size = 1
            self.output_size = 1
            self.noise_variance = params.noise_variance
        elif params.dataset == 'signs':
            self.input_channels = 3
            self.input_size = 64*64
            self.output_size = 6

        # create the layers in the network based on params
        self.hidden_sizes = params.hidden_sizes
        self.linears = nn.ModuleList([MFVI_Linear_Layer(self.input_size*self.input_channels, self.hidden_sizes[0], self.omega, prior_init)])
        self.linears.extend([MFVI_Linear_Layer(self.hidden_sizes[i], self.hidden_sizes[i+1], self.omega, prior_init) for i in range(0, len(self.hidden_sizes)-1)])
        self.linears.append(MFVI_Linear_Layer(self.hidden_sizes[-1], self.output_size, self.omega, prior_init))

    def get_KL_term(self):
        # calculate KL divergence between q and the prior for the entire network
        KL_term = 0
        for _, l in enumerate(self.linears):
            KL_term = KL_term + l.KL()
        return KL_term

    def forward(self, s, no_samples, shared_weights=False):
        s = s.view(-1, self.input_size*self.input_channels)
        for i, l in enumerate(self.linears):
            s = l(s, no_samples = no_samples, shared_weights = shared_weights)
            if i < len(self.linears) - 1:
                if self.activation_name == 'prelu':
                    s = self.activation(s, self.prelu_weight)
                else:
                    s = self.activation(s) 
                ##################################### THIS ISN'T WORKING - TRY A MASK INSTEAD
                #############################################################################
                # s_bump = bump(s)
                # no_dims = len(s.size())

                # no_bump_units = int(s_bump.size()[-1]/2) 
                # s_retanh = bump(s)

                # if no_dims == 2:
                #     s = torch
                #     s[:, :no_bump_units] = s_bump[:, :no_bump_units]
                #     s[:, no_bump_units:] = s_retanh[:, no_bump_units:]
                # elif no_dims == 3:
                #     s[:, :, :no_bump_units] = s_bump[:, :,no_bump_units]
                #     s[:, :, no_bump_units:] = s_retanh[:, :, no_bump_units:]
                # else: 
                #     print(no_dims)
                #     print('errrrrrrrrrrrrrrrrrrrrrrrror')
        if self.dataset == '1d_cosine' or self.dataset == 'prior_dataset': # make this more flexible
            # s has dimension (no_samples x batch_size x no_output=1)
            s = s.view(no_samples, -1) # (no_samples x batch_size)
            return s
        else:
            s = F.log_softmax(s, dim=2) # dimension (no_samples x batch_size x no_output)       
            return torch.mean(s, 0) # taking the expectation, dimension (batch_size x no_output)

    def return_weights(self):
        """return the weights and biases of the network"""
        # return a list of lists - the first list goes through layers, and each item is a list that has means in 0th place and S.D.'s in 1st place
        weights = []
        biases = []
        priors = []
        for _, l in enumerate(self.linears):
            layer_weights = []
            layer_biases = []
            layer_weights.append(l.W_mean.cpu().detach().numpy())
            layer_biases.append(l.b_mean.cpu().detach().numpy())
            layer_weights.append(np.sqrt(np.exp(l.W_logvar.cpu().detach().numpy())))
            layer_biases.append(np.sqrt(np.exp(l.b_logvar.cpu().detach().numpy())))
            weights.append(layer_weights)
            biases.append(layer_biases)
            n_input = l.W_mean.size()[0]
            priors.append(self.omega/np.sqrt(n_input))
        return weights, biases, priors

class fully_connected_Net(nn.Module):
    def __init__(self, params):
        super(fully_connected_Net, self).__init__()
        self.dataset = params.dataset

         # adjust input and output size depending on dataset used
        if params.dataset == 'mnist':
            self.input_channels = 1
            self.input_size = 28*28
            self.output_size = 10
        elif params.dataset == '1d_cosine' or params.dataset == 'prior_dataset':
            self.input_channels = 1
            self.input_size = 1
            self.output_size = 1
            self.noise_variance = params.noise_variance
        elif params.dataset == 'signs':
            self.input_channels = 3
            self.input_size = 64*64
            self.output_size = 6

        # create the layers in the network based on params
        self.hidden_sizes = params.hidden_sizes
        self.linears = nn.ModuleList([nn.Linear(self.input_size*self.input_channels, self.hidden_sizes[0])])
        self.linears.extend([nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i+1]) for i in range(0, len(self.hidden_sizes)-1)])
        self.linears.append(nn.Linear(self.hidden_sizes[-1], self.output_size))

    # def forward(self, s):
    def forward(self, *args, **kwargs): 
        # flatten the input then forward propagate
        s = args[0]
        s = s.view(-1, self.input_size*self.input_channels)
        for i, l in enumerate(self.linears):
            s = l(s)
            if i < len(self.linears) - 1:
                s = F.relu(s) 
        # output type depends on regression/classification task
        if self.dataset == '1d_cosine' or self.dataset == 'prior_dataset': # should make this more general
            batch_size = s.size()[0]
            s = s.view(batch_size)
            return s # dimensions (batch_size)
        else:
            return F.log_softmax(s, dim=1)

class Net(nn.Module):
    """
    This is the standard way to define your own network in PyTorch. You typically choose the components
    (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
    on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions

    such as F.relu, F.sigmoid, F.softmax, F.max_pool2d. Be careful to ensure your dimensions are correct after each
    step. You are encouraged to have a look at the network in pytorch/nlp/model/net.py to get a better sense of how
    you can go about defining your own network.

    The documentation for all the various components available o you is here: http://pytorch.org/docs/master/nn.html
    """

    def __init__(self, params):
        """
        We define an convolutional network that predicts the sign from an image. The components
        required are:

        - an embedding layer: this layer maps each index in range(params.vocab_size) to a params.embedding_dim vector
        - lstm: applying the LSTM on the sequential input returns an output for each token in the sentence
        - fc: a fully connected layer that converts the LSTM output for each token to a distribution over NER tags

        Args:
            params: (Params) contains num_channels
        """
        super(Net, self).__init__()
        self.num_channels = params.num_channels
        self.dataset = params.dataset
        # adjust input and output size depending on dataset used
        if params.dataset == 'mnist':
            self.input_channels = 1
            self.flatten_size = 4*7*7
            self.output_size = 10
        else:
            self.input_channels = 3
            self.flatten_size = 4*8*8
            self.output_size = 6
        
        # each of the convolution layers below have the arguments (input_channels, output_channels, filter_size,
        # stride, padding). We also include batch normalisation layers that help stabilise training.
        # For more details on how to use these layers, check out the documentation.
        self.conv1 = nn.Conv2d(self.input_channels, self.num_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels*2, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.num_channels*2)
        self.conv3 = nn.Conv2d(self.num_channels*2, self.num_channels*4, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(self.num_channels*4)

        # 2 fully connected layers to transform the output of the convolution layers to the final output
        self.fc1 = nn.Linear(self.flatten_size*self.num_channels, self.num_channels*4)
        self.fcbn1 = nn.BatchNorm1d(self.num_channels*4)
        self.fc2 = nn.Linear(self.num_channels*4, self.output_size)       
        self.dropout_rate = params.dropout_rate

    def forward(self, *args, **kwargs):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 3 x 64 x 64 .

        Returns:
            out: (Variable) dimension batch_size x 6 with the log probabilities for the labels of each image.

        Note: the dimensions after each step are provided
        """
        s = args[0]
        #                                                  -> batch_size x 3 x 64 x 64
        # we apply the convolution layers, followed by batch normalisation, maxpool and relu x 3
        s = self.bn1(self.conv1(s))                         # batch_size x num_channels x 64 x 64
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels x 32 x 32
        s = self.bn2(self.conv2(s))                         # batch_size x num_channels*2 x 32 x 32
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*2 x 16 x 16
        s = self.bn3(self.conv3(s))                         # batch_size x num_channels*4 x 16 x 16
        if self.dataset == 'mnist':
            s = F.relu(s)                                   # do one less maxpool on MNIST than on SIGNS
        else:
            s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*4 x 8 x 8

        # flatten the output for each image
        s = s.view(-1, self.flatten_size*self.num_channels)             # batch_size x 8*8*num_channels*4

        # apply 2 fully connected layers with dropout
        s = F.dropout(F.relu(self.fcbn1(self.fc1(s))), 
            p=self.dropout_rate, training=self.training)    # batch_size x self.num_channels*4
        s = self.fc2(s)                                     # batch_size x 6

        # apply log softmax on each image's output (this is recommended over applying softmax
        # since it is numerically more stable)
        return F.log_softmax(s, dim=1)

def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.

    Args:
        outputs: (Variable) dimension batch_size x 6 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    num_examples = outputs.size()[0]
    return -torch.sum(outputs[range(num_examples), labels])/num_examples # average log loss

def regression_loss_fn(**kwargs):
    outputs = kwargs['outputs']
    labels = kwargs['labels']
    model = kwargs['model']
    dataset_size = kwargs['dataset_size']
    # calculate the per datapoint log likelihood for regression
    const_term = 0.5*np.log(2*np.pi*model.noise_variance)   
    return dataset_size*torch.mean((1/(2*model.noise_variance))*(labels - outputs)**2) + np.asscalar(const_term)

def MFVI_loss_fn(outputs, labels, model, dataset_size): 
    # calculate the per-datapoint ELBO - this is printed as 'loss' on the test set but ELBO is fairly meaningless for test set
    batch_size = outputs.size()[0]
    reconstruction_loss = -torch.sum(outputs[range(batch_size), labels])/batch_size 
    KL_term = model.get_KL_term()/dataset_size
    return reconstruction_loss + KL_term 

def MAP_regression_loss_fn(outputs, labels, model, dataset_size, term): # ignore constant terms
    error = (1/(2*model.noise_variance))*torch.sum((labels - outputs)**2)
    L2_term = model.get_L2_term() ##################################################### hacking the L2 term
    loss = error + L2_term
    return {"loss": loss, "error": error, "L2": L2_term}

def MFVI_regression_loss_fn(outputs, labels, model, dataset_size, term = 'loss'): 
    # calculate the per datapoint ELBO for regression
    no_samples = outputs.size()[0]
    labels = labels.expand(no_samples, -1)
    const_term = 0.5*np.log(2*np.pi*model.noise_variance)
    reconstruction_loss = np.asscalar(const_term) + (1/(2*model.noise_variance))*torch.mean((labels - outputs)**2)
    KL_term = model.get_KL_term()/dataset_size
    loss = reconstruction_loss + KL_term 
    if term == 'loss':
        return loss
    elif term == 'all':
        return {"loss": loss, "reconstruction": reconstruction_loss, "KL": KL_term}

def MFVI_regression_ML_loss_fn(outputs, labels, model, dataset_size):
    """this loss only has the reconstruction term and is used to initialise the model"""
    no_samples = outputs.size()[0]
    labels = labels.expand(no_samples, -1)
    const_term = 0.5*np.log(2*np.pi*model.noise_variance)
    reconstruction_loss = np.asscalar(const_term) + (1/(2*model.noise_variance))*torch.mean((labels - outputs)**2)
    return reconstruction_loss 

def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels - or compute RMSE for regression problem

    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(labels.size)

def nll(outputs, labels):
    """compute the per-datapoint negative log likelihood for classification"""
    batch_size = labels.size
    return -np.sum(outputs[range(batch_size), labels])/batch_size 

# def regression_nll(outputs, labels):
#     """compute the per-datapoint negative log likelihood for regression"""
#     batch_size = labels.size
#     return 