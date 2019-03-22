# implement Laplace approximation using the outer-product approximation to the Hessian
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
import gc

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
                x = self.activation(x) ###### activation function very important for laplace
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

    def get_gradient(self, x):
        # REMEMBER TO ZERO GRAD BEFORE CALLING THIS FUNCTION

        # get a single vector of the gradient of the output wrt all parameters in the network 
        # EXCEPT noise parameter
        # forward pass
        output = self.forward(x) # output should be a scalar - batch size one here
        output = torch.squeeze(output)
        # backward pass
        output.backward()

        # fill gradient values into a single vector
        gradient = torch.cuda.DoubleTensor(self.no_params).fill_(0)
        start_index = 0
        for name, param in self.named_parameters():
            if name != 'noise_var_param': # dont do laplace for noise variance
                grad_vec = param.grad.detach().reshape(-1) # flatten into a vector
                end_index = start_index + grad_vec.size()[0]
                gradient[start_index:end_index] = grad_vec # fill into single vector
                start_index = start_index + grad_vec.size()[0]

        return gradient

    def get_parameter_vector(self): 
        # load all the parameters into a single numpy vector EXCEPT the noise variance
        parameter_vector = np.zeros(self.no_params)
        # fill parameter values into a single vector
        start_index = 0

        for name, param in self.named_parameters():
            if name != 'noise_var_param': # dont include noise variance
                param_vec = param.detach().reshape(-1) # flatten into a vector
                end_index = start_index + param_vec.size()[0]
                parameter_vector[start_index:end_index] = param_vec.cpu().numpy()
                start_index = start_index + param_vec.size()[0]
        return parameter_vector

    def get_P_vector(self):
        # get prior contribution to the Hessian
        P_vector = torch.cuda.DoubleTensor(self.no_params).fill_(0)
        P_vector[:self.dim_input*self.hidden_sizes[0]] = self.dim_input/(self.omega**2) # first weight matrix
        start_index = self.dim_input*self.hidden_sizes[0]
        for i in range(len(self.hidden_sizes) - 1):
            # bias vector
            end_index = start_index + self.hidden_sizes[i]
            P_vector[start_index:end_index] = 1 # biases have unit variance
            # weight matrix
            start_index = end_index
            end_index = start_index + self.hidden_sizes[i]*self.hidden_sizes[i+1]
            P_vector[start_index:end_index] = self.hidden_sizes[i]/(self.omega**2) 
            start_index = end_index
        # final weight matrix and last two biases
        # bias vector
        end_index = start_index + self.hidden_sizes[-1]
        P_vector[start_index:end_index] = 1 # biases have unit variance
        # weight matrix
        start_index = end_index
        end_index = start_index + self.hidden_sizes[-1]*1 # assume unit output
        P_vector[start_index:end_index] = self.hidden_sizes[-1]/(self.omega**2)
        # output bias
        start_index = end_index
        end_index = start_index + 1
        P_vector[start_index:end_index] = 1 # biases have unit variance
        return P_vector 

    def get_H(self, x_train, optimizer=None):
        # create 'sum of outer products' matrix
        # try subsampling or minibatching this 
        H = torch.cuda.DoubleTensor(self.no_params, self.no_params).fill_(0)
        for i in range(x_train.size()[0]): # for all training inputs
            # clear gradients
            optimizer.zero_grad()
            # get gradient of output wrt single training input
            x = x_train[i]
            x = torch.unsqueeze(x, 0)
            gradient = self.get_gradient(x)
            # form outer product
            outer = gradient.unsqueeze(1)*gradient.unsqueeze(0)
            H.add_(outer)
        return H

    def get_Ainv(self, train_inputs, optimizer=None): # invert the Hessian in 'parameter space' instead of 'data space'
        if self.learned_noise_var == True:
            noise_variance = self.get_noise_var(self.noise_var_param)
        else:
            noise_variance = self.noise_variance   

        H = self.get_H(train_inputs, optimizer)
        P = torch.diag(self.get_P_vector())
        
        # calculate and invert (negative) Hessian of posterior
        A = (1/noise_variance)*H + P 
        Ainv = torch.inverse(A) # just directly invert this and don't care about numerical stability
        return Ainv.detach()

    def linearised_laplace_direct(self, Ainv, test_inputs, optimizer=None): # get the predictive variances, one by one for now lol                      
        if self.learned_noise_var == True:
            noise_variance = self.get_noise_var(self.noise_var_param)
        else:
            noise_variance = self.noise_variance 
        
        # get list of test gradients
        no_test = test_inputs.size()[0]
        G = torch.cuda.DoubleTensor(self.no_params, no_test).fill_(0)
        for i in range(no_test):
            # clear gradients
            optimizer.zero_grad()
            # get gradient of output wrt single test input
            x = test_inputs[i]
            x = torch.unsqueeze(x, 0) # this may not be necessary if x is multidimensional
            gradient = self.get_gradient(x)
            # store in G
            G[:,i] = gradient
        # unsqueeze so its a batch of column vectors 
        G = torch.transpose(G, 0, 1)
        Gbatch = torch.unsqueeze(G, 2)
        Gt = torch.unsqueeze(G, 1)
        AinvG = torch.matmul(Ainv, Gbatch)
        gtAinvg = torch.squeeze(torch.matmul(Gt, AinvG))

        print('noise_var: {}'.format(noise_variance))
        print('gtAinvg: {}'.format(gtAinvg))

        predictive_var = noise_variance + gtAinvg
        print('predictive_var: {}'.format(predictive_var))
        return predictive_var.detach()

    def get_L(self, train_inputs, optimizer=None): # invert the Hessian in 'parameter space' instead of 'data space'
        if self.learned_noise_var == True:
            noise_variance = self.get_noise_var(self.noise_var_param)
        else:
            noise_variance = self.noise_variance   

        # get A
        H = self.get_H(train_inputs, optimizer)
        P = torch.diag(self.get_P_vector())
        A = (1/noise_variance)*H + P
        # symmetrise in case numerical issues
        A = (A + A.transpose(0,1))/2
        # jitter A
        A = A + torch.eye(self.no_params).double().cuda()*1e-6 
        # cholesky decompose 
        L = torch.potrf(A, upper=False) # lower triangular decomposition
        return L.detach()

    def linearised_laplace_direct_cholesky(self, L, test_inputs, optimizer=None):
        # do a numerically stable version of the algorithm
        if self.learned_noise_var == True:
            noise_variance = self.get_noise_var(self.noise_var_param)
        else:
            noise_variance = self.noise_variance 

        # get list of test gradients
        no_test = test_inputs.size()[0]
        G = torch.cuda.DoubleTensor(self.no_params, no_test).fill_(0)
        for i in range(no_test):
            # clear gradients
            optimizer.zero_grad()
            # get gradient of output wrt single test input
            x = test_inputs[i]
            x = torch.unsqueeze(x, 0) # this may not be necessary if x is multidimensional
            gradient = self.get_gradient(x)
            # store in G
            G[:,i] = gradient
            
        # backsolve for all columns
        LslashG = torch.trtrs(G, L, upper=False)[0]  

        # batch dot product
        predictive_var = noise_variance + torch.sum(LslashG**2, 0)
        return predictive_var.detach()
        
    def linearised_laplace(self, train_inputs, test_inputs, subsampling=None, optimizer=None): # return the posterior uncertainties for all the inputs

        if self.learned_noise_var == True:
            self.noise_variance = self.get_noise_var(self.noise_var_param)

        if subsampling == None: # don't subsample
            # form Jacobian matrix of train set, Z - use a for loop for now?
            no_train = train_inputs.size()[0]
            Z = torch.cuda.DoubleTensor(no_train, self.no_params).fill_(0)
            for i in range(no_train):
                # clear gradients
                optimizer.zero_grad()
                # get gradient of output wrt single training input
                x = train_inputs[i]
                # x = torch.unsqueeze(x, 0) # this may not be necessary if x is multidimensional
                gradient = self.get_gradient(x)
                # store in Z
                Z[i,:] = gradient
        else: # subsample the train set 
            no_train = subsampling
            Z = torch.cuda.DoubleTensor(no_train, self.no_params).fill_(0)    
            for i, sample in enumerate(np.random.choice(train_inputs.size()[0], no_train, replace=False)):
                # clear gradients
                optimizer.zero_grad()
                # get gradient of output wrt single training input
                x = train_inputs[sample]
                # x = torch.unsqueeze(x, 0) # this may not be necessary if x is multidimensional
                gradient = self.get_gradient(x)
                # store in Z, and scale to compensate the subsampling
                Z[i,:] = (train_inputs.size()[0]/no_train)*gradient

        # get list of test gradients
        no_test = test_inputs.size()[0]
        G = torch.cuda.DoubleTensor(self.no_params, no_test).fill_(0)
        for i in range(no_test):
            # clear gradients
            optimizer.zero_grad()
            # get gradient of output wrt single test input
            x = test_inputs[i]
            x = torch.unsqueeze(x, 0) # this may not be necessary if x is multidimensional
            gradient = self.get_gradient(x)
            # store in G
            G[:,i] = gradient
        # unsqueeze so its a batch of column vectors ###### maybe not necessary now?
        Gunsq = torch.unsqueeze(G, 1)

        # calculate ZPinvG as a batch
        G_batch = Gunsq.permute(2,0,1) # make the batch index first (no_test x no_params x 1)
        Pinv_vector = 1/self.get_P_vector().unsqueeze(1) # column vector
        
        PinvG = Pinv_vector*G_batch # batch elementwise multiply (no_test x no_params x 1)
        ZPinvG = torch.matmul(Z, PinvG.squeeze().transpose(0,1)) # batch matrix multiply - (no_test x no_train)

        # calculate the 'inner matrix' M and Cholesky decompose
        PinvZt = Pinv_vector*torch.transpose(Z, 0, 1) # diagonal matrix multiplication      

        M = torch.eye(no_train).cuda() + (1/(self.noise_variance))*torch.matmul(Z, PinvZt)
        #import pdb; pdb.set_trace()
        # symmetrize M - there may be numerical issues causing it to be non symmetric
        M = (M + M.transpose(0,1))/2
        # JITTER M
        M = M + torch.eye(no_train).double().cuda()*1e-6 
          
        U = torch.potrf(M, upper=True) # upper triangular decomposition

        # solve the triangular system
        V = torch.trtrs(ZPinvG, U, transpose=True)[0] # (no_train x no_test), some of the stuff might need transposing
        V = V.transpose(0,1) # (no_test x no_train)

        # dot products
        v = (-1/(self.noise_variance))*torch.bmm(V.view(no_test, 1, no_train), V.view(no_test, no_train, 1)) 

        # prior terms dot products        
        G_batch *= PinvG
        prior_terms = G_batch.sum(1)

        predictive_var = self.noise_variance + prior_terms.squeeze() + v.squeeze()
        return torch.squeeze(predictive_var)

    def unpack_sample(self, parameter_vector):
        """convert a numpy vector of parameters to a list of parameter tensors"""
        sample = []
        start_index = 0
        end_index = 0   
        # unpack first weight matrix and bias
        end_index = end_index + self.hidden_sizes[0]*self.dim_input
        weight_vector = parameter_vector[start_index:end_index]
        weight_matrix = weight_vector.reshape((self.hidden_sizes[0], self.dim_input))
        sample.append(torch.cuda.DoubleTensor(weight_matrix))
        start_index = start_index + self.hidden_sizes[0]*self.dim_input
        end_index = end_index + self.hidden_sizes[0]
        biases_vector = parameter_vector[start_index:end_index]
        sample.append(torch.cuda.DoubleTensor(biases_vector))
        start_index = start_index + self.hidden_sizes[0]
        for i in range(len(self.hidden_sizes)-1): 
            end_index = end_index + self.hidden_sizes[i]*self.hidden_sizes[i+1]
            weight_vector = parameter_vector[start_index:end_index]
            weight_matrix = weight_vector.reshape((self.hidden_sizes[i+1], self.hidden_sizes[i]))
            sample.append(torch.cuda.DoubleTensor(weight_matrix))
            start_index = start_index + self.hidden_sizes[i]*self.hidden_sizes[i+1]
            end_index = end_index + self.hidden_sizes[i+1]
            biases_vector = parameter_vector[start_index:end_index]
            sample.append(torch.cuda.DoubleTensor(biases_vector))
            start_index = start_index + self.hidden_sizes[i+1]
        # unpack output weight matrix and bias
        end_index = end_index + self.hidden_sizes[-1]
        weight_vector = parameter_vector[start_index:end_index]
        weight_matrix = weight_vector.reshape((1, self.hidden_sizes[-1]))
        sample.append(torch.cuda.DoubleTensor(weight_matrix))
        start_index = start_index + self.hidden_sizes[-1]
        biases_vector = parameter_vector[start_index:] # should reach the end of the parameters vector at this point
        sample.append(torch.cuda.DoubleTensor(biases_vector))
        return sample

def plot_cov(cov, directory, title=''):
    # plot covariance of Gaussian fit
    fig, ax = plt.subplots()
    im = ax.imshow(np.abs(cov) , interpolation='nearest', cmap=cm.Greys_r)
    filepath = os.path.join(directory, title + 'covariance.pdf')
    fig.savefig(filepath)
    plt.close()

    # plot correlation matrix using cov matrix 
    variance_vector = np.diag(cov)
    sd_vector = np.sqrt(variance_vector)
    outer_prod = np.outer(sd_vector, sd_vector)
    correlations = cov/outer_prod

    fig, ax = plt.subplots()
    im = ax.imshow(correlations , interpolation='nearest')
    fig.colorbar(im)
    filepath = os.path.join(directory, title + 'correlation.pdf')
    fig.savefig(filepath)
    plt.close()

def sample_gaussian(model, L, inputs, labels, train_mean, train_sd, no_samples):
    # inputs and labels assumed NOT normalised
    # sample from the Gaussian with covariance matrix Ainv and mean equal to the MAP solution, then return the function evaluations
    # scale the inputs
    inputs = inputs - train_mean[:-1] 
    inputs = inputs/train_sd[:-1]
    
    mean = torch.cuda.DoubleTensor(model.get_parameter_vector())
    # use cholesky factor of the precision matrix to sample from Laplace posterior by backsolving
    z = torch.cuda.DoubleTensor(mean.shape[0], no_samples).normal_()   
    sampled_parameters = torch.unsqueeze(mean,1) + torch.trtrs(z, L.transpose(0,1))[0] 
    sampled_parameters = sampled_parameters.transpose(0,1).data.cpu().numpy()
    # samples is a list of lists of tensors
    samples = []
    for i in range(no_samples):
        # unpack the parameter vector into a list of parameter tensors
        sample = model.unpack_sample(sampled_parameters[i,:])
        samples.append(sample)

    # fill the model with the sampled parameters and evaluate
    all_outputs = torch.cuda.DoubleTensor(inputs.shape[0], no_samples)
    for i, sample in enumerate(samples):
        j = 0 
        for name, param in model.named_parameters():
            if name != 'noise_var_param': # dont do laplace for noise variance
                param.data = sample[j] # fill in the model with these weights
                j=j+1
        # forward pass through the sampled model
        outputs = torch.squeeze(model(inputs))
        # scale the outputs
        outputs = outputs*train_sd[-1]
        outputs = outputs + train_mean[-1]
        all_outputs[:,i] = outputs
    
    # calculate summed log likelihoods for this batch of inputs
    noise_var = model.get_noise_var(model.noise_var_param)
    # scale the noise var because of the normalisation
    noise_var = noise_var*(train_sd[-1]**2) 
    error_term = ((all_outputs - torch.unsqueeze(labels,1))**2)/(2*noise_var) 
    exponents = -torch.log(torch.cuda.DoubleTensor([no_samples])) - 0.5*torch.log(2*3.1415926536*noise_var) - error_term
    LLs = torch.logsumexp(exponents, 1)
    sum_LLs = torch.sum(LLs)

    # put the old model parameters back, so that later optimisation is not affected
    sample = model.unpack_sample(mean.data.cpu().numpy())
    j = 0
    for name, param in model.named_parameters():
        if name != 'noise_var_param': # dont do laplace for noise variance
            param.data = sample[j] # fill in the model with these weights
            j=j+1

    # calculate the quantities needed to plot calibration curves and get RMSEs
    mean_prediction = torch.mean(all_outputs, 1)
    squared_error = torch.sum((labels - mean_prediction)**2)
    abs_errors = torch.abs(labels - mean_prediction)
    variances = noise_var + torch.mean(all_outputs**2, 1) - mean_prediction**2

    return sum_LLs, squared_error, abs_errors, variances 

def evaluate(model, x_test, y_test, train_mean, train_sd, laplace=False, x_train_normalised=None, subsampling=None, validation=None, optimizer=None, directory=None, name=None):
    
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

    # cholesky decompose A just once
    if laplace == True and (direct_invert == True or sample == True):
        L = model.get_L(x_train_normalised, optimizer)
       
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

        if (laplace == True) and (sample == True): # monte carlo sample from the Gaussian
            no_samples = no_samples_laplace
            log_likelihood, squared_error, errors, variances = sample_gaussian(model, L, inputs, labels, train_mean, train_sd, no_samples)
            sum_squared_error = sum_squared_error + squared_error

            if i != num_batches - 1: # this isn't the final batch
                predictive_sds[i*eval_batch_size:(i+1)*eval_batch_size] = np.sqrt(variances.data.cpu().numpy())
                abs_errors[i*eval_batch_size:(i+1)*eval_batch_size] = errors.data.cpu().numpy()
            else:
                predictive_sds[i*eval_batch_size:] = np.sqrt(variances.data.cpu().numpy())
                abs_errors[i*eval_batch_size:] = errors.data.cpu().numpy()

        else:
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
            if laplace == False: 
                noise_var = model.get_noise_var(model.noise_var_param)
                # scale the noise var because of the normalisation
                noise_var = noise_var*(train_sd[-1]**2)            
                log_likelihood = -0.5*actual_batch_size*torch.log(2*3.1415926536*noise_var) - (1/(2*noise_var))*squared_error
            else: # do Laplace approximation
                # get the predictive variances
                if direct_invert == True:
                    predictive_var = model.linearised_laplace_direct_cholesky(L, inputs, optimizer)
                else:
                    predictive_var = model.linearised_laplace(x_train_normalised, inputs, subsampling=subsampling, optimizer=optimizer)
                # scale predictive var because of the normalisation
                predictive_var = predictive_var*(train_sd[-1]**2)
                sigma2_term = -0.5*torch.sum(torch.log(2*3.1415926536*predictive_var))
                error_term = -torch.sum((1/(2*predictive_var))*((outputs - labels)**2))
                log_likelihood = sigma2_term + error_term     
                
                if i != num_batches - 1: # this isn't the final batch
                    predictive_sds[i*eval_batch_size:(i+1)*eval_batch_size] = np.sqrt(predictive_var.data.cpu().numpy())
                    abs_errors[i*eval_batch_size:(i+1)*eval_batch_size] = np.abs((outputs - labels).data.cpu().numpy())
                else:
                    predictive_sds[i*eval_batch_size:] = np.sqrt(predictive_var.data.cpu().numpy())
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

    mean_squared_error = sum_squared_error/testset_size
    mean_ll = sum_log_likelihood/testset_size

    return mean_squared_error, mean_ll[0]

def train(model, train_x, train_y, eval_x, eval_y, train_mean, train_sd, validation=False, minibatch_size=None, no_epochs=None, subsampling=None, optimizer=None, early_stopping=False): 
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
                    MAP_MSE, MAP_LL = evaluate(model, eval_x, eval_y, train_mean, train_sd, validation=True, optimizer=optimizer) # without laplace
                    lap_MSE, lap_LL = evaluate(model, eval_x, eval_y, train_mean, train_sd, laplace=True, x_train_normalised=train_x, subsampling=subsampling, validation=True, optimizer=optimizer) # with laplace
                    results_dict = {'MAP_MSE':MAP_MSE, 'MAP_LL':MAP_LL, 'lap_MSE':lap_MSE, 'lap_LL':lap_LL, 'no_epochs':epoch + 1}
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
        train(model, x_train_normalised, y_train_normalised, x_test, y_test, train_mean, train_sd, validation=False, minibatch_size=minibatch_size, no_epochs=no_epochs, subsampling=subsampling, optimizer=optimizer)
        MAP_MSE, MAP_LL = evaluate(model, x_test, y_test, train_mean, train_sd, validation=False, optimizer=optimizer) # without laplace
        lap_MSE, lap_LL = evaluate(model, x_test, y_test, train_mean, train_sd, laplace=True, x_train_normalised=x_train_normalised, subsampling=subsampling, validation=False, optimizer=optimizer, directory=results_dir, name=str(split)) # with laplace
        return MAP_MSE, MAP_LL, lap_LL, lap_MSE

    else: # this is validation time, do early stopping for hyperparam search
        results_dict_list = train(model, x_train_normalised, y_train_normalised, x_val_normalised, y_val_normalised, train_mean, train_sd, validation=True, minibatch_size=minibatch_size, no_epochs=no_epochs, subsampling=subsampling, optimizer=optimizer, early_stopping=True)
        return results_dict_list

def individual_tune_train(results_dir, standard_normal_prior, activation_function, hidden_sizes, omega_range, learning_rate_range, minibatch_size_range, no_epochs_range, input_dim, subsampling, noise_param_init, dataset):
    # do a grid search on each split separately, then evaluate on the test set
    # this grids over omega, minibatch_size and no_epochs
    
    # create array of values to grid search over - but don't repeat searches when doing early stopping
    list_hypers = [omega_range, learning_rate_range, minibatch_size_range]
    hyperparams = cartesian(list_hypers)
    MAP_RMSEs = np.zeros(no_splits)
    lap_RMSEs = np.zeros(no_splits)
    MAP_LLs = np.zeros(no_splits)
    lap_LLs = np.zeros(no_splits)

    for split in range(no_splits): 
        # find data location
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
                file.write('lap_RMSE: {} \n'.format(torch.sqrt(results_dict['lap_MSE'])))
                file.write('MAP_LL: {} \n'.format(results_dict['MAP_LL']))
                file.write('lap_LL: {} \n'.format(results_dict['lap_LL']))
                file.close() 
    
            # record the hyperparams that maximise validation set lap_LL
            if i == 0: # first hyperparam setting
                # find the best lap_LL in results_dict_list
                for k, results_dict in enumerate(results_dict_list):
                    if k == 0:
                        max_LL = results_dict['lap_LL']
                        best_no_epochs = results_dict['no_epochs']
                    else:
                        if float(results_dict['lap_LL']) > float(max_LL):
                            max_LL = results_dict['lap_LL']
                            best_no_epochs = results_dict['no_epochs']
                best_hyperparams = copy_hyperparams[i,:]
            else:
                for results_dict in results_dict_list:
                    if float(results_dict['lap_LL']) > float(max_LL):
                        max_LL = results_dict['lap_LL']
                        best_no_epochs = results_dict['no_epochs']
                        best_hyperparams = copy_hyperparams[i,:]
        
        # use the best hyperparams found to retrain on all the train data, and evaluate on the test set
        test = True # this is test time
        omega = best_hyperparams[0]
        learning_rate = best_hyperparams[1]
        minibatch_size = int(best_hyperparams[2])
        no_epochs = best_no_epochs

        MAP_MSE, MAP_LL, lap_LL, lap_MSE = individual_train(data_location, test, noise_variance, hidden_sizes,\
             omega, activation_function, learned_noise_var, input_dim, noise_param_init, learning_rate, no_epochs,\
                  standard_normal_prior, minibatch_size, results_dir, split)
        MAP_RMSEs[split] = torch.sqrt(MAP_MSE).data.cpu().numpy()
        lap_RMSEs[split] = torch.sqrt(lap_MSE).data.cpu().numpy()
        MAP_LLs[split] = MAP_LL.data.cpu().numpy()
        lap_LLs[split] = lap_LL.data.cpu().numpy()
        
        # record best hyperparams
        file = open(results_dir + '/best_hypers.txt','a') 
        file.write('split: {} \n'.format(str(split)))
        file.write('omega: {} \n'.format(omega))
        file.write('learning_rate: {} \n'.format(learning_rate))
        file.write('no_epochs: {} \n'.format(no_epochs))
        file.write('minibatch_size: {} \n'.format(minibatch_size))
        file.write('test_MAP_RMSE: {} \n'.format(MAP_RMSEs[split]))
        file.write('test_lap_RMSE: {} \n'.format(lap_RMSEs[split]))
        file.write('test_MAP_LL: {} \n'.format(MAP_LLs[split]))
        file.write('test_lap_LL: {} \n'.format(lap_LLs[split]))
        file.close() 

    # find the mean and std error of the RMSEs and LLs
    mean_MAP_RMSE = np.mean(MAP_RMSEs)
    sd_MAP_RMSE = np.std(MAP_RMSEs)
    mean_lap_RMSE = np.mean(lap_RMSEs)
    sd_lap_RMSE = np.std(lap_RMSEs)
    mean_MAP_LL = np.mean(MAP_LLs)
    sd_MAP_LL = np.std(MAP_LLs)
    mean_lap_LL = np.mean(lap_LLs)
    sd_lap_LL = np.std(lap_LLs)

    # save the answer
    file = open(results_dir + '/test_results.txt','w') 
    file.write('MAP_RMSEs: {} \n'.format(MAP_RMSEs))
    file.write('lap_RMSEs: {} \n'.format(lap_RMSEs))
    file.write('MAP_LLs: {} \n'.format(MAP_LLs))
    file.write('lap_LLs: {} \n'.format(lap_LLs))
    file.write('mean_MAP_RMSE: {} \n'.format(mean_MAP_RMSE))
    file.write('sd_MAP_RMSE: {} \n'.format(sd_MAP_RMSE))
    file.write('mean_lap_RMSE: {} \n'.format(mean_lap_RMSE))
    file.write('sd_lap_RMSE: {} \n'.format(sd_lap_RMSE))
    file.write('mean_MAP_LL: {} \n'.format(mean_MAP_LL))
    file.write('sd_MAP_LL: {} \n'.format(sd_MAP_LL))
    file.write('mean_lap_LL: {} \n'.format(mean_lap_LL))
    file.write('sd_lap_LL: {} \n'.format(sd_lap_LL))
    file.close() 

if __name__ == "__main__":

    # set RNG
    seed = 0
    np.random.seed(seed) 
    torch.manual_seed(seed) 

    input_dims = {'boston_housing': 13, 'concrete': 8, 'energy': 8, 'kin8nm': 8, 'power': 4, 'protein': 9, 'wine': 11, 'yacht': 6, 'naval': 16}
    datasets = ['naval', 'boston_housing', 'concrete', 'energy', 'kin8nm', 'power', 'protein', 'wine', 'yacht']

    # hyperparameters
    sample = False
    direct_invert = True
    standard_normal_prior = True
    activation_function = torch.tanh
    hidden_sizes = [50]
    no_samples_laplace = 100
    learned_noise_var = True
    subsampling = None
    noise_param_init = -1

    for dataset in datasets: 
        if dataset == 'protein':
            no_splits = 5
        else:
            no_splits = 20

        directory = './/experiments//' + dataset + '_yarin//double_Mar22'
        os.mkdir(directory)
        input_dim = input_dims[dataset]
        omega_range = [1.0, 2.0]
        minibatch_size_range = [100]
        learning_rate_range = [0.01, 0.001]
        no_epochs_range = [40, 100, 200, 400]

        # save text file with hyperparameters
        file = open(directory + '/hyperparameters.txt','w') 
        file.write('sample: {} \n'.format(sample))
        file.write('no_samples_laplace: {} \n'.format(no_samples_laplace))
        file.write('direct_invert: {} \n'.format(direct_invert))
        file.write('standard_normal_prior: {} \n'.format(standard_normal_prior))
        file.write('activation_function: {} \n'.format(activation_function.__name__))
        file.write('seed: {} \n'.format(seed))
        file.write('hidden_sizes: {} \n'.format(hidden_sizes))
        file.write('learned_noise_var: {} \n'.format(learned_noise_var))
        file.write('minibatch_size_range: {} \n'.format(minibatch_size_range))
        file.write('subsampling: {} \n'.format(subsampling))
        file.write('noise_param_init: {} \n'.format(noise_param_init))
        file.write('omega_range: {} \n'.format(omega_range))
        file.write('learning_rate_range: {} \n'.format(learning_rate_range))
        file.write('no_epochs_range: {} \n'.format(no_epochs_range))
        file.close() 

        all_RMSE = np.zeros(no_splits)
        all_MAPLL = np.zeros(no_splits)
        all_lapLL = np.zeros(no_splits)

        individual_tune_train(directory, standard_normal_prior, activation_function, hidden_sizes, omega_range, learning_rate_range, minibatch_size_range, no_epochs_range, input_dim, subsampling, noise_param_init, dataset)  

    


    

    




       



