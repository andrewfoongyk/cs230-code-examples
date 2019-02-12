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

def memReport():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())

class MLP(nn.Module):
    def __init__(self, noise_variance, hidden_sizes, omega, activation=torch.tanh, learned_noise_var=False, input_dim=None, noise_param_init=None, standard_normal_prior=None):
        super(MLP, self).__init__()
        self.standard_normal_prior = standard_normal_prior
        self.dim_input = input_dim 
        self.activation = activation
        self.omega = float(omega)
        self.learned_noise_var = learned_noise_var
        if learned_noise_var == False:
            self.noise_variance = torch.Tensor([noise_variance]).cuda()
        else:
            # self.noise_var_param = nn.Parameter(torch.Tensor([-5]).cuda()) # this seemed to work OK
            self.noise_var_param = nn.Parameter(torch.Tensor([noise_param_init]).cuda())
            self.noise_variance = self.get_noise_var(self.noise_var_param)
        self.hidden_sizes = hidden_sizes
        self.linears = nn.ModuleList([nn.Linear(input_dim, self.hidden_sizes[0])])
        self.linears.extend([nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i+1]) for i in range(0, len(self.hidden_sizes)-1)])
        self.linears.append(nn.Linear(self.hidden_sizes[-1], 1))
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
        gradient = torch.cuda.FloatTensor(self.no_params).fill_(0)
        start_index = 0
        for name, param in self.named_parameters():
            if name != 'noise_var_param': # dont do laplace for noise variance
                grad_vec = param.grad.detach().reshape(-1) # flatten into a vector
                end_index = start_index + grad_vec.size()[0]
                gradient[start_index:end_index] = grad_vec # fill into single vector
                start_index = start_index + grad_vec.size()[0]

        return gradient

    def get_parameter_vector(self): 
        # load all the parameters into a single numpy vector
        parameter_vector = np.zeros(self.no_params)
        # fill parameter values into a single vector
        start_index = 0
        for _, param in enumerate(self.parameters()):
            param_vec = param.detach().reshape(-1) # flatten into a vector
            end_index = start_index + param_vec.size()[0]
            parameter_vector[start_index:end_index] = param_vec.cpu().numpy()
            start_index = start_index + param_vec.size()[0]
        return parameter_vector

    def get_P_vector(self):
        # get prior contribution to the Hessian
        P_vector = torch.cuda.FloatTensor(self.no_params).fill_(0)
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
        H = torch.cuda.FloatTensor(self.no_params, self.no_params).fill_(0)
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
        G = torch.cuda.FloatTensor(self.no_params, no_test).fill_(0)
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

        #import pdb; pdb.set_trace()
        print('noise_var: {}'.format(noise_variance))
        print('gtAinvg: {}'.format(gtAinvg))

        predictive_var = noise_variance + gtAinvg
        print('predictive_var: {}'.format(predictive_var))
        return predictive_var.detach()

    #def linearised_laplace_direct_cholesky(self, ):
    # do a numerically stable version of the algorithm

    def linearised_laplace(self, train_inputs, test_inputs, subsampling=None, optimizer=None): # return the posterior uncertainties for all the inputs

        if self.learned_noise_var == True:
            self.noise_variance = self.get_noise_var(self.noise_var_param)

        if subsampling == None: # don't subsample
            # form Jacobian matrix of train set, Z - use a for loop for now?
            no_train = train_inputs.size()[0]
            Z = torch.cuda.FloatTensor(no_train, self.no_params).fill_(0)
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
            Z = torch.cuda.FloatTensor(no_train, self.no_params).fill_(0)    
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
        G = torch.cuda.FloatTensor(self.no_params, no_test).fill_(0)
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
        M = M + torch.eye(no_train).cuda()*M[0,0]*1e-6 ########
          
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

def unpack_sample(model, parameter_vector):
    """convert a numpy vector of parameters to a list of parameter tensors"""
    sample = []
    start_index = 0
    end_index = 0   
    # unpack first weight matrix and bias
    end_index = end_index + model.hidden_sizes[0]
    weight_vector = parameter_vector[start_index:end_index]
    weight_matrix = weight_vector.reshape((model.hidden_sizes[0], 1))
    sample.append(torch.Tensor(weight_matrix).cuda())
    start_index = start_index + model.hidden_sizes[0]
    end_index = end_index + model.hidden_sizes[0]
    biases_vector = parameter_vector[start_index:end_index]
    sample.append(torch.Tensor(biases_vector).cuda())
    start_index = start_index + model.hidden_sizes[0]
    for i in range(len(model.hidden_sizes)-1): 
        end_index = end_index + model.hidden_sizes[i]*model.hidden_sizes[i+1]
        weight_vector = parameter_vector[start_index:end_index]
        weight_matrix = weight_vector.reshape((model.hidden_sizes[i+1], model.hidden_sizes[i]))
        sample.append(torch.Tensor(weight_matrix).cuda())
        start_index = start_index + model.hidden_sizes[i]*model.hidden_sizes[i+1]
        end_index = end_index + model.hidden_sizes[i+1]
        biases_vector = parameter_vector[start_index:end_index]
        sample.append(torch.Tensor(biases_vector).cuda())
        start_index = start_index + model.hidden_sizes[i+1]
    # unpack output weight matrix and bias
    end_index = end_index + model.hidden_sizes[-1]
    weight_vector = parameter_vector[start_index:end_index]
    weight_matrix = weight_vector.reshape((1, model.hidden_sizes[-1]))
    sample.append(torch.Tensor(weight_matrix).cuda())
    start_index = start_index + model.hidden_sizes[-1]
    biases_vector = parameter_vector[start_index:] # should reach the end of the parameters vector at this point
    sample.append(torch.Tensor(biases_vector).cuda())
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

    # invert A just once
    if laplace == True and direct_invert == True:
        Ainv = model.get_Ainv(x_train_normalised, optimizer)

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
        if laplace == False: 
            noise_var = model.get_noise_var(model.noise_var_param)
            # scale the noise var because of the normalisation
            noise_var = noise_var*(train_sd[-1]**2)            
            log_likelihood = -0.5*actual_batch_size*torch.log(2*3.1415926536*noise_var) - (1/(2*noise_var))*squared_error
        else: # do Laplace approximation
            # get the predictive variances
            if direct_invert == True:
                predictive_var = model.linearised_laplace_direct(Ainv, inputs, optimizer)
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
            #import pdb; pdb.set_trace()
                    
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
                    results_dict = {'MAP_MSE':MAP_MSE, 'MAP_LL':MAP_LL, 'lap_LL':lap_LL, 'no_epochs':epoch + 1}
                    results_dict_list.append(results_dict)

    if early_stopping == True:
        return results_dict_list 

def train_all(test, noise_variance, hidden_sizes, omega, activation_function, learned_noise_var, input_dim, noise_param_init, standard_normal_prior):
    if test == True: # train on the train and val set combined, evaluate on test set
        # train on various splits
        for split in range(no_splits):
            print('BEGINNING SPLIT {}'.format(split))
            # dataset       
            data_location = '..//vision//data//boston_housing_yarin//boston_housing' + str(split) + '.pkl'

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

            train_mean = torch.Tensor(train_mean).cuda()
            train_sd = torch.Tensor(train_sd).cuda()

            x_train_normalised = torch.Tensor(train_set_normalised[:,:-1]).cuda()
            y_train_normalised = torch.Tensor(train_set_normalised[:,-1]).cuda()

            x_val_normalised = torch.Tensor(val_set_normalised[:,:-1]).cuda()
            y_val_normalised = torch.Tensor(val_set_normalised[:,-1]).cuda()

            # concatenate train and val set
            x_train_normalised = torch.cat((x_train_normalised, x_val_normalised), 0)
            y_train_normalised = torch.cat((y_train_normalised, y_val_normalised), 0)

            x_test = torch.Tensor(test_set[:,:-1]).cuda()
            y_test = torch.Tensor(test_set[:,-1]).cuda()

            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # train the model, and print out the test set log likelihood when training
            train(model, x_train_normalised, y_train_normalised, x_test, y_test, train_mean, train_sd, validation=False, minibatch_size=minibatch_size, no_epochs=no_epochs, subsampling=subsampling, optimizer=optimizer)
            
            # record the final TEST SET scores
            MAP_MSE, MAP_LL = evaluate(model, x_test, y_test, train_mean, train_sd, validation=False, optimizer=optimizer) # without laplace
            lap_MSE, lap_LL = evaluate(model, x_test, y_test, train_mean, train_sd, laplace=True, x_train_normalised=x_train_normalised, subsampling=subsampling, validation=False, optimizer=optimizer) # with laplace
            all_RMSE[split] = torch.sqrt(MAP_MSE).data.cpu().numpy()
            all_MAPLL[split] = MAP_LL.data.cpu().numpy()
            all_lapLL[split] = lap_LL.data.cpu().numpy()

    else: # train on train set, evaluate on val set,
        # train on various splits
        for split in range(no_splits):
            print('BEGINNING SPLIT {}'.format(split))
            # dataset       
            data_location = '..//vision//data//boston_housing_yarin//boston_housing' + str(split) + '.pkl'

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

            train_mean = torch.Tensor(train_mean).cuda()
            train_sd = torch.Tensor(train_sd).cuda()

            x_train_normalised = torch.Tensor(train_set_normalised[:,:-1]).cuda()
            y_train_normalised = torch.Tensor(train_set_normalised[:,-1]).cuda()

            x_val_normalised = torch.Tensor(val_set_normalised[:,:-1]).cuda()
            y_val_normalised = torch.Tensor(val_set_normalised[:,-1]).cuda()

            x_test = torch.Tensor(test_set[:,:-1]).cuda()
            y_test = torch.Tensor(test_set[:,-1]).cuda()

            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # train the model, and print out the validation set log likelihood when training
            train(model, x_train_normalised, y_train_normalised, x_val_normalised, y_val_normalised, train_mean, train_sd, validation=True, minibatch_size=minibatch_size, no_epochs=no_epochs, subsampling=subsampling, optimizer=optimizer)
            
            # record the final VALIDATION SET scores
            MAP_MSE, MAP_LL = evaluate(model, x_val_normalised, y_val_normalised, train_mean, train_sd, validation=True, optimizer=optimizer) # without laplace
            lap_MSE, lap_LL = evaluate(model, x_val_normalised, y_val_normalised, train_mean, train_sd, laplace=True, x_train_normalised=x_train_normalised, subsampling=subsampling, validation=True, optimizer=optimizer) # with laplace
            all_RMSE[split] = torch.sqrt(MAP_MSE).data.cpu().numpy()
            all_MAPLL[split] = MAP_LL.data.cpu().numpy()
            all_lapLL[split] = lap_LL.data.cpu().numpy()
    
    print('MSEs: {}'.format(all_RMSE))
    print('MAP LLs: {}'.format(all_MAPLL))
    print('lap LLs: {}'.format(all_lapLL))
    print('avg MSE: {}'.format(np.mean(all_RMSE)))
    print('MSE s.d.: {}'.format(np.std(all_RMSE)))
    print('avg MAP LL: {}'.format(np.mean(all_MAPLL)))
    print('MAP LL s.d.: {}'.format(np.std(all_MAPLL)))
    print('avg lap LL: {}'.format(np.mean(all_lapLL)))
    print('lap LL s.d.: {}'.format(np.std(all_lapLL)))

    # save text file with results
    file = open(directory + '/results.txt','w') 
    file.write('MSEs: {} \n'.format(all_RMSE))
    file.write('MAP LLs: {} \n'.format(all_MAPLL))
    file.write('lap LLs: {} \n'.format(all_lapLL))
    file.write('avg MSE: {} \n'.format(np.mean(all_RMSE)))
    file.write('MSE s.d.: {} \n'.format(np.std(all_RMSE)))
    file.write('avg MAP LL: {} \n'.format(np.mean(all_MAPLL)))
    file.write('MAP LL s.d.: {} \n'.format(np.std(all_MAPLL)))
    file.write('avg lap LL: {} \n'.format(np.mean(all_lapLL)))
    file.write('lap LL s.d.: {} \n'.format(np.std(all_lapLL)))
    file.close() 

def individual_train(data_location, test, noise_variance, hidden_sizes, omega, activation_function, learned_noise_var, input_dim, noise_param_init, learning_rate, no_epochs, standard_normal_prior, results_dir=None, split=None, early_stopping=False):
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

    train_mean = torch.Tensor(train_mean).cuda()
    train_sd = torch.Tensor(train_sd).cuda()

    x_train_normalised = torch.Tensor(train_set_normalised[:,:-1]).cuda()
    y_train_normalised = torch.Tensor(train_set_normalised[:,-1]).cuda()

    x_val_normalised = torch.Tensor(val_set_normalised[:,:-1]).cuda()
    y_val_normalised = torch.Tensor(val_set_normalised[:,-1]).cuda()

    if test == True: # combine train and val sets
        x_train_normalised = torch.cat((x_train_normalised, x_val_normalised), 0)
        y_train_normalised = torch.cat((y_train_normalised, y_val_normalised), 0)

    x_test = torch.Tensor(test_set[:,:-1]).cuda()
    y_test = torch.Tensor(test_set[:,-1]).cuda()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # train the model, and print out the validation set log likelihood when training

    if test == True: # this is test time, no early stopping needed
        train(model, x_train_normalised, y_train_normalised, x_test, y_test, train_mean, train_sd, validation=False, minibatch_size=minibatch_size, no_epochs=no_epochs, subsampling=subsampling, optimizer=optimizer)
        MAP_MSE, MAP_LL = evaluate(model, x_test, y_test, train_mean, train_sd, validation=False, optimizer=optimizer) # without laplace
        lap_MSE, lap_LL = evaluate(model, x_test, y_test, train_mean, train_sd, laplace=True, x_train_normalised=x_train_normalised, subsampling=subsampling, validation=False, optimizer=optimizer, directory=results_dir, name=str(split)) # with laplace
        return MAP_MSE, MAP_LL, lap_LL

    else: # this is validation time, do early stopping for hyperparam search
        results_dict_list = train(model, x_train_normalised, y_train_normalised, x_val_normalised, y_val_normalised, train_mean, train_sd, validation=True, minibatch_size=minibatch_size, no_epochs=no_epochs, subsampling=subsampling, optimizer=optimizer, early_stopping=True)
        return results_dict_list

def individual_tune_train(results_dir, standard_normal_prior, activation_function, hidden_sizes, omega_range, learning_rate_range, minibatch_size, no_epochs_range, input_dim, subsampling, noise_param_init):
    # do a grid search on each split separately, then evaluate on the test set
    # this grids over omega, minibatch_size and no_epochs
    
    # create array of values to grid search over - but don't repeat searches when doing early stopping
    list_hypers = [omega_range, learning_rate_range]
    hyperparams = cartesian(list_hypers)
    RMSEs = np.zeros(no_splits)
    MAP_LLs = np.zeros(no_splits)
    lap_LLs = np.zeros(no_splits)

    for split in range(no_splits): 
        # find data location
        data_location = '..//vision//data//energy_yarin//energy' + str(split) + '.pkl'
        test = False # do hyperparam grid search on validation set
        for i in range(hyperparams.shape[0]):
            # get hyperparams
            copy_hyperparams = deepcopy(hyperparams)
            omega = copy_hyperparams[i,0]
            learning_rate = copy_hyperparams[i,1]

            # train on one split, and validate            
            noise_variance = 0 # not using this parameter
            learned_noise_var = True # always true for UCI regression
            results_dict_list = individual_train(data_location, test, noise_variance, hidden_sizes, omega, activation_function, learned_noise_var, input_dim, noise_param_init, learning_rate, no_epochs_range, standard_normal_prior, early_stopping=True)
            
            # save text file with results
            for results_dict in results_dict_list:
                file = open(results_dir + '/results' + str(split) + '.txt','a') 
                file.write('omega, learning_rate: {} \n'.format(hyperparams[i,:]))
                file.write('no_epochs: {} \n'.format(results_dict['no_epochs']))
                file.write('MAP_RMSE: {} \n'.format(torch.sqrt(results_dict['MAP_MSE'])))
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
        no_epochs = best_no_epochs

        MAP_MSE, MAP_LL, lap_LL = individual_train(data_location, test, noise_variance, hidden_sizes, omega, activation_function, learned_noise_var, input_dim, noise_param_init, learning_rate, no_epochs, standard_normal_prior, results_dir, split)
        RMSEs[split] = torch.sqrt(MAP_MSE).data.cpu().numpy()
        MAP_LLs[split] = MAP_LL.data.cpu().numpy()
        lap_LLs[split] = lap_LL.data.cpu().numpy()
        
        # record best hyperparams
        file = open(results_dir + '/best_hypers.txt','a') 
        file.write('split: {} \n'.format(str(split)))
        file.write('omega: {} \n'.format(omega))
        file.write('learning_rate: {} \n'.format(learning_rate))
        file.write('no_epochs: {} \n'.format(no_epochs))
        file.write('test_RMSE: {} \n'.format(RMSEs[split]))
        file.write('test_MAP_LL: {} \n'.format(MAP_LLs[split]))
        file.write('test_lap_LL: {} \n'.format(lap_LLs[split]))
        file.close() 

    # find the mean and std error of the RMSEs and LLs
    mean_RMSE = np.mean(RMSEs)
    sd_RMSE = np.std(RMSEs)
    mean_MAP_LL = np.mean(MAP_LLs)
    sd_MAP_LL = np.std(MAP_LLs)
    mean_lap_LL = np.mean(lap_LLs)
    sd_lap_LL = np.std(lap_LLs)

    # save the answer
    file = open(results_dir + '/test_results.txt','w') 
    file.write('RMSEs: {} \n'.format(RMSEs))
    file.write('MAP_LLs: {} \n'.format(MAP_LLs))
    file.write('lap_LLs: {} \n'.format(lap_LLs))
    file.write('mean_RMSE: {} \n'.format(mean_RMSE))
    file.write('sd_RMSE: {} \n'.format(sd_RMSE))
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

    # hyperparameters
    no_splits = 20
    direct_invert = False
    standard_normal_prior = True
    activation_function = torch.tanh
    noise_variance = 0.01
    hidden_sizes = [50, 50]
    omega = 4
    learning_rate = 0.01
    learned_noise_var = True
    minibatch_size = 100
    no_epochs = 400
    input_dim = 8
    subsampling = 100
    noise_param_init = -1
    test = False
    directory = './/experiments//energy_yarin//subsampling'

    omega_range = [1.0, 2.0, 4.0]
    learning_rate_range = [0.01, 0.005, 0.001]
    no_epochs_range = [40, 100, 200]

    # save text file with hyperparameters
    file = open(directory + '/hyperparameters.txt','w') 
    file.write('direct_invert: {} \n'.format(direct_invert))
    file.write('standard_normal_prior: {} \n'.format(standard_normal_prior))
    file.write('activation_function: {} \n'.format(activation_function.__name__))
    file.write('seed: {} \n'.format(seed))
    file.write('hidden_sizes: {} \n'.format(hidden_sizes))
    file.write('omega: {} \n'.format(omega))
    file.write('learning_rate: {} \n'.format(learning_rate))
    file.write('learned_noise_var: {} \n'.format(learned_noise_var))
    file.write('minibatch_size: {} \n'.format(minibatch_size))
    file.write('no_epochs: {} \n'.format(no_epochs))
    file.write('subsampling: {} \n'.format(subsampling))
    file.write('noise_param_init: {} \n'.format(noise_param_init))
    file.write('test: {} \n'.format(test))
    file.write('omega_range: {} \n'.format(omega_range))
    file.write('learning_rate_range: {} \n'.format(learning_rate_range))
    file.write('no_epochs_range: {} \n'.format(no_epochs_range))
    file.close() 

    all_RMSE = np.zeros(no_splits)
    all_MAPLL = np.zeros(no_splits)
    all_lapLL = np.zeros(no_splits)

    #train_all(test, noise_variance, hidden_sizes, omega, activation_function, learned_noise_var, input_dim, noise_param_init, standard_normal_prior)

    individual_tune_train(directory, standard_normal_prior, activation_function, hidden_sizes, omega_range, learning_rate_range, minibatch_size, no_epochs_range, input_dim, subsampling, noise_param_init)  

    ############################################

    # evaluate the model on the test set
    # MAP_MSE, MAP_LL = evaluate(model, x_test, y_test, train_mean, train_sd, validation=False) # without laplace
    # lap_MSE, lap_NLL = evaluate(model, x_test, y_test, train_mean, train_sd, laplace=True, x_train_normalised=x_train_normalised, subsampling=None, validation=False) # with laplace
    # print('test MAP_RMSE: {}'.format(torch.sqrt(MAP_MSE)))
    # print('test MAP_LL: {}'.format(MAP_LL))
    # print('test lap_MSE: {}'.format(torch.sqrt(lap_MSE)))
    # print('test lap_NLL: {}'.format(lap_NLL))

    ############################################
        
    # # Laplace approximation
    # MAP = model.get_parameter_vector()

    # # get the fit using woodbury identity
    # # evaluate model on test points
    # N = 100 # number of test points
    # x_lower = -3
    # x_upper = 3
    # test_inputs_np = np.linspace(x_lower, x_upper, N)
    # # move to GPU if available
    # test_inputs = torch.FloatTensor(test_inputs_np)
    # test_inputs = test_inputs.cuda(async=True)
    # test_inputs = torch.unsqueeze(test_inputs, 1) 

    # predictive_mean = model(test_inputs)

    # ##########################################
    # # use O(no_data^3) matrix inversion

    # # dont subsample
    # predictive_var = model.linearised_laplace(x_train, test_inputs)
    # predictive_sd = torch.sqrt(predictive_var)
    # # plot
    # plot(test_inputs, predictive_mean, predictive_sd, directory)

    # # subsample and plot
    # for num_subsamples in [100, 70, 40, 20, 10, 5, 4, 3, 2, 1]:
    #     predictive_var = model.linearised_laplace(x_train, test_inputs, num_subsamples)
    #     predictive_sd = torch.sqrt(predictive_var)
    #     # plot
    #     plot(test_inputs, predictive_mean, predictive_sd, directory, title = '_subsample_' + str(num_subsamples))

    ##########################################

    # # get the fit without any subsampling
    # # get H
    # H = get_H(model, x_train, subsample=False, num_subsamples=None)
    # # print out the outer product Hessian
    # plot_cov(H.data.cpu().numpy(), directory, title='Hessian_outer_product_')

    # # get prior contribution to Hessian
    # P_vector = model.get_P_vector()
    # P = torch.diag(P_vector)
    # # calculate and invert (negative) Hessian of posterior
    # A = (1/model.noise_variance)*H + P 
    # Ainv = torch.inverse(A)    
    # # plot regression with error bars using linearisation
    # plot_reg(model, data_load, directory, iter_number=no_iters, linearise=True, Ainv=Ainv)
    # # plot covariance and correlation matrix
    # plot_cov(Ainv.data.cpu().numpy(), directory)

    ###########################################
    # try SVD's of different ranks

    # for R in [1,2,3,4,5]:
    #     # perform an SVD and keep only the R largest singular values
    #     U,S,V = torch.svd(H)
    #     H_approx = torch.zeros(model.no_params, model.no_params).cuda()
    #     for i in range(R):
    #         H_approx.add_(U[:,i].unsqueeze(1)*V[:,i].unsqueeze(0)*S[i])
    #     plot_cov(H_approx.data.cpu().numpy(), directory, title='Hessian_SVD_rank_' + str(R))

    #     # get prior contribution to Hessian
    #     P_vector = model.get_P_vector()
    #     P = torch.diag(P_vector)
    #     # calculate and invert (negative) Hessian of posterior
    #     A = (1/model.noise_variance)*H_approx + P 
    #     Ainv = torch.inverse(A)    
    #     # plot regression with error bars using linearisation
    #     plot_reg(model, data_load, directory, iter_number=no_iters, linearise=True, Ainv=Ainv, title='SVD_rank_' + str(R))
    #     # plot covariance and correlation matrix
    #     plot_cov(Ainv.data.cpu().numpy(), directory)

    ###########################################

    # # try minibatching
    # H = get_H(model, x_train, subsample=None, num_subsamples=None, minibatch=True, batch_size=100, no_batches=1)
    # # print out the outer product Hessian
    # plot_cov(H.data.cpu().numpy(), directory, title='Hessian_OP_minibatch_')
    # # calculate and invert (negative) Hessian of posterior
    # A = (1/model.noise_variance)*H + P 
    # Ainv = torch.inverse(A)    
    # # plot regression with error bars using linearisation
    # plot_reg(model, data_load, directory, iter_number=no_iters, linearise=True, Ainv=Ainv, title='minibatching')
    # # plot covariance and correlation matrix
    # plot_cov(Ainv.data.cpu().numpy(), directory)

    ###########################################

    # # get the fits with subsampling
    # for subsampling in [1, 2, 3, 5, 10, 20, 50, 70, 100]:
    #     # get H
    #     H = get_H(model, x_train, subsample=True, num_subsamples=subsampling)
    #     # print out the outer product Hessian
    #     plot_cov(H.data.cpu().numpy(), directory, title='Hessian_OP_subsampling_' + str(subsampling) + '_')
    #     # calculate and invert (negative) Hessian of posterior
    #     A = (1/model.noise_variance)*H + P 
    #     Ainv = torch.inverse(A)    
    #     # plot regression with error bars using linearisation
    #     plot_reg(model, data_load, directory, iter_number=no_iters, linearise=True, Ainv=Ainv, title='subsampling_' + str(subsampling))
    #     # plot covariance and correlation matrix
    #     plot_cov(Ainv.data.cpu().numpy(), directory)

    ##########################################

    # # find the eigenvalues of A
    # w, v = LA.eig(A.data.cpu().numpy())
    # w = np.sort(w)
    # print('eigenvalues: {}'.format(w))

    # # print the Jacobian
    # model.linearised_laplace(x_train, None)

    # print out A
    # plot_cov(A.data.cpu().numpy(), directory, title='A_')

    # plot regression with error bars using sampling  
    # for covscale in [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]:
    #    plot_reg(model, data_load, directory, iter_number=no_iters, linearise=False, Ainv=Ainv, sampling=True, covscale=covscale, mean=MAP)

    



    

    




       



