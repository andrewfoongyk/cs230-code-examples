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

class MLP(nn.Module):
    def __init__(self, noise_variance, hidden_sizes, omega, activation=torch.tanh, learned_noise_var=False):
        super(MLP, self).__init__()
        self.activation = activation
        self.omega = omega
        self.learned_noise_var = learned_noise_var
        if learned_noise_var == False:
            self.noise_variance = torch.Tensor([noise_variance]).cuda()
        else:
            self.noise_var_param = nn.Parameter(torch.Tensor([-5]).cuda())
            self.noise_variance = self.get_noise_var(self.noise_var_param)
        self.hidden_sizes = hidden_sizes
        self.linears = nn.ModuleList([nn.Linear(1, self.hidden_sizes[0])])
        self.linears.extend([nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i+1]) for i in range(0, len(self.hidden_sizes)-1)])
        self.linears.append(nn.Linear(self.hidden_sizes[-1], 1))
        print(self.linears)

        # calculate number of parameters in network
        no_params = 1*self.hidden_sizes[0] # first weight matrix
        for i in range(len(self.hidden_sizes)-1):
            no_params = no_params + self.hidden_sizes[i] + self.hidden_sizes[i]*self.hidden_sizes[i+1]
        no_params = no_params + self.hidden_sizes[-1] + self.hidden_sizes[-1]*1 + 1 # final weight matrix and last 2 biases
        self.no_params = no_params

    def get_noise_var(self, noise_var_param):
        return torch.log(1 + torch.exp(noise_var_param)) + 1e-5

    def forward(self, x):
        batch_size = x.size()[0]
        x = x.view(batch_size, 1)
        for i, l in enumerate(self.linears):
            x = l(x)
            if i < len(self.linears) - 1:
                x = self.activation(x) ###### activation function very important for laplace
        return x

    def get_U(self, inputs, labels, trainset_size):
        minibatch_size = labels.shape()[0]

        if self.learned_noise_var == True:
            self.noise_variance = self.get_noise_var(self.noise_var_param)
        outputs = self.forward(inputs)
        labels = labels.reshape(labels.size()[0], 1)
        L2_term = 0
        for _, l in enumerate(self.linears): # Neal's prior (bias has variance 1)
            n_inputs = l.weight.size()[0]
            single_layer_L2 = 0.5*(n_inputs/(self.omega**2))*torch.sum(l.weight**2) + 0.5*torch.sum(l.bias**2)
            L2_term = L2_term + single_layer_L2

        #### I'M HERREEEEE - Scale the loss function to accommodate minibatching!    
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
        # forward pass
        output = self.forward(x) # output should be a scalar - batch size one here
        output = torch.squeeze(output)
        # backward pass
        output.backward()

        # fill gradient values into a single vector
        gradient = torch.cuda.FloatTensor(self.no_params).fill_(0)
        start_index = 0
        for _, param in enumerate(self.parameters()):
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
        P_vector[:1*self.hidden_sizes[0]] = 1/(self.omega**2) # first weight matrix
        start_index = 1*self.hidden_sizes[0]
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

    def linearised_laplace(self, train_inputs, test_inputs, subsampling=None): # return the posterior uncertainties for all the inputs
        
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
                x = torch.unsqueeze(x, 0) # this may not be necessary if x is multidimensional
                gradient = model.get_gradient(x)
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
                x = torch.unsqueeze(x, 0) # this may not be necessary if x is multidimensional
                gradient = model.get_gradient(x)
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
            gradient = model.get_gradient(x)
            # store in G
            G[:,i] = gradient
        # unsqueeze so its a batch of column vectors ###### maybe not necessary now?
        Gunsq = torch.unsqueeze(G, 1)
        
        #import pdb; pdb.set_trace()

        # calculate ZPinvG as a batch
        G_batch = Gunsq.permute(2,0,1) # make the batch index first (no_test x no_params x 1)
        Pinv_vector = 1/self.get_P_vector().unsqueeze(1) # column vector
        
        PinvG = Pinv_vector*G_batch # batch elementwise multiply (no_test x no_params x 1)
        ZPinvG = torch.matmul(Z, PinvG.squeeze().transpose(0,1)) # batch matrix multiply - (no_test x no_train)

        # calculate the 'inner matrix' M and Cholesky decompose
        PinvZt = Pinv_vector*torch.transpose(Z, 0, 1) # diagonal matrix multiplication      

        M = torch.eye(no_train).cuda() + (1/(self.noise_variance))*torch.matmul(Z, PinvZt)

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
        #print(predictive_var)
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

def get_H(model, x_train, subsample=None, num_subsamples=None, minibatch=None, batch_size=None, no_batches=None):
    # create 'sum of outer products' matrix
    # try subsampling or minibatching this 
    H = torch.cuda.FloatTensor(model.no_params, model.no_params).fill_(0)
    if subsample == True:
        # subsampled version
        for i in np.random.choice(x_train.size()[0], num_subsamples, replace=False): # for a random subsample
            # clear gradients
            optimizer.zero_grad()
            # get gradient of output wrt single training input
            x = x_train[i]
            x = torch.unsqueeze(x, 0)
            gradient = model.get_gradient(x)
            # form outer product
            outer = gradient.unsqueeze(1)*gradient.unsqueeze(0)
            H.add_(outer)
        # scale H to account for subsampling
        H = H*(x_train.size()[0]/num_subsamples)
    elif minibatch == True: 
        # minibatched version
        for i in range(no_batches):
            # randomly select a minibatch
            gradient = torch.cuda.FloatTensor(model.no_params).fill_(0)
            for j in np.random.choice(x_train.size()[0], batch_size, replace=False):
                # clear gradients
                optimizer.zero_grad()
                # get gradient of output wrt single training input
                x = x_train[j]
                x = torch.unsqueeze(x, 0)
                gradient.add_(model.get_gradient(x)) # accumulate minibatch gradients
            gradient = gradient/batch_size # average gradient over batch
            # form outer product
            outer = gradient.unsqueeze(1)*gradient.unsqueeze(0)
            H.add_(outer)
        # scale H to account for minibatching
        H = H*(x_train.size()[0]/no_batches)   
    else:
        for i in range(x_train.size()[0]): # for all training inputs
            # clear gradients
            optimizer.zero_grad()
            # get gradient of output wrt single training input
            x = x_train[i]
            x = torch.unsqueeze(x, 0)
            gradient = model.get_gradient(x)
            # form outer product
            outer = gradient.unsqueeze(1)*gradient.unsqueeze(0)
            H.add_(outer)
    #print(H)
    return H

if __name__ == "__main__":

    # set RNG
    seed = 0
    np.random.seed(seed) # 0
    torch.manual_seed(seed) #  230

    # hyperparameters
    activation_function = torch.tanh
    noise_variance = 0.01
    hidden_sizes = [256, 256, 256]
    omega = 4
    learning_rate = 1e-3
    no_iters = 20001
    plot_iters = 1000
    learned_noise_var = True
    minibatch_size = 10
    no_epochs = 40
    #subsample = True
    #num_subsamples = 5

    directory = './/experiments//boston_housing'
    #data_location = './/experiments//2_points_init//prior_dataset.pkl'
    data_location = '..//vision//data//boston_housing//boston_housing.pkl'

    # save text file with hyperparameters
    file = open(directory + '/hyperparameters.txt','w') 
    file.write('activation_function: {} \n'.format(activation_function.__name__))
    file.write('seed: {} \n'.format(seed))
    file.write('noise_variance: {} \n'.format(noise_variance))
    file.write('hidden_sizes: {} \n'.format(hidden_sizes))
    file.write('omega: {} \n'.format(omega))
    file.write('learning_rate: {} \n'.format(learning_rate))
    file.write('no_iters: {} \n'.format(no_iters))
    file.write('learned_noise_var: {} \n'.format(learned_noise_var))
    #file.write('subsample: {} \n'.format(subsample))
    #file.write('num_subsamples: {} \n'.format(num_subsamples))
    file.close() 

    # model
    model = MLP(noise_variance, hidden_sizes, omega, activation=activation_function, learned_noise_var=learned_noise_var)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)
    model.to(device)
    for param in model.parameters():
        print(type(param.data), param.size())

    # get dataset
    with open(data_location, 'rb') as f:
        train_set, train_set_normalised, test_set, train_mean, train_sd = pickle.load(f)

    x_train_normalised = torch.Tensor(train_set[:,:-1]).cuda()
    y_train_normalised = torch.Tensor(train_set[:,-1]).cuda()

    trainset_size = y_train_normalised.size[0]

    x_test = torch.Tensor(test_set[:,:-1]).cuda()
    y_test = torch.Tensor(test_set[:,-1]).cuda()

    # train MAP network
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(no_epochs): # loop over epochs
        # calculate the number of batches in this epoch
        no_batches = int(np.ceil(trainset_size/minibatch_size))
        print('Beginning epoch {}'.format(epoch))
        with trange(no_batches) as t: # loop over trainset
            for i in t:
                # shuffle the dataset
                idx = torch.randperm(trainset_size)
                x_train_normalised = x_train_normalised[idx,:] 
                y_train_normalised = y_train_normalised[idx,:] 
                
                # fetch the batch, but only if there is enough datapoints left
                if (i+1)*minibatch_size <= trainset_size - 1:
                    x_train_batch = x_train_normalised[i*minibatch_size:(i+1)*minibatch_size]
                    y_train_batch = y_train_normalised[i*minibatch_size:(i+1)*minibatch_size]

                # forward pass and calculate loss
                loss = model.get_U(x_train_batch, y_train_batch, trainset_size=trainset_size)
            
                # clear previous gradients, compute gradients of all variables wrt loss
                optimizer.zero_grad()
                loss.backward()

                # perform updates using calculated gradients
                optimizer.step()

                # update tqdm 
                if i % 10 == 0:
                    t.set_postfix(loss=loss.item())

                # plot the regression
                if i % plot_iters == 0:
                    plot_reg(model, data_load, directory, i)
                    plot_reg(model, data_load, directory, i)

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

    



    

    




       



