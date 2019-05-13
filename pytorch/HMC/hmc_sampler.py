import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from copy import deepcopy
from tensorboardX import SummaryWriter
import os
import cProfile

def plot_regression(model, samples, directory, no_plot_samples):
    plt.figure()
    plt.rcParams.update({'font.size': 14})
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    axes = plt.gca()
    axes.set_xlim([-3, 3])
    axes.set_ylim([-3, 3])
    plt.plot(data_load[:,0], data_load[:,1], '+k', markersize=16)
    N = 1000
    x_lower = -6
    x_upper = 8
    x_values = np.linspace(x_lower, x_upper, N)
    test_inputs = torch.FloatTensor(x_values).cuda(async=True)

    # subsample for plotting only
    indeces = np.arange(0, len(samples))
    subsampled_indeces = np.random.choice(indeces, no_plot_samples, replace=False)
    subsampled_indeces = subsampled_indeces.astype(int)
    samples = [samples[i] for i in subsampled_indeces]

    # plot all the samples
    no_samp = len(samples)
    all_test_outputs = np.zeros((no_samp, N))
    for i, sample in enumerate(samples):
        for j, param in enumerate(model.parameters()):
            param.data = sample[j] # fill in the model with these weights
        # plot regression
        test_outputs = model(test_inputs)
        # convert back to np array
        test_outputs = test_outputs.data.cpu().numpy()
        test_outputs = test_outputs.reshape(N)
        plt.plot(x_values, test_outputs, linewidth=1)
        # save data for ensemble mean and s.d. calculation
        all_test_outputs[i,:] = test_outputs

    # calculate mean and variance
    mean = np.mean(all_test_outputs, 0)
    variance = model.noise_variance + np.mean(all_test_outputs**2, 0) - mean**2

    plt.plot(x_values, mean, color='b')
    plt.fill_between(x_values, mean + np.sqrt(variance), mean - np.sqrt(variance), color='b', alpha=0.3)

    filepath = os.path.join(directory, 'HMC_regression.pdf')
    plt.savefig(filepath)
    plt.close()

    # pickle everything as numpy arrays for posterity
    inputs_hmc = x_values
    mean_hmc = mean
    sd_hmc = np.sqrt(variance)

    pickle_location = os.path.join(directory, 'plot_hmc_relu')
    outfile = open(pickle_location, 'wb')
    pickle.dump(inputs_hmc, outfile)
    pickle.dump(mean_hmc, outfile)
    pickle.dump(sd_hmc, outfile)
    outfile.close()

def plot_covariance(hidden_sizes, samples, directory):
    # plot the empirical covariance matrix (use unbiased estimator?) and the correlation matrix
    
    # calculate number of parameters in the network
    no_params = hidden_sizes[0] # first weight matrix
    for i in range(len(hidden_sizes)-1):
        no_params = no_params + hidden_sizes[i] + hidden_sizes[i]*hidden_sizes[i+1]
    no_params = no_params + hidden_sizes[-1] + hidden_sizes[-1] + 1 # final weight matrix and last 2 biases

    sampled_parameters = np.zeros((no_params, len(samples)))
    for i, sample in enumerate(samples):
        # send parameters to numpy arrays and pack the parameters into a vector
        start_index = 0
        for _, param in enumerate(sample):
            if len(param.size()) > 1:
                param = torch.t(param) # nn.Linear does a transpose for some reason, so undo this
            param = param.cpu().detach().numpy()
            param = param.reshape(-1) # flatten into a vector
            end_index = start_index + param.size
            sampled_parameters[start_index:end_index, i] = param # fill into array
            start_index = start_index + param.size
    
    # calculate sample mean
    sample_mean = np.mean(sampled_parameters, axis=1)
    sample_mean = sample_mean.reshape(-1, 1) # numpy broadcasting thing
    centered_samples = sampled_parameters - sample_mean

    # calculate empirical covariances
    cov = np.zeros((no_params, no_params))
    for i in range(len(samples)):
        cov = cov + np.outer(centered_samples[:,i], centered_samples[:,i])
    cov = cov/len(samples) # max likelihood estimator of covariance

    fig, ax = plt.subplots()
    im = ax.imshow(np.abs(cov) , interpolation='nearest', cmap=cm.Greys_r)
    filepath = os.path.join(directory, 'covariance.pdf')
    fig.savefig(filepath)
    plt.close()

    # plot correlation matrix using cov matrix estimate
    variance_vector = np.diag(cov)
    sd_vector = np.sqrt(variance_vector)
    outer_prod = np.outer(sd_vector, sd_vector)
    correlations = cov/outer_prod

    fig, ax = plt.subplots()
    im = ax.imshow(correlations , interpolation='nearest')
    fig.colorbar(im)
    filepath = os.path.join(directory, 'correlation.pdf')
    fig.savefig(filepath)
    plt.close()

class MLP(nn.Module):
    def __init__(self, noise_variance, hidden_sizes, omega):
        super(MLP, self).__init__()
        self.omega = omega
        self.noise_variance = noise_variance
        self.hidden_sizes = hidden_sizes
        self.linears = nn.ModuleList([nn.Linear(1, self.hidden_sizes[0])])
        self.linears.extend([nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i+1]) for i in range(0, len(self.hidden_sizes)-1)])
        self.linears.append(nn.Linear(self.hidden_sizes[-1], 1))

        # ################ custom initialisation
        # self.linears[0].weight.data = torch.Tensor([[-1.414]])
        # self.linears[0].bias.data = torch.Tensor([0])
        # self.linears[1].weight.data = torch.Tensor([[-1.414]])
        # self.linears[1].bias.data = torch.Tensor([2])
        # ################

        ################ custom initialisation for 2 points in-between uncertainty
        # self.linears[0].weight.data = torch.Tensor([[1], [1]])
        # self.linears[0].bias.data = torch.Tensor([0.5, -0.5])
        # self.linears[1].weight.data = torch.Tensor([[1, 1]])
        # self.linears[1].bias.data = torch.Tensor([0.2])

        print(self.linears)
        
    def forward(self, x):
        batch_size = x.size()[0]
        x = x.view(batch_size, 1)
        for i, l in enumerate(self.linears):
            x = l(x)
            if i < len(self.linears) - 1:
                x = torch.tanh(x) ######################## playing with activation 
        return x

    def get_U(self, inputs, labels):
        outputs = self.forward(inputs)
        labels = labels.reshape(labels.size()[0], 1)
        L2_term = 0
        for _, l in enumerate(self.linears): # Neal's prior (bias has variance 1)
            n_inputs = l.weight.size()[0]
            single_layer_L2 = 0.5*(n_inputs/(self.omega**2))*torch.sum(l.weight**2) + 0.5*torch.sum(l.bias**2)
            L2_term = L2_term + single_layer_L2
        error = (1/(2*self.noise_variance))*torch.sum((labels - outputs)**2)
        U = error + L2_term
        return U

"""This version of HMC follows https://arxiv.org/pdf/1206.1901.pdf. Identity mass matrix used"""

class HMC_Sampler:
    def __init__(self, inputs, targets, step_size = 0.002, num_steps = 20, no_samples = 20000, burn_in = 1000, thinning = 1):
        self.step_size = step_size
        self.num_steps = num_steps
        self.no_samples = no_samples
        self.burn_in = burn_in
        self.inputs = inputs
        self.targets = targets
        self.thinning = thinning
        self.no_accept = 0

    def get_samples(self, model):
        """run the HMC sampler and save the samples"""
        print('Beginning burn-in phase of {} samples'.format(self.burn_in))
        # don't save the burn-in samples
        for i in tqdm(range(self.burn_in)):
            new_parameters, energy = self.HMC_transition(model)
            writer.add_scalar('Energy', energy, i)
            if i%100 == 0:
                if i != 0:
                    print('Acceptance rate: {}%'.format(self.no_accept))
                self.no_accept = 0
            for i, param in enumerate(model.parameters()):
                param.data = new_parameters[i]    
        print('Burn-in phase finished, collecting {} samples with thinning of {}.'.format(self.no_samples, self.thinning))  
        samples = []
        for i in tqdm(range(self.no_samples)):
            # get new parameters and use them to replace the old ones
            new_parameters, energy = self.HMC_transition(model)
            writer.add_scalar('Energy', energy, i + self.burn_in)
            for j, param in enumerate(model.parameters()):
                param.data = new_parameters[j] 

            # save the new parameters
            if i%self.thinning == 0:
                    samples.append(deepcopy(new_parameters))
            
            # print the acceptance rate
            if i%100 == 0:
                if i != 0:
                    print('Acceptance rate: {}%'.format(self.no_accept))
                self.no_accept = 0
        print('Done collecting samples')
        return samples 

    def HMC_transition(self, model):
        """perform one transition of the markov chain"""
        # randomise the step size and number of steps
        step_size = np.random.uniform(self.step_size[0], self.step_size[1])
        num_steps = np.random.randint(self.num_steps[0], self.num_steps[1] + 1)

        saved_params = deepcopy(list(model.parameters())) # list of all the parameter objects - positions
        p = [] # list of momenta
        for _, param in enumerate(model.parameters()): 
            param_size = param.size()
            # independent standard normal variates for corresponding momenta
            p.append(torch.cuda.FloatTensor(param_size).normal_(0, 1)) 
       
        # get gradients of U wrt parameters q
        U = model.get_U(self.inputs, self.targets) # get log posterior (up to constant)
        start_U = U.clone() # save the starting potential energy

        # save the starting kinetic energy
        start_K = torch.cuda.FloatTensor(1).fill_(0) # a zero 
        for momentum in p:
            start_K = start_K + torch.sum(momentum**2)/2

        U.backward() 

        # make half step for momentum at the beginning
        for i, momentum in enumerate(p):
            momentum += - step_size*list(model.parameters())[i].grad.data/2

        # alternate full steps for position and momentum
        for i in range(num_steps):            
            # make a full step for the position
            for l, param in enumerate(model.parameters()):
                param.data += step_size*p[l]

            # zero gradients of U wrt parameters q
            for _, param in enumerate(model.parameters()): 
                param.grad.data.zero_()
            # get gradients of U wrt parameters q
            U = model.get_U(self.inputs, self.targets) # get log posterior (up to constant)
            U.backward() 

            # make a full step for the momentum, except at end of trajectory <-- check this ################
            if not (i == (num_steps-1)):
                for j, momentum in enumerate(p):
                    momentum += - step_size*list(model.parameters())[j].grad.data

        # make a half step for momentum at the end
        for i, momentum in enumerate(p):
            momentum += - step_size*list(model.parameters())[i].grad.data/2

        # negate momentum at the end of trajectory to make the proposal symmetric
        # can probably skip this without effect

        # evaluate potential and kinetic energies at end of trajectory
        end_U = model.get_U(self.inputs, self.targets)

        end_K = torch.cuda.FloatTensor(1).fill_(0) # a zero 
        for momentum in p:
            end_K = end_K + torch.sum(momentum**2)/2

        # zero gradients of U wrt parameters q
        for _, param in enumerate(model.parameters()): 
            param.grad.data.zero_()

        # Accept or reject the state at end of trajectory, returning either the position 
        # at the end of the trajectory or the initial position

        if np.random.uniform(0, 1) < torch.exp(start_U - end_U + start_K - end_K).cpu().detach().numpy():
            self.no_accept = self.no_accept + 1
            return list(model.parameters()), end_U # accept
        else:
            return saved_params, start_U # reject


if __name__ == "__main__":

    # set RNG
    np.random.seed(0) # 0
    torch.manual_seed(230) #  230

    # hyperparameters
    noise_variance = 0.01 # 0.01
    hidden_sizes = [50]
    omega = 4

    burn_in = 10000
    no_samples = 20000
    no_saved_samples = 1000
    no_plot_samples = 100 #32
    step_size = [0.001, 0.0015]
    num_steps = [5, 10]

    directory = './/experiments//ICML_tanh2'
    #data_location = './/experiments//2_points_init//prior_dataset.pkl'
    data_location = '..//vision//data//1D_COSINE//1d_cosine_separated.pkl'

    # set up tensorboard
    tensorboard_path = os.path.join(directory, 'tensorboard')
    writer = SummaryWriter(tensorboard_path)

    # model
    net = MLP(noise_variance, hidden_sizes, omega)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)
    net.to(device)
    for param in net.parameters():
        print(type(param.data), param.size())

    # get dataset
    with open(data_location, 'rb') as f:
            data_load = pickle.load(f)

    #data_load = data_load[0] # just the points not the line

    x_train = torch.Tensor(data_load[:,0]).cuda()
    y_train = torch.Tensor(data_load[:,1]).cuda()

    # # profile the code
    # pr = cProfile.Profile()
    # pr.enable()

    # HMC sample  
    thinning = int(np.ceil(no_samples/no_saved_samples))
    sampler = HMC_Sampler(inputs = x_train, targets = y_train, step_size = step_size, num_steps = num_steps, 
        no_samples = no_samples, burn_in = burn_in, thinning=thinning)
    samples = sampler.get_samples(net) 
    no_saved_samples = len(samples) # the actual number of samples saved

    # pr.disable()
    # pr.print_stats()

    # pickle the samples
    filename = directory + '//HMC_samples'
    outfile = open(filename, 'wb')
    pickle.dump(samples, outfile)
    outfile.close()

    # plot and save plot of network output
    plot_regression(net, samples, directory, no_plot_samples)

    # plot empirical covariance
    plot_covariance(hidden_sizes, samples, directory)   

    # save text file with hyperparameters
    file = open(directory + '/hyperparameters.txt','w') 
    file.write('noise_variance: {} \n'.format(noise_variance))
    file.write('hidden_sizes: {} \n'.format(hidden_sizes))  
    file.write('omega: {} \n'.format(omega))   
    file.write('burn_in: {} \n'.format(burn_in))
    file.write('no_samples: {} \n'.format(no_samples)) 
    file.write('no_saved_samples: {} \n'.format(no_saved_samples))
    file.write('no_plot_samples: {} \n'.format(no_plot_samples))
    file.write('step_size: {} \n'.format(step_size))
    file.write('num_steps: {} \n'.format(num_steps))         
    file.close()  




