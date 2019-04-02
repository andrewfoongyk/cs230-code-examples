# sample mean field networks

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os

class MFVI_Linear_Layer(nn.Module):
    def __init__(self, n_input, n_output):
        super(MFVI_Linear_Layer, self).__init__()
        self.n_input = n_input
        self.n_output = n_output

        """initialise parameters, there is no prior in this case"""
        # weight parameters
        self.W_mean = nn.Parameter(torch.cuda.FloatTensor(n_input, n_output).normal_(0, 1)) # initialisation of weight means
        self.W_logvar = nn.Parameter(torch.cuda.FloatTensor(n_input, n_output).normal_(-3, 1)) # initialisation of weight logvariances 
        # bias parameters
        self.b_mean = nn.Parameter(torch.cuda.FloatTensor(n_output).normal_(0, 1)) # initialisation of bias means
        self.b_logvar = nn.Parameter(torch.cuda.FloatTensor(n_output).normal_(-3, 1)) # initialisation of bias logvariances (why uniform?)

        self.num_weights = n_input*n_output + n_output # number of weights and biases
 
    def forward(self, x, no_samples): # number of samples per forward pass
        """
        input is either (batch_size x no_input), if this is the first layer of the network, or (no_samples x batch_size x no_input), 
        and output is (no_samples x batch_size x no_output)
        """

        # can't use local reparam trick if we want to sample functions from the network. assume we will only do one test sample at a time
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
       
        return samples_activations

    def get_random(self, no_samples, batch_size):
        return Variable(torch.cuda.FloatTensor(no_samples, batch_size, self.n_output).normal_(0, 1)) # standard normal noise matrix

class MFVI_Net(nn.Module):
    def __init__(self, hidden_sizes):
        super(MFVI_Net, self).__init__()
        self.activation = F.relu
        self.input_size = 1
        self.output_size = 1

        # create the layers in the network based on params
        self.hidden_sizes = hidden_sizes
        self.linears = nn.ModuleList([MFVI_Linear_Layer(self.input_size, self.hidden_sizes[0])])
        self.linears.extend([MFVI_Linear_Layer(self.hidden_sizes[i], self.hidden_sizes[i+1]) for i in range(0, len(self.hidden_sizes)-1)])
        self.linears.append(MFVI_Linear_Layer(self.hidden_sizes[-1], self.output_size))

    def forward(self, s, no_samples):
        s = s.view(-1, self.input_size)
        for i, l in enumerate(self.linears):
            s = l(s, no_samples = no_samples)
            if i < len(self.linears) - 1:
                s = self.activation(s) 

        s = s.view(no_samples, -1) # (no_samples x batch_size)
        return s

def plot_reg(model, directory, title):

    # evaluate model on test points
    N = 1000 # number of test points
    x_lower = -6
    x_upper = 6
    test_inputs = np.linspace(x_lower, x_upper, N)

    # move to GPU if available
    test_inputs = torch.FloatTensor(test_inputs)
    test_inputs = test_inputs.cuda() 

    plt.figure(1)
    plt.clf() # clear figure

    # sample from the network then plot +/- 1 standard deviation range
    no_samp = 1000
    all_test_outputs = np.zeros((no_samp, N))
    for i in range(no_samp):
        test_outputs = model(test_inputs, no_samples = 1) # make all the datapoints in a batch use the same network weights
        # convert back to np array
        test_x = test_inputs.data.cpu().numpy()
        all_test_outputs[i,:] = test_outputs.data.cpu().numpy()
        if i % 100 == 0:
            plt.plot(test_x, all_test_outputs[i,:], linewidth=0.3)
    # calculate mean and variance
    mean = np.mean(all_test_outputs, 0)
    variance = np.mean(all_test_outputs**2, 0) - mean**2
    plt.plot(test_x, mean, color='b')
    plt.fill_between(test_x, mean + np.sqrt(variance), 
            mean - np.sqrt(variance), color='b', alpha=0.3)

    filename = title + '.pdf'
    filename = os.path.join(directory, filename)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title(title)
    plt.savefig(filename)
    plt.close() 

    # plot just the standard deviation
    plt.figure(1)
    plt.clf() # clear figure
    plt.plot(test_x, np.sqrt(variance), color='b')
    filename = title + '_standard_deviation.pdf'
    filename = os.path.join(directory, filename)
    plt.xlabel('$x$')
    plt.ylabel('standard deviation')
    plt.title(title)
    plt.savefig(filename)
    plt.close() 

if __name__ == "__main__":
    
    hidden_sizes = [10]

    for i in range(20):
        model = MFVI_Net(hidden_sizes)

        # plot samples from the network
        directory = 'experiments//wide'
        plot_reg(model, directory, 'mean_field_sample' + str(i))
