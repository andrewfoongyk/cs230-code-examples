"""Implement probabilistic meta-representations of neural networks"""
import numpy as np
from tqdm import tqdm
from tqdm import trange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import collections
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math
import pickle

class MLP(nn.Module):
    def __init__(self, hidden_sizes, input_size):
        super(MLP, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.input_size = input_size
        self.linears = nn.ModuleList([nn.Linear(self.input_size, self.hidden_sizes[0])])
        self.linears.extend([nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i+1]) for i in range(0, len(self.hidden_sizes)-1)])
        self.linears.append(nn.Linear(self.hidden_sizes[-1], 1)) # output size of 1, always
        
    def forward(self, x):
        ###### might have some batch size related issues?
        for i, l in enumerate(self.linears):
            x = l(x)
            if i < len(self.linears) - 1:
                x = F.relu(x) 
        return x

class UnitLatents(nn.Module):
    """module that holds the per-unit variational latents - has no forward method"""
    def __init__(self, no_units, unit_latent_dim):
        super(UnitLatents, self).__init__()
        self.unit_mean = nn.Parameter(torch.cuda.FloatTensor(no_units, unit_latent_dim).normal_(0,1))
        self.unit_logvar = nn.Parameter(torch.cuda.FloatTensor(no_units, unit_latent_dim).normal_(0,1e-1)) # tiny variance init

class MetaPriorNet(nn.Module):
    def __init__(self, noise_variance, layer_sizes, unit_latent_dim, network_latent_dim, weight_hypernet_hiddens, bias_hypernet_hiddens, weight_noise_inputs, bias_noise_inputs):
        super(MetaPriorNet, self).__init__()
        self.noise_variance = noise_variance
        self.layer_sizes = layer_sizes
        self.unit_latent_dim = unit_latent_dim
        self.network_latent_dim = network_latent_dim
        self.weight_hypernet_hiddens = weight_hypernet_hiddens
        self.bias_hypernet_hiddens = bias_hypernet_hiddens
        self.weight_noise_inputs = weight_noise_inputs
        self.bias_noise_inputs = bias_noise_inputs

        # calculate no of variational latent variables
        self.no_var_latents = 0
        for i in range(len(layer_sizes)):
            self.no_var_latents = self.no_var_latents + layer_sizes[i]*self.unit_latent_dim
        self.no_var_latents = self.no_var_latents + self.network_latent_dim

        # initialise variational latent parameter distributions for each unit (Z)
        self.unit_list = nn.ModuleList([UnitLatents(self.layer_sizes[0], self.unit_latent_dim)])
        self.unit_list.extend([UnitLatents(self.layer_sizes[i], self.unit_latent_dim) for i in range(1, len(self.layer_sizes))])

        # intialise network-wide variational latent parameters
        if self.network_latent_dim !=0:
            self.network_mean = nn.Parameter(torch.cuda.FloatTensor(self.network_latent_dim).normal_(0,1))
            self.network_logvar = nn.Parameter(torch.cuda.FloatTensor(self.network_latent_dim).normal_(-11.5,1e-10)) # tiny variance init

        # initialise the hypernetworks
        # weight hypernet
        weight_input_size = self.unit_latent_dim*2 + self.network_latent_dim + self.weight_noise_inputs
        self.weight_hypernet = MLP(hidden_sizes=self.weight_hypernet_hiddens, input_size=weight_input_size).cuda()

        # bias hypernet
        bias_input_size = self.unit_latent_dim + self.network_latent_dim + self.bias_noise_inputs
        self.bias_hypernet = MLP(hidden_sizes=self.bias_hypernet_hiddens, input_size=bias_input_size)
        
    def forward(self, x, no_samples):
        batch_size = x.size()[0]

        # sample variational latents for all units
        self.latent_list = []
        for _, layer in enumerate(self.unit_list):
            unit_means = layer.unit_mean
            unit_logvars = layer.unit_logvar
            unit_sds = torch.exp(unit_logvars/2)
            no_units = unit_means.size()[0]
            eps = Variable(torch.cuda.FloatTensor(no_samples, no_units, self.unit_latent_dim).normal_(0, 1)) # get standard gaussian random vars
            Z = eps*unit_sds.expand(no_samples, no_units, self.unit_latent_dim) + unit_means.expand(no_samples, no_units, self.unit_latent_dim)
            self.latent_list.append(Z)

        # sample network-wide variational latents
        if self.network_latent_dim != 0:
            eps = Variable(torch.cuda.FloatTensor(no_samples, self.network_latent_dim).normal_(0, 1))
            network_sds = torch.exp(self.network_logvar/2)
            network_latents = eps*network_sds.expand(no_samples, self.network_latent_dim) + self.network_mean.expand(no_samples, self.network_latent_dim)
            network_latents = network_latents.permute(1,0) # (latent_dim x no_samples)

        # create the weight matrices and bias vector using the hypernetwork
        for i in range(len(self.latent_list) - 1):
            # create weight matrices    
            input_latents = self.latent_list[i]
            output_latents = self.latent_list[i+1]
            # permute indices to (no_units x latent_dims x no_samples)
            input_latents = input_latents.permute(1,2,0)
            output_latents = output_latents.permute(1,2,0)
            no_inputs = input_latents.size()[0]
            no_outputs = output_latents.size()[0]

            # create "concatenation Gram matrix"
            expanded_input = input_latents.expand(no_outputs, no_inputs, self.unit_latent_dim, no_samples).permute(1,0,2,3)
            expanded_output = output_latents.expand(no_inputs, no_outputs, self.unit_latent_dim, no_samples)
            # 'expanded' should have dims (no_units_in x no_units_out x latent_dims x no_samples)
            concatenated = torch.cat((expanded_input, expanded_output), 2)
            
            # concatenate with the network latents
            if self.network_latent_dim != 0:
                expanded_network_latents = network_latents.expand(no_inputs, no_outputs, self.network_latent_dim, no_samples)
                concatenated = torch.cat((concatenated, expanded_network_latents), 2)
            
            # concatenate with the hypernet noise inputs
            if self.weight_noise_inputs != 0:
                eps = Variable(torch.cuda.FloatTensor(no_inputs, no_outputs, self.weight_noise_inputs, no_samples).normal_(0, 1))
                concatenated = torch.cat((concatenated, eps), 2)
    
            # reshape into one batch
            hypernet_input = concatenated.permute(0,1,3,2) # (no_inputs x no_outputs x no_samples x latent_size)
            hypernet_input = hypernet_input.reshape(no_inputs*no_outputs*no_samples, 2*self.unit_latent_dim + self.network_latent_dim + self.weight_noise_inputs)
            # forward pass through weight hypernet
            hypernet_output = self.weight_hypernet(hypernet_input)
            # reshape into a weight matrix with dims (no_inputs x no_outputs x no_samples)
            weight_matrix_samples = hypernet_output.reshape(no_inputs, no_outputs, no_samples)

            # create bias vectors - input units don't have biases but all the others do
            # concatenate with the network latents
            expanded_network_latents = network_latents.expand(no_outputs, self.network_latent_dim, no_samples)
            if self.network_latent_dim != 0:
                bias_concatenated = torch.cat((output_latents, expanded_network_latents), 1)
            else:
                bias_concatenated = output_latents
            
            # concatenate with the hypernet noise inputs
            if self.bias_noise_inputs != 0:
                eps = Variable(torch.cuda.FloatTensor(no_outputs, self.bias_noise_inputs, no_samples).normal_(0, 1))
                bias_concatenated = torch.cat((bias_concatenated, eps), 1)

            # reshape into one batch
            hypernet_input = bias_concatenated.permute(0,2,1) # (no_outputs x no_samples x latent_size)
            hypernet_input = hypernet_input.reshape(no_outputs*no_samples, self.unit_latent_dim + self.network_latent_dim + self.bias_noise_inputs)
            # forward pass through bias hypernet
            hypernet_output = self.bias_hypernet(hypernet_input)
            # reshape into bias vectors with dims (no_outputs x no_samples)
            bias_vector_samples =  hypernet_output.reshape(no_outputs, no_samples) ####### does this reshape right?

            # forward pass through one layer of the primary network
            # note: each sampled network is SHARED between all elements of a mini-batch
            if i == 0: # if this is the first input
                # x has dimensions (batch_size x input_dims) -> need to expand if this is the first input
                layer_input = x.expand(no_samples, batch_size, no_inputs)

            # else layer_input is already defined by previous pass through this loop
            bias_vector_samples = bias_vector_samples.expand(batch_size, no_outputs, no_samples)
            bias_vector_samples =  bias_vector_samples.permute(2,0,1) # (no_samples x batch_size x no_outputs)
            layer_output = torch.einsum('ios,sbi->sbo', (weight_matrix_samples, layer_input)) + bias_vector_samples
            layer_output = F.relu(layer_output)
            # next layer input
            layer_input = layer_output

        # final layer outputs
        return layer_output # (no_samples, batch_size, no_outputs) 

    def get_KL(self):
        # calculate KL term for all variational latents
        KL = torch.cuda.FloatTensor([0])
        # get contribution from units in each layer
        for _, layer in enumerate(self.unit_list):
            unit_means = layer.unit_mean
            unit_logvars = layer.unit_logvar
            unit_vars = torch.exp(unit_logvars)
            layer_KL = 0.5*(-torch.sum(unit_logvars) + torch.sum(unit_vars) + torch.sum(unit_means**2))
            KL.add_(layer_KL) # add in place

        # get contribution from network-wide variational latents
        network_vars = torch.exp(self.network_logvar)
        network_KL = 0.5*(-torch.sum(self.network_logvar) + torch.sum(network_vars) + torch.sum(self.network_mean**2))

        KL.add_(network_KL)
        KL.add_(-0.5*self.no_var_latents)
        return KL

def regression_loss(outputs, labels, model, noise_variance, dataset_size, term = 'loss'): 
    # calculate the per datapoint ELBO for regression
    no_samples = outputs.size()[0]
    batch_size = outputs.size()[1]
    output_dim = outputs.size()[2]
    labels = labels.expand(no_samples, batch_size, output_dim)
    const_term = 0.5*np.log(2*np.pi*noise_variance)
    reconstruction_loss = np.asscalar(const_term) + (1/(2*noise_variance))*torch.mean((labels - outputs)**2)
    KL_term = model.get_KL()/dataset_size
    loss = reconstruction_loss + KL_term 
    if term == 'loss':
        return loss
    elif term == 'all':
        return {"loss": loss, "reconstruction": reconstruction_loss, "KL": KL_term}

def plot_reg(model, data_load, directory, iter_number, no_samples):
    # evaluate model on test points
    N = 1000 # number of test points
    x_lower = -3
    x_upper = 3
    test_inputs_np = np.linspace(x_lower, x_upper, N)
    # move to GPU if available
    test_inputs = torch.FloatTensor(test_inputs_np)
    test_inputs = test_inputs.cuda(async=True)
    test_inputs = torch.unsqueeze(test_inputs, 1) 

    plt.figure(1)
    plt.clf() # clear figure

    plt.plot(data_load[:,0], data_load[:,1], '+k')
    test_outputs = model(test_inputs, no_samples) # (no_samples x batch_size x no_outputs)
    test_outputs = torch.squeeze(test_outputs) # (no_samples x batch_size)
    # convert back to np array
    test_y = test_outputs.data.cpu().numpy()
    # plot each sample function
    for i in range(no_samples):
        plt.plot(test_inputs_np, test_y[i,:], linewidth=0.3)
    # calculate mean and variance
    mean = np.mean(test_y, 0)
    variance = model.noise_variance + np.mean(test_y**2, 0) - mean**2 # is this right?
    plt.plot(test_inputs_np, mean, color='b')
    plt.fill_between(test_inputs_np, mean + np.sqrt(variance), 
            mean - np.sqrt(variance), color='b', alpha=0.3)
    
    filename = directory + '//regression_iteration_' + str(iter_number) + '.pdf' 
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":

    # set up tensorboard
    # tensorboard_path = os.path.join(args.model_dir, 'tensorboard')
    # writer = SummaryWriter(tensorboard_path)

    # use GPU if available
    params_cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    np.random.seed(0) # 0
    torch.manual_seed(230) #  230
    if params_cuda: torch.cuda.manual_seed(230)

    # hyperparameters
    noise_variance = 0.01
    layer_sizes = [1, 50, 50, 1]
    unit_latent_dim = 2
    network_latent_dim = 2
    weight_hypernet_hiddens = [30, 30]
    bias_hypernet_hiddens = [20, 20]
    weight_noise_inputs = 0
    bias_noise_inputs = 0
    no_samples = 10
    learning_rate = 0.001
    no_iters = 1000

    directory = './/experiments//1d_cosine_separated'
    data_location = '..//vision//data//1D_COSINE//1d_cosine_separated.pkl'

    # define the model
    model = MetaPriorNet(noise_variance, layer_sizes, unit_latent_dim, network_latent_dim,\
     weight_hypernet_hiddens, bias_hypernet_hiddens, weight_noise_inputs, bias_noise_inputs).cuda()

    # define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # print parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.size())

    # load the dataset
    with open(data_location, 'rb') as f:
            data_load = pickle.load(f)
    # convert to tensor and unsqueeze since the input dimension is 1
    x_train = torch.Tensor(data_load[:,0]).cuda()
    x_train = torch.unsqueeze(x_train, 1)
    y_train = torch.Tensor(data_load[:,1]).cuda()
    y_train = torch.unsqueeze(y_train, 1)
    dataset_size = x_train.size()[0]

    # training loop - put the entire dataset into one batch ###### try minibatching later?
    # set model to training mode 
    model.train()

    with trange(no_iters) as t:
        for i in t:
            # forward pass and calculate loss
            output = model(x_train, no_samples)
            loss = regression_loss(output, y_train, model, noise_variance, dataset_size)
           
            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # perform updates using calculated gradients
            optimizer.step()

            # update tqdm 
            if i % 10 == 0:
                t.set_postfix(loss=loss.data[0])

            # plot the regression
            if i % 100 == 0:
                plot_reg(model, data_load, directory, i, no_samples)

    # plot the latent representations?     
    # do tensorboard?


