# implement Laplace approximation using the outer-product approximation to the Hessian
import numpy as np
import torch
from tqdm import tqdm
from tqdm import trange
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from copy import deepcopy
from tensorboardX import SummaryWriter
import os
import cProfile

class MLP(nn.Module):
    def __init__(self, noise_variance, hidden_sizes, omega):
        super(MLP, self).__init__()
        self.omega = omega
        self.noise_variance = noise_variance
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

    def forward(self, x):
        batch_size = x.size()[0]
        x = x.view(batch_size, 1)
        for i, l in enumerate(self.linears):
            x = l(x)
            if i < len(self.linears) - 1:
                #x = torch.tanh(x)
                #x = x*torch.sigmoid(x) # swish
                x = F.relu(x) ###### activation function very important for laplace
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

    def get_P(self):
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
        return torch.diag(P_vector) 
                
def plot_reg(model, data_load, directory, iter_number=0, linearise=False, Ainv=0):
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

    test_outputs = model(test_inputs)
    test_y = test_outputs.data.cpu().numpy()
    test_y = np.squeeze(test_y)
    plt.plot(test_inputs_np, test_y, color='b')
       
    if linearise == True: # add error bars based on the linearisation in Bishop eq (5.173)
        predictive_var = np.zeros(N)
        for i in range(N): # for each test point
            # clear gradients
            optimizer.zero_grad()
            # get gradient of output wrt single training input
            x = test_inputs[i]
            x = torch.unsqueeze(x, 0)
            gradient = model.get_gradient(x)
            # calculate predictive variance
            predictive_var[i] = model.noise_variance + torch.matmul(gradient, torch.matmul(Ainv, gradient)).data.cpu().numpy()
        predictive_sd = np.sqrt(predictive_var)
        
        plt.fill_between(test_inputs_np, test_y + predictive_sd, 
                test_y - predictive_sd, color='b', alpha=0.3)
 
    if linearise == True:
        filename = directory + '//linearised_laplace.pdf' 
    else:
        filename = directory + '//regression_iteration_' + str(iter_number) + '.pdf' 
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.ylim([-3, 3])
    plt.xlim([-3, 3])
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":

    # set RNG
    seed = 0
    np.random.seed(0) # 0
    torch.manual_seed(0) #  230

    # hyperparameters
    noise_variance = 0.01
    hidden_sizes = [50]
    omega = 4
    learning_rate = 1e-3
    no_iters = 20001
    plot_iters = 1000

    directory = './/experiments//1d_cosine'
    #data_location = './/experiments//2_points_init//prior_dataset.pkl'
    data_location = '..//vision//data//1D_COSINE//1d_cosine_compressed.pkl'

    # save text file with hyperparameters
    file = open(directory + '/hyperparameters.txt','w') 
    file.write('seed: {} \n'.format(seed))
    file.write('noise_variance: {} \n'.format(noise_variance))
    file.write('hidden_sizes: {} \n'.format(hidden_sizes))
    file.write('omega: {} \n'.format(omega))
    file.write('learning_rate: {} \n'.format(learning_rate))
    file.write('no_iters: {} \n'.format(no_iters))
    file.close() 

    # # set up tensorboard
    # tensorboard_path = os.path.join(directory, 'tensorboard')
    # writer = SummaryWriter(tensorboard_path)

    # model
    model = MLP(noise_variance, hidden_sizes, omega)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)
    model.to(device)
    for param in model.parameters():
        print(type(param.data), param.size())

    # get dataset
    with open(data_location, 'rb') as f:
        data_load = pickle.load(f)

    x_train = torch.Tensor(data_load[:,0]).cuda()
    y_train = torch.Tensor(data_load[:,1]).cuda()

    # train MAP network
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    with trange(no_iters) as t:
        for i in t:
            # forward pass and calculate loss
            loss = model.get_U(x_train, y_train)
        
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

    # Laplace approximation

    # create 'sum of outer products' matrix
    H = torch.cuda.FloatTensor(model.no_params, model.no_params).fill_(0)
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

    # get prior contribution to Hessian
    P = model.get_P()

    # calculate and invert (negative) Hessian of posterior
    A = (1/model.noise_variance)*H + P
    print(torch.det(A))
    Ainv = torch.inverse(A)
    
    # plot regression with error bars using linearisation
    plot_reg(model, data_load, directory, iter_number=no_iters, linearise=True, Ainv=Ainv)




       



