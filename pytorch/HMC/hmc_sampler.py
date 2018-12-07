import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pickle
import matplotlib.pyplot as plt
from copy import deepcopy

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.noise_variance = 0.01
        self.linears = nn.ModuleList([nn.Linear(1,1)])
        # self.linears.extend([MAP_Linear_Layer(self.hidden_sizes[i], self.hidden_sizes[i+1]) for i in range(0, len(self.hidden_sizes)-1)])
        # self.linears.append(MAP_Linear_Layer(self.hidden_sizes[-1], self.output_size))
        
    def forward(self, x):
        x = x.view(100, 1)
        x = self.linears[0](x)
        return x

    def get_U(self, inputs, labels) :
        outputs = self.forward(inputs)
        L2_term = 0
        for i, l in enumerate(self.linears):
            single_layer_L2 = 0.5*(torch.sum(l.weight**2) + torch.sum(l.bias**2))
            L2_term = L2_term + single_layer_L2
        error = (1/(2*self.noise_variance))*torch.sum((labels - outputs)**2)
        U = error + L2_term
        print('U: {}'.format(U))
        return U

"""This version of HMC follows https://arxiv.org/pdf/1206.1901.pdf. Identity mass matrix used.
  Args:

  """

class HMC_Sampler:
    def __init__(self, inputs, targets, step_size = 0.002, num_steps = 20, no_samples = 20000, burn_in = 1000, thinning = 1):
        self.step_size = step_size
        self.num_steps = num_steps
        self.no_samples = no_samples
        self.burn_in = burn_in
        self.inputs = inputs
        self.targets = targets
        self.thinning = thinning

    def get_samples(self, model):
        """run the HMC sampler and save the samples"""
        print('Beginning burn-in phase of {} samples'.format(self.burn_in))
        # don't save the burn-in samples
        for _ in range(self.burn_in):
            new_parameters = self.HMC_transition(model)
            # print('new parameters:{}'.format(new_parameters))
            for i, param in enumerate(model.parameters()):
                param.data = new_parameters[i]    
        print('Burn-in phase finished, collecting {} samples.'.format(self.no_samples))  
        samples = []
        for i in range(self.no_samples):
            new_parameters = self.HMC_transition(model)
            for j, param in enumerate(model.parameters()):
                param.data = new_parameters[j]
                samples.append(new_parameters)
            if i%100 == 0:
                print('Sample {}'.format(i))
        return samples
        print('Done collecting samples')

    def HMC_transition(self, model):
        """perform one transition of the markov chain"""
        saved_params = deepcopy(list(model.parameters())) # list of all the parameter objects - positions
        p = [] # list of momenta
        for _, param in enumerate(model.parameters()): 
            param_size = param.size()
            # independent standard normal variates for corresponding momenta
            p.append(torch.cuda.FloatTensor(param_size).normal_(0, 1)) 

        #print(p)
       
        # get gradients of U wrt parameters q
        U = model.get_U(self.inputs, self.targets) # get log posterior (up to constant)
        # print('U:{}'.format(U))
        start_U = U # save the starting potential energy

        K = torch.cuda.FloatTensor(1).fill_(0) # a zero 
        for momentum in p:
            K = K + torch.sum(momentum**2)/2
        start_K = K # save the starting kinetic energy

        #print(start_K)

        U.backward() 

        # make half step for momentum at the beginning
        for i, momentum in enumerate(p):
            print('momentum: {}'.format(momentum))
            print('gradient: {}'.format(list(model.parameters())[i].grad.data))
            momentum = momentum - self.step_size*list(model.parameters())[i].grad.data/2
            print('momentum after: {}'.format(momentum))

        # alternate full steps for position and momentum
        for i in range(self.num_steps):            
            # make a full step for the position
            for i, param in enumerate(model.parameters()):
                param.data += self.step_size*p[i]

            # zero gradients of U wrt parameters q
            for _, param in enumerate(model.parameters()): 
                param.grad.data.zero_()
            # get gradients of U wrt parameters q
            U = model.get_U(self.inputs, self.targets) # get log posterior (up to constant)
            U.backward() 

            # make a full step for the momentum, except at end of trajectory
            if not (i == (self.num_steps-1)):
                for i, momentum in enumerate(p):
                    momentum = momentum - self.step_size*list(model.parameters())[i].grad.data

        # make a half step for momentum at the end
        for i, momentum in enumerate(p):
            momentum = momentum - self.step_size*list(model.parameters())[i].grad.data/2

        # negate momentum at the end of trajectory to make the proposal symmetric
        for _, momentum in enumerate(p):
            momentum = -momentum # can probably skip this without effect

        # evaluate potential and kinetic energies at end of trajectory
        end_U = model.get_U(self.inputs, self.targets)
        #end_K = self.get_K(p)

        K = torch.cuda.FloatTensor(1).fill_(0) # a zero 
        for momentum in p:
            K = K + torch.sum(momentum**2)/2
        end_K = K # save the starting kinetic energy

        # Accept or reject the state at end of trajectory, returning either the position 
        # at the end of the trajectory or the initial position

        print(start_U) 
        print(end_U)
        print(start_U - end_U + start_K - end_K)

        if np.random.uniform(0, 1) < torch.exp(start_U - end_U + start_K - end_K).cpu().detach().numpy():
            print('accept')
            #print(list(model.parameters()))
            return list(model.parameters()) # accept
        else:
            print('reject')
            #print(saved_params)
            return saved_params # reject


if __name__ == "__main__":

    # model
    net = MLP()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)
    net.to(device)
    print(list(net.parameters()))

    # get dataset
    # load 1d_cosine dataset
    with open('..//vision//data//1D_COSINE//1d_linear.pkl', 'rb') as f:
            data_load = pickle.load(f)

    # plot the dataset
    plt.figure()
    plt.plot(data_load[:,0], data_load[:,1], '+k')
    plt.show()

    x_train = torch.Tensor(data_load[:,0]).cuda()
    y_train = torch.Tensor(data_load[:,1]).cuda()

    # HMC sample 
    sampler = HMC_Sampler(inputs = x_train, targets = y_train, step_size = 0.0001, num_steps = 10, no_samples = 400, burn_in = 200, thinning = 1)
    samples = sampler.get_samples(net)

    for _, param in enumerate(net.parameters()):
        print(param)

    # print(samples)
    # plot network output

