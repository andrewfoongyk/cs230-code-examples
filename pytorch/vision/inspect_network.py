import os
import utils
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable

import model.net as net
import model.data_loader as data_loader
from plot_regression import plot_reg
import matplotlib.pyplot as plt
import scipy.stats
import math
from build_1d_dataset import Prior_Dataset_Builder
from train import Weight_Plot_Saver

from copy import deepcopy
import pickle

# load a network and inspect its weights
model_dir = 'experiments\mfvi_prior_wide' 
restore_file = 'last'

np.random.seed(0)
torch.manual_seed(230)

# unpickle the prior dataset
filename = os.path.join(model_dir, 'prior_dataset.pkl')
with open(filename, 'rb') as f:
    data, x_line, y_line = pickle.load(f)
print('unpickled prior dataset')

# plot prior dataset
plt.figure()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.plot(data[:,0], data[:,1], '+k')
plt.plot(x_line, y_line)
plt.title('dataset with ground truth')
plt.show()

# read params.json
json_path = os.path.join(model_dir, 'params.json')
assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
params = utils.Params(json_path)

params.cuda = torch.cuda.is_available()

# set up model and load the weights into it
model = net.MFVI_Net(params).cuda() # if params.cuda else net.MFVI_Net(params)
optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
utils.load_checkpoint(restore_path, model, optimizer)

# play with the weights in the model
weights, biases = model.return_weights()
# bias_indeces = np.argsort(biases[0][1]) # [1st layer][S.D.'s]
# print(biases[0][1][bias_indeces[0]])
# print(biases[0][1][bias_indeces[1]])
# print(biases[0][1][bias_indeces[2]])
# print(biases[0][1][bias_indeces[3]])
# print(bias_indeces[0]) # 6 for mfvi_prior_wide
# print(bias_indeces[1]) # 39
# print(bias_indeces[2]) # 74
# print(bias_indeces[3])

# input_indeces = np.argsort(weights[0][1][0][:]) # [1st layer][S.D.'s][row 0][all columns]
# print(weights[0][1].size)
# print(weights[0][1][0][input_indeces[0]])
# print(weights[0][1][0][input_indeces[1]])
# print(weights[0][1][0][input_indeces[2]])
# print(weights[0][1][0][input_indeces[3]])
# print(input_indeces[0]) # 
# print(input_indeces[1]) # 
# print(input_indeces[2]) # 
# print(input_indeces[3])

# output_weights = weights[1][1].reshape(256)
# output_indeces = np.argsort(output_weights) 
# print(output_weights[output_indeces[0]])
# print(output_weights[output_indeces[1]]) 
# print(output_weights[output_indeces[2]])
# print(output_weights[output_indeces[3]])
# print(output_indeces[0]) # 
# print(output_indeces[1]) # 
# print(output_indeces[2]) # 
# print(output_indeces[3])

# make list of units to monitor
no_units = params.hidden_sizes[0] ###### this might change
input_units = []
output_units = []
bias_units = []
for i in range(no_units):
    input_units.append((0, 0, i))
    output_units.append((1, i, 0))
    bias_units.append((0,i))

# plot the weights and biases along with the prior
weight_plot_save = Weight_Plot_Saver(model_dir)
weight_plot_save.save(model, 20000, input_units, name='input_weights_with_prior', parameter='weights', plot_prior=True)
weight_plot_save.save(model, 20000, output_units, name='output_weights_with_prior', parameter='weights', plot_prior=True)
weight_plot_save.save(model, 20000, bias_units, name='bias_with_prior', parameter='biases', plot_prior=True)

# use the model to make predictions and save a figure
plot_reg(model, model_dir, params, model_dir, title = 'inspection')

# save a copy of the output weights
output_mean_store = deepcopy(model.linears[1].W_mean)
output_logvar_store = deepcopy(model.linears[1].W_logvar)

# zero out the active units and make a prediction
model.linears[1].W_mean[6] = 0
model.linears[1].W_mean[39] = 0
model.linears[1].W_mean[74] = 0
model.linears[1].W_logvar[6] = -40
model.linears[1].W_logvar[39] = -40
model.linears[1].W_logvar[74] = -40
plot_reg(model, model_dir, params, model_dir, title = 'inspection_without_active')

# zero out the inactive units and make a prediction
for i in range(params.hidden_sizes[0]): # change
    model.linears[1].W_mean[i] = 0
    model.linears[1].W_logvar[i] = -40

model.linears[1].W_mean[6] = output_mean_store[6]
model.linears[1].W_mean[39] = output_mean_store[39]
model.linears[1].W_mean[74] = output_mean_store[74]
model.linears[1].W_logvar[6] = output_logvar_store[6]
model.linears[1].W_logvar[39] = output_logvar_store[39]
model.linears[1].W_logvar[74] = output_logvar_store[74]
plot_reg(model, model_dir, params, model_dir, title = 'inspection_without_inactive')

# print(output_mean_store)
# print(output_logvar_store)
# print(output_logvar_store[6])

# zero out the inactive units and make a prediction

