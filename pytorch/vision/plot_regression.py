# plot the output of regression network
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np

def plot_reg(model, data_dir, params, model_dir, epoch_number = 0, prior_draw=False, pretrain=False, title=None):
    # get training data
    if params.dataset == 'prior_dataset':
        filename = os.path.join(model_dir, 'prior_dataset.pkl')
        with open(filename, 'rb') as f:
            data, x_line, y_line = pickle.load(f)
        print('unpickled prior dataset')
    elif params.dataset == '1d_cosine': 
        # unpickle cosine dataset
        filename = os.path.join(data_dir, '1d_cosine_separated.pkl') ##################### use the separated version
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print('unpickled 1-D cosine dataset')
    
    X = data[:,0]
    Y = data[:,1]
    X = np.float32(X)
    Y = np.float32(Y)

    # evaluate model on test points
    N = 1000 # number of test points
    if prior_draw == True:
        x_lower = -15
        x_upper = 15
    elif params.dataset == 'prior_dataset':
        x_lower = -4
        x_upper = 4
    elif params.dataset == '1d_cosine':
        x_lower = -3
        x_upper = 3
    
    test_inputs = np.linspace(x_lower, x_upper, N)
    # move to GPU if available
    test_inputs = torch.FloatTensor(test_inputs)
    if params.cuda:
        test_inputs = test_inputs.cuda(async=True) 
    plt.figure(1)
    plt.clf() # clear figure
    if not params.model in ('mfvi', 'mfvi_prebias', 'weight_noise', 'map', 'fixed_mean_vi', 'fcvi'):
        # standard max likelihood network
        test_outputs = model(test_inputs)
        # convert back to np array
        test_inputs = test_inputs.data.cpu().numpy()
        test_outputs = test_outputs.data.cpu().numpy()
        # plot the dataset
        plt.figure()
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.plot(data[:,0], data[:,1], '+k')
        plt.plot(test_inputs, test_outputs)
        plt.title('ML neural network')
        plt.fill_between(test_inputs, test_outputs + np.sqrt(model.noise_variance), 
                test_outputs - np.sqrt(model.noise_variance), color='b', alpha=0.3)
        filename = os.path.join(model_dir, 'nn_reg_ML.pdf')
        plt.savefig(filename)
        plt.show()
    else: # MFVI neural network
        # sample from the network then plot +/- 1 standard deviation range
        no_samp = params.test_samples       
        plt.plot(data[:,0], data[:,1], '+k')
        all_test_outputs = np.zeros((no_samp, N))
        for i in range(no_samp):
            test_outputs = model(test_inputs, no_samples = 1, shared_weights = True) # make all the datapoints in a batch use the same network weights
            # convert back to np array
            test_x = test_inputs.data.cpu().numpy()
            all_test_outputs[i,:] = test_outputs.data.cpu().numpy()
            plt.plot(test_x, all_test_outputs[i,:], linewidth=0.3)
            # print(all_test_outputs[i,:])
        # calculate mean and variance
        mean = np.mean(all_test_outputs, 0)
        # print(mean)
        variance = model.noise_variance + np.mean(all_test_outputs**2, 0) - mean**2
        plt.plot(test_x, mean, color='b')
        plt.fill_between(test_x, mean + np.sqrt(variance), 
                mean - np.sqrt(variance), color='b', alpha=0.3)

        # pickle everything as numpy arrays for posterity
        inputs_mfvi = test_x
        mean_mfvi = mean
        sd_mfvi = np.sqrt(variance)

        pickle_location = os.path.join(model_dir, 'plot_mfvi_relu')
        outfile = open(pickle_location, 'wb')
        pickle.dump(inputs_mfvi, outfile)
        pickle.dump(mean_mfvi, outfile)
        pickle.dump(sd_mfvi, outfile)
        outfile.close()
        
        if title is not None:
            title = title
        else:
            if prior_draw == True:
                title = 'nn_reg_prior_draw'
            elif pretrain == True:
                title = 'nn_reg_pretrain_epoch' + str(epoch_number)
            else:
                title = 'nn_reg_epoch_' + str(epoch_number)

        filename = title + '.pdf'
        filename = os.path.join(model_dir, filename)
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title(title)
        if prior_draw == True:
            plt.ylim([-5, 5])
            plt.xlim([-5, 5])
        else:
            if params.dataset == 'prior_dataset': # automate this!!!
                plt.xlim([x_lower, x_upper])
                top = np.amax(y_line)
                bottom = np.amin(y_line)
                diff = top - bottom
                plt.ylim(np.floor(bottom - 0.25*diff), np.ceil(top + 0.25*diff))
                # plot the true line
                plt.plot(x_line, y_line, 'k', linewidth=1)
            elif params.dataset == '1d_cosine':
                plt.ylim([-3, 3])
                plt.xlim([-3, 3])
        plt.savefig(filename)
        plt.close() # get rid of this if you want to animate
