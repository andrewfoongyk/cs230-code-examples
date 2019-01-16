"""unpickle samples and plot pairwise marginals"""

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy
from scipy.stats import multivariate_normal
import os

from hmc_sampler import MLP

class Gaussian_fitter():
    """fit gaussians to the HMC samples while truncating the chain at various levels"""
    def __init__(self, model, samples, hidden_sizes, directory):
        self.directory = directory
        self.model = model
        self.hidden_sizes = hidden_sizes
        self.samples = samples
    
        # turn the samples into a numpy array
        # calculate number of parameters in the network
        self.no_params = hidden_sizes[0] # first weight matrix
        for i in range(len(hidden_sizes)-1):
            self.no_params = self.no_params + hidden_sizes[i] + hidden_sizes[i]*hidden_sizes[i+1]
        self.no_params = self.no_params + hidden_sizes[-1] + hidden_sizes[-1] + 1 # final weight matrix and last 2 biases

        self.sampled_parameters = np.zeros((self.no_params, len(samples)))
        for i, sample in enumerate(samples):
            # send parameters to numpy arrays and pack the parameters into a vector
            start_index = 0
            for _, param in enumerate(sample):
                if len(param.size()) > 1:
                    param = torch.t(param) # nn.Linear does a transpose for some reason, so undo this
                param = param.cpu().detach().numpy()
                param = param.reshape(-1) # flatten into a vector
                end_index = start_index + param.size
                #import pdb; pdb.set_trace()
                self.sampled_parameters[start_index:end_index, i] = param # fill into array
                start_index = start_index + param.size

    def moment_match(self, sampled_parameters):
        """get mean and covariance of the samples"""
        no_samples = sampled_parameters.shape[1]
        # calculate sample mean
        sample_mean = np.mean(sampled_parameters, axis=1)
        sample_mean = sample_mean.reshape(-1, 1) # numpy broadcasting thing
        centered_samples = sampled_parameters - sample_mean

        # calculate empirical covariances
        cov = np.zeros((self.no_params, self.no_params))
        for i in range(no_samples):
            cov = cov + np.outer(centered_samples[:,i], centered_samples[:,i])
        cov = cov/no_samples # max likelihood estimator of covariance
        sample_mean = sample_mean.reshape(self.no_params)
        return sample_mean, cov

    def plot_pairwise(self, theta1, theta2, chain_length, colour='blue', size=0.1, fit_gaussian = True):
        """plot pairwise marginals"""
        # truncate the chain
        truncated_samples = self.sampled_parameters[:, 0:chain_length]
        # choose two parameters to plot pairwise marginals
        sample1 = truncated_samples[theta1, :]
        sample2 = truncated_samples[theta2, :]

        # plot
        fig, ax = plt.subplots() 
        plt.xlabel('parameter ' + str(theta1))
        plt.ylabel('parameter ' + str(theta2))
        plt.scatter(sample1, sample2, s=size, c=colour)

        # plot a Gaussian fit
        if fit_gaussian == True:
            # moment match a Gaussian
            mean, cov = self.moment_match(truncated_samples)
            # marginalise out all parameters except theta1 and theta2
            mean = mean[[theta1, theta2]]
            mean = mean.reshape(2)
            cov = np.array([[cov[theta1, theta1], cov[theta1, theta2]],[cov[theta2, theta1], cov[theta2, theta2]]])

            # contour plot for Gaussian
            # get sensible limits
            max1 = np.max(sample1)
            min1 = np.min(sample1)
            range1 = max1 - min1
            max2 = np.max(sample2)
            min2 = np.min(sample2)
            range2 = max2 - min2

            delta = (range1 + range2)/200
            x = np.arange(min1 - 0.5*range1, max1 + 0.5*range1, delta)
            y = np.arange(min2 - 0.5*range2, max2 + 0.5*range2, delta)
            X, Y = np.meshgrid(x, y)
            # Pack X and Y into a single 3-dimensional array
            pos = np.empty(X.shape + (2,))
            pos[:, :, 0] = X
            pos[:, :, 1] = Y
            Z = self.multivariate_gaussian(pos, mean, cov) 
            ax.contour(X,Y,Z)

        filepath = os.path.join(directory, 'pairwise_' + str(theta1) + '_' + str(theta2) + '_chain_length_' + str(chain_length) + '.pdf')
        plt.savefig(filepath)
        plt.close()

    def multivariate_gaussian(self, pos, mu, Sigma):
        """Return the multivariate Gaussian distribution on array pos.

        pos is an array constructed by packing the meshed arrays of variables
        x_1, x_2, x_3, ..., x_k into its _last_ dimension.

        """

        n = mu.shape[0]
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        N = np.sqrt((2*np.pi)**n * Sigma_det)
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

        return np.exp(-fac / 2) / N

    def plot_regression_gaussian(self, data, chain_length, no_plot_samples = 32):
        """plot samples from the network after moment matching a Gaussian"""
        # truncate the chain
        truncated_samples = self.sampled_parameters[:, 0:chain_length]

        plt.figure()
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        axes = plt.gca()
        axes.set_xlim([-3, 3])
        axes.set_ylim([-3, 3])
        plt.plot(data[:,0], data[:,1], '+k')
        N = 1000
        x_lower = -3
        x_upper = 3
        x_values = np.linspace(x_lower, x_upper, N)
        test_inputs = torch.FloatTensor(x_values).cuda(async=True)

        # moment match to the samples in the chain
        mean, cov = self.moment_match(truncated_samples)
        # do mean field #################
        #cov = np.diag(np.diag(cov))

        # draw samples from the moment matched Gaussian
        sampled_parameters = np.random.multivariate_normal(mean, cov, no_plot_samples).transpose()
        
        # fit those samples back into the model
        samples = []
        for i in range(no_plot_samples):
            # unpack the parameter vector into a list of parameter tensors
            sample = self.unpack_sample(sampled_parameters[:, i])
            samples.append(sample)

        # plot all the samples
        all_test_outputs = np.zeros((no_plot_samples, N))
        for i, sample in enumerate(samples):
            for j, param in enumerate(self.model.parameters()):
                param.data = sample[j] # fill in the model with these weights
            # plot regression
            test_outputs = self.model(test_inputs)
            # convert back to np array
            test_outputs = test_outputs.data.cpu().numpy()
            test_outputs = test_outputs.reshape(N)
            plt.plot(x_values, test_outputs, linewidth=0.3)
            # save data for ensemble mean and s.d. calculation
            all_test_outputs[i,:] = test_outputs

        # calculate mean and variance
        mean = np.mean(all_test_outputs, 0)
        variance = self.model.noise_variance + np.mean(all_test_outputs**2, 0) - mean**2

        plt.plot(x_values, mean, color='b')
        plt.fill_between(x_values, mean + np.sqrt(variance), mean - np.sqrt(variance), color='b', alpha=0.3)

        filepath = os.path.join(directory, 'Gaussian_fit_regression_chain_length_' + str(chain_length) + '.pdf')
        plt.savefig(filepath)
        plt.close()
        
        #################################
        # # plot on top of original samples as a sanity check 
        # plt.figure()
        # # plot original
        # theta1 = 12
        # theta2 = 13
        # sample1 = truncated_samples[theta1, :]
        # sample2 = truncated_samples[theta2, :]
        # # plot
        # plt.subplots() 
        # plt.xlabel('parameter ' + str(theta1))
        # plt.ylabel('parameter ' + str(theta2))
        # plt.scatter(sample1, sample2, s=0.1, c='blue') # actual HMC samples
        # # plot Gaussian fit samples
        # gaussian_sample1 = sampled_parameters[theta1, :]
        # gaussian_sample2 = sampled_parameters[theta2, :]
        # plt.scatter(gaussian_sample1, gaussian_sample2, s=1, c='red') # gaussian samples
        # filepath = os.path.join(directory, 'sanity_check_' + str(theta1) + '_' + str(theta2) + '.pdf')
        # plt.savefig(filepath)
        # plt.close
        ##################################

    def unpack_sample(self, parameter_vector):
        """convert a numpy vector of parameters to a list of parameter tensors"""
        sample = []
        start_index = 0
        end_index = 0   
        # unpack first weight matrix and bias
        end_index = end_index + self.hidden_sizes[0]
        weight_vector = parameter_vector[start_index:end_index]
        weight_matrix = weight_vector.reshape((self.hidden_sizes[0], 1))
        sample.append(torch.Tensor(weight_matrix).cuda())
        start_index = start_index + self.hidden_sizes[0]
        end_index = end_index + self.hidden_sizes[0]
        biases_vector = parameter_vector[start_index:end_index]
        sample.append(torch.Tensor(biases_vector).cuda())
        start_index = start_index + self.hidden_sizes[0]
        for i in range(len(self.hidden_sizes)-1): 
            end_index = end_index + self.hidden_sizes[i]*self.hidden_sizes[i+1]
            weight_vector = parameter_vector[start_index:end_index]
            weight_matrix = weight_vector.reshape((self.hidden_sizes[i+1], self.hidden_sizes[i]))
            sample.append(torch.Tensor(weight_matrix).cuda())
            start_index = start_index + self.hidden_sizes[i]*self.hidden_sizes[i+1]
            end_index = end_index + self.hidden_sizes[i+1]
            biases_vector = parameter_vector[start_index:end_index]
            sample.append(torch.Tensor(biases_vector).cuda())
            start_index = start_index + self.hidden_sizes[i+1]
        # unpack output weight matrix and bias
        end_index = end_index + self.hidden_sizes[-1]
        weight_vector = parameter_vector[start_index:end_index]
        weight_matrix = weight_vector.reshape((1, self.hidden_sizes[-1]))
        sample.append(torch.Tensor(weight_matrix).cuda())
        start_index = start_index + self.hidden_sizes[-1]
        biases_vector = parameter_vector[start_index:] # should reach the end of the parameters vector at this point
        sample.append(torch.Tensor(biases_vector).cuda())
        return sample

    def plot_regression(self, data, chain_length, no_plot_samples = 32):
        """plot samples from the network"""
        # truncate the chain
        truncated_samples = self.samples[:chain_length]

        plt.figure()
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        axes = plt.gca()
        axes.set_xlim([-3, 3])
        axes.set_ylim([-3, 3])
        plt.plot(data[:,0], data[:,1], '+k')
        N = 1000
        x_lower = -3
        x_upper = 3
        x_values = np.linspace(x_lower, x_upper, N)
        test_inputs = torch.FloatTensor(x_values).cuda(async=True)

        # subsample for plotting only
        indeces = np.arange(0, len(truncated_samples))
        subsampled_indeces = np.random.choice(indeces, no_plot_samples, replace=False)
        subsampled_indeces = subsampled_indeces.astype(int)
        samples = [truncated_samples[i] for i in subsampled_indeces]

        # plot all the samples
        no_samp = len(samples)
        all_test_outputs = np.zeros((no_samp, N))
        for i, sample in enumerate(samples):
            for j, param in enumerate(self.model.parameters()):
                param.data = sample[j] # fill in the model with these weights
            # plot regression
            test_outputs = self.model(test_inputs)
            # convert back to np array
            test_outputs = test_outputs.data.cpu().numpy()
            test_outputs = test_outputs.reshape(N)
            plt.plot(x_values, test_outputs, linewidth=0.3)
            # save data for ensemble mean and s.d. calculation
            all_test_outputs[i,:] = test_outputs

        # calculate mean and variance
        mean = np.mean(all_test_outputs, 0)
        variance = self.model.noise_variance + np.mean(all_test_outputs**2, 0) - mean**2

        plt.plot(x_values, mean, color='b')
        plt.fill_between(x_values, mean + np.sqrt(variance), mean - np.sqrt(variance), color='b', alpha=0.3)

        filepath = os.path.join(directory, 'HMC_regression_chain_length_' + str(chain_length) + '.pdf')
        plt.savefig(filepath)
        plt.close()
        
def plot_pairwise(samples, hidden_sizes, theta1, theta2, directory):
    # extract samples and plot pairwise marginals

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

    # choose two parameters to plot pairwise marginals
    sample1 = sampled_parameters[theta1, :]
    sample2 = sampled_parameters[theta2, :]

    # plot
    plt.figure()
    plt.xlabel('parameter ' + str(theta1))
    plt.ylabel('parameter ' + str(theta2))
    plt.scatter(sample1, sample2, s=3, c='purple')
    filepath = os.path.join(directory, 'pairwise_' + str(theta1) + '_' + str(theta2) + '.pdf')
    plt.savefig(filepath)
    plt.close()

if __name__ == "__main__":    
    # set RNG
    np.random.seed(0) # 0
    torch.manual_seed(230) #  230

    directory = './/experiments//1d_cosine_separated2'
    # hyperparameters
    noise_variance = 0.01
    hidden_sizes = [50]
    omega = 4

    burn_in = 10000
    no_samples = 40000
    no_saved_samples = 40000
    no_plot_samples = 32
    step_size = [0.0005, 0.0015]
    num_steps = [5, 10]

    with open(directory + '//HMC_samples', 'rb') as f:
                samples = pickle.load(f)

    # get data for plotting purposes
    data_location = '..//vision//data//1D_COSINE//1d_cosine_separated.pkl'
    # get dataset
    with open(data_location, 'rb') as f:
            data_load = pickle.load(f)

    # theta1=2
    # theta2=3
    # plot_pairwise(samples, hidden_sizes, theta1, theta2, directory)

    # define model (make sure it's the same one used to generate the samples)
    model = MLP(noise_variance, hidden_sizes, omega)

    gauss_fit = Gaussian_fitter(model, samples, hidden_sizes, directory)
    for chain_length in [40000, 20000, 10000, 5000, 2500, 1000, 500, 250, 100, 50]:        
        gauss_fit.plot_pairwise(0, 50, chain_length, colour='blue')
        gauss_fit.plot_pairwise(0, 100, chain_length, colour='red')
        gauss_fit.plot_pairwise(50, 100, chain_length, colour='green')
        gauss_fit.plot_regression(data_load, chain_length)
        gauss_fit.plot_regression_gaussian(data_load, chain_length)