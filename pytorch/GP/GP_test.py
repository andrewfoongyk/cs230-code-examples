# GP regression

import numpy as np
from sklearn.metrics import pairwise_distances 
import matplotlib.pyplot as plt
import pickle

def make_K_SE(x, sigma, l):
    # using SE kernel def given in Murphy pg. 521
    # get pairwise distances
    km = pairwise_distances(x.reshape(-1,1))**2

    # turn into an RBF gram matrix
    km *= (-1.0 / (2.0*(l**2.0)))
    return np.exp(km)*sigma**2

def make_Kstar_SE(x, xstar, sigma=1, l=0.5):
    # get squared distance vector
    N = x.shape[0]
    xstar = np.tile(xstar, N)
    dists_sq = (x - xstar)**2

    # turn into gram vector
    v = dists_sq*(-1.0 / (2.0*(l**2.0)))
    return np.exp(v)*(sigma**2)

def SE_kernel(x1, x2, sigma, l):
    # evaluate kernel
    return np.exp((-1.0 / (2.0*(l**2.0)))*(x2-x1)**2)*sigma**2

def NN_kernel(x1, x2, kernel_params):
    """equivalent kernel for single layer neural net. expect x1, x2 both column vectors"""
    sigma_b = kernel_params['sigma_b']
    sigma_w = kernel_params['sigma_w']

    x1 = np.array([x1])
    x2 = np.array([x2])
    x2 = x2.T

    K12 = sigma_b**2 + (sigma_w**2)*x1*x2
    K11 = sigma_b**2 + (sigma_w**2)*x1*x1
    K22 = sigma_b**2 + (sigma_w**2)*x2*x2
    theta = np.arccos(K12/np.sqrt(np.multiply(K11, K22)))

    return sigma_b**2 + (sigma_w**2/(2*np.pi))*np.sqrt(np.multiply(K11, K22))*(np.sin(theta) + np.multiply((np.pi - theta), np.cos(theta)))

def GP_predict(x, y, xstar, kernel_function, kernel_params, sigma_n):
    K = kernel_function(x, x, kernel_params)

    # Algorithm 2.1 in Rasmussen and Williams
    L = np.linalg.cholesky(K + (sigma_n**2)*np.eye(len(x)))
    alpha = np.linalg.solve(L.T, (np.linalg.solve(L, y)))

    pred_mean = np.zeros(len(xstar))
    pred_var = np.zeros(len(xstar))
    # predictive mean and variance at a test point
    for i in range(len(xstar)):
        kstar = kernel_function(x, xstar[i], kernel_params)[0] # so i get a column vector?
        pred_mean[i] = np.dot(kstar, alpha)
        v = np.linalg.solve(L, kstar)
        pred_var[i] = kernel_function(xstar[i], xstar[i], kernel_params)[0] - np.dot(v,v)

    return pred_mean, pred_var

if __name__ == "__main__":
    filename = './/experiments//NNkernel.pdf'
    data_location = '..//vision//data//1D_COSINE//1d_cosine_separated.pkl'

    # get dataset
    with open(data_location, 'rb') as f:
            data_load = pickle.load(f)

    x = data_load[:,0]
    y = data_load[:,1]

    # hyperparameters
    sigma_n = 0.1 # noise standard deviation
    NN_params = {'sigma_b':1, 'sigma_w':4}
    SE_params = {'sigma':1, 'l':0.5}

    xstar = np.linspace(-3, 3, num=1000)

    pred_mean, pred_var = GP_predict(x, y, xstar, NN_kernel, NN_params, sigma_n)

    ##############
    # xstar = np.linspace(-3, 3, num=1000)
    # K = make_K_SE(x, sigma, l)

    # # Algorithm 2.1 in Rasmussen and Williams
    # L = np.linalg.cholesky(K + (sigma_n**2)*np.eye(len(x)))
    # alpha = np.linalg.solve(L.T, (np.linalg.solve(L, y)))

    # pred_mean = np.zeros(len(xstar))
    # pred_var = np.zeros(len(xstar))
    # # predictive mean and variance at a test point
    # for i in range(len(xstar)):
    #     kstar = make_Kstar_SE(x, xstar[i], sigma, l)
    #     pred_mean[i] = np.dot(kstar, alpha)
    #     v = np.linalg.solve(L, kstar)
    #     pred_var[i] = SE_kernel(xstar[i], xstar[i], sigma, l) - np.dot(v,v)
    #############

    # plot GP fit
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.ylim([-3, 3])
    plt.xlim([-3, 3])
    plt.plot(x, y, 'k+')
    plt.plot(xstar, pred_mean, color='b')
    plt.fill_between(xstar, pred_mean + np.sqrt(pred_var) + sigma_n, 
            pred_mean - np.sqrt(pred_var) - sigma_n, color='b', alpha=0.3)
    plt.savefig(filename)

    



