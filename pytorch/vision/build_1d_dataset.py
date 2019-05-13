# build noisy sine 1-D dataset

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import torch

class Prior_Dataset_Builder(object):
    """build and save 1D dataset by drawing from model prior"""
    def __init__(self, model, params, directory):
        N = params.prior_dataset_size
        x = np.linspace(-4, 4, 10000)
        x = torch.FloatTensor(x)
        if params.cuda:
            x = x.cuda(async=True) 
        y = model(x, no_samples = 1, shared_weights = True)[0] # make all the datapoints in a batch use the same network weights
        randn = np.random.randn(N)*np.sqrt(params.noise_variance)
        y_line = y.data.cpu().numpy()
        x_line = x.data.cpu().numpy()
        # choose N points to constitute the dataset
        # indices = np.random.choice(int(10000/2), N, replace=False) + int(10000/4)
        # generate separated points
        indices1 = np.random.choice(int(10000/8), (np.ceil(N/2)).astype(int), replace=False) + int(10000/4)
        indices2 = np.random.choice(int(10000/8), (N - np.ceil(N/2)).astype(int), replace=False) + int(10000*3/4)
        indices = np.concatenate((indices1, indices2))
        
        X = x_line[indices]
        Y = y_line[indices] + randn
        # merge dataset
        data = np.c_[X.reshape(len(X), -1), Y.reshape(len(Y), -1)] 
        filename = os.path.join(directory, 'prior_dataset.pkl')

        # pickle the dataset 
        with open(filename, 'wb') as f:
            pickle.dump([data, x_line, y_line], f)
    

        # unpickle the dataset
        with open(filename, 'rb') as f:
            data, x_line, y_line = pickle.load(f)

        # plot the dataset
        plt.figure()
        plt.plot(data[:,0], data[:,1], '+k')
        plt.plot(x_line, y_line)
        plt.title('dataset with ground truth')
        plt.savefig(os.path.join(directory, 'prior_draw_dataset.pdf'))
        plt.show()         

if __name__ == '__main__':
    n = 100 # number of datapoints
    noise_var = 0.01 # datapoint noise
    # X = np.random.uniform(-1, 1, n)
    X1 = np.random.uniform(-1, -0.7, 50)
    X2 = np.random.uniform(0.5, 1, 50)
    X = np.concatenate((X1, X2), axis=0)

    cosx = np.cos(4*X + 0.8)
    randn = np.random.randn(n)*np.sqrt(noise_var)
    Y = cosx + randn

    # def func(x):
    #     return np.cos(2*x + 0.8)

    # X = np.random.rand(n, 1) * 2 - 1
    # Y = func(X) + np.random.randn(n, 1) * 0.1
    # Xmean = np.mean(X)
    # Xstd = np.std(X)
    # X = (X - Xmean) / Xstd
    # Ymean = np.mean(Y)
    # Ystd = np.std(Y)
    # Y = (Y - Ymean) / Ystd

    # merge dataset
    data = np.c_[X.reshape(len(X), -1), Y.reshape(len(Y), -1)]

    # pickle the dataset 
    with open('data//1D_COSINE//1d_cosine_separated.pkl', 'wb') as f:
        pickle.dump(data, f)

    # unpickle the dataset
    with open('data//1D_COSINE//1d_cosine_separated.pkl', 'rb') as f:
        data_load = pickle.load(f)

    # plot the dataset
    plt.figure()
    plt.plot(data_load[:,0], data_load[:,1], '+k')
    plt.show()


