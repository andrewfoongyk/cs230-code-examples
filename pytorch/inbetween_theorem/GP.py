# GP regression

import numpy as np
from sklearn.metrics import pairwise_distances 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import os

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

def SE_kernel(x1, x2, kernel_params):
    sigma = kernel_params['sigma']
    l = kernel_params['l']
    # using SE kernel def given in Murphy pg. 521
    # get pairwise distances
    km = pairwise_distances(x.reshape(-1,1))**2

    # turn into an RBF gram matrix
    km *= (-1.0 / (2.0*(l**2.0)))
    return np.exp(km)*sigma**2
    # sigma = kernel_params['sigma']
    # l = kernel_params['l']
    # # evaluate kernel
    # return np.exp((-1.0 / (2.0*(l**2.0)))*(x2-x1)**2)*sigma**2

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

def NN_kernel_multidim(x1, x2, kernel_params):
    ############################# MAYBE BUGGY???
    """equivalent kernel for single layer neural net. expect x1, x2 both (N x D) matrices, with N points and D dimensions"""
    sigma_b = kernel_params['sigma_b']
    sigma_w = kernel_params['sigma_w']

    d_in = x1.shape[1]

    K12 = sigma_b**2 + (sigma_w**2)*x1 @ x2.T/d_in
    K11 = sigma_b**2 + (sigma_w**2)*x1 @ x1.T/d_in
    K22 = sigma_b**2 + (sigma_w**2)*x2 @ x2.T/d_in
    theta = np.arccos(K12/np.sqrt(np.multiply(K11, K22)))

    return sigma_b**2 + (sigma_w**2/(2*np.pi))*np.sqrt(np.multiply(K11, K22))*(np.sin(theta) + np.multiply((np.pi - theta), np.cos(theta)))

def NN_kernel_multidim_2(x1, x2, kernel_params):
    """test: use the vector version to construct the matrix one column at a time"""
    len_x1 = x1.shape[0]
    len_x2 = x2.shape[0]
    K = np.zeros((len_x1, len_x2))

    for i in range(len_x2):
        K[:,i] = np.squeeze(NN_kernel_multidim_vector(x1, x2[i].reshape(1, -1), kernel_params))

    return K

def NN_kernel_multidim_vector(x, xstar, kernel_params):
    """equivalent kernel for single layer neural net. 
    expect x an (N x D) matrix, with N points and D dimensions
    expect xstar a (1 x D) matrix, which is the test point"""
    sigma_b = kernel_params['sigma_b']
    sigma_w = kernel_params['sigma_w']

    N = x.shape[0]
    d_in = x.shape[1]

    #import pdb; pdb.set_trace()

    K12 = sigma_b**2 + (sigma_w**2) * x @ xstar.T / d_in
    K11 = sigma_b**2 + (sigma_w**2) * np.sum(x**2, axis=1) / d_in
    K22 = sigma_b**2 + (sigma_w**2) * np.sum(xstar**2, axis=1) / d_in

    K11 = K11.reshape(N, 1)
    K22 = K22[0]

    if np.min(K12/np.sqrt(K11 * K22)) < -1:
        import pdb; pdb.set_trace()
    if np.max(K12/np.sqrt(K11 * K22)) > 1:
        import pdb; pdb.set_trace()

    theta = np.arccos(K12/np.sqrt(K11 * K22))

    return sigma_b**2 + (sigma_w**2/(2*np.pi))*np.sqrt(K11 * K22)*(np.sin(theta) + np.multiply((np.pi - theta), np.cos(theta)))

def GP_sample(x, kernel_function, kernel_params, sigma_n, num_samples=1): # sample from the prior
    K = kernel_function(x, x, kernel_params)
    num_points = x.shape[0]
    mean = np.zeros(num_points)
    cov = K + (sigma_n**2) * np.eye(num_points)

    y = np.random.multivariate_normal(mean, cov, num_samples)
    return y

def GP_predict(x, y, xstar, kernel_function, kernel_function_vector, kernel_params, sigma_n):
    K = kernel_function(x, x, kernel_params)

    # Algorithm 2.1 in Rasmussen and Williams
    L = np.linalg.cholesky(K + (sigma_n**2)*np.eye(len(x)))
    alpha = np.linalg.solve(L.T, (np.linalg.solve(L, y)))

    k_star_stars = np.diag(kernel_function(xstar, xstar, kernel_params))

    pred_mean = np.zeros(len(xstar))
    pred_var = np.zeros(len(xstar))
    # predictive mean and variance at a test point
    for i in range(len(xstar)):
        #import pdb; pdb.set_trace()
        kstar = kernel_function_vector(x, xstar[i].reshape(1, -1), kernel_params) # so i get a column vector?
        kstar = np.squeeze(kstar)
        pred_mean[i] = np.dot(kstar, alpha)
        v = np.linalg.solve(L, kstar)
        #import pdb; pdb.set_trace()
        pred_var[i] = k_star_stars[i] - np.dot(v,v)

    return pred_mean, pred_var

if __name__ == "__main__":
    filename = './/experiments//NNkernel.pdf'
    data_location = '..//vision//data//1D_COSINE//1d_cosine_separated.pkl'

    # # get dataset
    # with open(data_location, 'rb') as f:
    #         data_load = pickle.load(f)

    # x = data_load[:,0]
    # y = data_load[:,1]

    # hyperparameters
    sigma_n = 0.1 # noise standard deviation
    NN_params = {'sigma_b':1, 'sigma_w':4} # sigma_w is the same as Neal's omega
    SE_params = {'sigma':1, 'l':0.5}

    # generate 2D grid of input points
    points_per_axis = 40
    x = np.linspace(-2, 2, points_per_axis)
    y = np.linspace(-2, 2, points_per_axis)
    x_grid, y_grid = np.meshgrid(x,y)
    x_grid_flattened = x_grid.reshape(-1,1)
    y_grid_flattened = y_grid.reshape(-1,1)
    xy_flattened = np.stack((np.squeeze(x_grid_flattened), np.squeeze(y_grid_flattened)), axis=-1)

    # evaluate at two separated clusters to form a dataset
    xy_data_1 = np.random.multivariate_normal([1, 1], 0.01*np.eye(2), 50)
    xy_data_2 = np.random.multivariate_normal([-1, -1], 0.01*np.eye(2), 50)
    xy_data = np.concatenate((xy_data_1, xy_data_2))

    # sample from prior and fit
    num_samples = 1
    output_data = GP_sample(xy_data, NN_kernel_multidim_2, NN_params, sigma_n, num_samples=num_samples)
    output_data = np.squeeze(output_data)
    pred_mean, pred_var = GP_predict(xy_data, output_data, xy_flattened, NN_kernel_multidim_2, NN_kernel_multidim_vector, NN_params, sigma_n)
    
    # plot 1D diagonal slice underneath
    lambdas = np.linspace(-2*np.sqrt(2), 2*np.sqrt(2), 500)
    xlambdas = np.zeros((500, 2))
    for i in range(500):
        xlambdas[i, :] = np.array([np.sqrt(2)/2, np.sqrt(2)/2]) * lambdas[i]
    # make predictions along the diagonal
    pred_mean_lambdas, pred_var_lambdas = GP_predict(xy_data, output_data, xlambdas, NN_kernel_multidim_2, NN_kernel_multidim_vector, NN_params, sigma_n)

    # project the xy_data onto lambda values
    unit_vec = [[np.sqrt(2)/2],[np.sqrt(2)/2]]
    projected_xy = xy_data @ unit_vec
    projected_xy = np.squeeze(projected_xy)

    # pickle everything as numpy arrays for posterity
    inputs = xy_data
    outputs = output_data

    pickle_location = 'experiments/prior_sample_2d/data.pkl'
    outfile = open(pickle_location, 'wb')
    pickle.dump(inputs, outfile)
    pickle.dump(outputs, outfile)
    pickle.dump(pred_mean, outfile)
    pickle.dump(pred_var, outfile)
    pickle.dump(lambdas, outfile)
    pickle.dump(xlambdas, outfile)
    pickle.dump(pred_mean_lambdas, outfile)
    pickle.dump(pred_var_lambdas, outfile)
    outfile.close()


    # MEAN PLOT
    # Plot figure with subplots of different sizes
    fig, axes = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios':[3, 1]}, figsize=(6.2, 8))
    cnt = axes[0].contourf(x_grid, y_grid, pred_mean.reshape(points_per_axis, points_per_axis), 200)
    axes[0].set_aspect('equal')
    axes[0].set_xlabel('$x_1$')
    axes[0].set_ylabel('$x_2$')
    axes[0].set_title('$\mathbb{E}[f(\mathbf{x})]$')
    for c in cnt.collections:
        c.set_edgecolor("face")
    axes[0].plot(xy_data[:,0], xy_data[:,1], 'r+')
    # plot diagonal line
    axes[0].plot([0, 1], [0, 1], 'k--', transform=axes[0].transAxes)

    # plot 1D slice
    axes[1].plot(lambdas, pred_mean_lambdas)
    plt.fill_between(lambdas, pred_mean_lambdas + 2 * np.sqrt(pred_var_lambdas), 
            pred_mean_lambdas - 2 * np.sqrt(pred_var_lambdas), color='b', alpha=0.3)
    axes[1].set_xlabel('$\lambda$')
    axes[1].set_ylabel('$f(\mathbf{x}(\lambda))$')
    # plot projected datapoints
    plt.plot(projected_xy, output_data, 'r+')

    bbox_ax_top = axes[0].get_position()
    bbox_ax_bottom = axes[1].get_position()

    cbar_im1a_ax = fig.add_axes([0.9, bbox_ax_top.y0, 0.02, bbox_ax_top.y1-bbox_ax_top.y0])
    cbar_im1a = plt.colorbar(cnt, cax=cbar_im1a_ax)

    plt.savefig('experiments/prior_sample_2d/pred_mean.pdf')
    plt.close()

    # VARIANCE PLOT
    # Plot figure with subplots of different sizes
    fig, axes = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios':[3, 1]}, figsize=(6.2, 8))
    cnt = axes[0].contourf(x_grid, y_grid, pred_var.reshape(points_per_axis, points_per_axis), 200)
    axes[0].set_aspect('equal')
    axes[0].set_xlabel('$x_1$')
    axes[0].set_ylabel('$x_2$')
    axes[0].set_title('Var$[f(\mathbf{x})]$')
    for c in cnt.collections:
        c.set_edgecolor("face")
    axes[0].plot(xy_data[:,0], xy_data[:,1], 'r+')
    # plot diagonal line
    axes[0].plot([0, 1], [0, 1], 'k--', transform=axes[0].transAxes)

    # plot 1D slice
    axes[1].plot(lambdas, pred_var_lambdas)
    axes[1].set_xlabel('$\lambda$')
    axes[1].set_ylabel('Var$[f(\mathbf{x}(\lambda))]$')
    axes[1].set_ylim(bottom=0)
    # plot projected datapoints
    axes[1].plot(projected_xy, np.zeros_like(projected_xy), 'r|', markersize=30)

    bbox_ax_top = axes[0].get_position()
    bbox_ax_bottom = axes[1].get_position()

    cbar_im1a_ax = fig.add_axes([0.9, bbox_ax_top.y0, 0.02, bbox_ax_top.y1-bbox_ax_top.y0])
    cbar_im1a = plt.colorbar(cnt, cax=cbar_im1a_ax)

    plt.savefig('experiments/prior_sample_2d/pred_var.pdf')
    plt.close()

    # SD PLOT
    # Plot figure with subplots of different sizes
    fig, axes = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios':[3, 1]}, figsize=(6.2, 8))
    cnt = axes[0].contourf(x_grid, y_grid, np.sqrt(pred_var).reshape(points_per_axis, points_per_axis), 200)
    axes[0].set_aspect('equal')
    axes[0].set_xlabel('$x_1$')
    axes[0].set_ylabel('$x_2$')
    axes[0].set_title('$\sigma[f(\mathbf{x})]$')
    for c in cnt.collections:
        c.set_edgecolor("face")
    axes[0].plot(xy_data[:,0], xy_data[:,1], 'r+')
    # plot diagonal line
    axes[0].plot([0, 1], [0, 1], 'k--', transform=axes[0].transAxes)

    # plot 1D slice
    axes[1].plot(lambdas, np.sqrt(pred_var_lambdas))
    axes[1].set_xlabel('$\lambda$')
    axes[1].set_ylabel('$\sigma[f(\mathbf{x}(\lambda))]$')
    axes[1].set_ylim(bottom=0)
    # plot projected datapoints
    axes[1].plot(projected_xy, np.zeros_like(projected_xy), 'r|', markersize=30)

    bbox_ax_top = axes[0].get_position()
    bbox_ax_bottom = axes[1].get_position()

    cbar_im1a_ax = fig.add_axes([0.9, bbox_ax_top.y0, 0.02, bbox_ax_top.y1-bbox_ax_top.y0])
    cbar_im1a = plt.colorbar(cnt, cax=cbar_im1a_ax)

    plt.savefig('experiments/prior_sample_2d/pred_sd.pdf')
    plt.close()




    # plt.figure(2)
    # pred_var = pred_var.reshape(points_per_axis, points_per_axis)
    # cnt = plt.contourf(x_grid, y_grid, pred_var, 40)
    # plt.scatter(xy_data[:,0], xy_data[:,1])
    # plt.colorbar()
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('predictive variance')
    # for c in cnt.collections:
    #     c.set_edgecolor("face")
    # plt.savefig('experiments/prior_sample_2d/pred_var.pdf')
    
    # plt.figure(3)
    # pred_sd = np.sqrt(pred_var)
    # cnt = plt.contourf(x_grid, y_grid, pred_sd, 40)
    # plt.scatter(xy_data[:,0], xy_data[:,1])
    # plt.colorbar()
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('predictive standard deviation')
    # for c in cnt.collections:
    #     c.set_edgecolor("face")
    # plt.savefig('experiments/prior_sample_2d/pred_sd.pdf')

    # # sample from the prior
    # num_samples = 1
    # output_flattened = GP_sample(xy_flattened, NN_kernel_multidim, NN_params, sigma_n, num_samples=num_samples)

    # # plot things in 2D
    # output = output_flattened.reshape(points_per_axis, points_per_axis)
    # cnt = plt.contourf(x_grid, y_grid, output, 40)
    # plt.colorbar()
    # plt.xlabel('x')
    # plt.ylabel('y')
    # # plt.show()
    # for c in cnt.collections:
    #     c.set_edgecolor("face")
    # plt.savefig('experiments/prior_sample_2d/NN_kernel.pdf')


    # x_star = np.linspace(-2, 2, num=1000)

    # # evaluate at two separated clusters to form a dataset
    # x_data_1 = np.random.multivariate_normal([1], [[0.01]], 50)
    # x_data_2 = np.random.multivariate_normal([-1], [[0.01]], 50)
    # x_data = np.concatenate((x_data_1, x_data_2))
    # x_data = np.squeeze(x_data)

    # # sample from the prior
    # num_samples = 1
    # y_data = GP_sample(x_data, NN_kernel, NN_params, sigma_n, num_samples=num_samples)
    # y_data = np.squeeze(y_data)
    # pred_mean, pred_var = GP_predict(x_data, y_data, x_star, NN_kernel, NN_params, sigma_n)

    # # plot GP fit
    # plt.figure(1)
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    # plt.xlim([-2, 2])
    # plt.plot(x_data, y_data, 'k+')
    # plt.plot(x_star, pred_mean, color='b')
    # plt.fill_between(x_star, pred_mean + 2 * np.sqrt(pred_var), 
    #         pred_mean - 2 * np.sqrt(pred_var), color='b', alpha=0.3)
    # plt.savefig('experiments/prior_sample_2d/predictive.pdf')

    # # plot variance
    # plt.figure(2)
    # plt.xlabel('$x$')
    # plt.ylabel('$Var[f(x)]$')
    # plt.xlim([-2, 2])
    # plt.plot(x_data, np.zeros_like(x_data), 'k|', markersize = 30)
    # plt.plot(x_star, pred_var, color='b')
    # plt.gca().set_ylim(bottom=0)
    # plt.savefig('experiments/prior_sample_2d/variance.pdf')

    # # plot standard deviation
    # plt.figure(3)
    # plt.xlabel('$x$')
    # plt.ylabel('$\sigma[f(x)]$')
    # plt.xlim([-2, 2])
    # plt.plot(x_data, np.zeros_like(x_data), 'k|', markersize = 30)
    # plt.plot(x_star, np.sqrt(pred_var), color='b')
    # plt.gca().set_ylim(bottom=0)
    # plt.savefig('experiments/prior_sample_2d/standard_deviation.pdf')

    # # pickle the datapoints
    # pickle_location = 'experiments/prior_sample_2d/data.pkl'
    # outfile = open(pickle_location, 'wb')
    # pickle.dump(x_data, outfile)
    # pickle.dump(y_data, outfile)
    # pickle.dump(pred_mean, outfile)
    # pickle.dump(pred_var, outfile)
    # outfile.close()


    # # plot GP fit
    # plt.figure(1)
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    # # plt.ylim([-2, 2])
    # plt.xlim([-2, 2])
    # for i in range(num_samples):
    #     plt.plot(x, y[i,:])
    # plt.savefig('experiments/prior_sample/NN_kernel.pdf')


    ##############
#     xstar = np.linspace(-3, 3, num=1000)
#     K = make_K_SE(x, sigma, l)

#     # Algorithm 2.1 in Rasmussen and Williams
#     L = np.linalg.cholesky(K + (sigma_n**2)*np.eye(len(x)))
#     alpha = np.linalg.solve(L.T, (np.linalg.solve(L, y)))

#     pred_mean = np.zeros(len(xstar))
#     pred_var = np.zeros(len(xstar))
#     # predictive mean and variance at a test point
#     for i in range(len(xstar)):
#         kstar = make_Kstar_SE(x, xstar[i], sigma, l)
#         pred_mean[i] = np.dot(kstar, alpha)
#         v = np.linalg.solve(L, kstar)
#         pred_var[i] = SE_kernel(xstar[i], xstar[i], sigma, l) - np.dot(v,v)
    #############

    # plot prior draws
#     no_samp = 10
#     x_in = np.linspace(-20, 20, 1000)
#     cov = NN_kernel(x_in, x_in, NN_params) + 1e-6 * np.eye(1000)
#     L = np.linalg.cholesky(cov)
#     sample = L @ np.random.randn(x_in.shape[0], no_samp)

#     # plot sample
#     for i in range(no_samp):
#         plt.plot(x_in, sample[:,i])
#     plt.show()

    # # plot GP fit
    # plt.figure(1)
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    # plt.ylim([-3, 3])
    # plt.xlim([-3, 3])
    # plt.plot(x, y, 'k+')
    # plt.plot(xstar, pred_mean, color='b')
    # plt.fill_between(xstar, pred_mean + np.sqrt(pred_var), 
    #         pred_mean - np.sqrt(pred_var), color='b', alpha=0.3)
    # plt.savefig('experiments/test_bound/predictive.pdf')

    # # plot just the variance
    # plt.figure(2)
    # plt.xlabel('$x$')
    # plt.ylabel('$Var[f(x)]$')
    # plt.ylim([0, 1.2])
    # plt.xlim([-2, 2])
    # plt.plot(x, np.zeros_like(x), 'k|', markersize=30)
    # plt.plot(xstar, pred_var, color='b')
    # plt.savefig('experiments/test_bound/variance.pdf')

    # # plot just the S.D.
    # plt.figure(3)
    # plt.xlabel('$x$')
    # plt.ylabel('$\sigma[f(x)]$')
    # plt.ylim([0, 1.2])
    # plt.xlim([-2, 2])
    # plt.plot(x, np.zeros_like(x), 'k|', markersize=30)
    # plt.plot(xstar, np.sqrt(pred_var), color='b')
    # plt.savefig('experiments/test_bound/sd.pdf')
    
    # plt.show()
    # plt.savefig(filename)

    # pickle everything as numpy arrays for posterity
#     inputs = xstar
#     mean = pred_mean
#     sd = np.sqrt(pred_var + sigma_n**2)

#     pickle_location = os.path.join('experiments', 'plot_GP')
#     outfile = open(pickle_location, 'wb')
#     pickle.dump(inputs, outfile)
#     pickle.dump(mean, outfile)
#     pickle.dump(sd, outfile)
#     outfile.close()

    



