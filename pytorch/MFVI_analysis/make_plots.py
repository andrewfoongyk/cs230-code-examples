# make variance plots 

import numpy as np 
import matplotlib.pyplot as plt 
import pickle 

if __name__ == "__main__":

    pred_location = './experiments/last_layer/'

    # make overlaid variance plot
    fig, ax = plt.subplots()

    # unpickle predictives
    f = open(pred_location + 'mfvi_last_layer', "rb")
    test_x = pickle.load(f)
    mean = pickle.load(f)
    sd = pickle.load(f)
    f.close()
    var_mfvi_last_layer = sd**2

    f = open(pred_location + 'mfvi_both_layers', "rb")
    test_x = pickle.load(f)
    mean = pickle.load(f)
    sd = pickle.load(f)
    f.close()
    var_mfvi_both_layers = sd**2

    f = open(pred_location + 'neural_linear', "rb")
    test_x = pickle.load(f)
    mean = pickle.load(f)
    sd = pickle.load(f)
    f.close()
    var_neural_linear = sd**2

    # unpickle data
    with open('..//vision//data//1D_COSINE//1d_cosine_separated.pkl', 'rb') as f:
            data_load = pickle.load(f)

    # plot the data
    ax.plot(data_load[:,0], np.zeros(100), '|', color='k', markersize=50, mew=0.3)

    # plot the predictive
    mfvi_out_line, = ax.plot(test_x, var_mfvi_last_layer, label='mean field output only')
    mfvi_line, = ax.plot(test_x, var_mfvi_both_layers, label='mean field all parameters')
    BLR_line, = ax.plot(test_x, var_neural_linear, label='full Gaussian output only')
    ax.set_xlim([-2,2])
    ax.set_ylim([0,0.25])
    ax.set_xlabel('$x$')
    ax.set_ylabel('predictive variance')
    # ax.set_aspect('box')
    plt.legend(handles=[mfvi_out_line, mfvi_line, BLR_line], loc='upper left')  
    fig.set_size_inches(4.8, 3.2)

    plt.tight_layout()
    plt.savefig('variance_comparison_presentation_3.pdf')

    

    # plot the data points
    # axMFVI.plot(data[:,0], data[:,1], '+k')
    # axMFVI.set_xlabel('$x$')
    # axMFVI.set_ylabel('$y$')
    # axMFVI.set_aspect('equal', 'box')

    # add label
    # axMFVI.text(0.15, 0.92, 'MFVI', horizontalalignment='center', verticalalignment='center', transform=axMFVI.transAxes, fontsize=14)
