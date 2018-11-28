"""Train the model"""

import argparse
import logging
import os
import cProfile

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

import utils
import model.net as net
import model.data_loader as data_loader
from evaluate import evaluate
from plot_regression import plot_reg

from tensorboardX import SummaryWriter
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.stats
import math

from build_1d_dataset import Prior_Dataset_Builder

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/64x64_SIGNS', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'

class Weight_Plotter(object):
    """ units is a list, each element is a tuple with (layer, weightrow, weightcol). look at first unit by default"""
    def __init__(self, model, units = [(0,0,0)]): 
        # set up weight animation
        self.fig_weights, self.axis_weights = plt.subplots(1, 1)
        self.axis_weights.set_xlim(-3, 3)
        self.axis_weights.set_ylim(0, 5)
        self.units = units
        self.styles = ['r-', 'g-', 'y-', 'm-', 'k-', 'c-']

        # open the figure window
        self.fig_weights.show()
        self.fig_weights.canvas.draw()

        # Get means and standard deviations to plot
        self.x_weights = np.linspace(-3, 3, 1000)
        weights, biases = model.return_weights()
        
        # plot all the weights in units
        for i, u in enumerate(units):
            mu = weights[u[0]][0][u[1]][u[2]] # [layer][mean/sd][weightrow][weightcol]
            sigma = weights[u[0]][1][u[1]][u[2]]
            y_weights = scipy.stats.norm.pdf(self.x_weights, mu, sigma)
            self.points = self.axis_weights.plot(self.x_weights, y_weights, self.styles[i%6], animated=True, linewidth=0.7)[0]

        # cache the background
        self.background = self.fig_weights.canvas.copy_from_bbox(self.axis_weights.bbox)

    def update(self, model): 
        # update the weight data to plot
        weights = model.return_weights() 

        # restore background
        self.fig_weights.canvas.restore_region(self.background)

        # plot all the weights in units
        for i, u in enumerate(self.units):
            mu = weights[u[0]][0][u[1]][u[2]] # [layer][mean/sd][weightrow][weightcol]
            sigma = weights[u[0]][1][u[1]][u[2]]
            y_weights = scipy.stats.norm.pdf(self.x_weights, mu, sigma)
            self.points = self.axis_weights.plot(self.x_weights, y_weights, self.styles[i%6], animated=True, linewidth=0.3)[0]
            self.axis_weights.draw_artist(self.points)

        # fill in the axes rectangle
        self.fig_weights.canvas.blit(self.axis_weights.bbox)

class Weight_Plot_Saver(object):
    """saves a snapshot of the weight distribution"""
    def __init__(self, directory):         
        # self.x_weights = np.linspace(-3, 3, 1000)
        self.styles = ['r-', 'g-', 'y-', 'm-', 'k-', 'c-']
        self.directory = directory

    def save(self, model, epoch, units, name = "", parameter = 'weights', plot_prior=True):
        # create a new figure
        self.fig_weights, self.axis_weights = plt.subplots(1, 1)
        
        # update the weight data to plot
        weights, biases, priors = model.return_weights()       
        prior_sd = priors[units[0][0]]
        self.axis_weights.set_ylim([0, 10/np.sqrt(2*np.pi*(prior_sd)**2)]) 
        if parameter == 'weights':
            self.x_weights = np.linspace(-4*prior_sd, 4*prior_sd, 1000)
        elif parameter == 'biases':
            self.x_weights = np.linspace(-4, 4, 1000) # since bias variances do not scale with size of input to layer

        if parameter == 'weights':
            # plot all the weights in units
            for i, u in enumerate(units):
                mu = weights[u[0]][0][u[1]][u[2]] # [layer][mean/sd][weightrow][weightcol]
                sigma = weights[u[0]][1][u[1]][u[2]]
                y_weights = scipy.stats.norm.pdf(self.x_weights, mu, sigma)
                self.axis_weights.plot(self.x_weights, y_weights, self.styles[i%6], linewidth=0.3)
        elif parameter == 'biases':
            # plot all the biases in units
            for i, u in enumerate(units):
                mu = biases[u[0]][0][u[1]] # [layer][mean/sd][biasrow]
                sigma = biases[u[0]][1][u[1]]
                y_weights = scipy.stats.norm.pdf(self.x_weights, mu, sigma)
                self.axis_weights.plot(self.x_weights, y_weights, self.styles[i%6], linewidth=0.3)

        # plot the prior 
        if plot_prior == True:
            if parameter == 'weights':
                y_prior = scipy.stats.norm.pdf(self.x_weights, 0, prior_sd)
                self.axis_weights.plot(self.x_weights, y_prior, 'k--', linewidth=1)
            elif parameter == 'biases': # under radford neal's prior scaling, we keep bias prior as standard normal always
                y_prior = scipy.stats.norm.pdf(self.x_weights, 0, 1)
                self.axis_weights.plot(self.x_weights, y_prior, 'k--', linewidth=1)
                
        # save to directory
        filepath = os.path.join(self.directory, name + '_epoch_' + str(epoch) + '.pdf')
        self.fig_weights.savefig(filepath)
        plt.close()
        
def train(model, optimizer, loss_fn, dataloader, metrics, params):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            # move to GPU if available
            if params.cuda:
                train_batch, labels_batch = train_batch.cuda(async=True), labels_batch.cuda(async=True)
            # convert to torch Variables
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

            # compute model output and loss
            output_batch = model(train_batch, no_samples = params.train_samples)
            loss = loss_fn(outputs = output_batch, labels = labels_batch, model = model, dataset_size = trainset_size)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric:metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)

def whole_dataset_train(model, optimizer, loss_fn, dataloader, metrics, params, map_pretrain=False):
    """train without minibatching"""
    # get entire dataset
    dataset = next(iter(dataloader))
    train_set = dataset[0]
    labels_set = dataset[1]

    # move to GPU if available
    if params.cuda:
        train_set, labels_set = train_set.cuda(async=True), labels_set.cuda(async=True)
    # convert to torch Variables
    train_set, labels_set = Variable(train_set), Variable(labels_set)

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()
    
    if map_pretrain == True:
        no_epochs = params.num_pretrain_epochs
    else:
        no_epochs = params.num_epochs
    
    # make list of units to monitor
    no_units = params.hidden_sizes[0] ########################## change
    input_units = []
    output_units = []
    bias_units = []
    for i in range(no_units):
        input_units.append((0, 0, i))
        output_units.append((len(params.hidden_sizes), i, 0)) # change depending on no. of layers
        bias_units.append((0,i))

    # plotter = Weight_Plotter(model, units = units) # initialise weight plotter
    weight_plot_save = Weight_Plot_Saver(args.model_dir)

    # Use tqdm for progress bar
    with tqdm(total = no_epochs) as t:
        for i in range(no_epochs):
            # compute model output and loss
            output_set = model(train_set, no_samples = params.train_samples)
            # print(type(output_set))
            loss_dict = loss_fn(outputs = output_set, labels = labels_set, model = model, dataset_size = trainset_size, term = 'all')
            loss = loss_dict['loss']

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()
            
            if params.model in ('mfvi', 'mfvi_prebias', 'weight_noise'):
                # log to tensorboard
                writer.add_scalar('Loss', loss, i)
                # loss_writer.add_scalar('Train', loss, i)
                writer.add_scalar('KL', loss_dict['KL'], i)
                # KL_writer.add_scalar('Train', loss_dict['KL'],i)
                writer.add_scalar('Reconstruction', loss_dict['reconstruction'], i)
                # reconstruction_writer.add_scalar('Train', loss_dict['reconstruction'],i)
            elif params.model == 'map':
                writer.add_scalar('Loss', loss, i)
                writer.add_scalar('L2', loss_dict['L2'], i)
                writer.add_scalar('Error', loss_dict['error'], i)
            elif params.model == 'fixed_mean_vi': 
                if loss_fn == net.MAP_regression_loss_fn: # pretraining
                    writer.add_scalar('Loss', loss, i)
                    writer.add_scalar('L2', loss_dict['L2'], i)
                    writer.add_scalar('Error', loss_dict['error'], i)
                else: # training the FMVI model
                    writer.add_scalar('Loss', loss, i + params.num_pretrain_epochs)
                    writer.add_scalar('KL', loss_dict['KL'], i + params.num_pretrain_epochs)
                    writer.add_scalar('Reconstruction', loss_dict['reconstruction'], i +params.num_pretrain_epochs)

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_set = output_set.data.cpu().numpy()
                labels_set_cpu = labels_set.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric:metrics[metric](output_set, labels_set_cpu)
                                 for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)

            # if i % 100 == 0:
                # update the weight data to plot for animations
                # plotter.update(model)

            if map_pretrain == True:
                label = 'pretrain_'
            else:
                label = ''

            # plot 
            if i % 1000 == 0: # save images
               plot_reg(model, args.data_dir, params, args.model_dir, epoch_number=i, pretrain=map_pretrain)
               weight_plot_save.save(model, epoch=i, units=input_units, name=label +'input_weights', parameter='weights')
               weight_plot_save.save(model, epoch=i, units=output_units, name=label +'output_weights', parameter='weights')
               weight_plot_save.save(model, epoch=i, units=bias_units, name=label + 'biases', parameter='biases')

            if i == no_epochs-1:
               weight_plot_save.save(model, epoch=i, units=input_units, name=label + 'input_weights', parameter='weights')
               weight_plot_save.save(model, epoch=i, units=output_units, name=label + 'output_weights', parameter='weights')
               weight_plot_save.save(model, epoch=i, units=bias_units, name=label + 'biases', parameter='biases')

            # update the average loss
            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()
        
def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, params, model_dir,
                       restore_file=None, map_pretrain=False):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

    if map_pretrain == 'true':
        num_epochs = params.num_pretrain_epochs
    else:
        num_epochs = params.num_epochs

    for epoch in range(num_epochs):
        if not params.regression == 'true': # classification task
            # Run one epoch
            logging.info("Epoch {}/{}".format(epoch + 1, num_epochs))

            # compute number of batches in one epoch (one full pass over the training set)
            train(model, optimizer, loss_fn, train_dataloader, metrics, params)

            # Evaluate for one epoch on validation set
            val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params)

            val_acc = val_metrics['accuracy']
            is_best = val_acc>=best_val_acc

            # Save weights
            utils.save_checkpoint({'epoch': epoch + 1,
                                'state_dict': model.state_dict(),
                                'optim_dict' : optimizer.state_dict()},
                                is_best=is_best,
                                checkpoint=model_dir)

            # If best_eval, best_save_path
            if is_best:
                logging.info("- Found new best accuracy")
                best_val_acc = val_acc

                # Save best val metrics in a json file in the model directory
                best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
                utils.save_dict_to_json(val_metrics, best_json_path)

            # Save latest val metrics in a json file in the model directory
            last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
            utils.save_dict_to_json(val_metrics, last_json_path)
        
        else: # regression task
            if whole_dataset_batch == True:
                # do not use minibatching
                whole_dataset_train(model, optimizer, loss_fn, train_dataloader, metrics, params, map_pretrain = map_pretrain)
                # Save weights
                utils.save_checkpoint({'epoch': epoch + 1,
                                    'state_dict': model.state_dict(),
                                    'optim_dict' : optimizer.state_dict()},
                                    is_best=True,
                                    checkpoint=model_dir)
                break # only go through this loop once 

            else:
                # Run one epoch
                logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

                # compute number of batches in one epoch (one full pass over the training set)
                train(model, optimizer, loss_fn, train_dataloader, metrics, params)
                
                # plot results for regression task
                if epoch % 100 == 0:
                    plot_reg(model, args.data_dir, params, args.model_dir)
                    # Save weights
                    utils.save_checkpoint({'epoch': epoch + 1,
                                        'state_dict': model.state_dict(),
                                        'optim_dict' : optimizer.state_dict()},
                                        is_best=True,
                                        checkpoint=model_dir)

if __name__ == '__main__':
    # profile the code
    # pr = cProfile.Profile()
    # pr.enable()

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Determine whether to use the entire dataset in one batch (good for small datasets like 1-D regression)
    if params.regression == 'true':
        whole_dataset_batch = True
    
    # set up tensorboard
    tensorboard_path = os.path.join(args.model_dir, 'tensorboard')
    writer = SummaryWriter(tensorboard_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    np.random.seed(0) # 0
    torch.manual_seed(230) #  230
    if params.cuda: torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # Define model to draw prior samples from
    if params.model in ('mfvi', 'map', 'fixed_mean_vi'): # use MFVI net to create prior draws for MAP net
        prior_model = net.MFVI_Net(params, prior_init=True).cuda() if params.cuda else net.MFVI_Net(params, prior_init=True)
    elif params.model == 'mfvi_prebias':
        prior_model = net.MFVI_Prebias_Net(params, prior_init=True).cuda() if params.cuda else net.MFVI_Prebias_Net(params, prior_init=True)
    elif params.model == 'weight_noise':
        prior_model = net.Weight_Noise_Net(params, prior_init=True).cuda() if params.cuda else net.Weight_Noise_Net(params, prior_init=True)

    if params.dataset == 'prior_dataset':
        # create dataset by drawing from prior
        make_prior_dataset = Prior_Dataset_Builder(prior_model, params, args.model_dir) # save prior dataset into the model directory 

    # draw samples from the prior to plot
    plot_reg(prior_model, args.data_dir, params, args.model_dir, epoch_number=0, prior_draw=True)

    # fetch dataloaders
    if params.dataset == 'mnist':
        dataloaders = data_loader.fetch_mnist_dataloader(['train', 'val'], args.data_dir, params)
        trainset_size = 60000
    elif params.dataset == '1d_cosine':
        dataloaders = data_loader.fetch_1d_cosine_dataloader(['train', 'val'], args.data_dir, params)
        trainset_size = 100 # make more flexible later?
    elif params.dataset == 'prior_dataset':
        dataloaders = data_loader.fetch_prior_draw_dataloader(['train', 'val'], args.model_dir, params) # store prior draws in model directory
        trainset_size = params.prior_dataset_size # does this work?
    else: # signs dataset
        dataloaders = data_loader.fetch_dataloader(['train', 'val'], args.data_dir, params)
        trainset_size = len(dataloaders['train']) # does this actually work?? - i don't think so

    print('trainset_size:{}'.format(trainset_size))

    train_dl = dataloaders['train']
    val_dl = dataloaders['val']

    logging.info("- done.")

    # Define the model and optimizer
    if params.model == 'fully_connected':
        model = net.fully_connected_Net(params).cuda() if params.cuda else net.fully_connected_Net(params)
    elif params.model == 'mfvi':
        model = net.MFVI_Net(params).cuda() if params.cuda else net.MFVI_Net(params)
    elif params.model == 'mfvi_prebias':
        model = net.MFVI_Prebias_Net(params).cuda() if params.cuda else net.MFVI_Prebias_Net(params)
    elif params.model == 'weight_noise':
        model = net.Weight_Noise_Net(params).cuda() if params.cuda else net.Weight_Noise_Net(params)
    elif params.model == 'map':
        model = net.MAP_Net(params).cuda() if params.cuda else net.MAP_Net(params)
    elif params.model == 'fixed_mean_vi':
        # initialise the MAP model first
        print('Initialising MAP model')
        model = net.MAP_Net(params).cuda() if params.cuda else net.MAP_Net(params)
    else:
        model = net.Net(params).cuda() if params.cuda else net.Net(params)
    print('cuda:{}'.format(params.cuda))
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    print(model)
    for param in model.parameters():
        print(type(param.data), param.size())

    # fetch loss function and metrics
    if params.model in ('mfvi', 'mfvi_prebias', 'weight_noise', 'fixed_mean_vi'):
        if params.dataset in ('1d_cosine', 'prior_dataset'): # regression
            loss_fn = net.MFVI_regression_loss_fn
        else: # classification
            loss_fn = net.MFVI_loss_fn
    elif params.model == 'map':
        # can only do MAP regression for now, add classification later
        loss_fn = net.MAP_regression_loss_fn
    else: # not bayesian
        if params.dataset in ('1d_cosine', 'prior_dataset'): # regression
            loss_fn = net.regression_loss_fn
            # loss_fn = torch.nn.MSELoss()
        else: # classification
            loss_fn = net.loss_fn

    # maintain all metrics required in this dictionary- these are used in the training and evaluation loops
    if params.dataset == '1d_cosine' or params.dataset == 'prior_dataset':
        metrics = {
            # decide on metrics later ##############
        }
    else:
        metrics = {
            'accuracy': accuracy,
            'nll': nll,
            # could add more metrics such as accuracy for each token type
        }

    if params.model == 'fixed_mean_vi': # pretrain the map solution and use it to initialise Fixed Mean VI
        print('Training MAP model')
        logging.info("Starting training for {} epoch(s)".format(params.num_pretrain_epochs))
        train_and_evaluate(model, train_dl, val_dl, optimizer, net.MAP_regression_loss_fn, metrics, params, args.model_dir,
                       args.restore_file, map_pretrain=True)
        map_model = model

        # initialise model with means from MAP model
        print('Initialising Fixed Mean VI Model')
        model = net.Fixed_Mean_VI_Net(params, map_model).cuda() if params.cuda else net.Fixed_Mean_VI_Net(params, map_model)
        optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
        print(model)
        for param in model.parameters():
            print(type(param.data), param.size())

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, metrics, params, args.model_dir,
                       args.restore_file)
    #plt.show()
    # plot results for regression task
    plot_reg(model, args.data_dir, params, args.model_dir, epoch_number=params.num_epochs)
    # weight_plot_save.save(model, epoch=params.num_epochs)

    # pr.disable()
    # pr.print_stats()
    
