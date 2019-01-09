import random
import os
import pickle
import gzip
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# borrowed from http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
# and http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# define a training image loader that specifies transforms on images. See documentation for more details.
train_transformer = transforms.Compose([
    transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)
    transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
    transforms.ToTensor()])  # transform it into a torch tensor

# loader for evaluation, no horizontal flip
eval_transformer = transforms.Compose([
    transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)
    transforms.ToTensor()])  # transform it into a torch tensor

class PriorDataset(Dataset):
    """load prior draw dataset"""
    def __init__(self, data_dir, split):
        # unpickle dataset
        filename = os.path.join(data_dir, 'prior_dataset.pkl')
        with open(filename, 'rb') as f:
            data, x_line, y_line = pickle.load(f)
        print('unpickled prior draw dataset')

        # regardless of whether this is train, val or test split, always use the whole dataset
        self.X = data[:,0]
        self.Y = data[:,1]
        self.X = np.float32(self.X)
        self.Y = np.float32(self.Y)

    def __len__(self):
        # return size of dataset
        return len(self.X)

    def __getitem__(self, idx):
        x_value = self.X[idx]
        y_value = self.Y[idx]
        return x_value, y_value

def fetch_prior_draw_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:
            dl = DataLoader(PriorDataset(data_dir, split=split), batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda)

            dataloaders[split] = dl

    return dataloaders

class CosineDataset(Dataset):
    """ load 1-D cosine dataset """
    def __init__(self, data_dir, split):
        # unpickle cosine dataset
        filename = os.path.join(data_dir, '1d_cosine_separated.pkl') ############# use separated version
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print('unpickled 1-D cosine dataset')

        # regardless of whether this is train, val or test split, always use the whole dataset
        self.X = data[:,0]
        self.Y = data[:,1]
        self.X = np.float32(self.X)
        self.Y = np.float32(self.Y)

    def __len__(self):
        # return size of dataset
        return len(self.X)

    def __getitem__(self, idx):
        x_value = self.X[idx]
        # print('x:{}'.format(x_value))
        y_value = self.Y[idx]
        # print('y:{}'.format(y_value))
        return x_value, y_value

def fetch_1d_cosine_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:
            dl = DataLoader(CosineDataset(data_dir, split=split), batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda)

            dataloaders[split] = dl

    return dataloaders

class MNISTDataset(Dataset):
    """ load the MNIST Dataset """
    def __init__(self, data_dir, split):
        
        # unpickle MNIST
        filename = os.path.join(data_dir, "mnist.pkl.gz")
        f = gzip.open(filename, 'rb')
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        f.close()
        print("unpickled MNIST")

        self.split = split
        if split == 'train':
            self.X = np.vstack((train_set[0], valid_set[0]))
            self.X = np.resize(self.X,(60000,1,28,28))
            self.Y = np.hstack((train_set[1], valid_set[1]))
        elif split == 'val':
            self.X = test_set[0]
            self.X = np.resize(self.X,(10000,1,28,28))
            self.Y = test_set[1]
    
    def __len__(self):
        # return size of dataset
        if self.split == 'train':
            return 60000 # 50k train, 10k valid, 10k test
        elif self.split == 'val':
            return 10000

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.Y[idx]
        return image, label

def fetch_mnist_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:
            dl = DataLoader(MNISTDataset(data_dir, split=split), batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda)

            dataloaders[split] = dl

    return dataloaders

        
class SIGNSDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, data_dir, transform):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        self.filenames = os.listdir(data_dir)
        self.filenames = [os.path.join(data_dir, f) for f in self.filenames if f.endswith('.jpg')]

        self.labels = [int(os.path.split(filename)[-1][0]) for filename in self.filenames]
        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        image = Image.open(self.filenames[idx])  # PIL image
        image = self.transform(image)
        return image, self.labels[idx]


def fetch_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:
            path = os.path.join(data_dir, "{}_signs".format(split))

            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                dl = DataLoader(SIGNSDataset(path, train_transformer), batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda)
            else:
                dl = DataLoader(SIGNSDataset(path, eval_transformer), batch_size=params.batch_size, shuffle=False,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)

            dataloaders[split] = dl

    return dataloaders
