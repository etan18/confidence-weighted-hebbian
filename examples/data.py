import logging
import os
import random

import numpy as np
from sklearn.decomposition import PCA
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms

from pytorch_hebbian import config, utils

PATH = os.path.dirname(os.path.abspath(__file__))

class PCADataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def get_data(params, dataset_name, subset=None, pca=False):
    load_test = 'train_all' in params and params['train_all']
    test_dataset = None

    # Loading the dataset and creating the data loaders and transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    if dataset_name == 'mnist':
        dataset = datasets.MNIST(root='data', download=True, transform=transform)
        if load_test:
            test_dataset = datasets.MNIST(root='data', download=True, train=False, transform=transform)
    elif dataset_name == 'mnist-fashion':
        dataset = datasets.mnist.FashionMNIST(root=config.DATASETS_DIR, download=True, transform=transform)
        if load_test:
            test_dataset = datasets.mnist.FashionMNIST(root=config.DATASETS_DIR, download=True, train=False,
                                                       transform=transform)
    elif dataset_name == "cifar-10":
        dataset = datasets.cifar.CIFAR10(root=config.DATASETS_DIR, download=True, transform=transform)
        
        if pca:
            # Collect all data into a single array
            images = np.stack([np.array(dataset[i][0]).flatten() for i in range(len(dataset))])  # Flatten each image
            mean_image = np.mean(images, axis=0)
            std_image = np.std(images, axis=0)
            normalized_images = (images - mean_image) / std_image  # Standardize
            
            # Perform PCA on the flattened images
            pca = PCA(whiten=True)  # Whitening enabled
            pca_images = pca.fit_transform(normalized_images)  # Shape: (n_samples, n_components)

            # Define the number of components to retain
            n_components = 3 * 32 * 32  # Retain the full dimensionality to match the input shape
            pca = PCA(n_components=n_components, whiten=True)
            pca_images = pca.fit_transform(normalized_images)

            # Reshape the PCA output back to (batch_size, channels, height, width)
            pca_images = pca_images.reshape(-1, 3, 32, 32)
            pca_transformed_images = torch.tensor(pca_images, dtype=torch.float32)
            labels = torch.tensor([dataset[i][1] for i in range(len(dataset))], dtype=torch.long)
            
            dataset = PCADataset(pca_transformed_images, labels)
        
        if load_test:
            test_dataset = datasets.cifar.CIFAR10(root=config.DATASETS_DIR, download=True, train=False,
                                                  transform=transform)
            
            if pca:
                test_images = np.stack([np.array(test_dataset[i][0]).flatten() for i in range(len(test_dataset))])
                normalized_test_images = (test_images - mean_image) / std_image
                
                pca_test_images = pca.transform(normalized_test_images)
                pca_test_images = pca_test_images.reshape(-1, 3, 32, 32)
                pca_test_tensor = torch.tensor(pca_test_images, dtype=torch.float32)
                
                test_labels = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))], dtype=torch.long)
                
                test_dataset = PCADataset(pca_test_tensor, test_labels)
                
                print("Pre-processed '{}'.".format(dataset_name))
            
    else:
        raise AttributeError('Dataset not found')

    if subset is not None and subset > 0:
        dataset = Subset(dataset, random.sample(range(len(dataset)), subset))

    if load_test:
        train_loader = DataLoader(dataset, batch_size=params['train_batch_size'], shuffle=True)
        val_loader = DataLoader(test_dataset, batch_size=params['val_batch_size'], shuffle=False)
    else:
        train_dataset, val_dataset = utils.split_dataset(dataset, val_split=params['val_split'])
        train_loader = DataLoader(train_dataset, batch_size=params['train_batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['val_batch_size'], shuffle=False)

    # Analyze dataset
    data_batch = next(iter(train_loader))[0]
    logging.debug("Data batch min: {:.4f}, max: {:.4f}.".format(torch.min(data_batch),
                                                                torch.max(data_batch)))
    logging.debug("Data batch mean: {1:.4f}, std: {0:.4f}.".format(*torch.std_mean(data_batch)))

    return train_loader, val_loader
