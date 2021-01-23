import pandas as pd
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader



def balance_dataframe(df, target):
    '''
    Creates the biggest possible dataframe from the given one and the target feature.
    Works on binary classification only, removes random samples to balance the 2 classes.
    '''

    # Get the least and most represented label
    counts = dict(df[target].value_counts())
    
    min_val = min(counts, key=counts.get)
    min_elements = counts[min_val]
    
    max_val = max(counts, key=counts.get)
    max_elements = counts[max_val]
    

    # Keep the same amount of rows for both labels
    num_remove = max_elements - min_elements
    to_remove = np.random.choice(df[df[target] == max_val].index, size=num_remove, replace=False)

    return df.drop(to_remove)
    
def train_test_split(dataset, ratio, batch_size):
    '''
    Splits a dataset into train and test sets with size depending on ratio.
    '''

    assert ratio >= 0
    assert ratio <= 1

    sizeDS = len(dataset)
    
    indices = list(range(sizeDS))
    np.random.shuffle(indices)
    train_size = int((1-ratio) * sizeDS)
    test_size = sizeDS - train_size

    train_indices, test_indices = indices[:train_size], indices[train_size:]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, test_loader

    
def balance_train_test_split(dataset, ratio, batch_size):
    '''
    Splits a balanced dataset into train and test sets with size depending on ratio.
    Both sets obtained (DataLoaders) are made to be balanced themselves. 
    Works on binary classification datas that have been previously balanced.
    '''
    
    assert ratio >= 0
    assert ratio <= 1
    
    size = len(dataset)
    
    class0_samples = [i for i in range(size) if dataset[i]['target'] == 0]
    class1_samples = list(set(range(size)) - set(class0_samples))
    
    len0 = len(class0_samples)
    assert len0 == len(class1_samples)
    
    # Compute target sizes for the 2 halves of the dataset
    train_size = int((1-ratio) * len0)
    test_size = len0 - train_size
    
    print(f"Train size: {train_size*2}")
    print(f"Test size: {test_size*2}")
    
    indices = list(range(len0))
    np.random.shuffle(indices)
    
    # Indices of the elements to take in class[0-1]_samples lists
    train_indices, test_indices = indices[train_size:], indices[:test_size]
    
    # Indices of the elements to take in the dataset
    d_train_indices = [class0_samples[i] for i in train_indices] + [class1_samples[i] for i in train_indices]
    d_test_indices = [class0_samples[i] for i in test_indices] + [class1_samples[i] for i in test_indices]
    
    train_sampler = SubsetRandomSampler(d_train_indices)
    test_sampler = SubsetRandomSampler(d_test_indices)
    
    # Data loaders for the split
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset, sampler=test_sampler)
    
    return train_loader, test_loader
