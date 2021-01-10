import torch
from torch.utils.data import Dataset
from skimage import io
import os
import pandas as pd

# Dataset storing images in grey level, corresponding to one channel of the original datas
class FullImageDataset(Dataset):
    
    def __init__(self, annotations, root_dir, extension='.jpg', transform=None):
        """
        Args:
            annotations(dataframe): annotations for the dataset's images
            root_dir(string): images directory
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.annotations = annotations
        self.root_dir = root_dir
        self.extension = extension
        self.transform = transform
        
    def __len__(self):
        """Length is nb of samples"""
        return len(self.annotations)
        
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
            
        im_name, target = self.annotations.iloc[index, :]
        im_path = os.path.join(self.root_dir, im_name + self.extension)
        image = io.imread(im_path)
        
        sample = {'image': image, 'target': target}

        if self.transform:
            sample = self.transform(sample)

        return sample 