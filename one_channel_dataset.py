import torch
from torch.utils.data import Dataset
from skimage import io
import os
import pandas as pd

# Dataset storing images in grey level, corresponding to one channel of the original datas
class OneChannelImageDataset(Dataset):
    
    def __init__(self, annotations, root_dir, extension='.jpg', channel=0, transform=None):
        """
        Args:
            annotations(dataframe): annotations for the dataset's images
            root_dir(string): images directory
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        assert(channel in (0,1,2))
        
        self.annotations = annotations
        self.root_dir = root_dir
        self.extension = extension
        self.channel = channel
        self.transform = transform
        
    def __len__(self):
        """Length is nb_captions / captions_per_image"""
        return len(self.annotations)
        
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
            
        im_name, target = self.annotations.iloc[index, :]
        im_path = os.path.join(self.root_dir, im_name + self.extension)
        image = io.imread(im_path)[:,:,self.channel]
        
        sample = {'image': image, 'target': target}

        if self.transform:
            sample = self.transform(sample)

        return sample 