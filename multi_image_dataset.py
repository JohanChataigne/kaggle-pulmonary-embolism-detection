import torch
from torch.utils.data import Dataset
from skimage import io
import os
import pandas as pd
import numpy as np

# Dataset storing images in grey level, corresponding to one channel of the original datas
class MultiImageDataset(Dataset):
    
    def __init__(self, annotations, root_dir, extension='.jpg', nb_images=50, transform=None):
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
        self.nb_images = nb_images
        self.series = dict(enumerate(dict(self.annotations['SeriesInstanceUID'].value_counts()).keys()))
        
    def __len__(self):
        """Length is nb of samples"""
        return len(self.series)
        
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
            
        serie = self.series[index]
        images = self.annotations.loc[self.annotations['SeriesInstanceUID'] == serie][['SOPInstanceUID','pe_present_on_image']]

        p_images = list(images.loc[images['pe_present_on_image'] == 1]['SOPInstanceUID'])
        n_images = list(images.loc[images['pe_present_on_image'] == 0]['SOPInstanceUID'])

        ratio = len(p_images)/len(images)

        nb_p_images = int(50 * ratio)
        nb_n_images = 50 - nb_p_images

        final_p_images = p_images[:nb_p_images]
        final_n_images = n_images[:nb_n_images]
        
        final_p_images = list(map(lambda x: self.root_dir+'/'+x+'.jpg', final_p_images))
        final_n_images = list(map(lambda x: self.root_dir+'/'+x+'.jpg', final_n_images))

        final_images = final_p_images + final_n_images
        final_target = list(self.annotations.loc[self.annotations['SeriesInstanceUID']==serie]['negative_exam_for_pe'])[0]
    
        final_images = np.asarray(list(map(lambda x: io.imread(x).reshape(3, 512, 512), final_images)))
        final_images = final_images.reshape(-1, 512, 512)
        
        sample = {'image': final_images, 'target': final_target}

        if self.transform:
            sample = self.transform(sample)

        return sample 