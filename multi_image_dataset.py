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
        self.studies = dict(enumerate(dict(self.annotations['StudyInstanceUID'].value_counts()).keys()))
    
        self.samples = dict()
        
        for key in self.studies.keys():
            
            self.samples[key] = self.init_sample(key)
        
        
    def init_sample(self, index):
        
        study = self.studies[index]
        images = self.annotations.loc[self.annotations['StudyInstanceUID'] == study][['SOPInstanceUID','pe_present_on_image']]

        # Separate positive and negative samples */
        p_images = list(images.loc[images['pe_present_on_image'] == 1]['SOPInstanceUID'])
        n_images = list(images.loc[images['pe_present_on_image'] == 0]['SOPInstanceUID'])

        # Compute ratio of positives
        ratio = len(p_images)/len(images)

        # Keep the same ratio while picking nb_images samples
        nb_p_images = int(self.nb_images * ratio)
        nb_n_images = self.nb_images - nb_p_images

        final_p_images = p_images[:nb_p_images]
        final_n_images = n_images[:nb_n_images]
        
        final_p_images = list(map(lambda x: self.root_dir+'/'+x+'.jpg', final_p_images))
        final_n_images = list(map(lambda x: self.root_dir+'/'+x+'.jpg', final_n_images))

        final_images = final_p_images + final_n_images
        final_target = list(self.annotations.loc[self.annotations['StudyInstanceUID']==study]['negative_exam_for_pe'])[0]
    
        final_images = np.asarray(list(map(lambda x: io.imread(x), final_images)))
        final_images = final_images.reshape(512, 512, -1)
        
        return {'image': final_images, 'target': final_target}
        
        
    def __len__(self):
        """Length is nb of samples"""
        return len(self.studies)
        
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
            
        sample = self.samples[index]

        if self.transform:
            sample = self.transform(sample)

        return sample 