import torch
from torch.utils.data import Dataset
from skimage import io
import os
import pandas as pd
import numpy as np
import random

# Dataset storing nb_images per study and study level labels for ill patients.
class MultiImageMultiLabelDataset(Dataset):
    
    def __init__(self, annotations, root_dir, extension='.jpg', nb_images=50, transform=None):
        """
        Args:
            annotations(dataframe): annotations for the dataset's images
            root_dir(string): images directory
            nb_images(int): number of images to keep for a patient
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

        final_p_images = random.sample(p_images, nb_p_images)
        final_n_images = random.sample(n_images, nb_n_images)
        
        final_p_images = list(map(lambda x: self.root_dir + x + self.extension, final_p_images))
        final_n_images = list(map(lambda x: self.root_dir + x + self.extension, final_n_images))
        
        final_images = final_p_images + final_n_images
        
        ratios = [list(self.annotations.loc[self.annotations['StudyInstanceUID']==study]['rv_lv_ratio_gte_1'])[0],
                        list(self.annotations.loc[self.annotations['StudyInstanceUID']==study]['rv_lv_ratio_lt_1'])[0]]

        locations = [list(self.annotations.loc[self.annotations['StudyInstanceUID']==study]['leftsided_pe'])[0],
                            list(self.annotations.loc[self.annotations['StudyInstanceUID']==study]['central_pe'])[0],
                            list(self.annotations.loc[self.annotations['StudyInstanceUID']==study]['rightsided_pe'])[0]]
        
        types = [list(self.annotations.loc[self.annotations['StudyInstanceUID']==study]['acute_pe'])[0],
                            list(self.annotations.loc[self.annotations['StudyInstanceUID']==study]['chronic_pe'])[0],
                            list(self.annotations.loc[self.annotations['StudyInstanceUID']==study]['acute_and_chronic_pe'])[0]]
        
        
        final_target = [ratios, locations, types]
        
        return {'image': final_images, 'target': final_target}
        
        
    def __len__(self):
        """Length is nb of samples"""
        return len(self.studies)
        
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
            
        sample = self.samples[index].copy()

        sample['image'] = np.asarray(list(map(lambda x: io.imread(x), sample['image'])))
        sample['image'] = sample['image'].reshape(512, 512, -1)

        if self.transform:
            sample = self.transform(sample)

        return sample 