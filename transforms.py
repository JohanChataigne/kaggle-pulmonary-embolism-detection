import torch

class ToTensor(object):
    """Convert ndarrays in a sample to Tensors."""

    def __call__(self, sample):
        image, target = sample['image'], sample['target']

        return {'image': torch.from_numpy(image),
                'target': target}
    
    
class Normalize(object):
    """Transfer pixel values from [0;255] to [0;1]"""

    def __call__(self, sample):
        image, target = sample['image'], sample['target']

        image = 1./255 * image
        
        return {'image': image,
                'target': target}