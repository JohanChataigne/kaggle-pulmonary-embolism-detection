import torch
from skimage import transform

class ToTensor(object):
    """Convert ndarrays in a sample to Tensors."""

    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        
        # Need to have 3 dimensions to go through a convnet
        if len(image.shape) < 3:
            image = torch.from_numpy(image).view(1, image.shape[0], image.shape[1])
        else:
            image = torch.from_numpy(image).view(image.shape[2], image.shape[0], image.shape[1])  
        
        return {'image': image,
                'target': target}
    
    
class Normalize(object):
    """Transfer pixel values from [0;255] to [0;1]"""

    def __call__(self, sample):
        image, target = sample['image'], sample['target']

        image = 1./255 * image
        
        return {'image': image,
                'target': target}

class Rescale(object):
    """Rescale the image in a sample to a given size. Usefull to have all samples of same shape in input of a CNN
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'target': sample['target']}