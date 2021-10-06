from skimage.io import imread, imsave, imshow
from PIL import Image
import glob, os, pdb, cv2 
import numpy as np
from torch.utils import data
from torchvision import transforms
from torch.utils.data import Sampler
from torch.utils.data import DataLoader

class dataset_TASK1(data.Dataset):
    def __init__(self, X, Y, image_size=512, crop_size=480, mode='Train'):
        super(dataset_TASK1,self).__init__()
        
        self.img = X
        self.gt  = Y
        self.mode= mode
        
        self.image_size = image_size
        self.crop_size  = crop_size
        
        self.mean_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(20, resample=Image.BILINEAR, center=(240, 240)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
        ])
    
        self.transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
        ])
        
    def __getitem__(self,index):
        im = Image.open(self.img[index])
        if self.mode == 'Train':
            im = self.transform(im)
        else :
            im = self.mean_transform(im)
        label = self.gt[index]
        
        return im, label
    
    
    def __len__(self):
        return len(self.img)
        