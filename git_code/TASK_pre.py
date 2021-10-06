import cv2
import os, glob
from torchvision import models
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from TASK1_dataset import *
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision import models as tm
from TASK1_model import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda'
file_name = '/home/john/Desktop/AMD Task 4 class/yared/test'
model_path= torch.load('/home/john/Desktop/AMD Task 4 class/yared/efficientnet_v1(colorjitter)/model_epoch_154.pth')
image     = glob.glob(file_name + '/*.jpg')
model = DFModel().cuda()

model.load_state_dict(model_path)
model.eval()
image_    = Image.open(image[0])
mean_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
        ])
Y = []

with torch.no_grad():
    for batch_idx, f in enumerate(image):
        name = os.path.basename(f)

        im = mean_transform(Image.open(f)).unsqueeze(0)
#        print(im.type, im.shape)
        
        pred = model(im.to(device))
        pred = torch.sigmoid(pred)
        pred = pred[0].cpu().numpy()

        Y.append([name, pred[1]])
    
pds = pd.DataFrame(Y)
pds.columns = ['FileName', 'glaucoma']
pds.to_csv('Classification_Results.csv', index=False)

    


