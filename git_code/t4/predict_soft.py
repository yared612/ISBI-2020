#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 01:41:54 2020

@author: yared
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 18:01:11 2020

@author: 20190524
"""

import argparse
import os
import numpy as np
import torch
import cv2, glob

from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *

from dataloaders.utils import decode_segmap
MI = np.load('meanimage.npy')

numClass={'pascal':3,
'coco':21,
'cityscapes':19}
classes = ['ex','he']
pre_n = 'ce_dil'
if not os.path.isdir('result/' + pre_n):
        os.mkdir('result/' + pre_n)
for cid, c in enumerate(classes):
    if not os.path.isdir('result/'+ pre_n +'/'+c):
        os.mkdir('result/'+ pre_n +'/' + c)

# parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")

# parser.add_argument('--backbone', type=str, default='resnet',choices=['resnet', 'xception', 'drn', 'mobilenet'],help='backbone name (default: resnet)')
# parser.add_argument('--out-stride', type=int, default=16,help='network output stride (default: 8)')
# parser.add_argument('--dataset', type=str, default='cityscapes', choices=['pascal', 'coco', 'cityscapes'],help='dataset name (default: pascal)')
# parser.add_argument('--sync-bn', type=bool, default=None,help='whether to use sync bn (default: auto)')
# parser.add_argument('--freeze-bn', type=bool, default=False,help='whether to freeze bn parameters (default: False)')
# parser.add_argument('--weightPath', type=str, default=None,help='put the path to resuming file if needed')
# parser.add_argument('--imgPath', type=str, default=None,help='put the path to resuming file if needed')
# parser.add_argument('--outPath', type=str, default=None,help='put the path to resuming file if needed')
# parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
# args = parser.parse_args()
cuda = torch.cuda.is_available()
cuda = False
nclass = numClass['pascal']
model = DeepLab(num_classes=nclass, backbone='resnet', output_stride=16, sync_bn=None, freeze_bn=False)
weight_dict=torch.load(r'/home/john/Desktop/DEEPLAB(Idr)/run/pascal/'+ pre_n +'/model_best.pth.tar', map_location='cpu')
model2 = DeepLab(num_classes=nclass, backbone='resnet', output_stride=16, sync_bn=None, freeze_bn=False)
weight_dict2=torch.load(r'/home/john/Desktop/DEEPLAB(Idr)/run/pascal/'+ pre_n +'2/model_best.pth.tar', map_location='cpu')
model3 = DeepLab(num_classes=nclass, backbone='resnet', output_stride=16, sync_bn=None, freeze_bn=False)
weight_dict3=torch.load(r'/home/john/Desktop/DEEPLAB(Idr)/run/pascal/'+ pre_n +'3/model_best.pth.tar', map_location='cpu')
model4 = DeepLab(num_classes=nclass, backbone='resnet', output_stride=16, sync_bn=None, freeze_bn=False)
weight_dict4=torch.load(r'/home/john/Desktop/DEEPLAB(Idr)/run/pascal/'+ pre_n +'4/model_best.pth.tar', map_location='cpu')
model5 = DeepLab(num_classes=nclass, backbone='resnet', output_stride=16, sync_bn=None, freeze_bn=False)
weight_dict5=torch.load(r'/home/john/Desktop/DEEPLAB(Idr)/run/pascal/'+ pre_n +'5/model_best.pth.tar', map_location='cpu')
if cuda:
    model = torch.nn.DataParallel(model, device_ids=[0])
    patch_replication_callback(model)
    model = model.cuda()
    model2 = torch.nn.DataParallel(model2, device_ids=[0])
    patch_replication_callback(model2)
    model2 = model2.cuda()
    model3 = torch.nn.DataParallel(model3, device_ids=[0])
    patch_replication_callback(model3)
    model3 = model3.cuda()
    model4 = torch.nn.DataParallel(model4, device_ids=[0])
    patch_replication_callback(model4)
    model4 = model4.cuda()
    model5 = torch.nn.DataParallel(model5, device_ids=[0])
    patch_replication_callback(model5)
    model5 = model5.cuda()
    
    
    model.module.load_state_dict(weight_dict['state_dict'])
    model2.module.load_state_dict2(weight_dict['state_dict'])
    model3.module.load_state_dict3(weight_dict['state_dict'])
    model4.module.load_state_dict4(weight_dict['state_dict'])
    model5.module.load_state_dict5(weight_dict['state_dict'])
else:
    model.load_state_dict(weight_dict['state_dict']) 
model.eval()
model2.eval()
model3.eval()
model4.eval()
model5.eval()


filenames = glob.glob(r'/home/john/Desktop/IDRiD(new)/test/ori/*.jpg')
kernel = np.ones((5,5),np.uint8)
kernel2 = np.ones((7,7),np.uint8)
for imgPath in filenames:
    fn = os.path.basename(imgPath)[:-4]
    outPath = 'result/'+ pre_n +'/' + fn +'.png'

    image = cv2.imread(imgPath)
    oriDim = image.shape
    image = cv2.resize(image, dsize=(513,513)) - MI
    image = image.astype(np.float32) / 255.
    image = image[:, :, ::-1]
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    
    
    
    for i in range(3):
        image[:, :, i] = image[:, :, i] - means[i]
        image[:, :, i] = image[:, :, i] / stds[i]

    image = torch.from_numpy(image.transpose((2, 0, 1)).astype(np.float32)).float().unsqueeze(0)

    if cuda:
        image = image.cuda()
        
    with torch.no_grad():
        output = model(image)
        output = output.data.cpu().numpy()
        output2 = model2(image)
        output2 = output2.data.cpu().numpy()
        output3 = model3(image)
        output3 = output3.data.cpu().numpy()
        output4 = model4(image)
        output4 = output4.data.cpu().numpy()
        output5 = model5(image)
        output5 = output5.data.cpu().numpy()
        
        output_s = (output +  output2 + output3 + output4 + output5)/5
        
        
        
        prediction = np.argmax(output_s, axis=1)[0]
   
        ps=[]
        for cid, c in enumerate(classes):
            mask = np.zeros((prediction.shape[0], prediction.shape[1]), np.uint8) +255
            mask[prediction == cid+1] = 0
            mask = cv2.morphologyEx(255-mask, cv2.MORPH_OPEN, kernel)
            mask = 255-cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel2)

            mask = cv2.resize(mask,dsize=(oriDim[1],oriDim[0]), interpolation=cv2.INTER_NEAREST)
            ps.append(np.mean(prediction == cid+1))
            cv2.imwrite('result/'+ pre_n +'/' + c + '/' + fn+'.png', mask)

        segmap = decode_segmap(prediction, dataset='pascal')
        segmap = (segmap*255).astype(np.uint8)
        segmap = cv2.resize(segmap,dsize=(oriDim[1],oriDim[0]))
        segmap = segmap[:, :, ::-1]
        cv2.imwrite(outPath, segmap)
    print('Done inference '+fn,'percentage:', ps)
exit(1)
