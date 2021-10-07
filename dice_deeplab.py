#import cv2
import os
import numpy as np
import glob
from skimage.io import imread,imshow
ex_pre_path = './deeplab_flod/ensemble/soft_vote/ce/ex/'
he_pre_path = './deeplab_flod/ensemble/soft_vote/ce/he/'
ex_gt_path = './test_gt/ex/'
he_gt_path = './test_gt/he/'
ex_im_filename = glob.glob(ex_pre_path + '*.png')
he_im_filename = glob.glob(he_pre_path + '*.png')
d = []
a = []
cc = 0
for k in ex_im_filename:
    n = k.split('/')[-1].split('.')[0]
    n1 = k.split('/')[-1].split('.')[1]
    pre = imread(ex_pre_path + n + '.' + n1 )
    gt = (imread(ex_gt_path + n + '_EX' + '.tif')*255)

    pre = (pre)/255
    gt = (gt)/255
    a = 2*gt*pre
    b = gt + pre
    dice = (np.sum(a)+0.000001)/(np.sum(b)+0.000001)
    d.append(dice)
    # print(dice)
    cc = cc + 1
    
d1 = []
a1 = []
cc1 = 0
for k1 in he_im_filename:
    n1 = k1.split('/')[-1].split('.')[0]
    n11 = k1.split('/')[-1].split('.')[1]
    pre1 = imread(he_pre_path + n1 + '.' + n11 )
    gt1 = (imread(he_gt_path + n1 + '_HE' + '.tif')*255)

    pre1 = (pre1)/255
    gt1 = (gt1)/255
    a1 = 2*gt1*pre1
    b1 = gt1 + pre1
    dice1 = (np.sum(a1)+0.000001)/(np.sum(b1)+0.000001)
    d1.append(dice1)
    # print(dice)
    cc1 = cc1 + 1

x = sum(d)/cc
x1 = sum(d1)/cc1
print('ex:',x)
print('he:',x1)
    
 
   