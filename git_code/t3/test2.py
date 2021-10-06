import numpy as np
from skimage.io import imread,imsave
import glob
import os
from sklearn.utils import shuffle
import keras as ks
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras import backend as K
from keras.losses import binary_crossentropy
from skimage.transform import resize
from mult_output_resnet import *
import pandas as pd
from pandas import DataFrame
im_path    = '/home/john/Downloads/AMD-Challenge2/Validation-400-images'

im_filename = glob.glob(im_path + '/*.jpg')
im_name = []
for i in im_filename:
    n = i.split('/')[-1].split('.')[0]
    im_name.append(n)
def read_data(im_name):
    X,y1,y2,y3,infor = [],[],[],[],[]
    h,w = 1024,1024
    for i in im_name:
        pic      = imread(im_path + '/' + i + '.jpg').astype(np.float)/255
        pic_size = pic.shape
        pic      = resize(pic,(w,h))
        

        
        X.append(pic)
        infor.append([pic_size,(1024,1024,3)])
       
    X  = np.stack(X, axis = 0)

    return X, y1,y2,y3,infor

out=list()
model = resnet()
model.load_weights('/home/john/Downloads/AMD-Challenge2/saved_models/axis_4096_256_2_v2.h5')
for i in range(0,len(im_name)):
    pic = read_data(im_name[i:i+1])
    ans = model.predict(pic[0])
    ans_ = [[np.argmax(ans[0])//64,np.argmax(ans[0])%64],[np.argmax(ans[1])//16,np.argmax(ans[1])%16],ans[2]]
    r_axis = [ans_[0][0]*16 + ans_[1][0]*1 + ans_[2][0][0] , ans_[0][1]*16 + ans_[1][1]*1  + ans_[2][0][1]]
    o_axis = [r_axis[1]*(pic[4][0][0][1]/1024),r_axis[0]*(pic[4][0][0][0]/1024)]
    out.append({'FileName':im_name[i] + '.jpg','Fovea_X':o_axis[0], 'Fovea_Y':o_axis[1]})
    df=DataFrame(out)
    df.to_csv('Fovea_Localization_Results_v3.csv', index=False) 

 

#print('re_axis = ',pic[4][0][2],'ori_axis =' , pic[4][0][3],'\n','pred_axis =', (r_axis[1],r_axis[0]),
#      'pred_o_axis =',o_axis)
#
#for i in range(0,len(im_name)):
#    pic = read_data(im_name[i:i+1])
#    ans = model.predict(pic[0])
#    ans_ = [[np.argmax(ans[0])//32,np.argmax(ans[0])%32],[np.argmax(ans[1])//16,np.argmax(ans[1])%16],[np.argmax(ans[2])//2,np.argmax(ans[2])%2],ans[3]]
#    r_axis = [ans_[0][0]*32 + ans_[1][0]*2 + ans_[2][0] + ans_[3][0][0] , ans_[0][1]*32 + ans_[1][1]*2 + ans_[2][1] + ans_[3][0][1]]
#    o_axis = [r_axis[1]*(pic[5][0][0][1]/1024),r_axis[0]*(pic[5][0][0][0]/1024)]
##    
###    out.append({'imgName':label[i][0], 'ori_Fovea_X':pic[5][0][3][0] , 'ori_Fovea_Y':pic[5][0][3][1] ,'pred_Fovea_X':o_axis[0], 'pred_Fovea_Y':o_axis[1]})
#    out.append({'FileName':im_name[i] + '.jpg','Fovea_X':o_axis[0], 'Fovea_Y':o_axis[1]})
#    df=DataFrame(out)
#
#    df.to_csv('Fovea_Localization_Results.csv', index=False)    