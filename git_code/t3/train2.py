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

im_path    = '/home/xuan/Desktop/new AMD DATA/AMD ori400'
label_path = '/home/xuan/Desktop/new AMD DATA/Fovea_location.xlsx'
label      = pd.read_excel(label_path).iloc[:,1:4]
label      = np.array(label).tolist()
def read_data(label):
    X,y1,y2,y3,y4,infor = [],[],[],[],[],[]
    h,w = 1024,1024
    for i in label:
        pic      = imread(im_path + '/' + i[0])
        pic_size = pic.shape
        pic      = resize(pic,(w,h))
        
        r_x, r_y       = h/pic_size[1], w/pic_size[0]
        axis_x, axis_y = i[1]*r_x, i[2]*r_y
        
        number_1       = int(axis_x//16 + axis_y//16 * 64)
        one_hot_1      = np.zeros([4096])
        one_hot_1[number_1]= 1
        
        number_2       = int((axis_x%64)//16 + (axis_y%64)//16 * 16)
        one_hot_2      = np.zeros([256])
        one_hot_2[number_2]= 1
        
        reg = np.array([axis_x%1,axis_y%1])
        
        X.append(pic)
        infor.append([pic_size,(1024,1024,3)])
        y1.append(one_hot_1)
        y2.append(one_hot_2)
        y3.append(reg)
    X  = np.stack(X, axis = 0)
    y1 = np.stack(y1, axis = 0)
    y2 = np.stack(y2, axis = 0)
    y3 = np.stack(y3, axis = 0)
    return X, y1,y2,y3,axis_x, axis_y

def data_gen_fn(im_name,batch_size):
    Im = shuffle(im_name)
    i = 0
    while True:
        start = i * batch_size
        if (start + batch_size) > len(Im):
            end   = len(Im)
            start2= 0
            end2  = start + batch_size - len(Im)
            Ims   = Im[start:end]+Im[start2:end2]
            out = read_data(Ims)
            yield (out[0], {'out1': out[1], 'out2': out[2], 'out3': out[3]})
        else :
            end   = start + batch_size
            out   = read_data(Im[start:end])
            yield (out[0], {'out1': out[1], 'out2': out[2], 'out3': out[3]})
        i = i + 1
        if (i * batch_size) >= len(Im):
            Im = shuffle(im_name)
            i = 0
            
            
            
def mult_loss(y_true, y_pred):
    l1 = K.categorical_crossentropy(y_true[0],y_pred[0])
    l2 = K.categorical_crossentropy(y_true[1],y_pred[1])
    l3 = K.categorical_crossentropy(y_true[2],y_pred[2])
    w1, w2, w3 = 64/(64+16+1), 16/(64+16+1), 1/(64+16+1)
    return w1*l1 + w2*l2 + w3*l3 

model = resnet(64)
#model.compile(optimizer=Adam(lr=1e-4), loss=mult_loss,metrics=['accuracy'])
model.load_weights('/home/xuan/Desktop/new AMD DATA/axis_detect/saved_models/axis_class.h5')
model.compile(optimizer=Adam(lr=1e-4), loss={'out1':'categorical_crossentropy',
                                             'out2':'categorical_crossentropy',
                                             'out3':'categorical_crossentropy'},
    
                                       loss_weights={'out1' : 64/(64+16+1),
                                                     'out2' : 16/(64+16+1),
                                                     'out3' : 1 /(64+16+1)}, 
                                       metrics=['accuracy'])
epochs = 500
batch_size = 4
data_gen = data_gen_fn(label,batch_size)
steps_per_epoch = int(np.ceil(len(label) / batch_size))
saved_dir = './saved_models'
model_name = 'axis_class.h5'
model_path = '/'.join((saved_dir, model_name))
checkpoint = ModelCheckpoint(model_path, monitor='loss', save_best_only=True, verbose=1)
csv_logger = CSVLogger('training.log')
history = model.fit_generator(data_gen,
                 epochs=epochs, verbose=True,
                 steps_per_epoch=steps_per_epoch,
                 callbacks=[checkpoint, csv_logger])

