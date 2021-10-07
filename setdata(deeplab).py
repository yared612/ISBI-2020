import numpy as np
import glob
import os
import pandas as pd
from skimage.io import imread,imsave,imshow
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

im_path = '/home/john/Desktop/IDRiD(new)/ori'
lab1_path = '/home/john/Desktop/IDRiD(new)/gt/ex'
lab2_path = '/home/john/Desktop/IDRiD(new)/gt/he'
#lab3_path = '/home/john/Desktop/paper_using/ADAM Dataset/Lesion_Masks/exudate/dilate'
#lab4_path = '/home/john/Desktop/paper_using/ADAM Dataset/Lesion_Masks/hemorrhage/dilate'

save_path = '/media/john/SP PHD U3/eye_paper/deeplab_flod/SegmentationClass'
#txt1_path = '/home/user/Documents/Lin Cheng/flod1'
#txt2_path = '/home/user/Documents/Lin Cheng/flod2'
#txt3_path = '/home/user/Documents/Lin Cheng/flod3'
#txt4_path = '/home/user/Documents/Lin Cheng/flod4'
#txt5_path = '/home/user/Documents/Lin Cheng/flod5'

im_filename = glob.glob(im_path + '/*.jpg')
im_name = []
for i in im_filename:
    n = i.split('/')[-1].split('.')[0]
    im_name.append(n)
# im_name.sort()
train, test = train_test_split(im_name,random_state=2, train_size=0.7)

def read_data(im_name):
    for i in im_name:
        lab1 = (imread(lab1_path + '/' + i + '.bmp')).astype(np.float)/255
        lab2 = (imread(lab2_path + '/' + i + '.bmp')).astype(np.float)/255
#        lab3 = (255-imread(lab3_path + '/' + i + '.bmp')).astype(np.float)/255
#        lab4 = (255-imread(lab4_path + '/' + i + '.bmp')).astype(np.float)/255        
        lab3 = lab1+lab2
        ans1 = np.where(lab1>0,1,0)
        ans2 = np.where(lab2>0,2,0)
        ans_for = ans1+ans2
        ans_com = np.where(ans_for==3,2,ans_for)
#        ans = ans_com*ans_for
#        ans3 = np.where(lab3>0,3,0)
#        ans4 = np.where(lab4>0,4,0)
        ans3 = np.where(lab3==0,0,0)        
#        ans = (ans1+ans2+ans3).astype(np.uint8)        
        ans = ans_com.copy().astype(np.uint8)
        imsave(save_path + '/' + i + '.png', ans)
    return ans
read_data(im_name)
#def k_flod_data(im_name,flod_num):
#    list1 = shuffle(im_name,random_state=1)
#    n = 5
#    m = int(len(list1)/n)
#    im = []
#    for i in range(0, len(list1), m):
#        im.append(list1[i:i+m])    
#    if flod_num==0:
#        Im = im[1]+im[2]+im[3]+im[4]
#        val = im[0]
#    elif flod_num==1:
#        Im = im[0]+im[2]+im[3]+im[4]
#        val = im[1]
#    elif flod_num==2:
#        Im = im[0]+im[1]+im[3]+im[4]
#        val = im[2]
#    elif flod_num==3:
#        Im = im[0]+im[1]+im[2]+im[4]
#        val = im[3]
#    elif flod_num==4:
#        Im = im[0]+im[1]+im[2]+im[3]
#        val = im[4]
#    return Im,val
# Im1,val1 = k_flod_data(train, 0)
# Im2,val2 = k_flod_data(train, 1)
# Im3,val3 = k_flod_data(train, 2)
# Im4,val4 = k_flod_data(train, 3)
# Im5,val5 = k_flod_data(train, 4)

# df_im1 = pd.DataFrame(Im1)
# df_val1 = pd.DataFrame(val1)
# df_im2 = pd.DataFrame(Im2)
# df_val2 = pd.DataFrame(val2)
# df_im3 = pd.DataFrame(Im3)
# df_val3 = pd.DataFrame(val3)
# df_im4 = pd.DataFrame(Im4)
# df_val4 = pd.DataFrame(val4)
# df_im5 = pd.DataFrame(Im5)
# df_val5 = pd.DataFrame(val5)


# df_im1.to_csv('/home/user/Documents/Lin Cheng/flod1/train.txt', header=None, index=None)
# df_val1.to_csv('/home/user/Documents/Lin Cheng/flod1/validation.txt', header=None, index=None)
# df_im2.to_csv('/home/user/Documents/Lin Cheng/flod2/train.txt', header=None, index=None)
# df_val2.to_csv('/home/user/Documents/Lin Cheng/flod2/validation.txt', header=None, index=None) 
# df_im3.to_csv('/home/user/Documents/Lin Cheng/flod3/train.txt', header=None, index=None)
# df_val3.to_csv('/home/user/Documents/Lin Cheng/flod3/validation.txt', header=None, index=None) 
# df_im4.to_csv('/home/user/Documents/Lin Cheng/flod4/train.txt', header=None, index=None)
# df_val4.to_csv('/home/user/Documents/Lin Cheng/flod4/validation.txt', header=None, index=None) 
# df_im5.to_csv('/home/user/Documents/Lin Cheng/flod5/train.txt', header=None, index=None)
# df_val5.to_csv('/home/user/Documents/Lin Cheng/flod5/validation.txt', header=None, index=None)  
    
