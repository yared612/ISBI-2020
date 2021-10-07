import glob
import numpy as np
from skimage.io import imread,imshow,imsave

'''soft'''
# path = glob.glob('./deeplab_flod/result/ori/focal/npy/*')
# q_path = './deeplab_flod/result/ori/'
# save_path = './deeplab_flod/ensemble/'

# for i in path:
#     name = i.split('/')[-1]
#     nn   = i.split('/')[-1].split('.')[0]
#     q_type = i.split('/')[-4]
#     q_name = i.split('/')[-3]
#     q_loss = i.split('/')[-3].split('_')[0]
    # img  = np.load(i)[0,:,:,:]
    # img2 = np.load(q_path + q_name + '2/npy/' + name)[0,:,:,:]
    # img3 = np.load(q_path + q_name + '3/npy/' + name)[0,:,:,:]
    # img4 = np.load(q_path + q_name + '4/npy/' + name)[0,:,:,:]
    # img5 = np.load(q_path + q_name + '5/npy/' + name)[0,:,:,:]
    
#     img_s = img + img2 + img3 + img4 + img5
#     ans = np.argmax(img_s,axis = 0)
#     ex_ans = np.where(ans==0,255,0)
#     he_ans = np.where(ans==1,255,0)
    
#     imsave(save_path + q_type + '/' + q_loss + '/soft_vote/ex/' + nn + '.png',ex_ans)
#     imsave(save_path + q_type + '/' + q_loss + '/soft_vote/he/' + nn + '.png',he_ans)


'''hard'''
path = glob.glob('/media/yared/SP PHD U3/eye_paper/deeplab_flod/result/dil/ce_dil/*.png')
q_path = './deeplab_flod/result/dil/'
save_path = './deeplab_flod/ensemble/'
for i in path:
    name = i.split('/')[-1]
    nn   = i.split('/')[-1].split('.')[0]
    q_type = i.split('/')[-3]
    q_name = i.split('/')[-2]
    q_loss = i.split('/')[-2].split('_')[0]
    img_ex  = imread(i)[:,:,0]/255
    img2_ex = imread(q_path + q_name + '2/' + name)[:,:,0]/255
    img3_ex = imread(q_path + q_name + '3/' + name)[:,:,0]/255
    img4_ex = imread(q_path + q_name + '4/' + name)[:,:,0]/255
    img5_ex = imread(q_path + q_name + '5/' + name)[:,:,0]/255
    
    img_he  = imread(i)[:,:,1]/255
    img2_he = imread(q_path + q_name + '2/' + name)[:,:,1]/255
    img3_he = imread(q_path + q_name + '3/' + name)[:,:,1]/255
    img4_he = imread(q_path + q_name + '4/' + name)[:,:,1]/255
    img5_he = imread(q_path + q_name + '5/' + name)[:,:,1]/255
    
    # ex
    ans_ex = (img_ex + img2_ex + img3_ex + img4_ex + img5_ex)
    ans_ex_f = np.where(ans_ex>=(3/5),255,0)
    
    # he
    ans_he = (img_he + img2_he + img3_he + img4_he + img5_he)
    ans_he_f = np.where(ans_he>=(3/5),255,0)
    
    imsave(save_path + '/hard_vote/' + q_name + '/ex/' + nn +'.png',ans_ex_f)
    imsave(save_path + '/hard_vote/' + q_name + '/he/' + nn +'.png',ans_he_f)