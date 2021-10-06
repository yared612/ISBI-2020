import cv2
import os, glob
from torchvision import models
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from torchvision import models as tm
from torchsummary import summary
from TASK1_model import *
from TASK1_dataset import *
from sklearn.model_selection import train_test_split
import pandas as pd
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    filename = 'C:/Users/tommy/Desktop/AMDc/noextended/AMD/'
    filename1 = 'C:/Users/tommy/Desktop/AMDc/noextended/Non-AMD/'
    file = []
    file.append(filename)
    file.append(filename1)
    X,y = [], []
    for i in file:
        X.append(glob.glob(i + '/*.jpg'))
    X = X[0] + X[1]
    for j in X:
        if (j.split('/')[-1]).split('\A')[0] == 'AMD' :
            y.append(1)
#        elif (j.split('/')[-1]).split('\i')[0] == 'AMD' :
#            y.append(1)
        else :
            y.append(0)
    
    parameter_list = {'batch_size':3,'epochs':200}
    train_X, val_X, train_y, val_y = train_test_split(X,y,test_size=0.1,random_state=0,shuffle=True)
    
    train_loader = torch.utils.data.DataLoader(
                        dataset_TASK1(train_X, train_y), 
                        batch_size=parameter_list['batch_size'], 
                        shuffle=True, 
                        num_workers=4
                        )
    
    val_loader   = torch.utils.data.DataLoader(
                        dataset_TASK1(val_X, val_y,mode = 'val'), 
                        batch_size=parameter_list['batch_size'], 
                        shuffle=True, 
                        num_workers=4
                        )
    model = DFModel().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    ce = torch.nn.CrossEntropyLoss( weight=torch.Tensor([1., 2.44]).to(device))
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    
    resume_ind = 0
    step = 0
    find_best = []
    find_best_name=['batch','loss','ACC']
    for epoch in range(parameter_list['epochs']): 
        ep_loss = 0.
        for batch_idx, (x,y) in enumerate(train_loader):
            running_loss = 0.0
            optimizer.zero_grad()
            pred = model(x.to(device))
            lab = y.to(device).long()
    
            loss = ce(pred, lab)
            loss.backward()
            optimizer.step()
            running_loss  +=  loss.item()
    
            if batch_idx % 50 == 1:  
                with torch.no_grad():
                    model.eval()
                    cnt, acc, acc2 = 0, 0., 0.
                    for ind2, (vx, vy) in enumerate(val_loader):
                        pred_prob = model(vx.to(device))
                        pred = torch.argmax(pred_prob, 1).cpu().numpy()
                        acc += np.mean(pred == vy.cpu().numpy())
    
                        cnt+=1
    
                    print('[epoch: %d, batch: %5d] loss: %.3f, ACC: %.3f' %
                          (epoch, batch_idx+resume_ind, running_loss, acc/cnt))
                    find_best.append([batch_idx+resume_ind,running_loss,acc/cnt])
                ep_loss += running_loss
                step+=1
                running_loss = 0.0
                model.train() 
                
        print('Model saved!')
        torch.save(model.state_dict(), 'checkpoint/model_epoch_%d.pth' %(epoch))
        print('Epoch averaged loss = %f' % ep_loss)
        find_best.append([epoch,'<<epoch || averaged loss>>',ep_loss])
    pd.DataFrame(columns=find_best_name,data=find_best).to_csv('C:\\Users\\tommy\\Desktop\\AMDc\\record.csv',index=False,encoding='gbk')