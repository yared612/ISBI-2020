import os
import torch.nn as nn
from torchsummary import summary
from efficientnet_pytorch import EfficientNet
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device='cuda'
class DFModel(nn.Module):
    def __init__(self):
        super(DFModel, self).__init__()
        

        self.model = EfficientNet.from_pretrained("efficientnet-b5", advprop=True)
#         self.model = cnn.extract_features()
        self.gap=nn.AdaptiveAvgPool2d(1)
#         self.model = nn.Sequential(*list(cnn.children())[:-1])
        self.intermediate = nn.Sequential(nn.Linear(2560, 256), nn.ReLU(inplace=True))
        self.last = nn.Sequential(nn.Linear(256, 2))
   
        
    
    def forward(self, data):
        
        x = self.model.extract_features(data)
        x = self.gap(x)
        x = x.view(x.size(0),-1)
        x = self.intermediate(x)
        x = self.last(x)
        return x

#model = DFModel().cuda()
#summary(model,(3,512,512))