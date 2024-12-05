import sys
sys.path.append('../conv_kan_helper')

from torch import nn
import torch.nn.functional as F
from conv_kan_helper.KANConv import KAN_Convolutional_Layer
import torchvision.models as models

class KAN_CNN_MLP_(nn.Module):
    def __init__(self,grid_size= 5):
        super().__init__()
        self.conv1 = KAN_Convolutional_Layer(
            kernel_size= (3,3),
            grid_size = grid_size,
            spline_order=3
            
        )

        self.conv2 = KAN_Convolutional_Layer(
            kernel_size = (3,3),
            grid_size= grid_size,
            spline_order=3

        )

        self.conv3 = KAN_Convolutional_Layer(
            kernel_size = (3,3),
            grid_size= grid_size,
            spline_order=3

        )

        self.pool1 = nn.MaxPool2d(
            kernel_size=(2, 2)
        )
        
        self.flat = nn.Flatten() 
        
        self.linear1 = nn.Linear(2916, 1024)
        self.linear2 = nn.Linear(1024, 64)
        self.linear3 = nn.Linear(64, 8)
        self.name = "KAN Conv Grid updated & 3 Layer MLP"


    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.flat(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = F.log_softmax(x, dim=1)
        return x
