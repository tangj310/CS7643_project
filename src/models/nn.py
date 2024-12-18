import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights
from torchvision.models.efficientnet import EfficientNet
from torchvision.models.resnet import ResNet




class CustomNN(nn.Module):
    """

    """
    def __init__(self, config):
        """
        Initialize the model with 8 classes target .
        
        """
        super(CustomNN, self).__init__()
        
        model_name = config['model_name']

        if model_name == 'ResNet':

            self.model = models.resnet50(
                pretrained=True
                )
  

        if model_name == 'EfficientNet':

            self.model = models.efficientnet_v2_l(
                pretrained=True
                )
            
    

        self.model.fc = nn.Sequential(
            nn.Linear(2048, 100)  # dense layer takes a 2048-dim input and outputs 100-dim
            ,nn.ReLU(inplace=True)  # ReLU activation introduces non-linearity
            ,nn.Dropout(0.1)  # common technique to mitigate overfitting
            ,nn.Linear(
                100 # input
                ,8 # final dense layer outputs 8-dim corresponding to our target classes
            ),
            )
        
    def get_model(self):
        """Return the vit_model."""
        return self.model