import torch.nn as nn
import torchvision.models as models
from torchvision.models.efficientnet import EfficientNet
from torchvision.models.resnet import ResNet

class CustomNN(nn.Module):
    """
    Custom ResNet50 model with modified head for multi-label classification.
    
    Architecture:
    - ResNet50 backbone (pretrained)
    - Custom classifier head:
        - Linear(2048, 100)
        - ReLU
        - Dropout(0.1)
        - Linear(100, num_classes)
    """
    def __init__(self, weights=ResNet50_Weights.DEFAULT):
        """
        Initialize the model with 8 classes target .
        
        """
        super().__init__()
        
        # Load pretrained ResNet50
        self.resnet = models.resnet50(weights=weights)
        
        # Replace the final fully connected layer
        self.resnet.fc = nn.Sequential(
            nn.Linear(2048, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(100, 8)
        )
    
    
    



