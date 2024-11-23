import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.models import VisionTransformer  # define own transformers
import yaml



class CustomLoss(nn.Module):
    """
    Custom simple ViT model with modified head for multi-label classification.
    
        
    """


    def __init__(self):

        self.criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for multi-label classification