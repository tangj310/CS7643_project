import torch
import torch.nn as nn
import yaml



class CustomLoss(nn.Module):
    """
    Custom simple ViT model with modified head for multi-label classification.
    
        
    """


    def __init__(self):
        super(CustomLoss, self).__init__() # Initialize the parent nn.Module
        self.criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for multi-label classification



    def get_loss(self):
        """Return the loss function."""
        return self.criterion