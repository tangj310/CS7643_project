import torch
import torch.nn as nn



class CustomLoss(nn.Module):
    """
    Custom simple ViT model with modified head for multi-label classification.
    
        
    """


    def __init__(self, config):
        super(CustomLoss, self).__init__() # Initialize the parent nn.Module

        loss_name = config['loss_name']

        if loss_name == 'CrossEntropyLoss':
            self.criterion = nn.CrossEntropyLoss()

        if loss_name == 'BCEWithLogitsLoss':
            self.criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for multi-label classification



    def get_loss(self):
        """Return the loss function."""
        return self.criterion