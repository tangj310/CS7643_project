import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from src.models.vit import CustomViT
from src.models.nn import CustomNN

class CustomOptimizer(nn.Module):
    """
    Custom simple ViT model with modified head for multi-label classification.
    
        
    """


    def __init__(self, config):

        super(CustomOptimizer, self).__init__()


        optimizer_name = config['optimizer_name']
        lr = config['optimizer_hyper_p']['lr']
        momentum = config['optimizer_hyper_p']['momentum']

        if optimizer_name == 'Adam':

            params = CustomViT(
                config=config
            ).model.parameters()

            self.optimizer = Adam(
                params=params
                ,lr=lr
                )
            
        if optimizer_name == 'SGD':

            params = CustomNN(
                config=config
            ).model.parameters()

            self.optimizer = SGD(
                params=params
                ,lr=lr
                ,momentum=momentum
                )
        

    def get_optimizer(self):
        """Return the optmizer."""
        return self.optimizer
