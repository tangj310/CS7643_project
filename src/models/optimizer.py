import torch
import torch.nn as nn
from torch.optim import Adam
from src.models.vit import CustomViT


class CustomOptimizer(nn.Module):
    """
    Custom simple ViT model with modified head for multi-label classification.
    
        
    """


    def __init__(self, config):

        super(CustomOptimizer, self).__init__()

        lr = config['adam_optimizer_hyper_p']['lr']


        params = CustomViT(
            config=config
        ).vit_model.parameters()

        self.adm_optimizer = Adam(
            params=params
            ,lr=lr
            )
        
    def get_optimizer(self):
        """Return the vit_model."""
        return self.adm_optimizer
