import torch
import torch.nn as nn
from torch.optim import Adam
from vit import CustomViT


class CustomOptimizer(nn.Module):
    """
    Custom simple ViT model with modified head for multi-label classification.
    
        
    """


    def __init__(self, config):

        super(CustomOptimizer, self).__init__()

        lr = config['adam_optimizer_hyper_p']['lr']

        params = CustomViT.vit_model.parameters()

        self.optimizer = Adam(
            params=params
            ,lr=lr
            )
