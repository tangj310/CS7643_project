import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.models import VisionTransformer  # define own transformers
import yaml



class CustomViT(nn.Module):
    """
    Custom simple ViT model with modified head for multi-label classification.
    
        
    """


    def __init__(self, config):
        """
        Initialize the model with 8 classes target .
        
        """
        super(CustomViT, self).__init__()
        
        # Extract parameters from the config
        # ViT hyper p
        image_size = config['vit_hyper_p']['image_size']
        patch_size = config['vit_hyper_p']['patch_size']
        num_layers = config['vit_hyper_p']['num_layers']
        num_heads = config['vit_hyper_p']['num_heads']
        hidden_dim = config['vit_hyper_p']['hidden_dim']
        mlp_dim = config['vit_hyper_p']['mlp_dim']
        dropout = config['vit_hyper_p']['dropout']
        num_classes = config['vit_hyper_p']['num_classes']


        self.model = VisionTransformer(
                                        image_size=image_size  # Input image size (224x224)
                                        ,patch_size=patch_size   # Patch size
                                        ,num_layers=num_layers
                                        ,num_classes=num_classes  # Number of classes for classification
                                        ,num_heads=num_heads     # Number of attention heads
                                        ,hidden_dim=hidden_dim
                                        ,mlp_dim=mlp_dim     # Dimension of MLP layers
                                        ,dropout=dropout     # Dropout rate
                                            )
        

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        # Access the last classification layer and modify it
        in_features = self.model.heads[0].in_features  # Access the first layer in 'heads'
        self.model.heads = nn.Linear(in_features, num_classes)

    def get_model(self):
        """Return the vit_model."""
        return self.model