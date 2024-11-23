import torch
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import yaml

# Load the YAML configuration
with open("configs/config.yaml", "r") as file:
    config = yaml.safe_load(file)




def train_model(
    model,
    train_dataloader,
    criterion,
    optimizer,
    num_epochs = 1,
    device = 'cpu',
    save_path = 'model_checkpoint/model.pth'
):
    """
    Train the model and track losses.
    
    Args:
        model: PyTorch model to train
        train_dataloader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer for updating weights
        num_epochs: Number of training epochs
        device: Device to train on ('cuda' or 'cpu')
        save_path: Path to save the trained model
    
    Returns:
        pd.Series: Series containing tracked losses
    """
    # Move model to device
    
    model = model.to(device)
    model.train()
    
    # Dictionary to track losses
    tracking_loss = {}
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        print(f"\nStarting epoch {epoch}")
        
        # Iterate through batches with progress bar
        for batch_n, batch in tqdm(
            enumerate(train_dataloader), 
            total=len(train_dataloader),
            desc=f'Epoch {epoch}/{num_epochs}'
        ):
            # Move batch to device
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            # 1) Zero gradients
            optimizer.zero_grad()
            
            # 2) Forward pass
            outputs = model(images)
            
            # 3) Compute loss
            loss = criterion(outputs, labels)
            tracking_loss[(epoch, batch_n)] = float(loss)
            
            # 4) Backward pass
            loss.backward()
            
            # 5) Update weights
            optimizer.step()
            
        # Print epoch loss
        epoch_loss = sum(v for k, v in tracking_loss.items() if k[0] == epoch) / len(train_dataloader)
        print(f"Epoch {epoch} average loss: {epoch_loss:.4f}")
    
    # Convert tracking_loss to pandas Series
    tracking_loss = pd.Series(tracking_loss)
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    tracking_loss.plot(alpha=0.2, label="loss")
    tracking_loss.rolling(
        center=True, 
        min_periods=1, 
        window=10
    ).mean().plot(label="loss (moving avg)")
    
    plt.xlabel("(Epoch, Batch)")
    plt.ylabel("Loss")
    plt.legend(loc=0)
    plt.title("Training Loss Over Time")
    
    # Save model
    torch.save(model, save_path)
    print(f"\nModel saved to {save_path}")
    
    return tracking_loss

def plot_loss(tracking_loss: pd.Series, save_path: str = None):
    """
    Plot training loss.
    
    Args:
        tracking_loss: Series containing loss values
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 5))
    tracking_loss.plot(alpha=0.2, label="loss")
    tracking_loss.rolling(
        center=True, 
        min_periods=1, 
        window=10
    ).mean().plot(label="loss (moving avg)")
    
    plt.xlabel("(Epoch, Batch)")
    plt.ylabel("Loss")
    plt.legend(loc=0)
    plt.title("Training Loss Over Time")
    
    if save_path:
        plt.savefig(save_path)
        print(f"Loss plot saved to {save_path}")

