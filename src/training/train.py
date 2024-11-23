import torch
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import yaml




# Load the YAML configuration
with open("configs/config.yaml", "r") as file:
    config = yaml.safe_load(file)


frac = config['data_split']['frac']
test_size = config['data_split']['test_size']



def data_load():
# change directory

    local_dir = 'E:/University-Georgia Tech/CS7643_deeplearning/groupproject/competition_VfIpjyh/'

    train_features = pd.read_csv(local_dir + "train_features.csv", index_col="id")
    test_features = pd.read_csv(local_dir + "test_features.csv", index_col="id")
    train_labels = pd.read_csv(local_dir + "train_labels.csv", index_col="id")

    train_features['filepath'] = train_features['filepath'].str.replace('/data', '', regex=False)
    test_features['filepath'] = test_features['filepath'].str.replace('/data', '', regex=False)

    train_features['filepath'] = local_dir + train_features['filepath']
    test_features['filepath'] = local_dir + test_features['filepath']

    return train_features, train_labels, test_features



def train_test_split():

    y = data_load()[1].sample(frac=frac, random_state=1)
    x = data_load()[0].loc[y.index].filepath.to_frame()

    # note that we are casting the species labels to an indicator/dummy matrix
    x_train, x_eval, y_train, y_eval = train_test_split(
        x, y, stratify=y, test_size=0.25
    )



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

