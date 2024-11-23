import torch
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import yaml
from datetime import date

from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from src.models.dataops import ImagesDataset
from src.models.vit import CustomViT
from src.models.resnet50 import CustomResNet50
from src.models.optimizer import CustomOptimizer
from src.models.loss import CustomLoss


# global params
str_today = str(date.today())


# Load the YAML configuration
with open("configs/config.yaml", "r") as file:
    config = yaml.safe_load(file)




def data_load():

    # load yaml parameters
    local_dir = config['data_load']['local_dir']


    train_features = pd.read_csv(local_dir + "train_features.csv", index_col="id")
    test_features = pd.read_csv(local_dir + "test_features.csv", index_col="id")
    train_labels = pd.read_csv(local_dir + "train_labels.csv", index_col="id")

    train_features['filepath'] = train_features['filepath'].str.replace('/data', '', regex=False)
    test_features['filepath'] = test_features['filepath'].str.replace('/data', '', regex=False)

    train_features['filepath'] = local_dir + train_features['filepath']
    test_features['filepath'] = local_dir + test_features['filepath']

    print('data load: succesful')
    return train_features, train_labels, test_features



def data_split(
        train_features
        ,train_labels
        ):

    # load yaml parameters
    frac = config['data_split']['frac']
    test_size = config['data_split']['test_size']

    y = train_labels.sample(frac=frac, random_state=1)
    x = train_features.loc[y.index].filepath.to_frame()

    # note that we are casting the species labels to an indicator/dummy matrix
    x_train, x_eval, y_train, y_eval = train_test_split(
        x
        ,y
        ,stratify=y
        ,test_size=test_size
    )


    print (x_train.shape, x_eval.shape, y_train.shape, y_eval.shape)
    print('data split: successful')
    return x_train, x_eval, y_train, y_eval



def data_preprocess(
        x_train
        ,y_train
        ):


    batch_size = config['data_preprocess']['batch_size']

    train_dataset = ImagesDataset(
        x_train
        ,y_train
        )
    
    train_dataloader = DataLoader(
        train_dataset
        ,batch_size=batch_size # can be adjusted in yaml
        )
    
    print('data preprocess: successful')
    return train_dataloader



def train_model(
        train_dataloader
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
    # Load the YAML configuration inside of train_model() not sure why need it but it will bug if not
    with open("configs/config.yaml", "r") as file:
        config = yaml.safe_load(file)



    num_epochs = config['train']['num_epochs']
    device = config['train']['device']
    model_save_path = config['train']['model_save_path']


    model = CustomViT(
        config=config
        ).get_vit_model()
    print('model get: successful')

    optimizer = CustomOptimizer(
        config=config
    ).get_optimizer()
    print('optmizer get: successful')

    criterion = CustomLoss().get_loss()
    print('loss function get: successful')

    str_today = str(date.today())



    # Training loop
    tracking_loss = {}

    for epoch in range(1, num_epochs + 1):
        print(f"Starting epoch {epoch}")

        # iterate through the dataloader batches. tqdm keeps track of progress.
        for batch_n, batch in tqdm(
            enumerate(train_dataloader), total=len(train_dataloader)
        ):
            print(batch_n) # total 2473 pics
            
            model.train()
            loss = 0.0
            for batch in train_dataloader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                # Forward pass
                outputs = model(images)

                loss = criterion(outputs, labels)
                
                # let's keep track of the loss by epoch and batch|
                tracking_loss[(epoch, batch_n)] = float(loss)
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss += loss.item()

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss / len(train_dataloader):.4f}")
    
    # Convert tracking_loss to pandas Series
    tracking_loss = pd.Series(tracking_loss)
    

    # Save model
    torch.save(model, f'{model_save_path}_{str_today}.pth')
    print(f"\nModel saved to f{model_save_path}_{str_today}")
    
    print('model training: successful')
    return tracking_loss



def plot_loss(
        tracking_loss: pd.Series
        ,save_path: str = None):
    """
    Plot training loss.
    
    Args:
        tracking_loss: Series containing loss values
        save_path: Optional path to save the plot
    """

    plt_pic_save_path = config['train']['plt_pic_save_path']

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
        plt.savefig(f'{plt_pic_save_path}_{str_today}.png')
        print(f"Loss plot saved to {plt_pic_save_path}_{str_today}")

    return

