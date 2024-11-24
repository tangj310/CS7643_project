
import torch
from torch.utils.data import DataLoader
from src.training.train import *
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
import yaml


# Load the YAML configuration
with open("configs/config.yaml", "r") as file:
    config = yaml.safe_load(file)

plt_pic_save_path = config['train']['plt_pic_save_path']



def main():


    train_features = data_load()[0]
    train_labels = data_load()[1]
    test_features = data_load()[2]


    x_train = data_split(
        train_features
        ,train_labels
    )[0]
    x_eval = data_split(
        train_features
        ,train_labels
    )[1]
    y_train = data_split(
        train_features
        ,train_labels
    )[2]
    y_eval = data_split(
        train_features
        ,train_labels
    )[3]


    train_dataloader = data_preprocess(
        x_train
        ,y_train
    )

    tracking_loss = train_model(
        train_dataloader
    )




    # Plot and save loss
    plot_loss(
        tracking_loss
        ,save_path=plt_pic_save_path)
        
    

if __name__ == "__main__":
    main()