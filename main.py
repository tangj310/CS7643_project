
import torch
from torch.utils.data import DataLoader
from src.data.dataset import ImagesDataset
from src.data.prepare_data import prepare_data
from src.models.resnet50 import CustomResNet50
from src.training.train import train_model, plot_loss
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss

def main():
    # Prepare datasets
    x_train, x_eval, y_train, y_eval, test_features = prepare_data(
        "data/train_features.csv",
        "data/train_labels.csv",
        "data/test_features.csv"
    )

    
    # Create datasets
    train_dataset = ImagesDataset(x_train, y_train)
    # eval_dataset = ImagesDataset(x_eval, y_eval)
    # test_dataset = ImagesDataset(test_features, None)



    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    

    # Initialize model, criterion, optimizer
    model = CustomResNet50()
    criterion = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)
        
     # Train model
    tracking_loss = train_model(
        model=model,
        train_dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=1,
        device='cpu',
        save_path='model_checkpoint/model.pth'
    )

    # Plot and save loss
    plot_loss(tracking_loss, save_path='training_loss.png')
        
    

if __name__ == "__main__":
    main()