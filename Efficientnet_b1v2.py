import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import os
import copy
import time
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
import argparse
import numpy as np


def train_model(model, criterion, optimizer, scheduler, dataloader, dataset_size, device, num_epochs=50, patience=5):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    early_stopping_counter = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        model.train()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloader['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_size['train']
        epoch_acc = running_corrects.double() / dataset_size['train']
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.item())

        print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        model.eval()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloader['val']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_size['val']
        epoch_acc = running_corrects.double() / dataset_size['val']
        val_losses.append(epoch_loss)
        val_accs.append(epoch_acc.item())

        print(f'Validation Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print(
                f'Early stopping after {patience} epochs with no improvement.')
            break

    model.load_state_dict(best_model_wts)
    time_elapsed = time.time() - since
    print(
        f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    return model, best_acc, (train_losses, val_losses, train_accs, val_accs)


def k_fold_cross_validation(k, model, dataset, batch_size, device, num_epochs, patience, class_names):
    kf = KFold(n_splits=k, shuffle=True)
    accuracies = []
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f'Fold {fold + 1}/{k}')

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False, num_workers=4)

        dataloaders = {'train': train_loader, 'val': val_loader}
        dataset_sizes = {'train': len(train_subset), 'val': len(val_subset)}

        model_copy = copy.deepcopy(model)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model_copy.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

        model_copy, best_acc, fold_result = train_model(
            model_copy, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs, patience)
        accuracies.append(best_acc)
        fold_results.append(fold_result)

        # Save the best model weights for each fold
        torch.save(model_copy.state_dict(), f'best_model_fold_{fold + 1}.pth')

    mean_acc = np.mean([acc.cpu().numpy() for acc in accuracies])
    std_acc = np.std([acc.cpu().numpy() for acc in accuracies])

    print(f'Mean Accuracy: {mean_acc:.4f}')
    print(f'Standard Deviation: {std_acc:.4f}')

    # Plotting
    for fold, (train_losses, val_losses, train_accs, val_accs) in enumerate(fold_results):
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'Loss Curve - Fold {fold + 1}')
        plt.savefig(f'loss_curve_fold_{fold + 1}.png')

        plt.figure(figsize=(10, 5))
        plt.plot(train_accs, label='Training Accuracy')
        plt.plot(val_accs, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title(f'Accuracy Curve - Fold {fold + 1}')
        plt.savefig(f'accuracy_curve_fold_{fold + 1}.png')

    return accuracies, mean_acc, std_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training and testing script with 5-fold cross-validation')
    parser.add_argument('--data_dir', type=str,
                        default='Potato Leaf Disease Dataset', help='path to the dataset')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size for training')
    parser.add_argument('--learning_rate', type=float,
                        default=0.001, help='learning rate for optimizer')
    parser.add_argument('--patience', type=int, default=50,
                        help='patience for early stopping')
    args = parser.parse_args()

    data_dir = args.data_dir
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    patience = args.patience

    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=os.path.join(
        data_dir, 'train'), transform=data_transforms)
    class_names = dataset.classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    k = 10
    accuracies, mean_acc, std_acc = k_fold_cross_validation(
        k, model, dataset, batch_size, device, num_epochs, patience, class_names)
