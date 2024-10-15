import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import os
import copy
import time
import matplotlib.pyplot as plt
import argparse
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=50, patience=5):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    early_stopping_counter = 0

    train_losses = []
    val_accs = []
    train_accs = []
    val_losses = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                scheduler.step()
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.item())
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1

                if early_stopping_counter >= patience:
                    print(f'Early stopping triggered after {epoch} epochs without improvement.')
                    break

        if early_stopping_counter >= patience:
            break

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.savefig('loss_curve.png')

    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')
    plt.savefig('accuracy_curve.png')

    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return best_acc

def k_fold_cross_validation(dataset, num_folds=10, num_epochs=50, batch_size=16, learning_rate=0.0001, patience=5):
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.arange(len(dataset)))):
        print(f'Fold {fold + 1}/{num_folds}')
        print('-' * 10)

        train_subsampler = Subset(dataset, train_idx)
        val_subsampler = Subset(dataset, val_idx)

        dataloaders = {
            'train': DataLoader(train_subsampler, batch_size=batch_size, shuffle=True, num_workers=4),
            'val': DataLoader(val_subsampler, batch_size=batch_size, shuffle=False, num_workers=4)
        }

        dataset_sizes = {'train': len(train_subsampler), 'val': len(val_subsampler)}
        class_names = dataset.classes

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = models.mnasnet1_3(pretrained=True)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

        best_acc = train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=num_epochs, patience=patience)

        torch.save(model.state_dict(), f'best_mnasnet1_3_model_fold_{fold + 1}.pth')

        fold_results.append(best_acc)

    mean_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)

    print(f'Average best accuracy: {mean_acc:.4f}')
    print(f'Standard deviation of best accuracy: {std_acc:.4f}')

    return fold_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training and validation script for a model on Potato Leaf Disease Dataset')
    parser.add_argument('--data_dir', type=str, default='Potato Leaf Disease Dataset', help='path to the dataset')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate for optimizer')
    parser.add_argument('--patience', type=int, default=50, help='patience for early stopping')
    args = parser.parse_args()

    data_dir = args.data_dir
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    patience = args.patience

    data_transforms = {
        'train': transforms.Compose([            
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    full_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=data_transforms['train'])

    fold_results = k_fold_cross_validation(full_dataset, num_folds=10, num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate, patience=patience)
