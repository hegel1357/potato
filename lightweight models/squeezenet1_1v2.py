import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Subset
import os
import copy
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torchvision.models import squeezenet1_1
from sklearn.model_selection import KFold

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
    return model, best_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script for a model on Potato Leaf Disease Dataset with 10-Fold Cross-Validation')
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

    dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=data_transforms['train'])
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_indices, val_indices) in enumerate(kfold.split(np.arange(len(dataset)))):
        print(f'\nFold {fold + 1}/{10}')

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        dataloaders = {
            'train': DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4),
            'val': DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)
        }

        dataset_sizes = {'train': len(train_subset), 'val': len(val_subset)}
        class_names = dataset.classes

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = squeezenet1_1(pretrained=True)
        model.num_classes = len(class_names)
        model.classifier[1] = nn.Conv2d(512, model.num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.classifier[2] = nn.AdaptiveAvgPool2d((1, 1))
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

        model, best_acc = train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=num_epochs, patience=patience)

        torch.save(model.state_dict(), f'best_squeezenet1_1_model_fold_{fold + 1}.pth')

        fold_results.append(best_acc.item())

    avg_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)

    print(f'\n10-Fold Cross-Validation Results:')
    print(f'Average Accuracy: {avg_acc:.4f}')
    print(f'Standard Deviation: {std_acc:.4f}')
