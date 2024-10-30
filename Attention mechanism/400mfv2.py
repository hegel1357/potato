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
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import numpy as np
from torchvision.models.regnet import RegNet_Y_400MF_Weights
from torchvision import transforms


# 訓練模型的函式


def train_model(model, criterion, optimizer, scheduler, dataloader, dataset_size, device, num_epochs=50):
    since = time.time()  # 記錄訓練開始時間
    best_model_wts = copy.deepcopy(model.state_dict())  # 儲存最佳模型權重
    best_acc = 0.0  # 初始化最佳準確度

    train_losses = []  # 訓練損失
    val_losses = []  # 驗證損失
    train_accs = []  # 訓練準確度
    val_accs = []  # 驗證準確度

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        model.train()  # 設定模型為訓練模式

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

        model.eval()  # 設定模型為驗證模式
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

        # 使用 ReduceLROnPlateau 根據驗證損失調整學習率
        scheduler.step(epoch_loss)

        # 獲取當前學習率
        current_lr = scheduler.get_last_lr()[0]
        print(f'Current learning rate: {current_lr:.6f}')

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    time_elapsed = time.time() - since
    print(
        f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    return model, best_acc, (train_losses, val_losses, train_accs, val_accs)


def k_fold_cross_validation(k, model, dataset, batch_size, device, num_epochs, class_names):
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
        optimizer = optim.AdamW(model_copy.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=15)  # 更新這裡

        model_copy, best_acc, fold_result = train_model(
            model_copy, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs)
        accuracies.append(best_acc)
        fold_results.append(fold_result)

        # 保存每個摺疊的最佳模型權重
        torch.save(model_copy.state_dict(), f'best_model_fold_{fold + 1}.pth')

    mean_acc = np.mean([acc.cpu().numpy() for acc in accuracies])
    std_acc = np.std([acc.cpu().numpy() for acc in accuracies])

    print(f'Mean Accuracy: {mean_acc:.4f}')
    print(f'Standard Deviation: {std_acc:.4f}')

    # 繪製圖表
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
        description='Training and testing script with 10-fold cross-validation')
    parser.add_argument('--data_dir', type=str,
                        default='Potato Leaf Disease Dataset', help='path to the dataset')
    parser.add_argument('--num_epochs', type=int, default=150,
                        help='number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size for training')
    parser.add_argument('--learning_rate', type=float,
                        default=0.001, help='learning rate for optimizer')
    args = parser.parse_args()

    data_dir = args.data_dir
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    # 資料增強的轉換操作
    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),                # 隨機裁切範圍，保持較大比例
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),     # 顏色變化
        transforms.RandomHorizontalFlip(p=0.5),                              # 隨機水平翻轉
        transforms.RandomVerticalFlip(p=0.2),                                # 隨機垂直翻轉
        transforms.RandomRotation(degrees=15),                               # 輕微旋轉
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=5),   # 平移與剪切
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])   # 標準化
    ])


    # 驗證集的轉換操作
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加載資料集
    dataset = datasets.ImageFolder(root=os.path.join(
        data_dir, 'train'), transform=data_transforms)
    class_names = dataset.classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 建立模型並加載預訓練權重
    model = models.regnet_y_400mf(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))  # 調整最後的全連接層以符合分類數
    model = model.to(device)

    k = 10  # 設置為5折交叉驗證
    accuracies, mean_acc, std_acc = k_fold_cross_validation(
        k, model, dataset, batch_size, device, num_epochs, class_names)
