import torch
import os
import time
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn  
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torchvision.models.regnet import RegNet_Y_400MF_Weights

def test_model(model, dataloader, dataset_size, device, class_names):
    model.eval()
    running_corrects = 0
    all_preds = []
    all_labels = []
    inference_times = []  # 用來存儲每張影像的推論時間

    img_counter = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            start_time = time.time()  # 開始計時
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            end_time = time.time()  # 結束計時

        running_corrects += torch.sum(preds == labels.data)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # 計算每張影像的推論時間
        for i in range(inputs.size(0)):
            inference_time = end_time - start_time
            inference_times.append(inference_time)  # 記錄推論時間
            print(f'Image: {dataloader.dataset.samples[img_counter][0]}')
            print(f'Predicted: {class_names[preds[i]]}, Actual: {class_names[labels.data[i]]}, Inference Time: {inference_time:.4f} seconds')
            img_counter += 1

    test_acc = running_corrects.double() / dataset_size
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    print(f'Test Acc: {test_acc:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print('Confusion Matrix:')
    print(cm)

    # 計算平均推論速度
    avg_inference_time = sum(inference_times) / len(inference_times)
    print(f'Average Inference Time per Image: {avg_inference_time:.4f} seconds')

    for i, class_name in enumerate(class_names):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        print(f'Class {class_name} - TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}')

if __name__ == '__main__':
    data_dir = 'Potato Leaf Disease Dataset'  # 修改為您的資料集路徑
    batch_size = 1  # 批量大小

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=data_transforms)
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    dataset_size = len(test_dataset)
    class_names = test_dataset.classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = models.regnet_y_400mf(weights=RegNet_Y_400MF_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    # 載入最佳權重
    model.load_state_dict(torch.load('best_model_fold_10.pth'))

    test_model(model, dataloader, dataset_size, device, class_names)
import torch
import os
import time
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn  
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torchvision.models.regnet import RegNet_Y_400MF_Weights

def test_model(model, dataloader, dataset_size, device, class_names):
    model.eval()
    running_corrects = 0
    all_preds = []
    all_labels = []
    inference_times = []  # 用來存儲每張影像的推論時間

    img_counter = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            start_time = time.time()  # 開始計時
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            end_time = time.time()  # 結束計時

        running_corrects += torch.sum(preds == labels.data)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # 計算每張影像的推論時間
        for i in range(inputs.size(0)):
            inference_time = end_time - start_time
            inference_times.append(inference_time)  # 記錄推論時間
            print(f'Image: {dataloader.dataset.samples[img_counter][0]}')
            print(f'Predicted: {class_names[preds[i]]}, Actual: {class_names[labels.data[i]]}, Inference Time: {inference_time:.4f} seconds')
            img_counter += 1

    test_acc = running_corrects.double() / dataset_size
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    print(f'Test Acc: {test_acc:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print('Confusion Matrix:')
    print(cm)

    # 計算平均推論速度
    avg_inference_time = sum(inference_times) / len(inference_times)
    print(f'Average Inference Time per Image: {avg_inference_time:.4f} seconds')

    for i, class_name in enumerate(class_names):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        print(f'Class {class_name} - TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}')

if __name__ == '__main__':
    data_dir = 'Potato Leaf Disease Dataset'  # 修改為您的資料集路徑
    batch_size = 1  # 批量大小

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=data_transforms)
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    dataset_size = len(test_dataset)
    class_names = test_dataset.classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = models.regnet_y_400mf(weights=RegNet_Y_400MF_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    # 載入最佳權重
    model.load_state_dict(torch.load('best_model_fold_10.pth'))

    test_model(model, dataloader, dataset_size, device, class_names)
