import torch
import torch.nn as nn
from torchvision import transforms, models
import os
import cv2
from PIL import Image
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_model(model_path, num_classes):
    model = models.regnet_y_400mf(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = preprocess(image).unsqueeze(0)
    return image

def predict(model, image, device, class_names, threshold=0.5):
    image = image.to(device)
    start_time = time.time()
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        max_prob, preds = torch.max(probs, 1)
    end_time = time.time()

    inference_time = end_time - start_time

    if max_prob.item() < threshold:
        return None, -1, max_prob.item(), inference_time

    return class_names[preds.item()], preds.item(), max_prob.item(), inference_time

def draw_prediction_on_image(image, true_label, prediction, confidence):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    text = f"True: {true_label}, Prediction: {prediction}, Confidence: {confidence:.2f}"
    cv2.putText(image, text, (10, 30), font, font_scale, (0, 255, 0), 2, cv2.LINE_AA)
    return image

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()

def calculate_class_metrics(TP, FP, FN, TN):
    accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy, precision, recall, f1_score

def calculate_weighted_average(metrics_per_class, class_weights):
    # Calculate weighted averages for each metric
    weighted_avg = np.zeros(len(metrics_per_class[0]))
    total_weight = sum(class_weights)

    for i in range(len(metrics_per_class[0])):
        weighted_avg[i] = sum([metrics_per_class[j][i] * class_weights[j] for j in range(len(class_weights))]) / total_weight

    return weighted_avg

if __name__ == "__main__":
    model_path = 'best_model_fold_1.pth'
    class_names = ['Bacteria', 'Fungi', 'Healthy', 'Nematode', 'Pest', 'Phytopthora', 'Virus']
    threshold = 0.5

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, len(class_names))
    model = model.to(device)

    test_dir = 'Potato Leaf Disease Dataset/test'
    result_dir = 'result'
    os.makedirs(result_dir, exist_ok=True)

    total_inference_time = 0
    image_count = 0

    cm = np.zeros((len(class_names), len(class_names)), dtype=int)

    # 每個類別的權重（可以根據樣本數進行調整）
    class_weights = [1] * len(class_names)  # 預設為等權重，你可以根據每類樣本數來調整權重

    metrics_per_class = []

    for disease_dir in os.listdir(test_dir):
        disease_path = os.path.join(test_dir, disease_dir)

        for img_name in os.listdir(disease_path):
            img_path = os.path.join(disease_path, img_name)

            try:
                image = cv2.imread(img_path)
                image_preprocessed = preprocess_image(image)
                prediction, pred_idx, max_prob, inference_time = predict(
                    model, image_preprocessed, device, class_names, threshold)

                total_inference_time += inference_time
                image_count += 1

                true_idx = class_names.index(disease_dir)

                if pred_idx == true_idx:
                    cm[true_idx, pred_idx] += 1
                else:
                    cm[true_idx, pred_idx] += 1

                if max_prob >= threshold:
                    result_class_dir = os.path.join(result_dir, disease_dir)
                    os.makedirs(result_class_dir, exist_ok=True)
                    image_with_prediction = draw_prediction_on_image(
                        image, disease_dir, prediction, max_prob)
                    result_img_path = os.path.join(result_class_dir, img_name)
                    cv2.imwrite(result_img_path, image_with_prediction)
                    print(f"Image: {img_name}, True Label: {disease_dir}, Prediction: {prediction}, Inference Time: {inference_time:.4f} seconds")

            except Exception as e:
                print(f"Error during prediction for {img_name}: {e}")
                continue

    # 計算每個類別的 TP、FP、FN 和 TN
    tp_per_class = [0] * len(class_names)
    fp_per_class = [0] * len(class_names)
    tn_per_class = [0] * len(class_names)
    fn_per_class = [0] * len(class_names)

    for idx, class_name in enumerate(class_names):
        TP = cm[idx, idx]
        FP = cm[:, idx].sum() - TP
        FN = cm[idx, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        
        tp_per_class[idx] = TP
        fp_per_class[idx] = FP
        fn_per_class[idx] = FN
        tn_per_class[idx] = TN

        accuracy, precision, recall, f1_score = calculate_class_metrics(TP, FP, FN, TN)

        metrics_per_class.append([accuracy, precision, recall, f1_score])

        print(f"Class: {class_name}")
        print(f"  True Positives (TP): {TP}")
        print(f"  False Positives (FP): {FP}")
        print(f"  True Negatives (TN): {TN}")
        print(f"  False Negatives (FN): {FN}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1_score:.4f}\n")

    # 計算加權平均
    weighted_avg = calculate_weighted_average(metrics_per_class, class_weights)

    print(f"Weighted Average Accuracy: {weighted_avg[0]:.4f}")
    print(f"Weighted Average Precision: {weighted_avg[1]:.4f}")
    print(f"Weighted Average Recall: {weighted_avg[2]:.4f}")
    print(f"Weighted Average F1 Score: {weighted_avg[3]:.4f}")

    avg_inference_time = total_inference_time / image_count if image_count > 0 else 0
    print(f"Average Inference Time: {avg_inference_time:.4f} seconds")

    plot_confusion_matrix(cm, class_names)
