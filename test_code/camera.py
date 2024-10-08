import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import os
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. 初始化模型
def load_model(model_path, num_classes):
    model = models.regnet_y_400mf(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    state_dict = torch.load(model_path)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"RuntimeError: {e}")
        # 如果形狀不匹配，重建輸出層
        model.fc = nn.Linear(num_ftrs, num_classes)
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and v.size() == model.state_dict()[k].size()}
        model.load_state_dict(pretrained_dict, strict=False)
    model.eval()
    return model

# 2. 圖像預處理
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = preprocess(image).unsqueeze(0)
    return image

# 3. 推論
def predict(model, image, device, class_names):
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        max_prob, preds = torch.max(probs, 1)
    
    return class_names[preds.item()], preds.item(), max_prob.item()

# 4. 主函數
def main(model_path, class_names, threshold=0.9):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, len(class_names))
    model = model.to(device)

    cap = cv2.VideoCapture(0)  # 開啟攝影機
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image from video.")
            break

        try:
            image_preprocessed = preprocess_image(frame)
            prediction, _, max_prob = predict(model, image_preprocessed, device, class_names)
        except Exception as e:
            print(f"Error during prediction: {e}")
            prediction = "Unknown"  # 預測失敗時，顯示"Unknown"

        # 在影像上顯示結果
        display_text = prediction if max_prob > threshold else f""
        cv2.putText(frame, display_text, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('Potato Leaf Disease Classification', frame)

        # 檢測按鍵事件
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = 'best_regnet_y_400_model.pth'
    class_names = ['Bacteria', 'Fungi', 'Healthy', 'Nematode', 'Pest', 'Phytopthora', 'Virus']
    main(model_path, class_names)
