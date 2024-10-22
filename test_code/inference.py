import torch
import torch.nn as nn
from torchvision import transforms, models
import os
import cv2
from PIL import Image
import time  # 引入 time 模組


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
    start_time = time.time()  # 記錄開始時間
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        max_prob, preds = torch.max(probs, 1)
    end_time = time.time()  # 記錄結束時間

    inference_time = end_time - start_time  # 計算推論時間

    if max_prob.item() < threshold:
        return None, -1, max_prob.item(), inference_time

    return class_names[preds.item()], preds.item(), max_prob.item(), inference_time


def draw_prediction_on_image(image, true_label, prediction, confidence):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2  # 增加字體大小
    text = f"True: {true_label}, prediction: {prediction} "
    cv2.putText(image, text, (10, 60), font,
                font_scale, (0, 255, 0), 2, cv2.LINE_AA)
    return image


if __name__ == "__main__":
    # 設置模型路徑和類別名稱
    model_path = 'best_model_fold_10.pth'
    class_names = ['Bacteria', 'Fungi', 'Healthy',
                   'Nematode', 'Pest', 'Phytopthora', 'Virus']
    threshold = 0.5  # 設置信心閾值

    # 載入模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, len(class_names))
    model = model.to(device)

    # 測試集資料夾路徑
    test_dir = 'Potato Leaf Disease Dataset/test'
    result_dir = 'result'
    os.makedirs(result_dir, exist_ok=True)

    # 初始化計數器
    tp_count = 0  # True Positive
    fp_count = 0  # False Positive
    fn_count = 0  # False Negative
    total_inference_time = 0  # 累積推論時間
    image_count = 0  # 計算影像數量

    # 遍歷測試集
    for disease_dir in os.listdir(test_dir):
        disease_path = os.path.join(test_dir, disease_dir)

        for img_name in os.listdir(disease_path):
            img_path = os.path.join(disease_path, img_name)

            try:
                image = cv2.imread(img_path)
                image_preprocessed = preprocess_image(image)
                prediction, pred_idx, max_prob, inference_time = predict(
                    model, image_preprocessed, device, class_names, threshold)

                # 累積推論時間
                total_inference_time += inference_time
                image_count += 1

                # 獲取真實類別索引
                true_idx = class_names.index(disease_dir)

                # 判斷預測結果類別是否和真實類別相符
                if pred_idx == true_idx:
                    if max_prob >= threshold:
                        tp_count += 1  # True Positive
                    else:
                        fn_count += 1  # False Negative
                else:
                    if max_prob >= threshold:
                        fp_count += 1  # False Positive

                # 打印每張圖片的預測結果並保存圖片（如果置信度大於等於閾值）
                if max_prob >= threshold:
                    # 創建真實類別的資料夾
                    result_class_dir = os.path.join(result_dir, disease_dir)
                    os.makedirs(result_class_dir, exist_ok=True)

                    # 在圖片上畫出預測結果和真實類別
                    image_with_prediction = draw_prediction_on_image(
                        image, disease_dir, prediction, max_prob)

                    # 保存圖片到真實類別的資料夾
                    result_img_path = os.path.join(result_class_dir, img_name)
                    cv2.imwrite(result_img_path, image_with_prediction)

                    print(
                        f"Image: {img_name}, True Label: {disease_dir}, Prediction: {prediction}, Inference Time: {inference_time:.4f} seconds")

            except Exception as e:
                print(f"Error during prediction for {img_name}: {e}")
                continue

    # 計算 Precision, Recall, 和 F1 Score
    precision = tp_count / \
        (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    recall = tp_count / \
        (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision +
                                           recall) if (precision + recall) > 0 else 0

    # 計算平均推論時間
    avg_inference_time = total_inference_time / \
        image_count if image_count > 0 else 0

    # 打印結果
    print(f"True Positive (TP): {tp_count}")
    print(f"False Positive (FP): {fp_count}")
    print(f"False Negative (FN): {fn_count}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"Average Inference Time: {avg_inference_time:.4f} seconds")
