
import os
from ultralytics import YOLO
import torch
import cv2
import os
import numpy as np
from PIL import Image, ImageDraw
import torch
# import matplotlib
# 强制不弹出窗口
# matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import albumentations as A
from tqdm import tqdm
import random

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------- yolo配置 ------------------
YOLO_MODEL_BEST_PATH = r'C:\Users\june.lin\Desktop\medicallmg\runs\detect\train11\weights\best.pt'
yolo_model = YOLO(YOLO_MODEL_BEST_PATH)

def yolo_roi_img(yolo_model,prediction_images_dir,crops_output_dir):
    '''
    YOLO模型预测，保存ROI图片
    输入：yolo模型、预测图片地址、roi保存地址、全图保存地址
    输出：无
    '''
    print(" Running predictions on your prediction dataset...")
    pred_image_files = [f for f in os.listdir(prediction_images_dir) if f.endswith(('.jpg', '.png'))]
    num_files_to_select = int(len(pred_image_files) / 5)
    random_selected_files = random.sample(pred_image_files, num_files_to_select)

    for img_file in random_selected_files:
        img_path = os.path.join(prediction_images_dir, img_file)
        results = yolo_model.predict(source=img_path, save=False, imgsz=640, device=DEVICE)

        # Draw predictions + ground truth
        img = Image.open(img_path).convert("RGB")

        final_conf = 0
        final_cls = None
        # Draw predicted boxes and save cropped predictions
        for idx, r in enumerate(results):
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = int(box.cls[0])
                conf = box.conf.item()
                if final_conf < conf:
                    final_conf = conf
                    final_cls = cls

                # Save cropped predicted ROI
                roi = img.crop((int(x1), int(y1), int(x2), int(y2)))
                crop_name = f"{os.path.splitext(img_file)[0][4:]}.jpg"
                roi.save(os.path.join(crops_output_dir, crop_name))

if __name__=="__main__":

    # Folder paths (modify as needed)
    script_full_path = os.path.abspath(__file__)
    data_path = os.path.join(os.path.dirname(__file__),'data')

    input_folder = r"C:\Users\june.lin\Desktop\medicallmg\data\冠脉病例GE-2\frames_flat" # 图片帧文件夹
    roi_folder = r"C:\Users\june.lin\Desktop\medicallmg\Medicallmg\U-Net\dataset\train-negative" # roi图片保存
    os.makedirs(roi_folder, exist_ok=True)

    # ==================== YOLO模型预测，获得ROI图片 =================== #
    yolo_roi_img(yolo_model, input_folder,roi_folder)