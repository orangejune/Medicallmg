import os
from ultralytics import YOLO
from PIL import Image, ImageDraw
import torch

def main(use_train = True):
    # ===== CONFIGURATION ===== #
    OUTPUT_DIR = "0717"
    DATASET_YAML = os.path.join(OUTPUT_DIR, "dataset.yaml")
    PREDICTION_IMAGES_DIR = os.path.join(OUTPUT_DIR, "images", "predict")
    PREDICTION_LABELS_DIR = os.path.join(OUTPUT_DIR, "labels", "predict")
    PREDICTION_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "predict_results")
    CROPS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "cropped_rois")

    # PREDICTION_IMAGES_DIR = r'C:\Users\june.lin\Desktop\medicallmg\data\test_video_frame'
    # PREDICTION_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "prediction_results_video")
    # CROPS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "cropped_rois_video")

    # Create output directories
    os.makedirs(PREDICTION_OUTPUT_DIR, exist_ok=True)
    os.makedirs(CROPS_OUTPUT_DIR, exist_ok=True)

    # ===== STEP 1: TRAIN YOLOv8 MODEL IF NOT EXISTS ===== #
    MODEL_BEST_PATH = r'C:\Users\june.lin\Desktop\medicallmg\runs\detect\train10\weights\best.pt'

    if torch.cuda.is_available():
        device = 0  # 使用第一块GPU
        print(f"GPU is available. Using device: cuda:{device}")
        print(f"GPU Name: {torch.cuda.get_device_name(device)}")
    else:
        device = 'cpu' # 如果没有GPU，则使用CPU
        print("GPU not found. Using device: cpu. Training will be slow.")

    if use_train == True:
        print(" Training YOLOv8 model ...")
        model = YOLO("yolov8n.pt")  # Start with YOLOv8 nano for speed
        model.train(data=DATASET_YAML, epochs=50, imgsz=640, device=device)
    else:
        print(f" Found existing trained model: {MODEL_BEST_PATH}")
        model = YOLO(MODEL_BEST_PATH)

    # ===== STEP 2: PREDICT ON NEW IMAGES ===== #
    print(" Running predictions on your prediction dataset...")
    pred_image_files = [f for f in os.listdir(PREDICTION_IMAGES_DIR) if f.endswith(('.jpg', '.png'))]

    for img_file in pred_image_files:
        img_path = os.path.join(PREDICTION_IMAGES_DIR, img_file)
        results = model.predict(source=img_path, save=False, imgsz=640, device=device)

        # Draw predictions + ground truth
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        # Draw ground truth boxes if label exists
        label_file = os.path.join(PREDICTION_LABELS_DIR, os.path.splitext(img_file)[0] + ".txt")
        if os.path.exists(label_file):
            with open(label_file, "r") as lf:
                for idx, line in enumerate(lf.readlines()):
                    parts = line.strip().split()
                    cls, x_center, y_center, width, height = map(float, parts)
                    w, h = img.size
                    x1 = int((x_center - width / 2) * w)
                    y1 = int((y_center - height / 2) * h)
                    x2 = int((x_center + width / 2) * w)
                    y2 = int((y_center + height / 2) * h)
                    color = "green" if int(cls) == 0 else "blue"
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                    # draw.text((x1, y1 - 10), f"GT {int(cls)}", fill=color)

                    # Save cropped ground truth ROI
                    roi = img.crop((x1, y1, x2, y2))
                    crop_name = f"{os.path.splitext(img_file)[0]}_GT_{idx}_cls{int(cls)}.png"
                    roi.save(os.path.join(CROPS_OUTPUT_DIR, crop_name))
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
                color = "red" if cls == 0 else "yellow"
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                draw.text((x1, y1 - 10), f"{conf:.2f}", fill=color)

                # Save cropped predicted ROI
                roi = img.crop((int(x1), int(y1), int(x2), int(y2)))
                crop_name = f"{os.path.splitext(img_file)[0]}_PRED_{idx}_cls{cls}.png"
                roi.save(os.path.join(CROPS_OUTPUT_DIR, crop_name))

        new_name = f'{img_file.split("_")[-3]}_conf{final_conf:.2f}_cls{final_cls}_{img_file.split("_")[-1]}'
        img.save(os.path.join(PREDICTION_OUTPUT_DIR, new_name))

    print(f" Prediction results saved to: {PREDICTION_OUTPUT_DIR}")
    print(f" Cropped ROIs saved to: {CROPS_OUTPUT_DIR}")
    print(" Training and prediction complete.")

if __name__ == '__main__':
    use_train = True
    main(use_train)