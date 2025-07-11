import os
from ultralytics import YOLO
from PIL import Image, ImageDraw

# ===== CONFIGURATION ===== #
OUTPUT_DIR = r"dataset"
DATASET_YAML = os.path.join(OUTPUT_DIR, "dataset.yaml")
PREDICTION_IMAGES_DIR = os.path.join(OUTPUT_DIR, "images", "predict")
PREDICTION_LABELS_DIR = os.path.join(OUTPUT_DIR, "labels", "predict")
PREDICTION_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "prediction_results")
CROPS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "cropped_rois")

# Create output directories
os.makedirs(PREDICTION_OUTPUT_DIR, exist_ok=True)
os.makedirs(CROPS_OUTPUT_DIR, exist_ok=True)

# ===== STEP 1: TRAIN YOLOv8 MODEL IF NOT EXISTS ===== #
MODEL_DIR = "runs/detect/train"
MODEL_BEST_PATH = os.path.join(MODEL_DIR, "weights", "besttest.pt")

if not os.path.exists(MODEL_BEST_PATH):
    print(" Training YOLOv8 model ...")
    model = YOLO("yolov8n.pt")  # Start with YOLOv8 nano for speed
    model.train(data=DATASET_YAML, epochs=50, imgsz=640)
    print(f" Model trained and saved to {MODEL_BEST_PATH}")
else:
    print(f" Found existing trained model: {MODEL_BEST_PATH}")
    model = YOLO(MODEL_BEST_PATH)

# ===== STEP 2: PREDICT ON NEW IMAGES ===== #
print(" Running predictions on your prediction dataset...")
pred_image_files = [f for f in os.listdir(PREDICTION_IMAGES_DIR) if f.endswith(('.jpg', '.png'))]

for img_file in pred_image_files:
    img_path = os.path.join(PREDICTION_IMAGES_DIR, img_file)
    results = model.predict(source=img_path, save=False, imgsz=640)

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
                draw.text((x1, y1 - 10), f"GT {int(cls)}", fill=color)

                # Save cropped ground truth ROI
                roi = img.crop((x1, y1, x2, y2))
                crop_name = f"{os.path.splitext(img_file)[0]}_GT_{idx}_cls{int(cls)}.png"
                roi.save(os.path.join(CROPS_OUTPUT_DIR, crop_name))

    # Draw predicted boxes and save cropped predictions
    for idx, r in enumerate(results):
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0])
            color = "red" if cls == 0 else "yellow"
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1, y1 - 10), f"Pred {cls}", fill=color)

            # Save cropped predicted ROI
            roi = img.crop((int(x1), int(y1), int(x2), int(y2)))
            crop_name = f"{os.path.splitext(img_file)[0]}_PRED_{idx}_cls{cls}.png"
            roi.save(os.path.join(CROPS_OUTPUT_DIR, crop_name))

    img.save(os.path.join(PREDICTION_OUTPUT_DIR, img_file))

print(f" Prediction results saved to: {PREDICTION_OUTPUT_DIR}")
print(f" Cropped ROIs saved to: {CROPS_OUTPUT_DIR}")
print(" Training and prediction complete.")
