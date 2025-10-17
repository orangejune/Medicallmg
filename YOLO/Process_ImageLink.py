import os
import shutil

# ===== CONFIGURATION ===== #

TRAIN_LABEL_DIR = "data/ROI-label/train_total"
PREDICT_LABEL_DIR = "data/ROI-label/test_total"
IMAGE_DIR = "data/ROI-label/img_total"
OUTPUT_DIR = "0717"

TRAIN_IMAGES_OUT = os.path.join(OUTPUT_DIR, "images", "train")
TRAIN_LABELS_OUT = os.path.join(OUTPUT_DIR, "labels", "train")
PREDICT_IMAGES_OUT = os.path.join(OUTPUT_DIR, "images", "predict")
PREDICT_LABELS_OUT = os.path.join(OUTPUT_DIR, "labels", "predict")

# Create YOLOv8 folder structure
os.makedirs(TRAIN_IMAGES_OUT, exist_ok=True)
os.makedirs(TRAIN_LABELS_OUT, exist_ok=True)
os.makedirs(PREDICT_IMAGES_OUT, exist_ok=True)
os.makedirs(PREDICT_LABELS_OUT, exist_ok=True)

def link_labels_and_images(label_dir, image_dir, images_out, labels_out, description):
    label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
    missing_images = []
    print(f" Linking {description} images and labels...")
    for label_file in label_files:
        base_name = os.path.splitext(label_file)[0]
        image_file_jpg = base_name + ".jpg"
        image_file_png = base_name + ".png"

        image_path = None
        if os.path.exists(os.path.join(image_dir, image_file_jpg)):
            image_path = os.path.join(image_dir, image_file_jpg)
        elif os.path.exists(os.path.join(image_dir, image_file_png)):
            image_path = os.path.join(image_dir, image_file_png)
        else:
            missing_images.append(base_name)
            print(f" Missing image for label: {label_file}")
            continue

        shutil.copy2(image_path, os.path.join(images_out, os.path.basename(image_path)))
        shutil.copy2(os.path.join(label_dir, label_file), os.path.join(labels_out, label_file))

    print(f"Linked {len(label_files) - len(missing_images)} {description} image-label pairs")
    if missing_images:
        print(f" {len(missing_images)} {description} label files had no matching image")

# Link training data
link_labels_and_images(TRAIN_LABEL_DIR, IMAGE_DIR, TRAIN_IMAGES_OUT, TRAIN_LABELS_OUT, "training")

# Link prediction data
link_labels_and_images(PREDICT_LABEL_DIR, IMAGE_DIR, PREDICT_IMAGES_OUT, PREDICT_LABELS_OUT, "prediction")

# Create dataset.yaml
with open(os.path.join(OUTPUT_DIR, "dataset.yaml"), "w", encoding="utf-8") as f:
    f.write(f"""
path: {OUTPUT_DIR}
train: images/train
val: images/predict
nc: 2
names: ['RCA', 'LCA']
""")
print(" dataset.yaml created for YOLOv8")
