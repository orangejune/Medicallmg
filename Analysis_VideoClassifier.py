# filter_and_classify_frames.py

# Updated and cleaned version of StructureTypeClassifier to ensure training and evaluation are consistent

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten, Dropout, Input, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt

from Process_CreateFrame import *

ROI = (100, 50, 950, 700)
IMG_SIZE = (ROI[3] - ROI[1], ROI[2] - ROI[0])  # (height, width)

class ImagePredictor:
    def __init__(self, model_path, label_file, input_folder, output_folder, conf_threshold=0.7):
        self.model = load_model(model_path)
        with open(label_file, "r", encoding="utf-8") as f:
            self.labels = [line.strip() for line in f.readlines()]
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.conf_threshold = conf_threshold
        os.makedirs(self.output_folder, exist_ok=True)

    def preprocess_image(self, path):
        img = Image.open(path).convert("RGB")
        img = np.array(img)
        crop_x1, crop_y1, crop_x2, crop_y2 = ROI
        img = img[crop_y1:crop_y2, crop_x1:crop_x2]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edge = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        X_raw = np.expand_dims(preprocess_input(img.astype(np.float32)), axis=0)
        X_line = np.expand_dims(np.expand_dims(edge.astype(np.float32) / 255.0, axis=0), axis=-1)
        return [X_raw, X_line]

    def predict_and_save(self):
        log_path = os.path.join(self.output_folder, "predicted_log.txt")
        with open(log_path, "w", encoding="utf-8") as log:
            for fname in sorted(os.listdir(self.input_folder)):
                if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                    continue
                path = os.path.join(self.input_folder, fname)
                try:
                    inputs = self.preprocess_image(path)
                    pred = self.model.predict(inputs, verbose=0)[0]
                    pred_idx = np.argmax(pred)
                    confidence = pred[pred_idx]
                    label = self.labels[pred_idx]

                    log.write(f"{fname}: predicted={label}, confidence={confidence:.2f}\n")

                    if confidence >= self.conf_threshold:
                        img = cv2.imread(path)
                        out_path = os.path.join(self.output_folder, f"{label}_{confidence:.2f}_{fname}")
                        cv2.imwrite(out_path, img)

                except Exception as e:
                    print(f"Failed to predict {path}: {e}")
                    log.write(f"{fname}: error={e}\n")

        print(f" Prediction complete. Log saved to {log_path}")

def load_balanced_image_paths(data_dir, label_map, balance=True):
    file_paths, labels = [], []
    class_image_dict = {}

    for label in label_map:
        class_dir = os.path.join(data_dir, label)
        images = [f for f in sorted(os.listdir(class_dir)) if f.lower().endswith(('jpg', 'png', 'jpeg'))]
        paths = [os.path.join(class_dir, fname) for fname in images]
        class_image_dict[label] = paths

    if balance:
        min_count = min(len(v) for v in class_image_dict.values())
        print(f"🔄 Balancing to {min_count} images per class")
    else:
        min_count = None
        print("📊 Using all available images per class")

    for label, paths in class_image_dict.items():
        if balance:
            paths = paths[:min_count]
        file_paths.extend(paths)
        labels.extend([label_map[label]] * len(paths))

    return file_paths, labels

if __name__ == "__main__":
    # Example usage of ImagePredictor
    extract_all_frames_with_unicode_paths("Data/Cardio/GEData/冠脉病例GE-1/冠脉病例GE-1/2025022wangdongyue/",
                                          "Data/Cardio/GEData/videotest") ##todo: add a video here

    predictor = ImagePredictor(
        model_path="latest_model.h5", ##todo: pretrained model
        label_file="filtered_frames/label_index.txt",
        input_folder="Data/Cardio/GEData/videotest",
        output_folder="predicted_outputs",
        conf_threshold=0.9
    )

    predictor.predict_and_save()

