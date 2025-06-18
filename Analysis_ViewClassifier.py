# Updated and cleaned version of StructureTypeClassifier to ensure training and evaluation are consistent

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten, Dropout, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image
import matplotlib.pyplot as plt

ROI = (100, 50, 950, 700)
IMG_SIZE = (ROI[3] - ROI[1], ROI[2] - ROI[0])  # (height, width)

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
        print(f" Balancing to {min_count} images per class")
    else:
        min_count = None
        print(" Using all available images per class")

    for label, paths in class_image_dict.items():
        if balance:
            paths = paths[:min_count]
        file_paths.extend(paths)
        labels.extend([label_map[label]] * len(paths))

    return file_paths, labels

class DualInputDataGenerator(Sequence):
    def __init__(self, file_paths, labels, img_size, batch_size, num_classes, shuffle=True):
        self.file_paths = file_paths
        self.labels = labels
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.file_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_paths = [self.file_paths[k] for k in indices]
        batch_labels = [self.labels[k] for k in indices]

        X_raw_list, X_line_list, Y_list = [], [], []
        for path, label in zip(batch_paths, batch_labels):
            try:
                img = Image.open(path).convert("RGB")
                img = np.array(img)
                crop_x1, crop_y1, crop_x2, crop_y2 = ROI
                img = img[crop_y1:crop_y2, crop_x1:crop_x2]
                line = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (5, 5), 0), 50, 150)
                X_raw_list.append(preprocess_input(img.astype(np.float32)))
                X_line_list.append(np.expand_dims(line.astype(np.float32) / 255.0, axis=-1))
                Y_list.append(label)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue

        # X_raw = np.expand_dims(np.array(X_raw_list), axis=0) if len(X_raw_list[0].shape) == 3 else np.array(X_raw_list)
        # X_line = np.expand_dims(np.array(X_line_list), axis=0) if len(X_line_list[0].shape) == 3 else np.array(X_line_list)

        X_raw = np.array(X_raw_list)  # shape: (batch_size, H, W, 3)
        X_line = np.array(X_line_list)  # shape: (batch_size, H, W, 1)

        Y = tf.keras.utils.to_categorical(Y_list, num_classes=self.num_classes)
        return [X_raw, X_line], Y

class StructureTypeClassifier:
    def __init__(self, data_dir, model_path, img_size, batch_size, epochs, num_classes, use_pretrained=False):
        self.data_dir = data_dir
        self.model_path = model_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_classes = num_classes
        self.use_pretrained = use_pretrained
        self.model = None

    def build_model(self):
        raw_input = Input(shape=self.img_size + (3,), name='raw_input')
        line_input = Input(shape=self.img_size + (1,), name='line_input')
        raw_base = MobileNetV2(include_top=False, input_tensor=raw_input, weights='imagenet')
        raw_base.trainable = False
        raw_features = GlobalAveragePooling2D()(raw_base.output)
        x = Conv2D(32, (3, 3), activation='relu')(line_input)
        x = MaxPooling2D(2, 2)(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D(2, 2)(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        merged = Concatenate()([raw_features, x])
        merged = Dense(128, activation='relu')(merged)
        output = Dense(self.num_classes, activation='softmax')(merged)
        self.model = Model(inputs=[raw_input, line_input], outputs=output)
        self.model.compile(optimizer=Adam(1e-4), loss=CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'])

    def preprocess_image_for_dual_input(self, path):
        img = Image.open(path).convert("RGB")
        img = np.array(img)
        crop_x1, crop_y1, crop_x2, crop_y2 = ROI
        img = img[crop_y1:crop_y2, crop_x1:crop_x2]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edge = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        X_raw = np.expand_dims(preprocess_input(img.astype(np.float32)), axis=0)
        X_line = np.expand_dims(np.expand_dims(edge.astype(np.float32) / 255.0, axis=0), axis=-1)
        return [X_raw, X_line]

    def train(self):
        fine_labels = sorted([f for f in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, f))])
        label_map = {label: idx for idx, label in enumerate(fine_labels)}
        inverse_label_map = {v: k for k, v in label_map.items()}


        file_paths, labels = [], []
        file_paths, labels = load_balanced_image_paths(self.data_dir, label_map, balance=False)  # or False

        # for label in fine_labels:
        #     class_dir = os.path.join(self.data_dir, label)
        #     for fname in sorted(os.listdir(class_dir)):
        #         if fname.lower().endswith(('jpg', 'png', 'jpeg')):
        #             file_paths.append(os.path.join(class_dir, fname))
        #             labels.append(label_map[label])

        train_gen = DualInputDataGenerator(file_paths, labels, self.img_size, self.batch_size, self.num_classes)
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
        class_weights = dict(enumerate(class_weights))

        checkpoint = ModelCheckpoint("latest_model.h5", save_best_only=False, verbose=1)
        history = self.model.fit(train_gen, epochs=self.epochs, class_weight=class_weights, callbacks=[checkpoint])

        # Save labels
        os.makedirs("filtered_frames", exist_ok=True)
        with open("filtered_frames/label_index.txt", "w", encoding="utf-8") as f:
            for name in fine_labels:
                f.write(name + "\n")

        # Save training predictions
        with open("trained_sorted.txt", "w", encoding="utf-8") as f:
            for path, true_idx in zip(file_paths, labels):
                try:
                    inputs = self.preprocess_image_for_dual_input(path)
                    pred = self.model.predict(inputs, verbose=0)[0]
                    pred_idx = np.argmax(pred)
                    confidence = pred[pred_idx]
                    f.write(f"{os.path.basename(path)}: true={inverse_label_map[true_idx]}, pred={inverse_label_map[pred_idx]}, conf={confidence:.2f}\n")
                except Exception as e:
                    print(f"Failed to predict {path}: {e}")

    def evaluate_misclassified(self):
        if os.path.exists("latest_model.h5"):
            self.model = tf.keras.models.load_model("latest_model.h5")
        else:
            print("️ latest_model.h5 not found. Using in-memory model.")

        with open("filtered_frames/label_index.txt", "r", encoding="utf-8") as f:
            fine_labels = [line.strip() for line in f.readlines()]

        label_map = {cls: i for i, cls in enumerate(fine_labels)}
        inverse_label_map = {v: k for k, v in label_map.items()}

        predictions = []
        for label in fine_labels:
            class_dir = os.path.join(self.data_dir, label)
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(('jpg', 'png', 'jpeg')):
                    path = os.path.join(class_dir, fname)
                    try:
                        inputs = self.preprocess_image_for_dual_input(path)
                        pred = self.model.predict(inputs, verbose=0)[0]
                        pred_idx = np.argmax(pred)
                        true_idx = label_map[label]
                        confidence = pred[pred_idx]
                        if pred_idx != true_idx:
                            predictions.append((path, inverse_label_map[true_idx], inverse_label_map[pred_idx], confidence))
                    except Exception as e:
                        print(f"Failed to evaluate {path}: {e}")

        predictions.sort(key=lambda x: os.path.basename(x[0]))
        with open("misclassified_sorted.txt", "w", encoding="utf-8") as f:
            for path, true_label, pred_label, conf in predictions:
                f.write(f"{os.path.basename(path)}: true={true_label}, pred={pred_label}, conf={conf:.2f}\n")
        print(" Misclassified samples saved to misclassified_sorted.txt")


if __name__ == "__main__":
    classifier = StructureTypeClassifier(
        data_dir="heart_cycles/labelnew2",
        model_path="structure_cnn.h5",
        img_size=IMG_SIZE,
        batch_size=16,
        epochs=5,
        num_classes=6,
        use_pretrained=False
    )

    # Check if model already trained
    if os.path.exists("latest_modelv2.h5"):
        print(" Found saved model. Skipping training. Proceeding with evaluation...")
        classifier.build_model()  # needed to initialize structure
        classifier.evaluate_misclassified()
    else:
        print(" No existing model found. Starting training...")
        classifier.build_model()
        classifier.train()
        classifier.evaluate_misclassified()
