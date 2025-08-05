# Updated and cleaned version of StructureTypeClassifier to ensure training and evaluation are consistent

import os
import cv2
import numpy as np
import tensorflow as tf
import json
import shutil
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten, Dropout, Input, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras.utils import Sequence
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
from functools import partial
import albumentations as A



ROI = (250, 150, 850, 550)
IMG_SIZE = (ROI[3] - ROI[1], ROI[2] - ROI[0])  # (height, width) ##todo

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
                line = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (5, 5), 0), 50, 150) # Canny边缘检测
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
        return (X_raw, X_line), Y
    
class DualInputDataGeneratorAug(Sequence):
    """
    一个集成了 Albumentations 数据增强功能的双输入数据生成器。
    """
    def __init__(self, file_paths, labels, img_size, batch_size, num_classes, shuffle=True, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.img_size = img_size # 注意：Albumentations 通常在增强后进行缩放
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.augment = augment
        
        # 定义增强管道
        if self.augment:
            self.transform = A.Compose([
                # --- 几何变换 (同时作用于图像和边缘图) ---
                A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=15, p=0.7),
                A.ElasticTransform(p=0.3, alpha=50, sigma=5),
                
                # --- 像素级变换 (只作用于原始图像) ---
                A.RandomBrightnessContrast(p=0.5),
                A.GaussNoise(p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                
                # --- 尺寸调整 (最后一步) ---
                # 注意：如果ROI裁剪后的尺寸不一，需要Resize
                A.Resize(height=self.img_size[0], width=self.img_size[1], always_apply=True),
            ], additional_targets={'line': 'mask'}) # 关键：让'line'像'mask'一样接受几何变换
        else:
            # 如果不增强，只做尺寸调整
            self.transform = A.Compose([
                A.Resize(height=self.img_size[0], width=self.img_size[1], always_apply=True),
            ], additional_targets={'line': 'mask'})

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
                
                # 假设 ROI 定义为 (x1, y1, x2, y2)
                # crop_x1, crop_y1, crop_x2, crop_y2 = ROI
                # img = img[crop_y1:crop_y2, crop_x1:crop_x2]
                
                # 先生成边缘图，再一起做增强
                gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
                line = cv2.Canny(blurred_img, 50, 150)

                # 应用增强
                # 传入 image 和 line，albumentations 会根据定义好的规则进行处理
                augmented = self.transform(image=img, line=line)
                augmented_img = augmented['image']
                augmented_line = augmented['line']
                
                # 预处理和格式化
                # 1. 对原始图像进行模型特定的预处理
                processed_img = preprocess_input(augmented_img.astype(np.float32))
                
                # 2. 对边缘图进行归一化并增加通道维度
                processed_line = np.expand_dims(augmented_line.astype(np.float32) / 255.0, axis=-1)
                
                X_raw_list.append(processed_img)
                X_line_list.append(processed_line)
                Y_list.append(label)
                
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue

        X_raw = np.array(X_raw_list)
        X_line = np.array(X_line_list)
        Y = tf.keras.utils.to_categorical(Y_list, num_classes=self.num_classes)
        
        return (X_raw, X_line), Y
    
class StructureTypeClassifier:
    def __init__(self, train_data_dir, test_data_dir, model_path, img_size, batch_size, epochs, num_classes, output_dir, use_pretrained=False):
        # self.data_dir = data_dir
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.model_path = model_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_classes = num_classes
        self.use_pretrained = use_pretrained
        self.model = None
        self.output_dir = output_dir

    def build_model(self):
        # 1. 双输入架构定义
        raw_input = Input(shape=self.img_size + (3,), name='raw_input')     # RGB彩色图像输入?
        line_input = Input(shape=self.img_size + (1,), name='line_input')   # 单通道线稿图输入
        
        # 2. 彩色图像处理分支（使用预训练MobileNetV2）
        raw_base = MobileNetV2(
            include_top=False,  # 排除顶层分类器
            input_tensor=raw_input, 
            weights='mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5'
        )
        raw_base.trainable = True  # 冻结预训练权重是否可训练（关键迁移学习技术）
        raw_features = GlobalAveragePooling2D()(raw_base.output)  # 将特征图转换为向量 [1280维]
        
        # 3. 线稿图像处理分支（自定义CNN）
        x = Conv2D(32, (3, 3), activation='relu')(line_input)    # 32通道3x3卷积
        x = MaxPooling2D(2, 2)(x)                               # 2x2最大池化→尺寸减半
        x = Conv2D(64, (3, 3), activation='relu')(x)            # 64通道3x3卷积
        x = MaxPooling2D(2, 2)(x)                               # 再次池化
        x = Flatten()(x)                                        # 展平特征图
        x = Dense(64, activation='relu')(x)                     # 全连接层→64维特征向量
        
        # 4. 特征融合与分类
        merged = Concatenate()([raw_features, x])              # 合并双分支特征 [1280+64维]
        merged = Dense(128, activation='relu')(merged)          # 融合特征的全连接层
        output = Dense(self.num_classes, activation='softmax')(merged)  # 输出分类概率
        
        # 5. 模型编译
        self.model = Model(inputs=[raw_input, line_input], outputs=output)
        self.model.compile(
            optimizer=Adam(1e-4),  # Adam优化器，学习率0.0001
            loss=CategoricalCrossentropy(label_smoothing=0.1),  # 分类交叉熵,标签平滑正则化,避免模型对标签过于自信，防止过拟合。
            metrics=['accuracy']
        )

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
        fine_labels = sorted([f for f in os.listdir(self.train_data_dir) if os.path.isdir(os.path.join(self.train_data_dir, f))])
        label_map = {label: idx for idx, label in enumerate(fine_labels)}
        inverse_label_map = {v: k for k, v in label_map.items()}


        train_paths, train_labels = [], []
        val_paths, val_labels = [], []
        train_paths, train_labels = load_balanced_image_paths(self.train_data_dir, label_map, balance=False)  # or False
        val_paths, val_labels = load_balanced_image_paths(self.test_data_dir, label_map, balance=False)  # or False

        # for label in fine_labels:
        #     class_dir = os.path.join(self.data_dir, label)
        #     for fname in sorted(os.listdir(class_dir)):
        #         if fname.lower().endswith(('jpg', 'png', 'jpeg')):
        #             file_paths.append(os.path.join(class_dir, fname))
        #             labels.append(label_map[label])

        # train_paths, val_paths, train_labels, val_labels = train_test_split(
        #     file_paths, labels, 
        #     test_size=0.2,               # 20%作为验证集
        #     stratify=labels,             # 保持类别平衡
        #     random_state=42              # 可重现的分割
        # )

        train_gen = DualInputDataGeneratorAug(train_paths, train_labels, self.img_size, self.batch_size, self.num_classes,shuffle=True, augment=True)  # 应用增强函数
        val_gen = DualInputDataGeneratorAug(val_paths, val_labels, self.img_size, self.batch_size, self.num_classes,shuffle=True, augment=False)
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
        class_weights = dict(enumerate(class_weights))

        # checkpoint = ModelCheckpoint(self.model_path, save_best_only=False, verbose=1)
        # history = self.model.fit(train_gen, epochs=self.epochs, class_weight=class_weights, callbacks=[checkpoint])

        # history = self.model.fit(
        #     train_gen,
        #     epochs=self.epochs,
        #     class_weight=class_weights,
        #     callbacks=[
        #         ModelCheckpoint(self.model_path, save_best_only=True, monitor='val_accuracy', verbose=1),
        #         EarlyStopping(monitor='val_loss', patience=5)
        #     ],
        #     validation_data=val_gen  # 添加验证数据
        # )
        history = self.model.fit(
            train_gen,  # 使用增强后的生成器
            epochs=self.epochs,
            class_weight=class_weights,
            callbacks=[
                # 添加更好的回调组合：
                ModelCheckpoint(
                    self.model_path, 
                    save_best_only=True,
                    save_weights_only=False,
                    monitor='val_accuracy',
                    mode='max',
                    verbose=1
                ),
                ReduceLROnPlateau(  # 添加学习率衰减
                    monitor='val_loss',
                    factor=0.1,
                    patience=3,
                    verbose=1,
                    mode='auto'
                ),
                EarlyStopping(  # 更早停止策略
                    monitor='val_loss',
                    patience=8,
                    restore_best_weights=True,
                    verbose=1
                ),
                TensorBoard(log_dir='./logs')  # 添加TensorBoard
            ],
            validation_data=val_gen,
            steps_per_epoch=len(train_gen),
            validation_steps=len(val_gen),
        )

        # Save train log
        # json_filepath = 'training_history.json'
        # with open(json_filepath, 'w') as f:
        #     json.dump(history.history, f, indent=4)
        history_df = pd.DataFrame(history.history)
        history_df['epoch'] = history.epoch
        csv_filepath = f'{self.output_dir}/training_history.csv' 
        history_df.to_csv(csv_filepath, index=False)
            
        # Save labels
        with open(f"{self.output_dir}/label_index.txt", "w", encoding="utf-8") as f:
            for name in fine_labels:
                f.write(name + "\n")

        # Save training predictions
        with open(f"{self.output_dir}/trained_sorted.txt", "w", encoding="utf-8") as f:
            for path, true_idx in tqdm(zip(train_paths, train_labels)):
                try:
                    inputs = self.preprocess_image_for_dual_input(path)
                    pred = self.model.predict(inputs, verbose=0)[0]
                    pred_idx = np.argmax(pred)
                    confidence = pred[pred_idx]
                    f.write(f"{os.path.basename(path)}: true={inverse_label_map[true_idx]}, pred={inverse_label_map[pred_idx]}, conf={confidence:.2f}\n")
                except Exception as e:
                    print(f"Failed to predict {path}: {e}")

    def evaluate_misclassified(self):
        if os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)
        else:
            print(f" {self.model_path} not found. Using in-memory model.")

        with open(f"{self.output_dir}/label_index.txt", "r", encoding="utf-8") as f:
            fine_labels = [line.strip() for line in f.readlines()]

        label_map = {cls: i for i, cls in enumerate(fine_labels)}
        inverse_label_map = {v: k for k, v in label_map.items()}

        predictions = []
        for label in fine_labels:
            class_dir = os.path.join(self.test_data_dir, label)
            for fname in tqdm(sorted(os.listdir(class_dir)), desc = f'{label}'):
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
        misclassified_dir = f"{self.output_dir}/misclassified_img"
        os.makedirs(misclassified_dir,exist_ok=True)
        with open(f"{self.output_dir}/misclassified_sorted.txt", "w", encoding="utf-8") as f:
            for path, true_label, pred_label, conf in predictions:
                f.write(f"{os.path.basename(path)}: true={true_label}, pred={pred_label}, conf={conf:.2f}\n")
                misclassified_path = os.path.join(misclassified_dir,f'true={true_label}_pred={pred_label}_conf={conf:.2f}.jpg')
                shutil.copy(path,misclassified_path)
        print(" Misclassified samples saved to misclassified_sorted.txt")


if __name__ == "__main__":
    output_dir = './0704_gen'
    os.makedirs(output_dir,exist_ok=True)
    # data_dir = "heart_cycles/small_test"
    train_data_dir = "heart_cycles/train"
    test_data_dir = "heart_cycles/test"
    model_path = f"{output_dir}/best_model.h5"
    classifier = StructureTypeClassifier(
        # data_dir=data_dir,
        train_data_dir=train_data_dir,
        test_data_dir=test_data_dir,
        model_path = model_path,
        img_size=IMG_SIZE,
        batch_size=16,
        epochs=50,
        num_classes=7,
        output_dir = output_dir,
        use_pretrained=False,
    )

    # Check if model already trained
    if os.path.exists(model_path):
        print(" Found saved model. Skipping training. Proceeding with evaluation...")
        classifier.build_model()  # needed to initialize structure
        classifier.evaluate_misclassified()
    else:
        print(" No existing model found. Starting training...")
        classifier.build_model()
        classifier.train()
        classifier.evaluate_misclassified()
