import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import os
from Analysis_ViewClassifier import DualInputDataGeneratorAug
import tensorflow as tf
from keras.utils import Sequence
from keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import albumentations as A

ROI = (250, 150, 850, 550)
IMG_SIZE = (ROI[3] - ROI[1], ROI[2] - ROI[0])  # (height, width) ##todo
img_size=IMG_SIZE
batch_size=4
epochs=50
num_classes=7
train_data_dir = "heart_cycles/train"

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

fine_labels = sorted([f for f in os.listdir(train_data_dir) if os.path.isdir(os.path.join(train_data_dir, f))])
label_map = {label: idx for idx, label in enumerate(fine_labels)}
inverse_label_map = {v: k for k, v in label_map.items()}


train_paths, train_labels = [], []
val_paths, val_labels = [], []
train_paths, train_labels = load_balanced_image_paths(train_data_dir, label_map, balance=False)  # or False

train_gen = DualInputDataGeneratorAug(train_paths, train_labels, img_size, batch_size, num_classes,shuffle=True, augment=True)  # 应用增强函数


# ===================================================================
# 4. 从生成器获取一批数据并进行可视化
# ===================================================================
# 获取第一个批次的数据
(batch_X_raw, batch_X_line), batch_Y = train_gen[0]

# 反归一化函数，用于正确显示被 preprocess_input 处理过的图像
def denormalize(img_array):
    # preprocess_input for MobileNetV2 scales pixels to [-1, 1]
    # To display, we scale it back to [0, 1]
    return (img_array + 1.0) / 2.0

# 可视化
num_images_to_show = batch_size
# 创建一个大的画布，每行显示3张图（原始图、边缘图、叠加图）
fig, axes = plt.subplots(num_images_to_show, 3, figsize=(12, 4 * num_images_to_show))
fig.suptitle("Data Augmentation Visualization", fontsize=16)

for i in range(num_images_to_show):
    raw_img = batch_X_raw[i]
    line_img = batch_X_line[i]
    
    # --- a. 显示增强后的原始图像 ---
    ax = axes[i, 0]
    # 反归一化以便 matplotlib 正确显示
    denormalized_raw_img = denormalize(raw_img)
    ax.imshow(denormalized_raw_img)
    ax.set_title(f"Augmented Raw Image {i}")
    ax.axis('off')

    # --- b. 显示增强后的边缘图 ---
    ax = axes[i, 1]
    # np.squeeze 去掉单通道维度 (H, W, 1) -> (H, W)
    ax.imshow(np.squeeze(line_img), cmap='gray')
    ax.set_title(f"Augmented Line Image {i}")
    ax.axis('off')
    
    # --- c. 显示叠加图 (最关键的检查!) ---
    ax = axes[i, 2]
    ax.imshow(denormalized_raw_img) # 先画底图
    # 在底图上叠加边缘图，使用红色并设置透明度
    ax.imshow(np.squeeze(line_img), cmap='Reds', alpha=0.5) 
    ax.set_title(f"Overlay Check {i}")
    ax.axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()