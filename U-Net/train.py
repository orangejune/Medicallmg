import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm

# --- 1. 配置 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
IMAGE_DIR = 'dataset/images'
MASK_DIR = 'dataset/masks'
ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
NUM_CLASSES = 2  # 0: background, 1: lumen
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 1e-4

# --- 2. 自定义数据集类 ---
class VesselDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_ids = os.listdir(image_dir)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask

# --- 3. 数据增强和加载 ---
# 定义训练和验证时的数据变换
# 训练集使用丰富的增强，验证集只做尺寸和类型转换
train_transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2(),
])

# 创建数据集实例
full_dataset = VesselDataset(IMAGE_DIR, MASK_DIR)

# 划分训练集和验证集 (例如 80/20)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# 应用不同的变换
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 4. 模型、损失函数、优化器 ---
model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    in_channels=1,  # 灰度图
    classes=NUM_CLASSES,
).to(DEVICE)

# DiceLoss在分割任务中表现优异
loss_fn = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 5. 训练循环 ---
best_val_loss = float('inf')
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE).long() # 损失函数需要long类型
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE).long()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            val_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    
    print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # 保存性能最好的模型
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print("-> Best model saved!")

print("训练完成！")