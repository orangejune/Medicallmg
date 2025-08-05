import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import os
import shutil
from tqdm import tqdm


# --- 配置 ---
MODEL_PATH = 'best_model.pth'
IMAGE_DIR = 'dataset/pre'  # 替换成你要预测的图像路径
RESULT_DIR = 'dataset/pre/result'
os.makedirs(RESULT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LUMEN_CLASS_ID = 1  # 血管腔的类别ID

# input_dir = 'dataset/roi_img_total'
# for img_name in os.listdir(input_dir):
#     old_file = os.path.join(input_dir, img_name)
#     new_file = os.path.join(IMAGE_DIR, img_name[4:])
#     shutil.copy(old_file,new_file)

# --- 加载模型 ---
model = smp.Unet(
    encoder_name='resnet34',
    encoder_weights=None, # 推理时不需要预训练权重
    in_channels=1,
    classes=2,
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# --- 图像预处理 (必须和验证集一致) ---
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2(),
])

# --- 推理函数 ---
def predict_and_get_polylines(image_path,save_path):
    # 1. 加载和预处理图像
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"Error: 无法加载图像 {image_path}")
        return [], []
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    
    transformed = transform(image=gray_img)
    input_tensor = transformed['image'].unsqueeze(0).to(DEVICE)

    # 2. 模型推理
    with torch.no_grad():
        logits = model(input_tensor)
    
    # 3. 获取预测掩码
    pred_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy().astype(np.uint8)
    
    # 4. 尺寸恢复到原始大小
    pred_mask_resized = cv2.resize(pred_mask, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 5. 提取血管腔轮廓
    # 创建一个二值图，只保留血管腔
    lumen_binary_mask = np.zeros_like(pred_mask_resized)
    lumen_binary_mask[pred_mask_resized == LUMEN_CLASS_ID] = 255
    
    # 查找轮廓
    all_contours, _ = cv2.findContours(lumen_binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = []
    min_contour_area = 50
    for cnt in all_contours:
        # 仅处理面积大于阈值的轮廓
        if cv2.contourArea(cnt) > min_contour_area:
            contours.append(cnt)

    
    # 6. 可视化
    result_img = original_img.copy()
    cv2.drawContours(result_img, contours, -1, (0, 255, 0), 1) # 用绿色线画出轮廓
    
    # 显示结果
    # cv2.imshow("Original Image", original_img)
    # cv2.imshow("Predicted Mask", pred_mask_resized * 100) # 乘以一个系数方便观察
    # cv2.imshow("Result with Polylines", result_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    titles = ['Original Image', 'Predicted Mask', 'Result with Polylines']
    images = [original_img, pred_mask_resized * 100, result_img]

    for i, ax in enumerate(axes):
        if images[i].ndim == 3:
            ax.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(images[i], cmap='gray')
        ax.set_title(titles[i])
        ax.axis('off')
    
    plt.tight_layout()
    try:
        if save_path:
            # 保存图像到文件
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            # 如果不保存，则显示窗口
            plt.show()
    finally:
        # 无论保存还是显示，最后都必须关闭图形以释放内存
        plt.close(fig)

    return contours

if __name__ == '__main__':
    for img_name in tqdm(os.listdir(IMAGE_DIR)):
        if img_name.endswith(('jpg','png')):
            img_path = os.path.join(IMAGE_DIR,img_name)
            save_path = os.path.join(RESULT_DIR,f'result_{img_name}')
            polylines = predict_and_get_polylines(img_path,save_path)
            # print(f"{img_name} 成功提取了 {len(polylines)} 条血管腔轮廓线。")