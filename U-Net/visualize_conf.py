import cv2
import os
import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- 配置 ---
MODEL_PATH = 'best_model.pth'
IMAGE_DIR = 'dataset/predict_test'  # 替换成你要预测的图像路径
RESULT_DIR = 'dataset/predict_test/conf2'
os.makedirs(RESULT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LUMEN_CLASS_ID = 1  # 血管腔的类别ID

# --- 模型加载 ---
# 确保模型结构和训练时一致
model = smp.Unet(
    encoder_name='resnet34',
    encoder_weights=None,
    in_channels=1,
    classes=2,
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# --- 图像预处理 ---
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2(),
])

# --- 增强后的推理函数 ---
def predict_and_analyze(image_path, result_dir=None, conf_threshold=0.7):
    """
    对图像进行推理，提取轮廓，并分析边界质量。

    返回:
        - contours: 筛选后的轮廓列表。
        - scores: 每个轮廓对应的平均边界置信度分数。
    """
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
    
    # 3. 计算概率和置信度 (核心改动)
    # probs 的形状: [1, 2, 256, 256]
    probs = F.softmax(logits, dim=1)
    
    # 4. 获取预测掩码和置信度图
    # pred_mask 是每个像素最可能的类别索引, shape: [256, 256]
    pred_mask = torch.argmax(probs, dim=1).squeeze().cpu().numpy().astype(np.uint8)
    
    # confidence_map 是每个像素被预测为其最终类别时的置信度, shape: [256, 256]
    # torch.max 返回 (values, indices)，我们只需要 values
    confidence_map = torch.max(probs, dim=1)[0].squeeze().cpu().numpy()

    # 5. 尺寸恢复到原始大小
    pred_mask_resized = cv2.resize(pred_mask, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    confidence_map_resized = cv2.resize(confidence_map, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_LINEAR)

    # 6. 提取血管腔轮廓
    lumen_binary_mask = np.zeros_like(pred_mask_resized, dtype=np.uint8)
    lumen_binary_mask[pred_mask_resized == LUMEN_CLASS_ID] = 255
    
    all_contours, _ = cv2.findContours(lumen_binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 7. 量化并筛选轮廓
    final_contours = []
    contour_scores = []
    min_contour_area = 50
    for cnt in all_contours:
        if cv2.contourArea(cnt) > min_contour_area:
            # 计算这条轮廓上的平均置信度
            mask = np.zeros(confidence_map_resized.shape, np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, 1) # 只画出轮廓线
            
            # 使用轮廓线作为掩码，从置信度图中提取数值
            contour_pixels_confidence = confidence_map_resized[mask == 255]
            
            if len(contour_pixels_confidence) > 0:
                avg_confidence = np.mean(contour_pixels_confidence)
            else:
                avg_confidence = 0
            
            # 根据置信度阈值筛选轮廓
            if avg_confidence >= conf_threshold:
                final_contours.append(cnt)
                contour_scores.append(avg_confidence)

    # 8. 可视化
    result_img = original_img.copy()
    cv2.drawContours(result_img, final_contours, -1, (0, 255, 0), 2) # 用绿色画出高质量轮廓
    
    # 将灰度置信度图转换为彩色热力图，方便观察
    confidence_heatmap = cv2.applyColorMap((confidence_map_resized * 255).astype(np.uint8), cv2.COLORMAP_HOT)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"Image: {image_path.split('/')[-1]} | Contour Confidence Threshold: {conf_threshold}")

    titles = ['Original Image', 'Predicted Mask', 'Confidence Heatmap', 'Final Result']
    images = [original_img, pred_mask_resized * 255, confidence_heatmap, result_img]

    for i, ax in enumerate(axes):
        if images[i].ndim == 3:
            ax.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(images[i], cmap='gray')
        ax.set_title(titles[i])
        ax.axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    try:
        if result_dir:
            save_path = os.path.join(result_dir, f'{img_name.split("_")[-3]}_conf{max(contour_scores):.4f}.jpg')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"结果已保存到: {save_path}")
        else:
            plt.show()
    finally:
        plt.close(fig)

    return final_contours, contour_scores

# --- 使用示例 ---
if __name__ == '__main__':
    for img_name in os.listdir(IMAGE_DIR):
        if img_name.endswith('jpg'):
            img_path = os.path.join(IMAGE_DIR,img_name)
            # save_path = os.path.join(RESULT_DIR,f'result_{img_name}')

            # 调用新函数
            contours, scores = predict_and_analyze(
                image_path=img_path,
                result_dir=RESULT_DIR,
                conf_threshold=0.70 # 设定一个置信度阈值
            )

            if contours:
                print(f"\n{img_path} 成功提取了 {len(contours)} 条高质量轮廓。")
                for i, score in enumerate(scores):
                    print(f"  - 轮廓 {i+1} 的平均边界置信度: {score:.4f}")
            else:
                print(f"\n{img_path} 未能提取到满足置信度阈值的轮廓。")