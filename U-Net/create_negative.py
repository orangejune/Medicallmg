import cv2
import numpy as np
import os

# --- 配置 ---
NEGATIVE_IMAGE_DIR = 'path/to/your/negative_images' # 存放负样本图片的文件夹
OUTPUT_MASK_DIR = 'dataset/masks' # 最终掩码输出的文件夹

# 确保输出目录存在
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

print("开始为负样本生成全黑掩码...")

for img_name in os.listdir(NEGATIVE_IMAGE_DIR):
    img_path = os.path.join(NEGATIVE_IMAGE_DIR, img_name)
    
    # 读取图片以获取其尺寸
    img = cv2.imread(img_path)
    if img is None:
        print(f"警告: 无法读取 {img_name}, 跳过。")
        continue
        
    height, width, _ = img.shape
    
    # 创建一个与原图等大的全黑掩码
    black_mask = np.zeros((height, width), dtype=np.uint8)
    
    # 定义输出路径
    mask_filename = os.path.splitext(img_name)[0] + '.png'
    output_path = os.path.join(OUTPUT_MASK_DIR, mask_filename)
    
    # 保存全黑掩码
    cv2.imwrite(output_path, black_mask)

print("处理完成！")