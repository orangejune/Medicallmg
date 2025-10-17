import os
import json
import numpy as np
import cv2
from tqdm import tqdm
import shutil

'''
U-Net需要的是图像和其对应的掩码图 (Mask)。掩码图是一张与原图等大的单通道图像，其中每个像素的值代表该像素的类别ID。

'''
# --- 配置 ---
file_path = os.path.dirname(__file__)
RAW_DATA_DIR = os.path.join(file_path,'dataset/train-positive')  # 存放原始图片和LabelMe JSON文件的目录
OUTPUT_DIR = os.path.join(file_path,'dataset')     # 处理后数据的存放目录

# 定义标签到类别ID的映射
# ！！！重要：这个顺序决定了填充的顺序，把要被覆盖的类别放前面
LABEL_TO_ID = {
    "_background_": 0,
    "lumen": 1
}

# --- 脚本主逻辑 ---
def process_labelme_json():
    # 创建输出目录
    images_output_dir = os.path.join(OUTPUT_DIR, 'images')
    masks_output_dir = os.path.join(OUTPUT_DIR, 'masks')
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(masks_output_dir, exist_ok=True)

    json_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.json')]
    
    print(f"找到 {len(json_files)} 个JSON文件，开始处理...")

    for json_file in tqdm(json_files):
        json_path = os.path.join(RAW_DATA_DIR, json_file)

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 获取图像尺寸和文件名
        image_height = data['imageHeight']
        image_width = data['imageWidth']
        image_filename = data['imagePath']
        
        # 创建一个全黑的背景掩码 (ID=0)
        mask = np.zeros((image_height, image_width), dtype=np.uint8)

        # 对标签进行排序，确保'vessel_wall'在'lumen'之前被绘制
        # 这样lumen就会覆盖在vessel_wall之上
        sorted_shapes = sorted(data['shapes'], key=lambda s: LABEL_TO_ID.get(s['label'], 99))

        for shape in sorted_shapes:
            label = shape['label']
            if label in LABEL_TO_ID:
                points = np.array(shape['points'], dtype=np.int32)
                class_id = LABEL_TO_ID[label]
                # 使用OpenCV的fillPoly进行填充
                cv2.fillPoly(mask, [points], color=class_id)
        
        # 保存原始图像和生成的掩码
        original_image_path = os.path.join(RAW_DATA_DIR, image_filename)
        # 确保源图片存在
        if os.path.exists(original_image_path):
            # 将原始图像复制到目标文件夹, 并换成png
            image_filename = os.path.splitext(image_filename)[0] + '.png'
            output_image_path = os.path.join(images_output_dir, image_filename)
            shutil.copy2(original_image_path, output_image_path)
            
            # 保存掩码图
            mask_filename = image_filename
            output_mask_path = os.path.join(masks_output_dir, mask_filename)
            is_success, buffer = cv2.imencode('.png', mask)
            if is_success:
                buffer.tofile(output_mask_path)
            else:
                print(f"\n警告: 无法编码掩码图像 {mask_filename}")
        else:
            print(f"警告: 找不到图像 {original_image_path}")

    print("处理完成！")
    print(f"图像保存在: {images_output_dir}")
    print(f"掩码保存在: {masks_output_dir}")

if __name__ == '__main__':
    process_labelme_json()