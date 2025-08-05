import os
import cv2
from tqdm import tqdm

# --- 配置 (请确保与 train.py 一致) ---
IMAGE_DIR = 'dataset/images'
MASK_DIR = 'dataset/masks'

def run_dataset_sanity_check():
    """
    遍历数据集中的所有文件，检查是否存在加载问题。
    """
    print("--- 开始数据集健康检查 ---")
    
    image_files = os.listdir(IMAGE_DIR)
    
    # 过滤掉非图像文件，如 .DS_Store 或 Thumbs.db
    valid_image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
    
    if len(valid_image_files) == 0:
        print("错误: 在 'images' 目录中没有找到任何有效的图像文件。")
        return

    print(f"在 {IMAGE_DIR} 中找到 {len(valid_image_files)} 个图像文件。现在开始逐一检查...")

    has_error = False
    for img_name in tqdm(valid_image_files, desc="Checking files"):
        img_path = os.path.join(IMAGE_DIR, img_name)
        mask_path = os.path.join(MASK_DIR, img_name) # 假设掩码和图像同名同扩展名

        # 检查1: 图像文件本身是否存在且可读
        image = cv2.imread(img_path)
        if image is None:
            print(f"\n[错误!] 无法加载图像文件: {img_path}")
            print("  -> 可能原因: 文件已损坏, 或者路径中有特殊字符且未正确处理。")
            has_error = True
            continue # 继续检查下一个，找出所有问题

        # 检查2: 对应的掩码文件是否存在
        if not os.path.exists(mask_path):
            print(f"\n[错误!] 找不到与图像 '{img_name}' 对应的掩码文件。")
            print(f"  -> 期望路径: {mask_path}")
            has_error = True
            continue

        # 检查3: 掩码文件是否可读
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            print(f"\n[错误!] 无法加载掩码文件: {mask_path}")
            print(f"  -> 可能原因: 文件已损坏。")
            has_error = True
            continue

    if not has_error:
        print("\n--- 检查完成：恭喜！您的数据集看起来是健康的。---")
    else:
        print("\n--- 检查完成：发现问题！请根据上面的错误提示修复您的数据集。---")

if __name__ == '__main__':
    run_dataset_sanity_check()