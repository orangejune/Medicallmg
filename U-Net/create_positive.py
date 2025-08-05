import os
import cv2
import numpy as np

def crop_roi_from_yolo_annotations():
    """
    根据YOLO格式的标注文件，从原图中裁剪出ROI区域并保存。
    """
    # 1. 设置文件路径
    base_path = r"C:\Users\june.lin\Desktop\medicallmg\dataset"
    img_dir = os.path.join(base_path, "img_total")
    label_dir = os.path.join(base_path, "test_total")
    output_dir = os.path.join(base_path, "roi_img_test")

    # 2. 检查并创建输出文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"成功创建输出文件夹: {output_dir}")

    # 获取所有标注文件的列表
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    
    if not label_files:
        print(f"警告: 在 '{label_dir}' 中没有找到任何 .txt 标注文件。")
        return

    print(f"开始处理 {len(label_files)} 个标注文件...")
    processed_count = 0
    skipped_count = 0

    # 3. 遍历每个标注文件
    for label_filename in label_files:
        # 从标注文件名中获取不带扩展名的基本名，如 'image001'
        base_name = os.path.splitext(label_filename)[0]
        label_path = os.path.join(label_dir, label_filename)

        # 尝试查找对应的图像文件（支持多种常见格式）
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif']:
            potential_img_path = os.path.join(img_dir, base_name + ext)
            if os.path.exists(potential_img_path):
                img_path = potential_img_path
                break
        
        if not img_path:
            print(f"警告: 找不到与标注 '{label_filename}' 对应的图像，已跳过。")
            skipped_count += 1
            continue

        # 4. 读取图像和标注信息
        try:
            # img = cv2.imread(img_path) #中文无效
            # 读取图像
            img_buffer = np.fromfile(img_path, dtype=np.uint8)
            # 2. 使用OpenCV从内存缓冲区解码图像数据
            img = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
            
            if img is None:
                print(f"警告: 无法读取图像文件 '{img_path}'，可能已损坏，已跳过。")
                skipped_count += 1
                continue
            
            img_h, img_w, _ = img.shape

            # 读取YOLO标注文件
            with open(label_path, 'r', encoding='utf-8') as f:
                line = f.readline().strip()
                if not line:
                    print(f"警告: 标注文件 '{label_filename}' 为空，已跳过。")
                    skipped_count += 1
                    continue
                
                parts = line.split()
                # 假设格式为: class_id center_x center_y width height
                class_id, cx_rel, cy_rel, w_rel, h_rel = map(float, parts)

            # 5. 将YOLO相对坐标转换为绝对像素坐标
            # 计算边界框的中心点和宽高（像素值）
            box_w_abs = int(w_rel * img_w)
            box_h_abs = int(h_rel * img_h)
            cx_abs = int(cx_rel * img_w)
            cy_abs = int(cy_rel * img_h)

            # 计算边界框的左上角和右下角坐标 (x1, y1, x2, y2)
            x1 = int(cx_abs - box_w_abs / 2)
            y1 = int(cy_abs - box_h_abs / 2)
            x2 = int(cx_abs + box_w_abs / 2)
            y2 = int(cy_abs + box_h_abs / 2)

            # 边界检查，确保裁剪区域不会超出图像范围
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w, x2)
            y2 = min(img_h, y2)

            # 6. 裁剪图像
            roi_img = img[y1:y2, x1:x2]

            if roi_img.size == 0:
                print(f"警告: 在图像 '{os.path.basename(img_path)}' 中计算出的ROI区域为空，已跳过。")
                skipped_count += 1
                continue

            # 7. 保存裁剪后的图像
            # 使用原图的文件名（包括扩展名）来保存
            output_filename = os.path.basename(img_path)[4:]
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, roi_img) 
            # 如果带中文出现乱码
            # is_success, buffer = cv2.imencode('.png', roi_img)
            # if is_success:
            #     buffer.tofile(output_path)
            
            processed_count += 1

        except Exception as e:
            print(f"处理文件 '{label_filename}' 时发生错误: {e}")
            skipped_count += 1

    print("\n--- 处理完成 ---")
    print(f"成功处理并保存: {processed_count} 个文件")
    print(f"跳过或失败: {skipped_count} 个文件")
    print(f"裁剪后的图片已保存至: {output_dir}")

# --- 运行脚本 ---
if __name__ == "__main__":
    crop_roi_from_yolo_annotations()