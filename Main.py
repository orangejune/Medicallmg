import os
from utils.Analysis_FrameSplit import *
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
# 强制不弹出窗口
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import albumentations as A
from tqdm import tqdm
from analysis.Analysis_D import calculate_average_diameter, find_and_visualize_max_diameter, find_and_visualize_max_diameter_improved
from analysis.Analysis_DistanceRatio import cal_distance_ratio
import torch.nn.functional as F
# 不弹出窗口,除非show()
plt.ioff()
from datetime import date
from analysis.score import score_vessel_boundary
from analysis.DicomFrameImg import convert_dicom_to_jpg
from analysis.DicomRatio import get_corrected_pixel_spacing
import shutil
"""
打分：yolo训练后的置信度/骨架评分
"""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# tag = date.today().strftime('%m%d')
tag = '0814'

# ----------------- yolo配置 ------------------
YOLO_MODEL_BEST_PATH = 'YOLO_model_result/best.pt'
yolo_model = YOLO(YOLO_MODEL_BEST_PATH)

# ---------------- u-net配置 --------------------
UNET_MODEL_PATH = f'U-Net_model_result/unet_best_model_{tag}.pth'
unet_model = smp.Unet(
    encoder_name='resnet34',
    encoder_weights=None, # 推理时不需要预训练权重
    in_channels=1,
    classes=2,
).to(DEVICE)
unet_model.load_state_dict(torch.load(UNET_MODEL_PATH, map_location=DEVICE))
unet_model.eval()

LUMEN_CLASS_ID = 1  # 血管腔的类别ID
# --- 图像预处理 (必须和验证集一致) ---
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2(),
])

# 打分的权重
scoring_weights = {
    'continuity': 0.15,  # 边界是否连续
    'smoothness': 0.40,  # 边界是否平滑无锯齿
    'tubularity': 0.35,  # 形态需要清晰
    'simplicity': 0.10   # 分割需要干净
}

def yolo_roi_img(yolo_model,prediction_images_dir,crops_output_dir,prediction_output_dir=None):
    '''
    YOLO模型预测，保存ROI图片
    输入：yolo模型、预测图片地址、roi保存地址、全图保存地址
    输出：无
    '''
    print(" Running predictions on your prediction dataset...")
    pred_image_files = [f for f in os.listdir(prediction_images_dir) if f.endswith(('.jpg', '.png'))]

    for img_file in pred_image_files:
        img_path = os.path.join(prediction_images_dir, img_file)
        results = yolo_model.predict(source=img_path, save=False, imgsz=640, device=DEVICE)

        # Draw predictions + ground truth
        img = Image.open(img_path).convert("RGB")

        final_conf = 0
        final_cls = None
        # Draw predicted boxes and save cropped predictions
        for idx, r in enumerate(results):
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = int(box.cls[0])
                conf = box.conf.item()
                if final_conf < conf:
                    final_conf = conf
                    final_cls = cls

                # Save cropped predicted ROI
                roi = img.crop((int(x1), int(y1), int(x2), int(y2)))
                crop_name = f"{os.path.splitext(img_file)[0]}.jpg"
                roi.save(os.path.join(crops_output_dir, crop_name))

        if prediction_output_dir:
            new_name = f'{img_file.split("_")[-3]}_conf{final_conf:.2f}_cls{final_cls}_{img_file.split("_")[-1]}'
            img.save(os.path.join(prediction_output_dir, new_name))

def predict_and_get_contour(image_path,save_folder=None,img_name=None):
    
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
        logits = unet_model(input_tensor)
    
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

    # 2. 判断是否找到了轮廓，并找出面积最大的轮廓
    if all_contours:
        score_result = score_vessel_boundary(lumen_binary_mask, scoring_weights)
        final_score = score_result['final_score']
        # 通过轮廓面积(cv2.contourArea)来找到最大的轮廓
        largest_contour = max(all_contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < 500:
            return None, None, None

        # 3. 创建一个与原图大小相同的黑色背景
        largest_contour_mask = np.zeros_like(lumen_binary_mask)

        # 4. 在黑色背景上绘制(填充)这个最大的轮廓
        # 参数: [图像], [轮廓列表], 轮廓索引(-1表示绘制所有), 颜色(255白色), 厚度(cv2.FILLED表示填充)
        cv2.drawContours(largest_contour_mask, [largest_contour], -1, 255, cv2.FILLED)

        # 计算这条轮廓上的平均置信度
        mask = np.zeros(confidence_map_resized.shape, np.uint8)
        cv2.drawContours(mask, [largest_contour], -1, 255, 1) # 只画出轮廓线
        # 使用轮廓线作为掩码，从置信度图中提取数值
        contour_pixels_confidence = confidence_map_resized[mask == 255]
        if len(contour_pixels_confidence) > 0:
            avg_confidence = np.mean(contour_pixels_confidence)
        else:
            avg_confidence = 0

        # 6. 可视化
        result_img = original_img.copy()
        cv2.drawContours(result_img, largest_contour, -1, (0, 0, 255), 3) # 用绿色线画出轮廓

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
            if save_folder:
                save_path = os.path.join(save_folder,f'score{final_score:.2f}_conf{avg_confidence:.2f}_{img_name}')
                # 保存图像到文件
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            else:
                # 如果不保存，则显示窗口
                plt.show()
        finally:
            # 无论保存还是显示，最后都必须关闭图形以释放内存
            plt.close(fig)

        return largest_contour_mask, largest_contour, final_score
    else:
        return None, None, None

    
if __name__ == "__main__":

    # Folder paths (modify as needed)
    script_full_path = os.path.abspath(__file__)
    data_path = os.path.join(os.path.dirname(__file__),'data4')
    # media_names = [i.split('.')[0] for i in os.listdir(data_path) if i.endswith('avi')]
    file_names = ['P8SCBO02']
    # media_names = ['Image196_type1-a','Image377_type3a']
    for file_name in file_names:
        file_path = f"{data_path}/{file_name}" # 视频
        input_folder = f"{data_path}/{file_name}_frame" # 图片帧文件夹
        roi_folder = f"{data_path}/{file_name}_roi" # roi图片保存
        os.makedirs(roi_folder, exist_ok=True)
        # yolo_img_path = f'{output_folder}/yolo_imgs' #yolo全图保存
        # os.makedirs(yolo_img_path, exist_ok=True)
        contour_folder = f"{data_path}/{file_name}_contour_{tag}_sc" #边界可视化结果保存
        os.makedirs(contour_folder, exist_ok=True)
        result_folder = f"{data_path}/{file_name}_result_{tag}" #测量结果可视化结果保存
        os.makedirs(result_folder, exist_ok=True)


        # ====================== dicom转成图片帧 ======================== #
        # if not os.path.isdir(input_folder) or not os.listdir(input_folder):
        #     os.makedirs(input_folder, exist_ok=True)
        #     extract_frames(video_path, input_folder, frame_interval=1)

        if not os.path.isdir(input_folder) or not os.listdir(input_folder):
            os.makedirs(input_folder, exist_ok=True)
            convert_dicom_to_jpg(file_path, input_folder)

        # ====================== 获取dicom像素距离 ======================== #
        pixel_spacing,_ = get_corrected_pixel_spacing(file_path)

        # ==================== YOLO模型预测，获得ROI图片 =================== #
        yolo_roi_img(yolo_model, input_folder,roi_folder)

        # ==================== U-net模型预测，获得边界 =================== #
        roi_imgs = os.listdir(roi_folder)
        # roi_imgs = ['frame_0003.jpg','frame_0024.jpg','frame_0124.jpg']
        for img_name in tqdm(roi_imgs):
            ori_img = os.path.join(input_folder,img_name)
            roi_path = os.path.join(roi_folder,img_name)
            lumen_binary_mask, contour, final_score = predict_and_get_contour(roi_path,contour_folder,img_name)

            # ==================== 处理边界 =================== #
            
            if contour is None:
                continue
            else:
                print('开始处理边界……')
                if pixel_spacing:
                    img_name = img_name.split('.')[0]
                    avg_save_path = f'{result_folder}/{img_name}_avg.jpg'
                    max_save_path = f'{result_folder}/{img_name}_max.jpg'
                    average_diameter_in_mm = calculate_average_diameter(lumen_binary_mask, pixel_spacing, avg_save_path)
                    max_diameter_in_mm = find_and_visualize_max_diameter_improved(lumen_binary_mask,roi_path, pixel_spacing, max_save_path) # 最大距离和位置不太准，待优化
                    # pixel_spacing = cal_distance_ratio(ori_img)
                    # print(pixel_spacing)
                    print(f'像素间距为：{pixel_spacing:.2f}')
                    print(f'平均直径为：{average_diameter_in_mm:.2f}mm, 最大直径为：{max_diameter_in_mm:.2f}mm')
                
                    print('1')
            
    # 最后可以多张图测量取平均？
    print("Processing complete. Check the 'output' folder for results.")