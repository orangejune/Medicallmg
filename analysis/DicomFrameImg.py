import pydicom
import numpy as np
import cv2
import os
from tqdm import tqdm

def convert_ybr_to_rgb(ybr_frame):
    """
    将YBR_FULL格式的图像数据转换为RGB。
    使用正确的转换矩阵。
    """
    # 创建转换矩阵 (根据DICOM标准 YBR_FULL to RGB)
    y = ybr_frame[:, :, 0].astype(np.float32)
    cb = ybr_frame[:, :, 1].astype(np.float32) - 128.0
    cr = ybr_frame[:, :, 2].astype(np.float32) - 128.0

    # YBR to RGB 转换公式
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb

    # 将值限制在0-255范围内并组合
    rgb_frame = np.stack([r, g, b], axis=-1)
    rgb_frame = np.clip(rgb_frame, 0, 255).astype(np.uint8)
    
    return rgb_frame

# 1. 读取DICOM文件
def convert_dicom_to_jpg(dicom_file_path, output_dir):
    ds = pydicom.dcmread(dicom_file_path)

    print(f"色彩解释: {ds.PhotometricInterpretation}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    if hasattr(ds, 'NumberOfFrames') and ds.NumberOfFrames > 1:
        print(f"这是一个多帧文件，包含 {ds.NumberOfFrames} 帧。")
        frames = ds.pixel_array # 解码压缩的图像数据，用时长
        print(f"原始数据形状: {frames.shape}")

        # 获取尺寸
        num_frames, height, width, channels = frames.shape

        # 2. 处理并保存每一帧为JPG
        for i in tqdm(range(num_frames)):
            frame_data = frames[i]

            # 使用归一化处理
            normalized_frame = cv2.normalize(frame_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            if normalized_frame.ndim == 3:
                bgr_frame = cv2.cvtColor(normalized_frame, cv2.COLOR_RGB2BGR)
            else:
                bgr_frame = normalized_frame

            # 保存为JPG文件
            output_filename = os.path.join(output_dir, f"frame_{i:04d}.jpg")
            cv2.imwrite(output_filename, bgr_frame)
            

        print(f"所有帧已保存到目录: {output_dir}")

    else:
        print("这不是一个多帧DICOM文件。")
        
        # 如果是单帧DICOM，也保存为图片
        if hasattr(ds, 'pixel_array'):
            frame_data = ds.pixel_array
            
            # 处理单帧图像
            normalized_frame = cv2.normalize(frame_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            if normalized_frame.ndim == 3:
                bgr_frame = cv2.cvtColor(normalized_frame, cv2.COLOR_RGB2BGR)
            else:
                bgr_frame = normalized_frame
            
            output_filename = os.path.join(output_dir, "single_frame.jpg")
            cv2.imwrite(output_filename, bgr_frame)
            print(f"单帧图像已保存为: {output_filename}")

if __name__ == "__main__":
    dicom_file_path = r"data3\P8SCBO02"
    output_dir = "dicom_frames"
    convert_dicom_to_jpg(dicom_file_path, output_dir)
