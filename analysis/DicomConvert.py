import pydicom
import numpy as np
import cv2

# 将dcm转为视频

def convert_ybr_to_rgb(ybr_frame):
    """
    将YBR_FULL格式的图像数据转换为RGB。
    使用正确的转换矩阵。
    """
    # 创建转换矩阵 (根据DICOM标准 YBR_FULL to RGB)
    # 注意：OpenCV是BGR顺序，所以我们需要RGB然后转换，或者直接调整矩阵输出BGR
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
dicom_file_path = r"F:\medicalimg0815\0828data\2025.8(2).29-KD+冠脉\GEMS_IMG\2025_AUG\29\KL115845\P8TC3UO2"
ds = pydicom.dcmread(dicom_file_path)

print(f"色彩解释: {ds.PhotometricInterpretation}") # 确认是YBR_FULL_422

if hasattr(ds, 'NumberOfFrames') and ds.NumberOfFrames > 1:
    print(f"这是一个多帧文件，包含 {ds.NumberOfFrames} 帧。")
    frames = ds.pixel_array
    print(f"原始数据形状: {frames.shape}")

    # 获取尺寸
    num_frames, height, width, channels = frames.shape

    # 2. 创建视频写入器 - 现在肯定是彩色视频
    output_filename = '29-P8TC3UO2_video.avi'
    fps = 25
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height), isColor=True)

    # 3. 处理并写入每一帧
    for i in range(num_frames):
        frame_data = frames[i]
        
        print(f"处理第 {i+1}/{num_frames} 帧...")
        
        # 关键步骤：将YBR_FULL转换为RGB
        if ds.PhotometricInterpretation == 'YBR_FULL_422':
            # 注意：pydicom可能已经将422采样转换为了全分辨率，但格式标记仍是YBR_FULL_422
            rgb_frame = convert_ybr_to_rgb(frame_data)
            # 将RGB转换为OpenCV需要的BGR
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        else:
            # 如果是其他格式，使用之前的处理方法
            normalized_frame = cv2.normalize(frame_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            if normalized_frame.ndim == 3:
                bgr_frame = cv2.cvtColor(normalized_frame, cv2.COLOR_RGB2BGR)
            else:
                bgr_frame = normalized_frame

        # 写入转换后的BGR帧
        out.write(bgr_frame)

    out.release()
    print(f"颜色正确的视频已保存为: {output_filename}")

else:
    print("这不是一个多帧DICOM文件。")