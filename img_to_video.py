import cv2
from tqdm  import tqdm
import os

def convert_images_to_video(images, video_name):
    # 视频编码器和相关参数
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 30  # 每秒帧数
    size = cv2.imread(images[0]).shape[::-1]  # 图片大小

    # 创建VideoWriter对象
    out = cv2.VideoWriter(filename=video_name, fourcc=fourcc, fps=fps, frameSize=size[1:])
    images = sorted(images)
    # 将图片帧写入视频
    for image in tqdm(images, 'read images...'):
        frame = cv2.imread(image)
        out.write(frame)

    # 释放VideoWriter对象
    out.release()

images = []
dir = r'C:\Users\june.lin\Desktop\medicallmg\dataset_0710\prediction_results_video'
for f in tqdm(os.listdir(dir), 'frame_to_image...'):
    if f.startswith("Image02"):
        images.append(os.path.join(dir,f))

video_name = f'Image02_test.avi'
convert_images_to_video(images, video_name)