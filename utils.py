import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


def process_region(region):
    """
    Apply Otsu's thresholding and extract contours from a region.

    Parameters:
        region (numpy.ndarray): Image region.

    Returns:
        list: List of detected contours.
    """
    _, thresh = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, thresh

def visualize_contours_on_image(image, contours, save_path=None):
    """
    Visualizes contours, midline, and perpendicular lines overlaid on the original image.
    - Saves the visualization as an image file.
    - Displays the image using OpenCV.

    Args:
        image (numpy.ndarray): Original grayscale image.
        contours (list of numpy.ndarray): Full contours (yellow).
        save_path (str, optional): Path to save the output image.

    Returns:
        None: Saves and displays the image.
    """
    # Convert grayscale image to BGR for visualization
    if len(image.shape) == 2:
        annotated_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        annotated_image = image.copy()

    # Draw reordered valid contours in Green
    for line in contours:
        cv2.polylines(annotated_image, [line], isClosed=False, color=(0, 255, 0), thickness=1)

    # Save the annotated image
    if save_path:
        cv2.imwrite(save_path, annotated_image)
        print(f"Visualization saved as: {save_path}")

    # Display the image
    cv2.imshow("Contours and Distance Visualization", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def extract_vessel_wall_boundary(gray_img: np.ndarray, save_path: str = None):
    """
    从心脏冠状动脉灰度图中提取两侧血管壁的边界。

    此版本特点:
    - 目标: 提取高亮度的血管壁。
    - 简化流程: 省略了形态学操作。
    - 策略: 通过面积筛选来过滤噪声轮廓。

    处理流程:
    1. 高斯模糊平滑图像。
    2. 使用单一阈值分割出高亮的血管壁。
    3. 找到所有轮廓，并根据面积进行筛选。
    4. 对所有有效轮廓进行多边形逼近简化。
    5. (可选) 可视化处理过程。

    Args:
        gray_img (np.ndarray): 输入的单通道灰度图像。
        visualize (bool): 是否显示处理过程的可视化图像。

    Returns:
        list: 包含一个或多个简化后轮廓(polyline)的列表。
    """
    if gray_img is None:
        print("错误：输入的图像为空。")
        return []

    # --- 1. 预处理：高斯模糊 ---
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # --- 2. 分割：阈值分割提取高亮区域 (血管壁) ---
    # 这是一个关键参数。根据图像，血管壁是非常亮的区域。
    # 我们选择一个较高的阈值来分离它们。
    wall_thresh_value = 90  # 血管壁灰度下限，需要根据实际情况微调
    _, wall_mask = cv2.threshold(blurred_img, wall_thresh_value, 255, cv2.THRESH_BINARY)

    # --- 3. 目标提取：找到所有轮廓并按面积筛选 ---
    # 因为跳过了形态学处理，这里会找到很多轮廓，包括噪声
    contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("警告：在当前阈值下未找到任何轮廓。")
        return []
        
    polylines = []
    min_contour_area = 150  # 最小轮廓面积，用于过滤噪声，可调整
    epsilon_factor = 0 # 控制轮廓简化的程度

    for cnt in contours:
        # 仅处理面积大于阈值的轮廓
        if cv2.contourArea(cnt) > min_contour_area:
            # --- 4. 轮廓简化 ---
            perimeter = cv2.arcLength(cnt, True)
            epsilon = epsilon_factor * perimeter
            approx_poly = cv2.approxPolyDP(cnt, epsilon, True)
            polylines.append(approx_poly)

    # --- 5. 可视化展示 ---
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 可视化流程调整为4步
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    result_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    # 用不同颜色绘制多个轮廓，以示区分
    if polylines:
        cv2.drawContours(result_img, polylines, -1, (0, 255, 0), 1)

    titles = [
        '1. 原始灰度图', '2. 高斯模糊后', 
        '3. 阈值分割 (提取血管壁)', '4. 最终边界提取结果'
    ]
    images = [
        gray_img, blurred_img, 
        wall_mask, result_img
    ]

    for i, ax in enumerate(axes):
        if images[i].ndim == 3:
            ax.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(images[i], cmap='gray')
        ax.set_title(titles[i])
        ax.axis('off')
    
    plt.tight_layout()
    if not save_path:
        plt.show()
    else:
        plt.savefig(save_path)

    return polylines