import cv2
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt

def calculate_average_diameter(binary_mask, pixel_spacing, save_path):
    """
    从血管的二值掩码图像计算平均内径。

    参数:
    mask_path (str): 二值掩码图像的文件路径。
    pixel_spacing: 像素距离

    返回:
    float: 血管的平均内径（单位：mm）。
    """
    # # 1. 加载掩码图像并进行预处理
    # # 以灰度模式加载
    # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # if mask is None:
    #     raise FileNotFoundError(f"无法找到或打开图像: {mask_path}")

    # # 确保掩码是二值的 (前景为255, 背景为0)
    # _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # scikit-image的skeletonize需要布尔类型的数组
    binary_mask_bool = binary_mask > 0

    # 2. 计算距离变换
    # CV_DIST_L2 表示计算欧氏距离
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)

    # 3. 提取骨架
    # 使用 scikit-image 的 skeletonize 函数，效果更好
    skeleton = skeletonize(binary_mask_bool)
    # 将布尔型骨架转换回 uint8 图像，方便处理
    skeleton_img = skeleton.astype(np.uint8) * 255

    # 4. 在骨架位置上提取半径值
    # np.where(skeleton) 返回骨架像素的坐标
    # dist_transform[skeleton] 直接使用布尔索引提取所有骨架位置上的距离值
    radii = dist_transform[skeleton]
    
    # 过滤掉半径为0的点（可能出现在骨架末端接触边界处）
    radii = radii[radii > 0]

    if len(radii) == 0:
        return 0.0 # 如果没有找到有效的骨架点

    # 5. 计算平均直径
    # 直径是半径的2倍
    average_diameter = np.mean(radii) * 2
    average_diameter_in_mm = average_diameter * pixel_spacing
    
    # --- 可视化部分 ---
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    overlay_img = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    # 将骨架部分（skeleton_img中的白色像素）在彩色图上标记为红色
    overlay_img[skeleton_img == 255] = [0, 0, 255]
    plt.imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
    plt.title('Skeleton Overlay (Red)')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    # 使用 viridis colormap 显示距离值，值越大的地方越亮
    plt.imshow(dist_transform, cmap='viridis')
    plt.colorbar(label='Distance to boundary')
    plt.title('Distance Transform')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    # 将骨架叠加在距离变换图上，更直观
    overlay = cv2.cvtColor(dist_transform.astype(np.float32) / dist_transform.max() * 255, cv2.COLOR_GRAY2BGR)
    overlay[skeleton, 1] = 255 # 将骨架位置标记为绿色
    plt.imshow(overlay)
    plt.title(f'Skeleton on Distance Map\nAvg. Diameter: {average_diameter:.2f} pixels, {average_diameter_in_mm:.2f}mm')
    plt.axis('off')
    
    plt.tight_layout()
    # plt.show()
    plt.savefig(save_path, dpi=150)

    return average_diameter_in_mm

def find_and_visualize_max_diameter(binary_mask, original_image_path):
    """
    计算血管的最大内径，并在原始图像上标注其位置。

    参数:
    mask_path (str): 二值掩码图像的文件路径。
    original_image_path (str): 原始灰度图像的路径，用于可视化。
    """
    original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)

    binary_mask_bool = binary_mask > 0

    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    skeleton = skeletonize(binary_mask_bool)
    
    # --- 新增部分：寻找最大直径 ---

    # 2. 定位最大半径及其坐标
    radii = dist_transform[skeleton]
    if len(radii) == 0:
        print("未在掩码中找到骨架。")
        return

    # 找到最大半径的值
    max_radius = np.max(radii)
    max_diameter = max_radius * 2
    
    # 找到最大半径在 'radii' 数组中的索引
    max_radius_idx = np.argmax(radii)
    
    # 获取所有骨架点的坐标 (row, col) -> (y, x)
    skeleton_coords = np.argwhere(skeleton)
    
    # 使用索引找到最大半径点的坐标
    center_y, center_x = skeleton_coords[max_radius_idx]
    center_point = (center_x, center_y)

    # 3. 计算局部方向 (使用PCA)
    # 在中心点周围取一个邻域 (例如 30x30 像素)
    half_win = 15
    y_start, y_end = max(0, center_y - half_win), min(skeleton.shape[0], center_y + half_win)
    x_start, x_end = max(0, center_x - half_win), min(skeleton.shape[1], center_x + half_win)
    
    local_skeleton_patch = skeleton[y_start:y_end, x_start:x_end]
    local_coords = np.argwhere(local_skeleton_patch).astype(np.float32)
    
    # 将局部坐标转换回全局坐标
    local_coords[:, 0] += y_start
    local_coords[:, 1] += x_start

    # 使用 PCA 计算局部方向
    if len(local_coords) >= 2:
        # OpenCV PCA需要 (N, 2) 格式，且数据类型为 float32
        # 注意OpenCV PCA的输入是 (x,y) 格式，所以我们交换列
        mean, eigenvectors = cv2.PCACompute(local_coords[:, ::-1], mean=None)
        tangent_vector = eigenvectors[0] # 主方向即为切线方向
    else:
        # 如果点太少，无法计算PCA，则使用一个默认的水平方向
        tangent_vector = np.array([1.0, 0.0])

    # 4. 计算直径端点
    # 法线向量（垂直于切线）
    normal_vector = np.array([-tangent_vector[1], tangent_vector[0]])
    
    # 计算端点
    p1 = (center_point + max_radius * normal_vector).astype(int)
    p2 = (center_point - max_radius * normal_vector).astype(int)

    # 5. 可视化
    # 将原始图像转为 BGR 彩色图，以便绘制彩色线条
    vis_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

    # 在图像上绘制最大直径线 (红色粗线)
    cv2.line(vis_image, tuple(p1), tuple(p2), (0, 0, 255), 2) # (B,G,R) -> Red
    
    # 标记中心点 (一个小的绿色圆)
    cv2.circle(vis_image, center_point, 3, (0, 255, 0), -1) # Green

    # 使用 matplotlib 显示
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)) # Matplotlib expects RGB
    plt.title(f"Max Diameter Location\nMax Diameter: {max_diameter:.2f} pixels")
    plt.axis('off')
    plt.show()

    print(f"计算出的血管最大内径为: {max_diameter:.2f} 像素")
    print(f"最大内径的中心点坐标: (x={center_x}, y={center_y})")

    return max_diameter

def find_and_visualize_max_diameter_improved(binary_mask, original_image_path, pixel_spacing, save_path):
    """
    计算血管的最大内径，并对比两种计算方法。
    方法1: max_radius * 2
    方法2: 计算绘制出的直径线段的实际长度 (更精确)
    """
    original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        print(f"错误: 无法读取原始图像 {original_image_path}")
        return None

    binary_mask_bool = binary_mask > 0
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    skeleton = skeletonize(binary_mask_bool)
    
    radii = dist_transform[skeleton]
    if len(radii) == 0:
        print("未在掩码中找到骨架。")
        return None

    max_radius = np.max(radii)
    
    # --- 方法1：简单乘以2 (原始方法) ---
    max_diameter_simple = max_radius * 2
    max_diameter_simple_in_mm = max_diameter_simple * pixel_spacing
    
    # --- 定位与方向计算 (与原代码相同) ---
    max_radius_idx = np.argmax(radii)
    skeleton_coords = np.argwhere(skeleton)
    center_y, center_x = skeleton_coords[max_radius_idx]
    center_point = (center_x, center_y)

    half_win = 15
    y_start, y_end = max(0, center_y - half_win), min(skeleton.shape[0], center_y + half_win)
    x_start, x_end = max(0, center_x - half_win), min(skeleton.shape[1], center_x + half_win)
    
    local_skeleton_patch = skeleton[y_start:y_end, x_start:x_end]
    local_coords = np.argwhere(local_skeleton_patch)
    
    if len(local_coords) < 2:
        tangent_vector = np.array([1.0, 0.0]) # 默认水平方向
    else:
        # 将局部坐标转换回全局坐标
        local_coords = local_coords.astype(np.float32)
        local_coords[:, 0] += y_start
        local_coords[:, 1] += x_start
        mean, eigenvectors = cv2.PCACompute(local_coords[:, ::-1], mean=None)
        tangent_vector = eigenvectors[0]

    normal_vector = np.array([-tangent_vector[1], tangent_vector[0]])
    
    # 计算端点 p1 和 p2
    # 注意：这里我们先用浮点数计算，最后再取整用于绘图
    p1_float = center_point + max_radius * normal_vector
    p2_float = center_point - max_radius * normal_vector

    # --- 方法2：计算线段 (p1, p2) 的欧氏距离 (您的提议，更精确) ---
    max_diameter_line = np.linalg.norm(p1_float - p2_float)
    max_diameter_line_in_mm = max_diameter_line * pixel_spacing

    # --- 可视化 ---
    p1_int = tuple(p1_float.astype(int))
    p2_int = tuple(p2_float.astype(int))
    
    vis_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    cv2.line(vis_image, p1_int, p2_int, (0, 0, 255), 2)
    cv2.circle(vis_image, center_point, 3, (0, 255, 0), -1)

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    title_text = (f"Max Diameter Location\n"
                  f"Method 1 (radius*2): {max_diameter_simple:.2f} px, {max_diameter_simple_in_mm:.2f} mm\n"
                  f"Method 2 (line length): {max_diameter_line:.2f} px, {max_diameter_line_in_mm:.2f} mm")
    plt.title(title_text)
    plt.axis('off')
    # plt.show() 
    plt.savefig(save_path, dpi=150)

    print("--- 直径计算结果对比 ---")
    print(f"方法1 (简单乘以2): {max_diameter_simple:.4f} 像素, {max_diameter_simple_in_mm:.2f} mm")
    print(f"方法2 (计算线段长度): {max_diameter_line:.4f} 像素, {max_diameter_line_in_mm:.2f} mm")

    # 返回更精确的值
    return max_diameter_line_in_mm
# --- 使用示例 ---
# 请将 'Predicted_Mask.png' 替换为您掩码图像的实际路径
if __name__ == "__main__":
    try:
        mask_image_path = 'Predicted Mask.png' # 假设图片在当前目录下
        avg_diameter_pixels = calculate_average_diameter(mask_image_path)
        print(f"计算出的血管平均内径为: {avg_diameter_pixels:.2f} 像素")
        
        # 如果您知道图像的物理尺寸，可以转换为毫米
        # 例如：如果一个像素代表 0.1 毫米
        # pixel_spacing_mm = 0.1
        # avg_diameter_mm = avg_diameter_pixels * pixel_spacing_mm
        # print(f"计算出的血管平均内径为: {avg_diameter_mm:.2f} mm")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"处理过程中发生错误: {e}")