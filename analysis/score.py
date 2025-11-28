import cv2
import numpy as np
from skimage.morphology import skeletonize

def calculate_boundary_quality(binary_mask, weights):
    """
    计算给定二值掩码(binary_mask)的边界质量得分。

    Args:
        binary_mask (np.ndarray): 输入的二值图像 (dtype=np.uint8, 值为 0 或 255)。
        weights (dict): 包含四个评分标准权重的字典。
                        e.g., {'count': 0.1, 'area': 0.4, 'skeleton': 0.2, 'parallelism': 0.3}

    Returns:
        dict: 包含最终综合得分和各分项得分的字典。
              如果无法处理（如没有轮廓），返回包含错误信息的字典。
    """
    # --- 0. 初始化和预检查 ---
    if not isinstance(binary_mask, np.ndarray) or binary_mask.ndim != 2:
        return {"error": "输入必须是一个2D NumPy数组。"}
    if np.sum(binary_mask) == 0:
        return {"error": "输入图像为纯黑，无法评分。", "final_score": 0}

    # --- 1. 边界数量和面积评分 (Contour Count & Area Score) ---
    all_contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not all_contours:
        return {"error": "未找到任何轮廓。", "final_score": 0}

    # 1.1 边界数量评分 (越少越好)
    # 理想情况是1个轮廓。多于1个则快速降低分数。
    score_count = 1.0 / len(all_contours)

    # 1.2 面积评分 (越大越好)
    # 我们只关心最大轮廓的面积
    largest_contour = max(all_contours, key=cv2.contourArea)
    largest_area = cv2.contourArea(largest_contour)
    total_area = binary_mask.shape[0] * binary_mask.shape[1]
    # 面积得分通过归一化得到，表示其占整个图像的比例
    score_area = largest_area / total_area

    # --- 2. 骨架复杂度评分 (Skeleton Simplicity Score) ---
    # 2.1 生成仅包含最大轮廓的掩码
    main_object_mask = np.zeros_like(binary_mask)
    cv2.drawContours(main_object_mask, [largest_contour], -1, 255, cv2.FILLED)

    # 2.2 提取骨架
    # skimage的skeletonize需要布尔值或0/1的图像
    skeleton = skeletonize(main_object_mask / 255)
    skeleton_img = (skeleton * 255).astype(np.uint8)
    skeleton_pixels = np.sum(skeleton)
    
    score_skeleton = 0.5 # 默认值，在无法计算时使用
    if skeleton_pixels > 0:
        # 2.3 计算节点数量 (分支点+端点)
        # 使用3x3卷积核计算每个骨架点的邻居数
        kernel = np.ones((3, 3), np.uint8)
        neighbors = cv2.filter2D(skeleton_img / 255, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        # 提取骨架点处的邻居计数值 (减1是去掉自身)
        node_values = neighbors[skeleton] - 1
        
        # 端点: 邻居数为1。分支点: 邻居数 > 2。
        num_endpoints = np.sum(node_values == 1)
        num_branchpoints = np.sum(node_values > 2)
        total_nodes = num_endpoints + num_branchpoints

        # 2.4 计算骨架复杂度得分 (节点密度越低越好)
        # 节点数/骨架总长度，得到节点密度
        node_density = total_nodes / skeleton_pixels
        # 使用 1 - density 作为得分，密度越高得分越低
        score_skeleton = max(0, 1 - node_density)

    # --- 3. 边界与骨架平行度评分 (Parallelism Score) ---
    # 3.1 计算距离变换图
    # 图中每个点的值是它到最近的0像素（边界）的距离
    dist_transform = cv2.distanceTransform(main_object_mask, cv2.DIST_L2, 5)

    score_parallelism = 0.5 # 默认值
    if skeleton_pixels > 0:
        # 3.2 提取骨架点处的距离值 (即骨架到边界的距离列表)
        skeleton_distances = dist_transform[skeleton]
        
        # 3.3 计算平行度得分 (距离分布越均匀越好)
        # 使用变异系数 (CV = std / mean) 来衡量均匀度。CV越小，越均匀。
        mean_dist = np.mean(skeleton_distances)
        std_dist = np.std(skeleton_distances)

        if mean_dist > 0:
            coeff_variation = std_dist / mean_dist
            # 使用 1 - CV 作为分数。CV越大（越不平行），分数越低。
            score_parallelism = max(0, 1 - coeff_variation)

    # --- 4. 计算最终综合得分 ---
    final_score = (weights['count'] * score_count +
                   weights['area'] * score_area +
                   weights['skeleton'] * score_skeleton +
                   weights['parallelism'] * score_parallelism)

    return {
        "final_score": final_score,
        "sub_scores": {
            "count": score_count,
            "area": score_area,
            "skeleton_simplicity": score_skeleton,
            "parallelism": score_parallelism
        },
        "details": {
            "contour_count": len(all_contours),
            "largest_area_ratio": score_area,
            "skeleton_nodes": total_nodes if 'total_nodes' in locals() else 'N/A',
            "parallelism_cv": coeff_variation if 'coeff_variation' in locals() else 'N/A'
        }
    }

import cv2
import numpy as np
from skimage.morphology import skeletonize

def score_vessel_boundary(binary_mask, weights, smoothness_epsilon=0.005):
    """
    针对血管边界质量进行评分。

    Args:
        binary_mask (np.ndarray): 血管腔的二值掩码 (dtype=np.uint8, 值为 0 或 255)。
        weights (dict): 五个评分标准的权重。
                        e.g., {'continuity': 0.2, 'smoothness': 0.3, 'tubularity': 0.2, 'simplicity': 0.1, 'area': 0.2}
        smoothness_epsilon (float): 用于轮廓近似的精度参数，值越小要求越平滑。

    Returns:
        dict: 包含最终综合得分和各分项得分的字典。
    """
    if np.sum(binary_mask) == 0:
        return {"error": "输入图像为纯黑，无法评分。", "final_score": 0}

    # --- 1. 连续性评分 (Continuity Score) ---
    all_contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not all_contours:
        return {"error": "未找到任何轮廓。", "final_score": 0}
    
    # 理想情况是1个轮廓，代表血管段是连续的
    score_continuity = 1.0 / len(all_contours)
    
    # 后续所有分析都基于最大的轮廓
    largest_contour = max(all_contours, key=cv2.contourArea)
    main_object_mask = np.zeros_like(binary_mask)
    cv2.drawContours(main_object_mask, [largest_contour], -1, 255, cv2.FILLED)
    
    # --- 新增：2. 面积评分 (Area Score) ---
    # 计算最大轮廓的面积并标准化到图像总面积
    largest_area = cv2.contourArea(largest_contour)
    total_area = binary_mask.shape[0] * binary_mask.shape[1]
    score_area = largest_area / total_area

    # --- 3. 平滑度评分 (Smoothness Score) ---
    # 使用多边形近似来衡量平滑度。平滑的曲线可以用更少的顶点来近似。
    arc_length = cv2.arcLength(largest_contour, True)
    # epsilon 决定了近似的精度
    epsilon = smoothness_epsilon * arc_length
    approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # 平滑度得分: 近似顶点数 / 原始顶点数。比值越小，说明越平滑。
    # 我们用 1 - (比值) 来让得分范围在0-1之间，且越高越好。
    # 加上一个很小的数避免除以零
    original_points = len(largest_contour)
    approx_points = len(approx_contour)
    score_smoothness = 1.0 - (approx_points / (original_points + 1e-6))
    score_smoothness = max(0, score_smoothness) # 确保不为负

    # --- 4. 管状形态/平行度评分 (Tubularity/Parallelism Score) ---
    dist_transform = cv2.distanceTransform(main_object_mask, cv2.DIST_L2, 5)
    
    skeleton = skeletonize(main_object_mask / 255)
    skeleton_pixels = np.sum(skeleton)
    
    score_tubularity = 0.5 # 默认分
    coeff_variation = 'N/A'
    if skeleton_pixels > 0:
        skeleton_distances = dist_transform[skeleton]
        mean_dist = np.mean(skeleton_distances)
        std_dist = np.std(skeleton_distances)
        if mean_dist > 0:
            coeff_variation = std_dist / mean_dist
            # CV越小，管壁越平行，得分越高
            score_tubularity = max(0, 1 - coeff_variation)

    # --- 5. 形态复杂度评分 (Simplicity Score) ---
    skeleton_img = (skeleton * 255).astype(np.uint8)
    score_simplicity = 0.5 # 默认分
    total_nodes = 'N/A'
    if skeleton_pixels > 0:
        kernel = np.ones((3, 3), np.uint8)
        neighbors = cv2.filter2D(skeleton_img / 255, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        node_values = neighbors[skeleton] - 1
        
        num_branchpoints = np.sum(node_values > 2) # 只关心分支点，因为端点是必然存在的
        
        # 复杂度以分支点密度来衡量
        node_density = num_branchpoints / skeleton_pixels
        score_simplicity = max(0, 1 - node_density * 10) # 乘以10放大密度效应

    # --- 6. 计算最终综合得分 ---
    final_score = (weights['continuity'] * score_continuity +
                   weights['smoothness'] * score_smoothness +
                   weights['tubularity'] * score_tubularity +
                   weights['simplicity'] * score_simplicity +
                   weights['area'] * score_area)

    return {
        "final_score": final_score,
        "sub_scores": {
            "continuity": score_continuity,
            "smoothness": score_smoothness,
            "tubularity": score_tubularity,
            "simplicity": score_simplicity,
            "area": score_area  # 返回新的评分项
        },
        "details": {
            "contour_count": len(all_contours),
            "smoothness_approx_pts_ratio": f"{approx_points}/{original_points}",
            "tubularity_cv": coeff_variation,
            "simplicity_branch_points": num_branchpoints if 'num_branchpoints' in locals() else 'N/A',
            "largest_area_ratio": score_area  # 添加面积比例详情
        }
    }


# --- 使用示例 ---
if __name__ == '__main__':
    # 针对血管场景，我们可能更看重平滑度和管状形态
    scoring_weights = {
        'continuity': 0.15,  # 边界是否连续
        'smoothness': 0.40,  # 边界是否平滑无锯齿
        'tubularity': 0.35,  # 是否呈均匀管状
        'simplicity': 0.10   # 整体形态是否简单
    }

    # 创建一个"好"的血管边界: 单一，平滑，管状
    good_vessel = np.zeros((300, 500), dtype=np.uint8)
    pts = np.array([[50,150], [150,120], [250,140], [350,130], [450,160]], np.int32)
    cv2.polylines(good_vessel, [pts], isClosed=False, color=255, thickness=40, lineType=cv2.LINE_AA) # 用粗线条画出管状

    # 创建一个"差"的血管边界: 不连续，有锯齿，粗细不均
    bad_vessel = np.zeros((300, 500), dtype=np.uint8)
    # 第一段，较粗
    cv2.line(bad_vessel, (50, 150), (180, 130), 255, 30)
    # 第二段，有锯齿，很细
    pts_bad = np.array([[220,140], [250,180], [280,130], [310,190], [340,120]], np.int32)
    cv2.polylines(bad_vessel, [pts_bad], isClosed=False, color=255, thickness=10)
    # 一些噪声
    cv2.circle(bad_vessel, (400, 80), 15, 255, -1)


    # 评分
    good_score = score_vessel_boundary(good_vessel, scoring_weights)
    bad_score = score_vessel_boundary(bad_vessel, scoring_weights)
    
    print("--- '好'血管边界的评分结果 ---")
    print(f"最终得分: {good_score['final_score']:.4f}")
    print(f"分项得分: {good_score['sub_scores']}")
    print(f"详细数据: {good_score['details']}\n")

    print("--- '差'血管边界的评分结果 ---")
    print(f"最终得分: {bad_score['final_score']:.4f}")
    print(f"分项得分: {bad_score['sub_scores']}")
    print(f"详细数据: {bad_score['details']}")
