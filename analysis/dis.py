import math

def calculate_distance(coord1, coord2):
    """
    计算两个坐标点之间的直线距离
    
    参数:
    coord1: 第一个坐标点的元组 (x1, y1)
    coord2: 第二个坐标点的元组 (x2, y2)
    
    返回:
    两点之间的距离
    """
    x1, y1 = coord1
    x2, y2 = coord2
    
    # 使用欧几里得距离公式
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

# 示例使用
point1 = (438, 366)
point2 = (453, 396)

distance = calculate_distance(point1, point2)
print(f"点 {point1} 和点 {point2} 之间的距离是: {distance:.2f}")