import os
import re
from collections import defaultdict
import pandas as pd

# --- 1. 配置信息 ---
# 输入文件名
INPUT_FILE = '0703_gen/misclassified_sorted.txt'

def create_misclassification_matrix():
    """
    主分析函数：读取、解析并以表格形式统计错分结果。
    """
    print("--- 开始分析错分文件 ---")

    # --- 2. 检查文件是否存在 ---
    if not os.path.exists(INPUT_FILE):
        print(f"错误：文件 '{INPUT_FILE}' 不存在。请确保脚本与该文件在同一目录下。")
        return

    # --- 3. 准备数据结构 ---
    # 结构: { '真实类别': { '错分类别': 次数 } }
    misclassification_counts = defaultdict(lambda: defaultdict(int))
    # 收集所有出现过的类别，用于构建表格的行列索引
    all_true_labels = set()
    all_pred_labels = set()
    processed_lines = 0

    # --- 4. 读取并解析文件 ---
    print(f"正在读取并解析文件: '{INPUT_FILE}'...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                match = re.search(r'true=([^,]+),\s*pred=([^,]+),', line)
                if match:
                    true_label = match.group(1).strip()
                    pred_label = match.group(2).strip()

                    # 累加计数
                    misclassification_counts[true_label][pred_label] += 1
                    # 记录所有类别
                    all_true_labels.add(true_label)
                    all_pred_labels.add(pred_label)
                    processed_lines += 1
                else:
                    print(f"警告：无法解析行: '{line}'")

    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return
        
    if processed_lines == 0:
        print("\n文件中没有找到任何有效的错分记录。")
        return

    # --- 5. 创建并填充表格 (DataFrame) ---
    print("正在生成错分矩阵...")
    
    # 排序以保证表格行列顺序固定
    sorted_true = sorted(list(all_true_labels))
    sorted_pred = sorted(list(all_pred_labels))
    
    # 创建一个以0填充的DataFrame，行列为所有出现过的类别
    df = pd.DataFrame(0, index=sorted_pred, columns=sorted_true)

    # 遍历统计结果，填充DataFrame
    for true_label, pred_dict in misclassification_counts.items():
        for pred_label, count in pred_dict.items():
            # 使用 .loc 进行精确赋值
            df.loc[pred_label, true_label] = count

    # --- 6. 格式化并输出结果 ---
    print("\n" + "="*50)
    print("                错分矩阵统计结果")
    print("="*50)
    print("\n说明：")
    print(" - 表格的【列】代表样本的【真实类别】")
    print(" - 表格的【行】代表模型【错误的预测类别】")
    print(" - 数字表示错分的次数\n")

    # 使用 to_string() 保证即使行列很多也能完整显示
    print(df.to_string())
    
    # --- 7. 输出总计 ---
    print("\n" + "="*50)
    print("                按真实类别统计总错分数")
    print("="*50)
    # 计算每一列（真实类别）的总错分数
    total_errors_per_class = df.sum(axis=0)
    total_errors_per_class.name = "总错分数"
    print(total_errors_per_class.to_string())
    
    print(f"\n\n--- 分析完成，共处理了 {processed_lines} 条错分记录。 ---")


if __name__ == "__main__":
    # 为了运行此脚本，请确保已安装 pandas: pip install pandas
    create_misclassification_matrix()