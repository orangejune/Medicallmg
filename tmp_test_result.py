import pandas as pd
import os
import shutil
import re

# --- 1. 配置信息 ---
# CSV 文件路径
CSV_FILE = 'test_image_statistics.csv'
# 包含视频名的列名
COLUMN_NAME = '视频名列表'
# 源图片文件夹
SOURCE_DIR = '0703_gen/predicted_outputs'
# 目标结果文件夹
DEST_DIR = '0703_gen/test_result'

def main():
    """
    主执行函数
    """
    print("--- 任务开始 ---")

    # --- 2. 检查源文件和文件夹是否存在 ---
    if not os.path.exists(CSV_FILE):
        print(f"错误：CSV文件 '{CSV_FILE}' 不存在，请检查文件名或路径。")
        return
    if not os.path.isdir(SOURCE_DIR):
        print(f"错误：源文件夹 '{SOURCE_DIR}' 不存在，请检查文件夹名称。")
        return

    # --- 3. 从CSV中读取并提取唯一的Image标识符 ---
    try:
        df = pd.read_csv(CSV_FILE)
        if COLUMN_NAME not in df.columns:
            print(f"错误：在CSV文件中找不到名为 '{COLUMN_NAME}' 的列。")
            return
            
        print(f"成功读取 '{CSV_FILE}'。正在从 '{COLUMN_NAME}' 列中提取标识符...")

        # 使用正则表达式 'Image\d+' 来匹配 'Image' 跟着一串数字的模式
        # 使用 set 来自动去重，提高效率
        image_identifiers = set()
        for video_name in df[COLUMN_NAME].dropna(): # dropna() 避免处理空值
            for i in video_name.split(';'):
                # str() 确保数据是字符串类型
                match = re.search(r'Image\d+', str(i))
                if match:
                    image_identifiers.add(match.group(0))

        if not image_identifiers:
            print("警告：在指定的列中没有找到任何 'ImageXXX' 格式的标识符。")
            return

        print(f"成功提取 {len(image_identifiers)} 个唯一的标识符。")
        # print("标识符列表:", sorted(list(image_identifiers))) # 如果需要，可以取消此行注释查看所有标识符

    except Exception as e:
        print(f"读取或处理CSV文件时出错: {e}")
        return

    # --- 4. 创建目标文件夹 ---
    os.makedirs(DEST_DIR, exist_ok=True)
    print(f"目标文件夹 '{DEST_DIR}' 已准备就绪。")

    # --- 5. 遍历源文件夹，查找并复制文件 ---
    print(f"正在 '{SOURCE_DIR}' 中搜索匹配的文件并复制到 '{DEST_DIR}'...")
    
    copied_count = 0
    total_files_in_source = 0

    for filename in os.listdir(SOURCE_DIR):
        # 构造完整的文件路径
        source_path = os.path.join(SOURCE_DIR, filename)

        # 确保它是一个文件而不是文件夹
        if os.path.isfile(source_path):
            total_files_in_source += 1
            # 检查文件名是否包含任何一个我们找到的标识符
            for identifier in image_identifiers:
                if identifier in filename:
                    # 如果匹配成功，构造目标路径并复制文件
                    destination_path = os.path.join(DEST_DIR, filename)
                    print(f"  - 匹配成功: 正在复制 '{filename}'...")
                    shutil.copy2(source_path, destination_path) # copy2 会同时复制元数据
                    copied_count += 1
                    # 找到一个匹配后就跳出内层循环，处理下一个文件
                    break
    
    print("\n--- 任务完成 ---")
    print(f"已检查源文件夹中的 {total_files_in_source} 个项目。")
    print(f"共复制了 {copied_count} 个文件到 '{DEST_DIR}' 文件夹。")


if __name__ == "__main__":
    main()