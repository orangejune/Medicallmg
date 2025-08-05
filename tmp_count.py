import os
import csv
from pathlib import Path

def analyze_image_folders(root_directory, output_csv_path):
    """
    统计指定根目录下各子文件夹中的图片数量、视频来源数量，并输出为CSV。

    Args:
        root_directory (str): 要扫描的根目录路径。
        output_csv_path (str): 输出CSV文件的保存路径。
    """
    print(f"开始扫描目录: {root_directory}")
    
    # 检查根目录是否存在
    root_path = Path(root_directory)
    if not root_path.is_dir():
        print(f"错误：目录 '{root_directory}' 不存在或不是一个有效的目录。")
        return

    # 用于存储最终结果的列表
    results = []

    # 遍历根目录下的所有子文件夹
    for folder_path in root_path.iterdir():
        if folder_path.is_dir():
            print(f"  正在处理文件夹: {folder_path.name}")
            
            image_count = 0
            # 使用 set 来自动去重，存储视频名
            video_names = set()

            # 查找所有.jpg图片文件 (可根据需要添加如.png, .jpeg等)
            for image_file in folder_path.glob('*.jpg'):
                image_count += 1
                
                # 从文件名中提取视频名
                # 文件名示例: 冠脉病例GE-1_20250312hechenyu_Image354_frame_00035.jpg
                # 我们需要 '_frame_' 之前的部分
                file_stem = image_file.stem  # 获取不带扩展名的文件名
                if '_frame_' in file_stem:
                    video_name = file_stem.rsplit('_frame_', 1)[0]
                    video_names.add(video_name)
            
            # 如果文件夹内有图片，则记录结果
            if image_count > 0:
                # --- 主要修正点在这里 ---
                # 将字典的 key 修改为与 fieldnames 匹配的中文
                results.append({
                    '文件夹路径': str(folder_path),
                    '类名':str(folder_path).split("\\")[-1],
                    '图片总数': image_count,
                    '视频总数': len(video_names),
                    '视频名列表': "; ".join(sorted(list(video_names)))
                })

    # 如果没有找到任何结果，则提前结束
    if not results:
        print("未在任何子文件夹中找到符合条件的图片。")
        return

    # 将结果写入CSV文件
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            # 使用 'utf-8-sig' 编码可以确保中文在Excel中正常显示
            # 定义CSV的列标题
            fieldnames = ['类名','图片总数', '视频总数', '文件夹路径', '视频名列表']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n统计完成！结果已保存到: {output_csv_path}")

    except IOError as e:
        print(f"\n错误：无法写入CSV文件。原因: {e}")


# --- 主程序入口 ---
if __name__ == "__main__":
    # --- 请在这里配置 ---
    
    # 1. 设置您要扫描的根目录路径
    #    Windows示例: "C:/Users/YourUser/Desktop/MyImages"
    #    macOS/Linux示例: "/home/user/my_images"
    ROOT_DIR = r"C:\Users\june.lin\Desktop\medicallmg\heart_cycles\test" 
    
    # 2. 设置输出的CSV文件名和路径
    OUTPUT_CSV = "test_image_statistics.csv"
    
    # --- 运行分析函数 ---
    analyze_image_folders(ROOT_DIR, OUTPUT_CSV)