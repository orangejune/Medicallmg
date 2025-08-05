import os
import json
import glob

# --- 配置区 ---
# 请将此路径替换为你的 JSON 文件所在的文件夹路径
# 注意：Windows 路径中建议使用正斜杠 '/' 或者双反斜杠 '\\'
json_folder_path = r'C:\Users\june.lin\Desktop\medicallmg\dataset\trained_sample'
# --- 配置区结束 ---

def batch_process_json_files(folder_path):
    """
    批量处理指定文件夹下的所有 JSON 文件，
    修改 imagePath 字段，移除其值的前8个字符。
    """
    # 1. 检查路径是否存在
    if not os.path.isdir(folder_path):
        print(f"错误：文件夹路径不存在 -> {folder_path}")
        return

    # 2. 查找所有 .json 文件
    # 使用 recursive=True 可以查找子文件夹中的文件，如果不需要可以去掉
    json_files = glob.glob(os.path.join(folder_path, '**/*.json'), recursive=True)

    if not json_files:
        print(f"在 '{folder_path}' 及其子目录中没有找到任何 .json 文件。")
        return

    print(f"总共找到 {len(json_files)} 个 JSON 文件。开始处理...")
    processed_count = 0
    skipped_count = 0

    # 3. 循环处理每个文件
    for file_path in json_files:
        try:
            # --- 智能读取文件，兼容 UTF-8 和 GBK 编码 ---
            data = None
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except UnicodeDecodeError:
                # 如果 UTF-8 解码失败，尝试使用 GBK（Windows 默认编码）
                with open(file_path, 'r', encoding='gbk') as f:
                    data = json.load(f)
            
            # --- 核心修改逻辑 ---
            if 'imagePath' in data and isinstance(data['imagePath'], str):
                original_path = data['imagePath']
                
                # 检查长度是否足够，避免切片错误
                if len(original_path) >= 4:
                    # 从第8个字符开始截取到末尾，即删除前4个字符
                    modified_path = original_path[4:]
                    data['imagePath'] = modified_path

                    # --- 以标准的 UTF-8 格式写回文件 ---
                    # ensure_ascii=False 确保中文字符能正确显示而非编码
                    # indent=2 使 JSON 文件格式化，易于阅读
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    
                    print(f"✅ 处理成功: {os.path.basename(file_path)}")
                    print(f"   - 原路径: {original_path}")
                    print(f"   - 新路径: {modified_path}")
                    processed_count += 1
                else:
                    print(f"⚠️ 跳过(路径太短): {os.path.basename(file_path)}")
                    skipped_count += 1
            else:
                print(f"⚠️ 跳过(无imagePath): {os.path.basename(file_path)}")
                skipped_count += 1

        except Exception as e:
            print(f"❌ 处理失败: {os.path.basename(file_path)}，错误: {e}")
            skipped_count += 1
            
    print("\n--- 处理完毕 ---")
    print(f"成功处理: {processed_count} 个文件")
    print(f"跳过或失败: {skipped_count} 个文件")


# --- 程序入口 ---
if __name__ == "__main__":
    batch_process_json_files(json_folder_path)