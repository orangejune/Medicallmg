import os
import json
import glob
'''
去掉前四个中文名，去掉json中imagePath的前四个中文字符
'''
def batch_process_json_files(folder_path):
    """
    批量处理指定文件夹下的所有 JSON 文件，
    修改 imagePath 字段，移除其值的前4个字符。
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
                    # 从第4个字符开始截取到末尾，即删除前4个字符
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

def batch_rename_files(folder_path):
    """
    批量重命名指定文件夹下的所有文件，删除每个文件名的前4个字符（去掉中文字符）。

    :param folder_path: 目标文件夹的路径
    """
    # 检查路径是否存在
    if not os.path.isdir(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在。")
        return

    print(f"开始处理文件夹: {folder_path}\n")
    
    # 遍历文件夹中的所有条目
    for filename in os.listdir(folder_path):
        old_path = os.path.join(folder_path, filename)

        # 确保只处理文件，跳过子文件夹
        if os.path.isfile(old_path):
            # 检查文件名长度是否足够
            if len(filename) > 4:
                # 构建新文件名 (从第5个字符开始截取)
                new_filename = filename[4:]
                new_path = os.path.join(folder_path, new_filename)

                # 检查新文件名是否已存在，防止覆盖
                if os.path.exists(new_path):
                    print(f"跳过: 新文件名 '{new_filename}' 已存在。 (原文件: '{filename}')")
                    continue
                
                # 执行重命名
                try:
                    os.rename(old_path, new_path)
                    print(f"成功: '{filename}'  ->  '{new_filename}'")
                except OSError as e:
                    print(f"错误: 重命名 '{filename}' 时发生错误: {e}")

            else:
                # 如果文件名长度不足4，则跳过
                print(f"跳过: 文件名 '{filename}' 长度不足4个字符。")
    
    print("\n处理完成。")


if __name__ == "__main__":
    target_folder = r"C:\Users\june.lin\Desktop\medicallmg\dataset\trained_sample" 
    # 去掉文件名前四个中文字符
    batch_rename_files(target_folder)
    # 去掉json中imagePath的前四个中文字符
    # batch_process_json_files(target_folder)