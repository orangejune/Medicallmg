import os

def remove_lines_starting_with_zero(directory_path):
    """
    读取指定文件夹下所有txt文件，删除其中以数字0开头的行，并保存。

    Args:
        directory_path (str): 包含txt文件的文件夹路径。
    """
    if not os.path.isdir(directory_path):
        print(f"错误：路径 '{directory_path}' 不是一个有效的文件夹。")
        return

    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            print(f"正在处理文件: {file_path}")

            lines_to_keep = []
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        # strip() 移除行首尾的空白字符，包括换行符
                        stripped_line = line.strip()
                        # 检查过滤条件：行不为空，并且第一个字符不是 '0'
                        if stripped_line and not stripped_line.startswith('2 '):
                            lines_to_keep.append(line) # 保留原始的行，包含换行符
                        if stripped_line and stripped_line.startswith('2 '):
                            lines_to_keep.append(f'0 {line[2:]}') # 保留原始的行，包含换行符
                            print(line, f'0 {line[2:]}')
                        # else:
                            # print(f"  删除行: {line.strip()}") # 可选：打印被删除的行

                # 检查是否有需要保存的内容，避免写入空文件（除非原文件就是空）
                if lines_to_keep or not os.path.getsize(file_path) == 0:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.writelines(lines_to_keep)
                    print(f"  文件 '{filename}' 处理完成。")
                else:
                    print(f"  文件 '{filename}' 在过滤后变为空，未写入。")

            except Exception as e:
                print(f"  处理文件 '{filename}' 时发生错误: {e}")

# --- 使用示例 ---
if __name__ == "__main__":
    # 请将 YOUR_FOLDER_PATH 替换为你实际的文件夹路径
    # 例如: "/path/to/your/txt/files" (Linux/macOS)
    # 或: "C:/Users/YourUser/Documents/TextFiles" (Windows)
    total_path=r'C:\Users\june.lin\Desktop\medicallmg\data\ROI-label\test'
    for filename in os.listdir(total_path):
        folder_to_process = os.path.join(total_path, filename)

        # 如果你想让脚本询问文件夹路径，可以取消下面的注释
        # folder_to_process = input("请输入包含txt文件的文件夹路径: ")

        remove_lines_starting_with_zero(folder_to_process)
    print("\n所有txt文件处理完毕。")