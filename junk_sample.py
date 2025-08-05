import os
import re
import random
import shutil
from collections import defaultdict

def parse_filename(filename):
    """从文件名中解析出视频名和帧号。"""
    match = re.match(r'(.+)_frame_(\d+)\.jpg', filename)
    if match:
        video_name = match.group(1)
        frame_number = int(match.group(2))
        return video_name, frame_number
    return None, None

def find_frame_chunks(file_paths):
    """将文件按视频分组，并识别出每个视频内的连续帧块。"""
    # (此函数无需改动，保持原样)
    videos = defaultdict(list)
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        video_name, frame_number = parse_filename(filename)
        if video_name:
            videos[video_name].append({'path': file_path, 'frame': frame_number})

    video_chunks = defaultdict(list)
    for video_name, frames in videos.items():
        sorted_frames = sorted(frames, key=lambda x: x['frame'])
        if not sorted_frames:
            continue
        current_chunk = [sorted_frames[0]]
        for i in range(1, len(sorted_frames)):
            if sorted_frames[i]['frame'] - sorted_frames[i-1]['frame'] > 1:
                video_chunks[video_name].append(current_chunk)
                current_chunk = []
            current_chunk.append(sorted_frames[i])
        video_chunks[video_name].append(current_chunk)
    return video_chunks

def get_candidate_samples(video_chunks, min_chunk_size=30):
    """从每个大于等于min_chunk_size的帧块的中间位置选择一个候选样本。"""
    # (此函数无需改动，保持原样)
    candidates = []
    print(f"\nIdentifying candidate frames from chunks of at least {min_chunk_size} frames...")
    for video_name, chunks in video_chunks.items():
        video_candidates = []
        for chunk in chunks:
            if len(chunk) >= min_chunk_size:
                middle_index = len(chunk) // 2
                candidate_frame = chunk[middle_index]
                video_candidates.append(candidate_frame)
        if video_candidates:
            candidates.append({'video_name': video_name, 'samples': video_candidates})
            print(f"  - Found {len(video_candidates)} candidates for video '{video_name}'")
    return candidates

def select_final_samples(candidates, total_samples):
    """从候选集中选取最终的样本，优先保证视频覆盖率。"""
    # (此函数无需改动，保持原样)
    final_samples = []
    
    if not candidates:
        print("Error: No valid candidates found. Check your `min_chunk_size` or data.")
        return []

    candidate_pool = {c['video_name']: list(c['samples']) for c in candidates}
    
    print(f"\nPhase 1: Ensuring coverage for all {len(candidate_pool)} videos...")
    videos_with_candidates = list(candidate_pool.keys())
    random.shuffle(videos_with_candidates)

    for video_name in videos_with_candidates:
        if len(final_samples) >= total_samples:
            break
        if candidate_pool[video_name]:
            sample = random.choice(candidate_pool[video_name])
            final_samples.append(sample['path'])
            candidate_pool[video_name].remove(sample)
    
    print(f"Phase 1 complete. Collected {len(final_samples)} samples.")

    print(f"\nPhase 2: Filling remaining spots to reach {total_samples} total samples...")
    while len(final_samples) < total_samples:
        available_videos = [name for name, samples in candidate_pool.items() if samples]
        if not available_videos:
            print(f"Warning: Candidate pool exhausted. Collected {len(final_samples)} samples, less than the target {total_samples}.")
            break
            
        video_to_sample = random.choice(available_videos)
        sample = random.choice(candidate_pool[video_to_sample])
        final_samples.append(sample['path'])
        candidate_pool[video_to_sample].remove(sample)

    print(f"Phase 2 complete. Final sample count: {len(final_samples)}")
    return final_samples


# --- 主程序 ---
if __name__ == "__main__":
    # 1. 设置参数
    negative_samples_dir = "D:/medicallmg/冠脉病例GE-2/frames_flat"   # <--- 修改为你的负样本文件夹路径
    destination_dir = "./heart_cycles/test/junk"  # <--- 2. 新增：定义目标文件夹路径
    total_samples_to_select = 300
    min_continuous_chunk_size = 20

    # 2. 获取所有负样本文件路径
    try:
        all_files = [os.path.join(negative_samples_dir, f) for f in os.listdir(negative_samples_dir) if f.endswith('.jpg')]
        if not all_files:
            print(f"Error: No .jpg files found in '{negative_samples_dir}'. Please check the path.")
            exit()
        print(f"Found {len(all_files)} total negative sample files.")
    except FileNotFoundError:
        print(f"Error: The directory '{negative_samples_dir}' was not found.")
        exit()

    # 3. 执行采样流程
    frame_chunks_by_video = find_frame_chunks(all_files)
    candidate_frames = get_candidate_samples(frame_chunks_by_video, min_chunk_size=min_continuous_chunk_size)
    selected_files = select_final_samples(candidate_frames, total_samples=total_samples_to_select)

    # 4. 输出结果并复制文件 <--- 这里是主要修改部分
    if selected_files:
        print("\n--- Copying Selected Sample Files ---")
        
        # 4a. 创建目标文件夹（如果不存在）
        os.makedirs(destination_dir, exist_ok=True)
        print(f"Files will be copied to: '{destination_dir}'")
        
        # 4b. 遍历并复制文件
        copied_count = 0
        for src_path in sorted(selected_files):
            try:
                # 构建目标路径，文件名保持不变
                filename = os.path.basename(src_path)
                dest_path = os.path.join(destination_dir, filename)
                
                # 执行复制
                shutil.copy(src_path, dest_path)
                print(f"  Copied: {filename}")
                copied_count += 1
            except Exception as e:
                print(f"  Error copying {os.path.basename(src_path)}: {e}")

        print(f"\nSuccessfully copied {copied_count} out of {len(selected_files)} selected files.")
        
        # 4c. (可选) 将选中的源文件路径列表保存到文件
        output_file = "selected_negative_samples_source_paths.txt"
        with open(output_file, "w", encoding="utf-8") as f_out:
            for file_path in sorted(selected_files):
                f_out.write(f"{file_path}\n")
        print(f"The list of original source paths has been saved to '{output_file}'")
    else:
        print("\nNo files were selected, so no files were copied.")