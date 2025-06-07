import os
import cv2
from PIL import Image
import numpy as np


def extract_all_frames_with_unicode_paths(video_root, output_folder, video_extensions=(".mp4", ".avi", ".mov", ".mkv")):
    # Resolve absolute paths
    video_root = os.path.abspath(video_root)
    output_folder = os.path.abspath(output_folder)

    print("Searching videos in:", video_root)
    print("Saving frames to:", output_folder)
    os.makedirs(output_folder, exist_ok=True)

    for dirpath, _, filenames in os.walk(video_root):
        for filename in filenames:
            if filename.lower().endswith(video_extensions):
                video_path = os.path.join(dirpath, filename)

                # Construct a unique prefix using relative path
                relative_path = os.path.relpath(dirpath, video_root).replace(os.sep, "_")
                video_name = os.path.splitext(filename)[0]
                prefix = f"{relative_path}_{video_name}" if relative_path != "." else video_name

                print(f"▶️ Processing video: {video_path}")
                cap = cv2.VideoCapture(video_path)
                frame_idx = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Convert to RGB for Pillow
                    try:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(rgb_frame)
                        frame_filename = os.path.join(output_folder, f"{prefix}_frame_{frame_idx:05d}.jpg")
                        image.save(frame_filename)
                        print(f" Saved: {frame_filename}")
                    except Exception as e:
                        print(f" Failed to save {prefix}_frame_{frame_idx:05d}: {e}")

                    frame_idx += 1

                cap.release()
                print(f" Finished {prefix}, total frames: {frame_idx}")


if __name__ == "__main__":
    # Replace with your folder paths
    video_root_folder = r"Data\Cardio\GEData\冠脉病例GE-1"
    frame_output_folder = os.path.join(video_root_folder, "frames_flat")

    extract_all_frames_with_unicode_paths(video_root_folder, frame_output_folder)
