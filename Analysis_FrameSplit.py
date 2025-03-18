import cv2
import os


def extract_frames(video_path, output_folder, frame_interval=1):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames to {output_folder}")


# Example usage
extract_frames(r"D:\Cardio\Videos\Media1.mp4", r"D:\Cardio\Videos\Media1", frame_interval=1)
