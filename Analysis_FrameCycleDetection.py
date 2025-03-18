import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def detect_heartbeat_cycles(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()

    if not ret:
        print("Error: Cannot read video.")
        return []

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    motion_intensity = []
    frame_numbers = []

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        motion_magnitude = np.linalg.norm(flow, axis=2).mean()
        motion_intensity.append(motion_magnitude)
        frame_numbers.append(frame_count)

        prev_gray = gray
        frame_count += 1

    cap.release()

    # Normalize motion data
    motion_intensity = np.array(motion_intensity)
    motion_intensity = (motion_intensity - np.min(motion_intensity)) / (
                np.max(motion_intensity) - np.min(motion_intensity))

    # Detect peaks (heart contraction points)
    peaks, _ = find_peaks(motion_intensity, height=0.5, distance=20)  # Adjust height & distance as needed

    # Get start and end frames for each cycle
    cycle_frames = [(peaks[i], peaks[i + 1]) for i in range(len(peaks) - 1)]

    # Print detected cycle frames
    print("\nDetected Heartbeat Cycles (Start Frame, End Frame):")
    for i, (start, end) in enumerate(cycle_frames):
        print(f"Cycle {i + 1}: Start = {start}, End = {end}")

    # Plot motion intensity with detected cycles
    plt.figure(figsize=(12, 6))
    plt.plot(frame_numbers, motion_intensity, label="Motion Intensity")
    plt.plot(peaks, motion_intensity[peaks], "ro", label="Detected Peaks (Contractions)")

    # Annotate peak frame numbers
    for peak in peaks:
        plt.text(peak, motion_intensity[peak], str(peak), fontsize=9, color="red")

    plt.xlabel("Frame Number")
    plt.ylabel("Normalized Motion")
    plt.title("Detected Heartbeat Cycles")
    plt.legend()
    plt.show()

    return cycle_frames, peaks




import os

def split_video_by_cycles(video_path, cycle_frames, output_folder="heart_cycles"):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get video FPS

    for i, (start_frame, end_frame) in enumerate(cycle_frames):
        start_time = start_frame / fps
        end_time = end_frame / fps
        output_file = os.path.join(output_folder, f"cycle_{i+1}.mp4")

        command = f'ffmpeg -i "{video_path}" -ss {start_time:.2f} -to {end_time:.2f} -c:v libx264 -c:a aac "{output_file}"'
        os.system(command)

        print(f"Saved {output_file} (Start Frame: {start_frame}, End Frame: {end_frame})")

    cap.release()
    print("\nVideo segmentation complete.")

# Example usage
video_path = r"D:\Cardio\Videos\Media2.mp4"
cycle_frames, peak_frames = detect_heartbeat_cycles(video_path)

# Print peak frames separately
print("\nDetected Peak Frames (Contractions):", peak_frames)

# Example usage
# split_video_by_cycles(video_path, cycle_frames)

