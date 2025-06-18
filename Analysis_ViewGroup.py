# view_classifier.py — ML-BASED GROUPING OF LINE FEATURES with ROI Support and ROI Visualization

import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler


def create_center_weight_mask(size=64, sigma=0.4):
    ax = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(ax, ax)
    gaussian = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return gaussian / gaussian.max()


def extract_binary_line_images(video_path, save_dir=None, roi=None):
    frames = []
    cap = cv2.VideoCapture(video_path)
    idx = 0
    weight_mask = create_center_weight_mask()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        original_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if roi is not None:
            x, y, w, h = roi
            gray = gray[y:y + h, x:x + w]
            cv2.rectangle(original_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        edges = cv2.Canny(gray, threshold1=50, threshold2=150)
        resized = cv2.resize(edges, (64, 64)).astype(np.float32) / 255.0
        weighted = resized * weight_mask
        frames.append(weighted.flatten())

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            edge_filename = os.path.join(save_dir, f"line_{idx:04d}.png")
            cv2.imwrite(edge_filename, edges)
            frame_filename = os.path.join(save_dir, f"roi_frame_{idx:04d}.png")
            cv2.imwrite(frame_filename, original_frame)
        idx += 1
    cap.release()
    return np.array(frames)


class MLLineViewGroupSplitter:
    def __init__(self, n_clusters=10, min_group_size=3):
        self.n_clusters = n_clusters
        self.min_group_size = min_group_size

    def split(self, line_features):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(line_features)
        clustering = AgglomerativeClustering(n_clusters=self.n_clusters)
        labels = clustering.fit_predict(X_scaled)

        groups = []
        current_label = labels[0]
        start = 0
        for i in range(1, len(labels)):
            if labels[i] != current_label:
                if i - start >= self.min_group_size:
                    groups.append((start, i - 1))
                start = i
                current_label = labels[i]
        if len(labels) - start >= self.min_group_size:
            groups.append((start, len(labels) - 1))
        return groups


def save_grouped_line_frames(video_path, groups, output_base="line_grouped_frames"):
    os.makedirs(output_base, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    all_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()

    for i, (start, end) in enumerate(groups):
        group_dir = os.path.join(output_base, f"group_{i:02d}")
        os.makedirs(group_dir, exist_ok=True)
        for j in range(start, end + 1):
            out_path = os.path.join(group_dir, f"frame_{j:04d}.png")
            cv2.imwrite(out_path, all_frames[j])


def main():
    video_path = "heart_cycles/Media1.mp4"
    output_base = "line_grouped_frames"
    line_image_dir = "line_edges"
    roi = (100, 50, 500, 350)  # (x, y, w, h) region of interest

    print("Extracting line features from ROI and saving edge + ROI overlay images...")
    line_features = extract_binary_line_images(video_path, save_dir=line_image_dir, roi=roi)

    print("Clustering line-based features using ML...")
    splitter = MLLineViewGroupSplitter(n_clusters=10, min_group_size=3)
    groups = splitter.split(line_features)

    print("Saving grouped original frames...")
    save_grouped_line_frames(video_path, groups, output_base)

    print("Done. View groups saved as:")
    for i, (start, end) in enumerate(groups):
        print(f"Group {i}: frames {start}–{end} → {output_base}/group_{i:02d}/")


if __name__ == "__main__":
    main()


##todo: coraser extraction maybe? no need to determine the view type ; also the grouping seems to be too fine level..
