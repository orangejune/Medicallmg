import cv2
import numpy as np
import os
import random
import networkx as nx

# Folder paths (modify as needed)
input_folder = r"D:\Cardio\Videos\Media1"
output_folder = r"D:\Cardio\Videos\Media1out"

os.makedirs(output_folder, exist_ok=True)

# Define ROI coordinates (top-left and bottom-right)
roi_top_left = (110, 100)
roi_bottom_right = (640, 500)

close_gap_threshold = 10  # Maximum distance to close gaps between polylines


def simplified_skeletonize(img):
    """Extracts a simplified skeleton using controlled erosion and thinning."""
    # Apply stronger Gaussian Blur to smooth structures
    # img = cv2.GaussianBlur(img, (5, 5), 0)
    #
    # # Apply morphological opening to remove fine details
    # kernel = np.ones((3, 3), np.uint8)
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # Use cv2 thinning method if available
    if hasattr(cv2.ximgproc, 'thinning'):
        skeleton = cv2.ximgproc.thinning(img)
    else:
        # Controlled skeletonization using erosion
        size = np.array(img.shape)
        skeleton = np.zeros(size, np.uint8)
        eroded = np.copy(img)
        temp = np.zeros(size, np.uint8)

        while cv2.countNonZero(eroded) > 0:
            eroded = cv2.erode(eroded, kernel)
            temp = cv2.dilate(eroded, kernel)
            temp = cv2.subtract(img, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)

    return skeleton


# def convert_skeleton_to_pixel_map(skeleton):
#     """Converts the skeletonized image into a pixel-based representation."""
#     pixel_map = np.zeros_like(skeleton)
#     contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     cv2.drawContours(pixel_map, contours, -1, 255, thickness=1)
#     return pixel_map


# def extract_polylines_from_pixels(pixel_map):
#     """Extracts polylines from a cleaned pixel representation."""
#     contours, _ = cv2.findContours(pixel_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     return contours

def convert_skeleton_to_polylines(skeleton, min_branch_length=0):
    """Converts a cleaned skeleton (2D array) directly into vector polylines, removing short branches."""
    height, width = skeleton.shape
    visited = np.zeros((height, width), dtype=bool)
    polylines = []

    # Define movement directions (8-connectivity)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def count_neighbors(x, y):
        """Counts the number of foreground neighbors of a given pixel."""
        count = 0
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < height and 0 <= ny < width and skeleton[nx, ny] > 0:
                count += 1
        return count

    def trace_polyline(start):
        """Traces a single connected polyline from a starting pixel, ensuring uniqueness."""
        path = []
        stack = [start]

        while stack:
            x, y = stack.pop()
            if visited[x, y]:
                continue
            visited[x, y] = True
            path.append((y, x))  # Store as (col, row) for OpenCV format

            # Explore neighbors
            next_steps = []
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < height and 0 <= ny < width and skeleton[nx, ny] > 0 and not visited[nx, ny]:
                    next_steps.append((nx, ny))

            if len(next_steps) == 1:
                stack.append(next_steps[0])

        return np.array(path, dtype=np.int32) if len(path) > min_branch_length else None

    # Remove short branches before tracing
    for i in range(height):
        for j in range(width):
            if skeleton[i, j] > 0 and count_neighbors(i, j) <= 1:
                skeleton[i, j] = 0  # Remove short dangles

    # Iterate through the cleaned skeleton
    for i in range(height):
        for j in range(width):
            if skeleton[i, j] > 0 and not visited[i, j]:
                polyline = trace_polyline((i, j))
                if polyline is not None:
                    polylines.append(polyline)

    return polylines


def merge_nearby_polylines(polylines, merge_threshold=10):
    """
    Merges polylines that have endpoints close to each other.
    - merge_threshold: Maximum Euclidean distance to merge polylines.
    """
    merged_polylines = []
    used = set()  # Track merged polylines

    def distance(pt1, pt2):
        """Compute Euclidean distance between two points."""
        return np.linalg.norm(np.array(pt1) - np.array(pt2))

    for i, poly1 in enumerate(polylines):
        if i in used:
            continue
        merged_poly = poly1.tolist()
        used.add(i)

        for j, poly2 in enumerate(polylines):
            if j in used or i == j:
                continue

            # Check if poly1's end is close to poly2's start
            if distance(merged_poly[-1], poly2[0]) < merge_threshold:
                merged_poly.extend(poly2.tolist())  # Merge them
                used.add(j)

            # Check if poly1's start is close to poly2's end (reverse needed)
            elif distance(merged_poly[0], poly2[-1]) < merge_threshold:
                merged_poly = poly2.tolist() + merged_poly  # Merge in reverse order
                used.add(j)

        merged_polylines.append(np.array(merged_poly, dtype=np.int32))

    return merged_polylines


def visualize_polylines(image, polylines):
    """Draws polylines with different colors."""
    img_with_polylines = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for polyline in polylines:
        if len(polyline) < 2:
            continue  # Ignore very short segments

        # Assign a unique random color
        color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        cv2.polylines(img_with_polylines, [polyline], isClosed=False, color=color, thickness=1)

    return img_with_polylines


for filename in os.listdir(input_folder):
    if filename.lower().endswith(".jpg"):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        base_name = os.path.splitext(filename)[0]  # Extract filename without extension

        # Convert to grayscale and apply ROI
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        roi_gray = gray[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]]

        # Apply Gaussian Blur (5x5) to remove small details
        blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)

        # **Enhanced Thresholding: CLAHE + Otsu's**
        clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(16, 16))
        enhanced = clahe.apply(blurred)

        _, bright_regions = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological closing to merge fragmented strokes
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(bright_regions, cv2.MORPH_CLOSE, kernel)

        # Extract simplified skeletonized centerline
        skeleton = simplified_skeletonize(closed)

        # Convert skeleton to a cleaned pixel map
        # pixel_map = convert_skeleton_to_pixel_map(skeleton)

        # Extract polylines from the cleaned pixel map
        polylines = convert_skeleton_to_polylines(skeleton)
        polylines = merge_nearby_polylines(polylines, merge_threshold=10)  # Adjust threshold as needed

        # **Visualize extracted polylines**
        img_with_polylines = visualize_polylines(roi_gray, polylines)

        # **Save visualization of extracted polylines**
        polylines_output_path = os.path.join(output_folder, f"{base_name}_final_polylines.jpg")
        cv2.imwrite(polylines_output_path, img_with_polylines)

        # **Save simplified skeleton image**
        skeleton_output_path = os.path.join(output_folder, f"{base_name}_simplified_skeleton.jpg")
        cv2.imwrite(skeleton_output_path, skeleton)

print("Processing complete. Check the 'output' folder for results.")
